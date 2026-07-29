import json
import time
import uuid
from copy import deepcopy
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional, Dict

from loguru import logger

from eff.agent.base import BaseAgent, is_valid_agent_history_message
from eff.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from eff.data_model.simulation import SimulationRun, TerminationReason
from eff.data_model.tasks import Task
from eff.environment.db import DB
from eff.environment.environment import Environment
from eff.user.base import BaseUser, is_valid_user_history_message
from eff.user.user_simulator import UserSimulator, UserState
from eff.utils.llm_utils import get_cost
from eff.utils.utils import format_time, get_now, DATA_DIR
from eff.config import DEFAULT_LANGUAGE


class Role(str, Enum):
    AGENT = "agent"
    USER = "user"
    ENV = "env"


def get_default_first_agent_message(language: str = None) -> AssistantMessage:
    """Get the default first agent message based on language"""
    if language is None:
        language = DEFAULT_LANGUAGE
    
    content = "你好，请问需要什么服务？" if language == "chinese" else "Hello, how can I help you?"
    return AssistantMessage(
        role="assistant", content=content, cost=0.0
    )


class Orchestrator:
    """
    Orchestrator for the simulation given a task.
    Passes messages between the Agent, User, and Environment.
    """

    def __init__(
        self,
        domain: str,
        agent: BaseAgent,
        user: BaseUser,
        environment: Environment,
        task: Task,
        max_steps: int = 100,
        max_errors: int = 10,
        seed: Optional[int] = None,
        solo_mode: bool = False,
        language: str = None,
        enable_dynamic_events: bool = False,
    ):
        self.domain = domain
        self.agent = agent
        self.user = user
        self.environment = environment
        self.task = task
        self.seed = seed
        self.solo_mode = solo_mode
        self.language = language
        self.enable_dynamic_events = enable_dynamic_events
        self.fired_event_ids: set = set()
        self.agent_state: Optional[Any] = None
        self.user_state: Optional[UserState] = None
        self.trajectory: list[Message] = []
        self.max_steps = max_steps
        self.max_errors = max_errors
        self.step_count = 0
        self.done = False
        self.termination_reason: Optional[TerminationReason] = None
        self.num_errors = 0
        self.from_role: Optional[Role] = None
        self.to_role: Optional[Role] = None
        self.message: Optional[Message] = None
        # ── 循环/浪费检测 ──
        # A. 同名同参数死循环（已实现）
        self._recent_tool_calls: list[str] = []
        self._max_consecutive_same_tool: int = 5
        # A2. 同 tool name 连续调用（不看 args）：拦截"同一工具堆 query 变体打转"
        # 例：连续 search_pois({"query":"X"}) → search_pois({"query":"X 武汉"}) → ...
        # （args 一直变绕过 A，但单一工具堆失败明显是 stuck）
        self._recent_tool_call_names: list[str] = []
        self._max_consecutive_same_tool_name: int = 8
        # B. Token 预算上限（防止上下文爆炸）
        self._cumulative_agent_tokens: int = 0
        self._max_token_budget: int = 700_000  # 兼顾 Anthropic tokenizer 中文 ~1.4× 系数（500K × 1.4 ≈ 700K）
        # C. 同名工具全局计数（防探索循环）
        self._tool_name_counts: dict = {}
        self._max_same_tool_total: int = 15  # 同一工具在整个 sim 中最多调 15 次
        # D. 连续非 user 消息计数（防 agent-env 乒乓）
        self._consecutive_non_user: int = 0
        self._max_consecutive_non_user: int = 20

    def initialize(self):
        """
        Initialize the orchestrator.
        - If the tasks specifies an initial state, use it to initialize the environment.
        - Initialize the agent and user states.
        - Send the first message (default message from the agent to the user).
        """
        message_history = (
            deepcopy(self.task.message_history)
            if self.task is not None and self.task.message_history is not None
            else []
        )
        for msg in message_history:
            msg.turn_idx = None

        message_history = self._add_timestamps(message_history)

        if self.seed is not None:
            self.agent.set_seed(self.seed)
            self.user.set_seed(self.seed)

        if len(message_history) > 0:
            self.validate_message_history(message_history)

            last_message = message_history[-1]
            if isinstance(last_message, AssistantMessage):
                self.from_role = Role.AGENT
                if not last_message.is_tool_call():
                    self.to_role = Role.USER
                else:
                    self.to_role = Role.ENV
                self.agent_state = self.agent.get_init_state(
                    message_history=[
                        msg
                        for msg in message_history
                        if is_valid_agent_history_message(msg)
                    ]
                )
                self.user_state = self.user.get_init_state(
                    message_history=[
                        msg
                        for msg in message_history[:-1]
                        if is_valid_user_history_message(msg)
                    ]
                )
                self.message = last_message
                if self.agent.is_stop(last_message):
                    self.done = True
                    self.termination_reason = TerminationReason.AGENT_STOP
            elif isinstance(last_message, UserMessage):
                self.from_role = Role.USER
                if not last_message.is_tool_call():
                    self.to_role = Role.AGENT
                else:
                    self.to_role = Role.ENV
                self.user_state = self.user.get_init_state(
                    message_history=[
                        msg
                        for msg in message_history
                        if is_valid_user_history_message(msg)
                    ]
                )
                self.agent_state = self.agent.get_init_state(
                    message_history=[
                        msg
                        for msg in message_history[:-1]
                        if is_valid_agent_history_message(msg)
                    ]
                )
                self.message = last_message
                self.done = UserSimulator.is_stop(last_message)
                if self.done:
                    self.termination_reason = TerminationReason.USER_STOP
            elif isinstance(last_message, ToolMessage):
                self.from_role = Role.ENV
                if last_message.requestor == "assistant":
                    self.to_role = Role.AGENT
                    self.agent_state = self.agent.get_init_state(
                        message_history=[
                            msg
                            for msg in message_history[:-1]
                            if is_valid_agent_history_message(msg)
                        ]
                    )
                    self.user_state = self.user.get_init_state(
                        message_history=[
                            msg
                            for msg in message_history
                            if is_valid_user_history_message(msg)
                        ]
                    )
                else:
                    self.to_role = Role.USER
                    self.agent_state = self.agent.get_init_state(
                        message_history=[
                            msg
                            for msg in message_history
                            if is_valid_agent_history_message(msg)
                        ]
                    )
                    self.user_state = self.user.get_init_state(
                        message_history=[
                            msg
                            for msg in message_history[:-1]
                            if is_valid_user_history_message(msg)
                        ]
                    )
                self.message = last_message
            else:
                raise ValueError(
                    f"Last message should be of type AssistantMessage, UserMessage, or ToolMessage, got {type(last_message)}"
                )
            self.trajectory = message_history

        else:
            self.agent_state = self.agent.get_init_state()
            self.user_state = self.user.get_init_state()
            first_message = deepcopy(get_default_first_agent_message(self.language))
            first_message.timestamp = get_now()

            if self.task is not None and self.task.initial_user_utterance:
                # Pre-inject the fixed first user message so it bypasses the user simulator
                user_message = UserMessage(role="user", content=self.task.initial_user_utterance)
                user_message.timestamp = get_now()
                self.trajectory = [first_message, user_message]
                self.message = user_message
                self.from_role = Role.USER
                self.to_role = Role.AGENT
            else:
                self.trajectory = [first_message]
                self.message = first_message
                self.from_role = Role.AGENT
                self.to_role = Role.USER

        # with open(DATA_DIR / "tools" / f"{self.domain}.json", 'w') as f:
        #     json.dump([tools.openai_schema for tools in self.agent.tools], f, indent=4, ensure_ascii=False)

    def run(self) -> SimulationRun:
        """
        Run the simulation.

        Returns:
            SimulationRun: The simulation run.
        """
        start_time = get_now()
        start = time.perf_counter()
        self.initialize()
        while not self.done:
            self.step()
            if self.step_count >= self.max_steps:
                self.done = True
                self.termination_reason = TerminationReason.MAX_STEPS
            if self.num_errors >= self.max_errors:
                self.done = True
                self.termination_reason = TerminationReason.TOO_MANY_ERRORS
            if self._check_tool_loop():
                self.done = True
                self.termination_reason = TerminationReason.TOOL_LOOP_DETECTED
                logger.warning(
                    f"Tool loop detected at step {self.step_count}: "
                    f"agent called the same tool with same args {self._max_consecutive_same_tool}+ times consecutively. "
                    f"Last call: {self._recent_tool_calls[-1] if self._recent_tool_calls else '?'}"
                )
            elif self._check_tool_name_loop():
                self.done = True
                self.termination_reason = TerminationReason.TOOL_LOOP_DETECTED
                stuck_name = self._recent_tool_call_names[-1] if self._recent_tool_call_names else "?"
                logger.warning(
                    f"Tool name loop detected at step {self.step_count}: "
                    f"agent called '{stuck_name}' {self._max_consecutive_same_tool_name}+ times consecutively "
                    f"(args varied but no other tools / no user turn in between)."
                )
            # B. Token 预算检查
            if self._cumulative_agent_tokens >= self._max_token_budget:
                self.done = True
                self.termination_reason = TerminationReason.TOKEN_BUDGET_EXCEEDED
                logger.warning(
                    f"Token budget exceeded at step {self.step_count}: "
                    f"{self._cumulative_agent_tokens:,} tokens >= {self._max_token_budget:,} budget"
                )
        duration = time.perf_counter() - start
        messages = self.get_trajectory()
        res = get_cost(messages)
        if res is None:
            agent_cost, user_cost = None, None
        else:
            agent_cost, user_cost = res

        simulation_run = SimulationRun(
            id=str(uuid.uuid4()),
            task_id=self.task.id,
            start_time=start_time,
            end_time=get_now(),
            duration=duration,
            termination_reason=self.termination_reason.value,
            reward_info=None,
            user_cost=user_cost,
            agent_cost=agent_cost,
            messages=messages,
            seed=self.seed,
            states=self.get_states(self.environment.tools.db, self.environment.tools.db.time)
        )
        return simulation_run

    def step(self):
        """
        Perform one step of the simulation.
        Sends self.message from self.from_role to self.to_role
        This can either be a message from agent to user/environment, environment to agent, or user to agent
        Updates self.trajectory
        """
        if self.done:
            raise ValueError("Simulation is done")
        logger.debug(
            f"Step {self.step_count}. Sending message from {self.from_role} to {self.to_role}"
        )
        logger.debug(
            f"Step {self.step_count}.\nFrom role: {self.from_role}\nTo role: {self.to_role}\nMessage: {self.message}"
        )
        if self.from_role in [Role.AGENT, Role.ENV] and self.to_role == Role.USER:
            user_msg, self.user_state = self.user.generate_next_message(
                self.message, self.user_state
            )
            user_msg.validate()
            if UserSimulator.is_stop(user_msg):
                self.done = True
                self.termination_reason = TerminationReason.USER_STOP
            self.trajectory.append(user_msg)
            self.message = user_msg
            self.from_role = Role.USER
            # D. user 消息重置非 user 计数
            self._consecutive_non_user = 0
            if user_msg.is_tool_call():
                self.to_role = Role.ENV
            else:
                self.to_role = Role.AGENT
        elif (
            self.from_role == Role.USER or self.from_role == Role.ENV
        ) and self.to_role == Role.AGENT:
            # If the message is from the user, inject user_id into the content sent to
            # the agent LLM without modifying the stored trajectory message.
            msg_for_agent = self.message
            if isinstance(self.message, UserMessage) and self.message.content is not None:
                user_id = (self.task.environment or {}).get("user_id") if self.task else None
                if user_id:
                    msg_for_agent = self.message.model_copy()
                    msg_for_agent.content = f"{self.message.content}\n[user_id: {user_id}]"

            # Retry up to 3 times if agent generates invalid message
            max_retries = 3
            retry_count = 0
            agent_msg = None
            original_agent_state = deepcopy(self.agent_state)  # Save original state

            while retry_count < max_retries:
                # Use a copy of the original state for each retry
                current_agent_state = deepcopy(original_agent_state)
                agent_msg, updated_agent_state = self.agent.generate_next_message(
                    msg_for_agent, current_agent_state
                )
                
                # Check if the message is valid (has text content or is tool call)
                if agent_msg.has_text_content() or agent_msg.is_tool_call():
                    # Only update the actual agent state if we get a valid message
                    self.agent_state = updated_agent_state
                    break
                
                retry_count += 1
                logger.warning(f"Agent generated invalid message (attempt {retry_count}/{max_retries}): {agent_msg}")
            
            # If all retries failed, terminate with INVALID_AGENT_MESSAGE
            if retry_count >= max_retries:
                self.done = True
                self.termination_reason = TerminationReason.INVALID_AGENT_MESSAGE
                return
            
            agent_msg.validate()
            if self.agent.is_stop(agent_msg):
                self.done = True
                self.termination_reason = TerminationReason.AGENT_STOP
            self.trajectory.append(agent_msg)
            self.message = agent_msg
            self.from_role = Role.AGENT
            # B. 累计 agent token
            if hasattr(agent_msg, 'usage') and agent_msg.usage:
                self._cumulative_agent_tokens += agent_msg.usage.get('prompt_tokens', 0)
                self._cumulative_agent_tokens += agent_msg.usage.get('completion_tokens', 0)
            # D. 连续非 user 计数
            self._consecutive_non_user += 1
            if agent_msg.is_tool_call():
                self.to_role = Role.ENV
                # A. 记录工具调用用于循环检测
                for tc in agent_msg.tool_calls:
                    if getattr(tc, "requestor", "assistant") == "assistant":
                        try:
                            canonical = f"{tc.name}|{json.dumps(tc.arguments or {}, sort_keys=True, ensure_ascii=False, default=str)}"
                        except Exception:
                            canonical = f"{tc.name}|{str(tc.arguments)}"
                        self._recent_tool_calls.append(canonical)
                        # A2. 同 tool name 连续序列（args 不计）
                        self._recent_tool_call_names.append(tc.name)
                        # C. 同名工具全局计数
                        self._tool_name_counts[tc.name] = self._tool_name_counts.get(tc.name, 0) + 1
                        if self._tool_name_counts[tc.name] >= self._max_same_tool_total:
                            logger.warning(
                                f"Tool overuse: {tc.name} called {self._tool_name_counts[tc.name]} times "
                                f"(limit={self._max_same_tool_total}). Not terminating but flagging."
                            )
            else:
                self.to_role = Role.USER
                # 非工具调用消息打断连续序列
                self._recent_tool_calls.clear()
                self._recent_tool_call_names.clear()
        elif self.from_role in [Role.AGENT, Role.USER] and self.to_role == Role.ENV:
            if not self.message.is_tool_call():
                raise ValueError("Agent or User should send tool call to environment")
            # Capture tool names before processing, for dynamic event triggers
            called_tool_names = [tc.name for tc in self.message.tool_calls]
            tool_msgs = []
            for tool_call in self.message.tool_calls:
                tool_msg = self.environment.get_response(tool_call)
                tool_msgs.append(tool_msg)
                # Increment error count if tool call failed
                if tool_msg.error:
                    self.num_errors += 1
            assert len(self.message.tool_calls) == len(tool_msgs), (
                "Number of tool calls and tool messages should be the same"
            )
            self.trajectory.extend(tool_msgs)
            if (
                len(tool_msgs) > 1
            ):
                self.message = MultiToolMessage(
                    role="tool",
                    tool_messages=tool_msgs,
                )
            else:
                self.message = tool_msgs[0]
            self.to_role = self.from_role
            self.from_role = Role.ENV
            # Fire dynamic events after tool calls complete
            if self.enable_dynamic_events and self.task.dynamic_events:
                self._check_and_fire_events(called_tool_names=called_tool_names)
        else:
            raise ValueError(
                f"Invalid role combination. From role: {self.from_role}, To role: {self.to_role}"
            )
        self.step_count += 1

    def _check_tool_loop(self) -> bool:
        """检测 agent 是否陷入工具调用循环。
        如果最近 _max_consecutive_same_tool 次调用都是同一个 (name, args)，返回 True。
        避免 Opus 等模型在 cascading-recovery 场景中反复轮询 get_ride_real_time_status 几百次。
        """
        n = self._max_consecutive_same_tool
        calls = self._recent_tool_calls
        if len(calls) < n:
            return False
        last_n = calls[-n:]
        return len(set(last_n)) == 1

    def _check_tool_name_loop(self) -> bool:
        """检测 agent 是否堆同一工具的 args 变体打转（不看 args 一致性，仅看 tool name）。
        当最近 _max_consecutive_same_tool_name 次调用全是同一个 tool name 且中间未穿插
        其它工具或 user 轮次（轮次切换会 clear 该列表），返回 True。
        典型场景：连续 search_pois({"query":"X"}) → ({"query":"X 城市"}) → ({"query":"X 区"}) ...
        Args 一直变会绕过 _check_tool_loop（要求 args 完全相同），但单一工具堆失败明显是 stuck。
        """
        n = self._max_consecutive_same_tool_name
        names = self._recent_tool_call_names
        if len(names) < n:
            return False
        last_n = names[-n:]
        return len(set(last_n)) == 1

    def _check_and_fire_events(self, called_tool_names: list = None):
        """Check dynamic event triggers and fire matching events by patching the DB.

        Two injection modes:
        - `notification`: system-style push (prefix "[系统通知]"); simulates 司机取消/溢价等系统事件
        - `user_inject`: user utterance verbatim (no prefix); simulates user's deterministic
                         mid-task change of mind (改时间/取消/加需求等), 避免 user-llm 发挥不稳定
        """
        for event in (self.task.dynamic_events or []):
            if event.event_id in self.fired_event_ids:
                continue
            trigger = event.trigger
            should_fire = False
            if trigger.type == "after_tool_call" and called_tool_names:
                should_fire = trigger.value in called_tool_names
            elif trigger.type == "after_step":
                should_fire = (self.step_count == int(trigger.value))
            if should_fire:
                self._apply_db_patch(event.db_patch or {})
                self.fired_event_ids.add(event.event_id)
                logger.info(f"[DynamicEvent] Fired: {event.event_id} — {event.description}")
                # 构造注入消息：优先 user_inject（user 自然语言），否则 notification（系统通知）
                inject_text = None
                if getattr(event, 'user_inject', None):
                    inject_text = event.user_inject
                    log_label = 'user_inject'
                elif event.notification:
                    inject_text = f"[系统通知] {event.notification}"
                    log_label = 'notification'
                if inject_text:
                    inject_msg = UserMessage(role="user", content=inject_text)
                    self.trajectory.append(inject_msg)
                    # BUG FIX: 当 self.message 是 tool 响应时（after_tool_call 触发），
                    # 直接覆盖 self.message 会让 tool_msgs 丢失（OpenAI API 报"tool_call 没响应"400）。
                    # 修复：先把 tool_msgs 注入 agent_state，再覆盖 self.message。
                    if self.agent_state is not None and hasattr(self.agent_state, "messages"):
                        if isinstance(self.message, MultiToolMessage):
                            self.agent_state.messages.extend(self.message.tool_messages)
                        elif isinstance(self.message, ToolMessage):
                            self.agent_state.messages.append(self.message)
                    self.message = inject_msg
                    self.from_role = Role.USER
                    self.to_role = Role.AGENT
                    logger.info(f"[DynamicEvent] {log_label} pushed to agent: {inject_text}")

    def _apply_db_patch(self, patch: dict):
        """Apply a db_patch dict to the environment DB."""
        db = self.environment.tools.db
        for key, value in patch.items():
            if key == "__latest_ride_order_patch__":
                # Patch the most recently updated ride order
                if hasattr(db, "ride_orders") and db.ride_orders:
                    latest_id = max(
                        db.ride_orders,
                        key=lambda oid: db.ride_orders[oid].get("update_time", "")
                    )
                    db.ride_orders[latest_id].update(value)
            elif hasattr(db, key):
                setattr(db, key, value)

    def get_trajectory(self) -> list[Message]:
        """
        Get the trajectory of the simulation.
        The trajectory is sorted by timestamp, turn_idx are added to messages, trajectory is returned.
        """
        messages: list[Message] = sorted(
            deepcopy(self.trajectory),
            key=lambda x: x.timestamp,
        )
        trajectory = []
        for i, msg in enumerate(messages):
            msg = deepcopy(msg)
            msg.turn_idx = i
            trajectory.append(msg)
        return trajectory
    
    def get_states(self, db: DB, env_time: str) -> Dict[str, Any]:
        """
        Split the states into a dictionary.
        """
        from eff.utils import str_to_datetime

        states = []
        if hasattr(db, "orders") and db.orders:
            states += list(db.orders.values())
        if hasattr(db, "books") and db.books:
            states += list(db.books.values())
        if hasattr(db, "reservations") and db.reservations:
            states += list(db.reservations.values())

        # Support ride-hailing domain ride_orders (dict of dicts)
        if hasattr(db, "ride_orders") and db.ride_orders:
            for order_dict in db.ride_orders.values():
                states.append(order_dict)

        states_dict = {"old_states": [], "new_states": []}
        for state in states:
            # ride_orders are plain dicts, others are pydantic models
            update_time = state.get("update_time") if isinstance(state, dict) else getattr(state, "update_time", None)
            if update_time and env_time:
                try:
                    if str_to_datetime(update_time) < str_to_datetime(env_time):
                        states_dict["old_states"].append(state)
                    else:
                        states_dict["new_states"].append(state)
                except (ValueError, TypeError):
                    states_dict["new_states"].append(state)
            else:
                states_dict["new_states"].append(state)
        return states_dict

    @classmethod
    def validate_message_history(cls, message_history: list[Message]):
        """
        Validate a message history.
            - Should only contain AssistantMessage, UserMessage, ToolMessage
            - All assistant/user messages should be either to user or tool call, not both.
            - If n tool calls are made by a participant, exactly n tool messages should follow with requestor matching the participant.
        """
        num_expected_tool_messages = 0
        requestor = None
        for msg in message_history:
            if isinstance(msg, AssistantMessage) or isinstance(msg, UserMessage):
                msg.validate()
                if msg.is_tool_call():
                    if num_expected_tool_messages > 0:
                        raise ValueError(
                            f"{num_expected_tool_messages} tool messages are missing. Got {msg.role} message."
                        )
                    num_expected_tool_messages = len(msg.tool_calls)
                    requestor = msg.role
                else:
                    num_expected_tool_messages = 0
                    requestor = None
            elif isinstance(msg, ToolMessage):
                if num_expected_tool_messages == 0 or requestor is None:
                    raise ValueError("No tool messages expected.")
                if requestor != msg.requestor:
                    raise ValueError(
                        f"Got tool message from {msg.requestor}, expected {requestor}."
                    )
                num_expected_tool_messages -= 1
            else:
                raise ValueError(f"Invalid message type: {type(msg)}")

    def _add_timestamps(
        self, message_history: list[Message]
    ) -> list[tuple[str, Message]]:
        """
        Add timestamps to the message history.
        This is used to sort the messages by timestamp.
        """
        time_offset = datetime.now() - timedelta(seconds=len(message_history))
        for i, msg in enumerate(message_history):
            msg.timestamp = format_time(time_offset + timedelta(seconds=i))
        return message_history
