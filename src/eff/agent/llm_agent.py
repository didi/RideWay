from copy import deepcopy
from datetime import datetime
from typing import List, Optional

from loguru import logger
from pydantic import BaseModel

from eff.agent.base import (
    LocalAgent,
    ValidAgentInputMessage,
    is_valid_agent_history_message,
)
from eff.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
)
from eff.environment.tool import Tool
from eff.utils.llm_utils import generate
from eff.utils.utils import get_now, get_weekday
from eff.prompts import get_prompts


def _parse_task_time(time_str: Optional[str]) -> Optional[datetime]:
    """解析 task.environment.time，返回 datetime 或 None。支持秒和分钟两种精度。"""
    if not time_str:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(time_str, fmt)
        except ValueError:
            continue
    return None


class LLMAgentState(BaseModel):
    """The state of the agent."""

    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]


class LLMAgent(LocalAgent[LLMAgentState]):
    """
    An LLM agent that can be used to solve a task.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        time=None,
        city_name: Optional[str] = None,
        city_id=None,
        enable_think: bool = False,
        language: str = None
    ):
        """
        Initialize the LLMAgent.
        """
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.llm = llm
        self.llm_args = deepcopy(llm_args) if llm_args is not None else {}
        self.language = language
        self.time = time + " " + get_weekday(time, language) if time else None
        # 时间流动：解析 task 起始时间为 datetime；首次 generate 时记录现实时刻
        self._task_base_time: Optional[datetime] = _parse_task_time(time)
        self._task_real_start: Optional[datetime] = None
        self.city_name = city_name or '未知'
        self.city_id = city_id if city_id is not None else '未知'
        self.enable_think = enable_think

    def _current_sim_time_str(self) -> str:
        """当前对话推进到 task 内的"现在时刻"= task 起始时间 + (现实当前时刻 - 首次 generate 时的现实时刻)"""
        if self._task_base_time and self._task_real_start:
            elapsed = datetime.now() - self._task_real_start
            sim_time = self._task_base_time + elapsed
            date_str = sim_time.strftime("%Y-%m-%d %H:%M:%S")
            weekday = get_weekday(date_str, self.language)
            return f"{date_str} {weekday}"
        return self.time if self.time else get_now("%Y-%m-%d %H:%M:%S")

    @property
    def system_prompt(self) -> str:
        return self.domain_policy.format(
            time=self._current_sim_time_str(),
            city_name=self.city_name,
            city_id=self.city_id,
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> LLMAgentState:
        """Get the initial state of the agent.

        Args:
            message_history: The message history of the conversation.

        Returns:
            The initial state of the agent.
        """
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )


        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        """
        Respond to a user or tool message.
        """
        # 首次 generate 时记录 task 真实开始时刻（即 agent 第一次说话前的时刻）
        if self._task_real_start is None:
            self._task_real_start = datetime.now()
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        # 每轮重新生成 system_messages —— 让 time 跟着对话推进
        state.system_messages = [SystemMessage(role="system", content=self.system_prompt)]
        messages = state.system_messages + state.messages

        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            enable_think=self.enable_think,
            **self.llm_args,
        )
        state.messages.append(assistant_message)

        return assistant_message, state


    def set_seed(self, seed: int):
        """Set the seed for the LLM."""
        if self.llm is None:
            raise ValueError("LLM is not set")
        cur_seed = self.llm_args.get("seed", None)
        if cur_seed is not None:
            logger.warning(f"Seed is already set to {cur_seed}, resetting it to {seed}")
        self.llm_args["seed"] = seed




class LLMSoloAgent(LocalAgent[LLMAgentState]):
    """
    An LLM agent that can be used to solve a task without any interaction with the customer.
    The task need to specify a ticket format.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        time=None,
        city_name: Optional[str] = None,
        city_id=None,
        enable_think: bool = False,
        language: str = None
    ):
        """
        Initialize the LLMAgent.
        """
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.llm = llm
        self.llm_args = deepcopy(llm_args) if llm_args is not None else {}
        self.language = language
        self.time = time + " " + get_weekday(time, language) if time else None
        # 时间流动：与 LLMAgent 一致
        self._task_base_time: Optional[datetime] = _parse_task_time(time)
        self._task_real_start: Optional[datetime] = None
        self.city_name = city_name or '未知'
        self.city_id = city_id if city_id is not None else '未知'
        self.enable_think = enable_think

    def _current_sim_time_str(self) -> str:
        if self._task_base_time and self._task_real_start:
            elapsed = datetime.now() - self._task_real_start
            sim_time = self._task_base_time + elapsed
            date_str = sim_time.strftime("%Y-%m-%d %H:%M:%S")
            weekday = get_weekday(date_str, self.language)
            return f"{date_str} {weekday}"
        return self.time if self.time else get_now("%Y-%m-%d %H:%M:%S")

    @property
    def system_prompt(self) -> str:
        prompts = get_prompts()
        rendered_time = self._current_sim_time_str()
        tpl = prompts.solo_agent_system_prompt
        try:
            return tpl.format(time=rendered_time, city_name=self.city_name, city_id=self.city_id)
        except KeyError:
            return tpl.format(time=rendered_time)

    @classmethod
    def is_stop(cls, message: AssistantMessage) -> bool:
        """Check if the message is a stop message."""
        if message.content is None:
            return False
        return cls.STOP_TOKEN in message.content

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> LLMAgentState:
        """Get the initial state of the agent.

        Args:
            message_history: The message history of the conversation.

        Returns:
            The initial state of the agent.
        """
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def generate_next_message(
        self, message: Optional[ValidAgentInputMessage], state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        """
        Respond to a user or tool message.
        """
        # 首次 generate 时记录 task 真实开始时刻
        if self._task_real_start is None:
            self._task_real_start = datetime.now()
        # if isinstance(message, UserMessage):
        #     raise ValueError("LLMSoloAgent does not support user messages.")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        elif message is None:
            assert len(state.messages) == 0, "Message history should be empty"
        else:
            state.messages.append(message)
        # 每轮重新生成 system_messages —— 让 time 跟着对话推进
        state.system_messages = [SystemMessage(role="system", content=self.system_prompt)]
        messages = state.system_messages + state.messages
        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            tool_choice="auto",
            enable_think=self.enable_think,
            **self.llm_args,
        )
        if not assistant_message.is_tool_call() and not self.is_stop(assistant_message):
            raise ValueError("LLMSoloAgent only supports tool calls before ###STOP###.")
        state.messages.append(assistant_message)
        return assistant_message, state

    def set_seed(self, seed: int):
        """Set the seed for the LLM."""
        if self.llm is None:
            raise ValueError("LLM is not set")
        cur_seed = self.llm_args.get("seed", None)
        if cur_seed is not None:
            logger.warning(f"Seed is already set to {cur_seed}, resetting it to {seed}")
        self.llm_args["seed"] = seed