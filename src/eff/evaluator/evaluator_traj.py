import json
import copy
from typing import List

from eff.config import DEFAULT_LLM_EVALUATOR, models
from eff.data_model.message import UserMessage, SystemMessage, Message
from eff.evaluator.evaluator_base import EvaluatorBase
from eff.data_model.simulation import RewardInfo, NLRubricCheck
from eff.data_model.tasks import RewardType, Task, EvaluationCriteria
from eff.utils.llm_utils import generate
from eff.utils import evaluator_extracter, get_weekday
from eff.prompts import get_prompts


def _sum_usage(usages: list) -> dict:
    """Aggregate a list of usage dicts (may contain None) into one."""
    total = {"prompt_tokens": 0, "completion_tokens": 0}
    for u in usages:
        if u:
            total["prompt_tokens"] += u.get("prompt_tokens", 0)
            total["completion_tokens"] += u.get("completion_tokens", 0)
    return total


class TrajectoryEvaluator(EvaluatorBase):
    """
    基于对话轨迹的评估器，使用 LLM 对 agent 与用户的对话进行质量评估。

    提供四种评估策略：
    1. calculate_reward              - 滑动窗口 + rubric：将对话分窗口，逐窗口评估预定义的 rubric 列表
    2. calculate_reward_full_traj_rubric   - 全轨迹 + rubric：一次性评估完整对话
    3. calculate_reward_sliding_wo_rubric  - 滑动窗口 + 无 rubric：分窗口，LLM 自行判断，带 memory
    4. calculate_reward_full_traj_wo_rubric - 全轨迹 + 无 rubric：一次性评估，LLM 自行判断

    核心概念：
    - rubric：预定义的评估准则（如"agent 应正确查询订单"），从 task.evaluation_criteria 中提取
    - 滑动窗口：将长对话切分为固定大小、带重叠的窗口，避免超出 LLM 上下文限制
    - reward：最终评估分数，所有 rubric 全部满足时为 1.0，否则为 0.0
    """

    @classmethod
    def calculate_reward(
        cls,
        task: Task,
        full_trajectory: List[Message],
        final_state: dict,
        window_size: int = 10,
        overlap: int = 2,
        llm_evaluator: str = None,
        llm_args_evaluator: dict = None,
        language: str = None,
    ) -> RewardInfo:
        """
        【策略一】滑动窗口 + rubric 评估。

        计算流程：
        1. 快速返回检查：无评估标准 或 无 rubric 时直接返回 reward=1.0
        2. 构建环境信息 env_info（系统时间、星期几等）
        3. 初始化 rubric 状态：从 evaluation_criteria 中提取所有 rubric，初始状态均为"未满足"
        4. 将完整对话切分为滑动窗口（窗口大小 window_size，重叠 overlap 条消息）
        5. 逐窗口调用 LLM 评估：
           - 将当前窗口内容 + 当前 rubric 状态发送给 LLM
           - LLM 返回每条 rubric 的最新判断（是否满足 + 理由）
           - 后续窗口的评估结果可以覆盖/翻转前面窗口的判断
        6. 汇总：所有 rubric 全部满足 → reward=1.0，否则 reward=0.0
           rubric_score 为满足比例（如 3/5 = 0.6），记录在 reward_breakdown 中

        Args:
            task: 任务定义，包含 evaluation_criteria（rubric 列表）和用户指令
            full_trajectory: 完整对话消息列表
            final_state: 仿真结束时的状态快照
            window_size: 每个窗口包含的消息数（默认 10）
            overlap: 相邻窗口重叠的消息数（默认 2），用于保持上下文连贯
            llm_evaluator: 评估用 LLM 模型名
            llm_args_evaluator: LLM 额外参数
            language: prompt 语言

        Returns:
            RewardInfo: 包含 reward、nl_rubrics（每条 rubric 判定详情）、
                       reward_breakdown（rubric 满足比例）、window_evaluations（各窗口评估详情）
        """
        if task.evaluation_criteria is None:
            return RewardInfo(
                reward=1.0,
                nl_rubrics=[],
                info={"note": "No evaluation criteria"},
                reward_breakdown={RewardType.NL_ASSERTION: 1.0},
            )

        evaluation_criteria = task.evaluation_criteria
        if not evaluation_criteria.overall_rubrics:
            return RewardInfo(
                reward=1.0,
                nl_rubrics=[],
                info={"note": "No rubric to evaluate"},
                reward_breakdown={RewardType.NL_ASSERTION: 1.0},
            )

        # 构建环境信息（系统时间 + 星期几），会注入到评估 prompt 中
        env_info = {
            "system_time": "",
            "database": []
        }

        if hasattr(task, 'environment') and task.environment:
            time_str = task.environment.get("time", "")
            if time_str:
                weekday = get_weekday(time_str, language)
                env_info["system_time"] = f"{time_str} {weekday or ''}"

        # 注入最终 ride_orders 状态，让 LLM 评估时能直接看到实际订单结果
        final_ride_orders = cls._extract_final_ride_orders(final_state)
        if final_ride_orders:
            env_info["final_ride_orders"] = final_ride_orders

        # 初始化所有 rubric 状态为 False（未满足）
        current_rubric_states = cls._initialize_rubric_states(evaluation_criteria)

        # 动态事件 rubric 自动处理：
        # 未注入动态事件 → [动态事件] 前缀的 rubric 直接设为通过，不送 LLM 评估
        # 已注入动态事件 → 去掉前缀，交 LLM 正常评估
        current_rubric_states = cls._auto_pass_dynamic_rubrics(task, current_rubric_states, full_trajectory)
        # 记录自动通过 / 自动 NA 的 rubric，全部窗口跑完后强制锁定，防止 LLM 误翻
        auto_passed_idx, auto_na_idx = cls._collect_auto_locked(current_rubric_states)

        # 将完整对话切分为滑动窗口
        windows = cls._create_sliding_windows(full_trajectory, window_size, overlap)

        step = window_size - overlap
        window_evaluations = []

        # 逐窗口评估：每个窗口的评估结果会更新 rubric 状态
        for i, window in enumerate(windows):
            print(f"Processing window {i+1}/{len(windows)} with {len(window)} messages")
            window_start_idx = i * step
            current_rubric_states, window_eval_info = cls._evaluate_window(
                env_info, task, window, current_rubric_states, i+1, len(windows), window_start_idx,
                llm_evaluator, llm_args_evaluator, language
            )
            window_evaluations.append(window_eval_info)

        # 强制锁定自动通过 / 自动 NA 的 rubric
        current_rubric_states = cls._lock_auto(current_rubric_states, auto_passed_idx, auto_na_idx)

        # [v4 暂时禁用] 结构化校验 required_orders：若最终订单不符合预期，强制将对应 rubric 设为 False
        # current_rubric_states = cls._apply_required_orders_check(
        #     evaluation_criteria, final_ride_orders, current_rubric_states
        # )

        # 将最终 rubric 状态转换为 NLRubricCheck 对象列表
        final_nl_rubric_checks = cls._convert_states_to_checks(current_rubric_states)

        # 计算最终 reward：not_applicable 的 rubric 不纳入评分；全部 N/A 视为任务未完成 → 0.0
        applicable_checks = [r for r in final_nl_rubric_checks if r.met is not None]
        if applicable_checks:
            all_expectations_met = all(result.met for result in applicable_checks)
            rubric_score = sum(1.0 if r.met else 0.0 for r in applicable_checks) / len(applicable_checks)
        else:
            all_expectations_met = False
            rubric_score = 0.0
        reward = 1.0 if all_expectations_met else 0.0

        evaluator_usage = _sum_usage([w.get("usage") for w in window_evaluations])

        # 若 evaluation_criteria 中有 enhance_rubrics，在原结果之外独立评估一遍
        enhance_reward_info = None
        enhance_rubrics = getattr(evaluation_criteria, "enhance_rubrics", None)
        if enhance_rubrics:
            print(f"[enhance] Evaluating {len(enhance_rubrics)} enhance_rubrics ...")
            enhance_reward_info = cls._evaluate_enhance_rubrics(
                task, enhance_rubrics, full_trajectory, env_info,
                window_size, overlap, llm_evaluator, llm_args_evaluator, language,
            )

        return RewardInfo(
            reward=reward,
            nl_rubrics=final_nl_rubric_checks,
            reward_breakdown={RewardType.NL_ASSERTION: rubric_score},
            info={"evaluation_method": "sliding_window", "num_windows": len(windows), "window_size": window_size},
            window_evaluations=window_evaluations,
            evaluator_usage=evaluator_usage,
            enhance_reward_info=enhance_reward_info,
        )

    @classmethod
    def _extract_final_ride_orders(cls, final_state: dict) -> list:
        """从 final_state 提取所有实际创建的 ride_orders（new_states + old_states 中含 order_id 的条目）"""
        if not final_state:
            return []
        all_states = final_state.get("new_states", []) + final_state.get("old_states", [])
        return [s for s in all_states if isinstance(s, dict) and "order_id" in s]

    # [已下线] _apply_required_orders_check：expected_states.required_orders 结构化校验。
    # call sites 早已注释、reward 不再依赖 expected_states，整方法随之移除（需恢复见 git 历史）。

    # POI 类字段别名映射 → 实际订单中的提取路径
    _START_POI_KEYS = frozenset({"start_name", "start_poi_name", "start_poi", "start", "start_address"})
    _END_POI_KEYS = frozenset({"end_name", "end_poi_name", "end_poi", "end", "end_address", "destination"})
    _VIA_POI_KEYS = frozenset({"passpoint_name", "passpoint", "via_poi", "via", "waypoint"})
    # 不参与结构校验、留给 NL rubric 判断的字段
    _SKIP_KEYS = frozenset({"task_type", "car_preference", "car_type_preference", "tags", "preference"})

    @staticmethod
    def _normalize_poi_name(s: str) -> str:
        """归一化 POI 名称用于模糊匹配：去全/半角括号差异、去多余空白、转小写。"""
        if not isinstance(s, str):
            return ""
        return (
            s.replace("（", "(").replace("）", ")")
             .replace(" ", "").replace("　", "")
             .lower()
        )

    @classmethod
    def _poi_match(cls, expected: str, actual: str) -> bool:
        """模糊匹配 POI 名称。

        归一化后通过即视为匹配（满足任一条件即可）：
        1. 子串包含（任一方向）
        2. 序列相似度 >= 0.75（容忍 "维修(...)" vs "维修店(...)" 之类的小字差异）
        """
        if not expected or not actual:
            return False
        e = cls._normalize_poi_name(expected)
        a = cls._normalize_poi_name(actual)
        if not e or not a:
            return False
        if e in a or a in e:
            return True
        from difflib import SequenceMatcher
        return SequenceMatcher(None, e, a).ratio() >= 0.75

    @classmethod
    def _match_required_field(cls, last_order: dict, key: str, expected) -> tuple:
        """对单个 required 字段做匹配，返回 (是否匹配, 失败原因)。"""
        if last_order is None:
            return False, f"未创建任何订单（期望 {key}={expected}）"

        # 跳过非订单结构字段
        if key in cls._SKIP_KEYS:
            return True, ""

        # POI 类字段：当期望值是字符串时，走 POI 别名解析（即便 key 与订单 dict 字段同名）
        is_str_expected = isinstance(expected, str)

        if key in cls._START_POI_KEYS and is_str_expected:
            actual = (last_order.get("start_poi") or {}).get("displayname", "")
            if cls._poi_match(expected, actual):
                return True, ""
            return False, f"起点不匹配：期望含 '{expected}'，实际为 '{actual}'"

        if key in cls._END_POI_KEYS and is_str_expected:
            actual = (last_order.get("end_poi") or {}).get("displayname", "")
            if cls._poi_match(expected, actual):
                return True, ""
            return False, f"终点不匹配：期望含 '{expected}'，实际为 '{actual}'"

        if key in cls._VIA_POI_KEYS and is_str_expected:
            via_list = last_order.get("via_pois") or []
            actuals = [(p or {}).get("displayname", "") for p in via_list]
            if any(cls._poi_match(expected, a) for a in actuals):
                return True, ""
            return False, f"途经点不匹配：期望含 '{expected}'，实际为 {actuals or '无'}"

        # 默认严格相等
        actual = last_order.get(key)
        if actual == expected:
            return True, ""
        return False, f"{key} 不匹配：期望 {expected!r}，实际 {actual!r}"

    @staticmethod
    def _summarize_order_for_error(order: dict) -> dict:
        """生成订单的可读摘要，用于错误信息（包含关键 POI 字段）。"""
        if not order:
            return {}
        return {
            "order_id": order.get("order_id"),
            "status": order.get("status"),
            "start_poi": (order.get("start_poi") or {}).get("displayname", ""),
            "end_poi": (order.get("end_poi") or {}).get("displayname", ""),
            "via_pois": [(p or {}).get("displayname", "") for p in (order.get("via_pois") or [])],
            "departure_time": order.get("departure_time"),
            "create_time": order.get("create_time"),
        }

    # rubric 文本命中这些关键词时，说明 rubric 依赖某个事件发生后的反应（如"司机取消后…"）
    _EVENT_DEPENDENT_KWS: list[str] = [
        "司机取消", "取消后", "动态加价", "加价后", "动态溢价", "溢价后",
        "司机已取消", "订单被取消", "重试时换", "在重试时", "司机违约",
        "司机迟到", "被取消", "取消原订单", "再次取消", "重新叫车", "重新匹配",
    ]
    # 额外的正则模式（而非纯子串匹配）：捕捉"至少 2 次 create"、"调用 N 次"这类
    # 表述——这类 rubric 隐含"有失败/重试场景"，而失败/重试本身是事件触发的结果
    _EVENT_DEPENDENT_PATTERNS: list[str] = [
        r"至少.*\d+.*次.*create",
        r"至少.*调用.*\d+.*次",
        r"调用.*\d+.*次.*create",
        r"(说明|表明).*(有|存在).*重试",
    ]

    _AUTO_PASS_JUSTIFICATION: str = "未注入动态事件，此条自动通过"
    _AUTO_NA_JUSTIFICATION: str = (
        "task 声明了 dynamic_events 但仿真中未触发（未检测到 [系统通知] 注入），"
        "rubric 依赖事件反应，转为 not_applicable"
    )

    @classmethod
    def _detect_event_fired(cls, full_trajectory) -> bool:
        """检测仿真中是否真的触发了动态事件。
        orchestrator 在事件触发后会向 trajectory 注入 UserMessage(content="[系统通知] ...")。
        evaluator 据此判断 has_fired，而非检查 task.environment.ride_events
        （后者是 task 定义层的初始字段，运行时事件触发不会改写它）。
        """
        if not full_trajectory:
            return False
        for msg in full_trajectory:
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            if isinstance(content, str) and "[系统通知]" in content:
                return True
        return False

    @classmethod
    def _collect_auto_locked(cls, rubric_states: dict) -> tuple[set, set]:
        """返回 (auto_passed_idx, auto_na_idx)：需要在 LLM 评估后强制锁定的 rubric 索引集。"""
        auto_passed = {
            k for k, v in rubric_states.items()
            if v.get("meetExpectation") is True
            and v.get("justification") == cls._AUTO_PASS_JUSTIFICATION
        }
        auto_na = {
            k for k, v in rubric_states.items()
            if v.get("meetExpectation") == "not_applicable"
            and v.get("justification") == cls._AUTO_NA_JUSTIFICATION
        }
        return auto_passed, auto_na

    @classmethod
    def _lock_auto(cls, rubric_states: dict, auto_passed: set, auto_na: set) -> dict:
        """在 LLM 评估返回后强制把自动通过 / 自动 NA 的 rubric 重置，防止 LLM 翻改。"""
        for k in auto_passed:
            if k in rubric_states:
                rubric_states[k]["meetExpectation"] = True
                rubric_states[k]["justification"] = cls._AUTO_PASS_JUSTIFICATION
        for k in auto_na:
            if k in rubric_states:
                rubric_states[k]["meetExpectation"] = "not_applicable"
                rubric_states[k]["justification"] = cls._AUTO_NA_JUSTIFICATION
        return rubric_states

    @classmethod
    def _auto_pass_dynamic_rubrics(cls, task: Task, rubric_states: dict, full_trajectory=None) -> dict:
        """处理 rubric 里对动态事件的依赖，分两条路径：

        路径 1（显式 [动态事件] 前缀）：
          - 仿真中未触发事件 → 去掉前缀 + 直接通过（met=True）
          - 仿真中已触发事件 → 去掉前缀，交 LLM 正常评估

        路径 2（无前缀，但文本命中事件依赖关键词）：
          - task 声明了 dynamic_events 但仿真中未触发
          - → 直接标 not_applicable，不送 LLM（agent 没机会响应事件）

        has_fired 判定：从 full_trajectory 中检测 [系统通知] 注入，
        而非读 task.environment.ride_events（该字段运行时不会被填充）。
        """
        DYNAMIC_PREFIX = "[动态事件]"

        has_fired = cls._detect_event_fired(full_trajectory)
        has_declared = bool(getattr(task, "dynamic_events", None))

        for state in rubric_states.values():
            text = state["rubric"]

            if text.startswith(DYNAMIC_PREFIX):
                state["rubric"] = text[len(DYNAMIC_PREFIX):]
                if not has_fired:
                    state["meetExpectation"] = True
                    state["justification"] = cls._AUTO_PASS_JUSTIFICATION
                continue

            # 路径 2：无前缀但命中事件依赖关键词/正则模式，且事件未实际触发
            import re as _re
            kw_hit = any(kw in text for kw in cls._EVENT_DEPENDENT_KWS)
            pat_hit = any(_re.search(p, text) for p in cls._EVENT_DEPENDENT_PATTERNS)
            if has_declared and not has_fired and (kw_hit or pat_hit):
                state["meetExpectation"] = "not_applicable"
                state["justification"] = cls._AUTO_NA_JUSTIFICATION

        return rubric_states

    @classmethod
    def _initialize_rubric_states(cls, evaluation_criteria: EvaluationCriteria) -> dict:
        """
        从 evaluation_criteria 中提取所有 rubric，初始化为"未满足"状态。

        会从两个来源收集 rubric：
        1. expected_states 中每个状态的 state_rubrics（如"agent 应查询订单状态"）
        2. overall_rubrics（全局评估准则，如"agent 应使用礼貌用语"）

        使用 seen_rubrics 集合去重，确保相同 rubric 文本只出现一次。

        Returns:
            dict: 形如 {"rubric_0": {"rubric": "...", "justification": "Not evaluated yet", "meetExpectation": False}, ...}
        """
        rubric_states = {}
        rubric_idx = 0
        seen_rubrics = set()

        # [v4 暂时禁用] 跳过 expected_states.state_rubrics，只评估 overall_rubrics
        # if evaluation_criteria.expected_states:
        #     for expected_state in evaluation_criteria.expected_states:
        #         if hasattr(expected_state, 'state_rubrics') and expected_state.state_rubrics:
        #             for rubric in expected_state.state_rubrics:
        #                 if rubric in seen_rubrics:
        #                     continue
        #                 seen_rubrics.add(rubric)
        #
        #                 key = f"rubric_{rubric_idx}"
        #                 rubric_states[key] = {
        #                     "rubric": rubric,
        #                     "justification": "Not evaluated yet",
        #                     "meetExpectation": False
        #                 }
        #                 rubric_idx += 1

        if evaluation_criteria.overall_rubrics:
            for rubric in evaluation_criteria.overall_rubrics:
                if rubric in seen_rubrics:
                    continue
                seen_rubrics.add(rubric)

                key = f"rubric_{rubric_idx}"
                rubric_states[key] = {
                    "rubric": rubric,
                    "justification": "Not evaluated yet",
                    "meetExpectation": False
                }
                rubric_idx += 1

        return rubric_states

    @classmethod
    def _initialize_rubric_states_from_list(cls, rubrics: List[str]) -> dict:
        """从 rubric 字符串列表初始化 rubric 状态，所有条目初始均为"未满足"。
        用于 enhance_rubrics 等不走 EvaluationCriteria 结构的额外评估。
        """
        rubric_states = {}
        for idx, rubric in enumerate(rubrics):
            key = f"rubric_{idx}"
            rubric_states[key] = {
                "rubric": rubric,
                "justification": "Not evaluated yet",
                "meetExpectation": False,
            }
        return rubric_states

    @classmethod
    def _evaluate_enhance_rubrics(
        cls,
        task: "Task",
        enhance_rubrics: List[str],
        full_trajectory: List[Message],
        env_info: dict,
        window_size: int = 10,
        overlap: int = 2,
        llm_evaluator: str = None,
        llm_args_evaluator: dict = None,
        language: str = None,
    ) -> "RewardInfo":
        """对 enhance_rubrics 进行独立的滑动窗口评估，返回独立的 RewardInfo。

        原评估逻辑完全不变；此方法仅在原结果之外额外评估 enhance_rubrics 中的每条 rubric，
        使用与 calculate_reward 相同的滑动窗口 + LLM 逐 rubric 判断方式。

        Args:
            task: 任务定义（用于指令、动态事件检测）
            enhance_rubrics: 要评估的 rubric 文本列表
            full_trajectory: 完整对话消息列表
            env_info: 已构建的环境信息（系统时间、ride_orders 等）
            window_size / overlap: 滑动窗口参数，与主评估保持一致
            llm_evaluator / llm_args_evaluator / language: 与主评估相同的 LLM 配置

        Returns:
            RewardInfo: 仅含 enhance_rubrics 评估结果的独立 RewardInfo（enhance_reward_info 为 None）
        """
        enhance_states = cls._initialize_rubric_states_from_list(enhance_rubrics)
        enhance_states = cls._auto_pass_dynamic_rubrics(task, enhance_states, full_trajectory)
        auto_passed_idx, auto_na_idx = cls._collect_auto_locked(enhance_states)

        def _lock_auto_pass(states):
            return cls._lock_auto(states, auto_passed_idx, auto_na_idx)

        # 与主评估保持一致的自适应策略：≤80 条消息走 full-traj（避免跨窗口继承问题），
        # 超过 80 条走滑动窗口（避免超出 LLM context）
        msg_count = len(full_trajectory) if full_trajectory else 0
        if msg_count <= 80:
            enhance_states, traj_eval_info = cls._evaluate_trajectory(
                env_info, task, full_trajectory, enhance_states,
                llm_evaluator, llm_args_evaluator, language,
            )
            enhance_states = _lock_auto_pass(enhance_states)
            final_checks = cls._convert_states_to_checks(enhance_states)
            applicable = [r for r in final_checks if r.met is not None]
            all_met = all(r.met for r in applicable) if applicable else False
            score = (
                sum(1.0 if r.met else 0.0 for r in applicable) / len(applicable)
                if applicable else 0.0
            )
            return RewardInfo(
                reward=1.0 if all_met else 0.0,
                nl_rubrics=final_checks,
                reward_breakdown={RewardType.NL_ASSERTION: score},
                info={"evaluation_method": "enhance_rubrics_full_trajectory"},
                evaluator_usage=traj_eval_info.get("usage"),
            )

        # fallback：长对话走滑动窗口
        windows = cls._create_sliding_windows(full_trajectory, window_size, overlap)
        step = window_size - overlap
        window_evaluations = []

        for i, window in enumerate(windows):
            window_start_idx = i * step
            enhance_states, window_eval_info = cls._evaluate_window(
                env_info, task, window, enhance_states, i + 1, len(windows), window_start_idx,
                llm_evaluator, llm_args_evaluator, language,
            )
            window_evaluations.append(window_eval_info)

        enhance_states = _lock_auto_pass(enhance_states)
        final_checks = cls._convert_states_to_checks(enhance_states)
        applicable = [r for r in final_checks if r.met is not None]
        all_met = all(r.met for r in applicable) if applicable else False
        score = (
            sum(1.0 if r.met else 0.0 for r in applicable) / len(applicable)
            if applicable else 0.0
        )
        evaluator_usage = _sum_usage([w.get("usage") for w in window_evaluations])

        return RewardInfo(
            reward=1.0 if all_met else 0.0,
            nl_rubrics=final_checks,
            reward_breakdown={RewardType.NL_ASSERTION: score},
            info={"evaluation_method": "enhance_rubrics_sliding_window", "num_windows": len(windows)},
            window_evaluations=window_evaluations,
            evaluator_usage=evaluator_usage,
        )

    @classmethod
    def _create_sliding_windows(cls, messages: List[Message], window_size: int, overlap: int = 2) -> List[List[Message]]:
        """
        将消息列表切分为滑动窗口。

        窗口按 step = window_size - overlap 步长滑动，相邻窗口之间有 overlap 条消息重叠，
        以保持评估时的上下文连贯性。如果消息总数不超过 window_size，则返回单个窗口。

        示例：messages=[m0..m19], window_size=10, overlap=2 → step=8
              窗口: [m0..m9], [m8..m17], [m16..m19]

        Args:
            messages: 完整消息列表
            window_size: 每个窗口的消息数量
            overlap: 相邻窗口重叠的消息数量

        Returns:
            List[List[Message]]: 窗口列表，每个窗口是一个消息子列表
        """
        if len(messages) <= window_size:
            return [messages]

        windows = []
        step = window_size - overlap

        i = 0
        while i < len(messages):
            window = messages[i:i + window_size]
            if len(window) > 0:
                windows.append(window)

            if i + window_size >= len(messages):
                break

            i += step

        return windows

    @classmethod
    def _evaluate_window(
        cls,
        env_info: dict,
        task: Task,
        window: List[Message],
        current_states: dict,
        window_idx: int,
        total_windows: int,
        window_start_idx: int = 0,
        llm_evaluator: str = None,
        llm_args_evaluator: dict = None,
        language: str = None,
    ) -> tuple[dict, dict]:
        """
        评估单个窗口，更新 rubric 状态。

        流程：
        1. 将窗口内的消息格式化为带全局序号的文本（如 "[5] user: 你好"）
        2. 将当前所有 rubric 的状态序列化为 JSON
        3. 使用评估 prompt 模板构建 system_prompt 和 user_prompt
        4. 调用 LLM 生成评估结果
        5. 从 LLM 回复中解析每条 rubric 的判定（meetExpectation + justification）
        6. 更新 rubric 状态（后续窗口可以翻转前面的判断）

        Args:
            env_info: 环境信息（系统时间等）
            task: 任务定义
            window: 当前窗口的消息列表
            current_states: 当前所有 rubric 的状态
            window_idx: 当前窗口序号（从 1 开始）
            total_windows: 总窗口数
            window_start_idx: 当前窗口第一条消息在完整对话中的全局索引
            llm_evaluator: 评估模型名
            llm_args_evaluator: 模型参数
            language: prompt 语言

        Returns:
            tuple: (updated_states: 更新后的 rubric 状态, window_evaluation_info: 本窗口的评估详情)
        """
        if llm_evaluator is None:
            llm_evaluator = DEFAULT_LLM_EVALUATOR
        if llm_args_evaluator is None:
            llm_args_evaluator = models[DEFAULT_LLM_EVALUATOR]

        # 格式化窗口消息为文本，带全局消息序号
        window_content = cls._format_window_content(window, window_start_idx)

        # 将当前 rubric 状态序列化为 JSON 格式，供 LLM 参考
        current_rubrics_str = cls._format_current_rubrics(current_states)

        # 构建评估 prompt
        prompts = get_prompts(language)
        system_prompt = prompts.sliding_window_eval_template.format(
            env_info=env_info,
            user_instruction=task.instructions,
            window_idx=window_idx,
            total_windows=total_windows
        )

        user_prompt = f"""
# Input
<window_content>
{window_content}
</window_content>

<current_rubrics>
{current_rubrics_str}
</current_rubrics>
"""

        messages = [
            SystemMessage(role="system", content=system_prompt),
            UserMessage(role="user", content=user_prompt),
        ]

        # 调用 LLM 进行评估
        assistant_message = generate(
            model=llm_evaluator,
            messages=messages,
            **llm_args_evaluator,
        )

        # 记录本窗口的评估过程信息（prompt + LLM 回复），用于后续调试和分析
        window_evaluation_info = {
            "window_idx": window_idx,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "assistant_message_content": assistant_message.content,
            "assistent_message_usage": assistant_message.usage,
            "usage": assistant_message.usage,
        }

        # 深拷贝当前状态，避免修改原始数据
        updated_states = copy.deepcopy(current_states)
        # 从 LLM 回复中解析结构化评估结果
        result_data = evaluator_extracter(assistant_message.content)

        if result_data:
            # 逐条更新 rubric 的判定结果和理由
            for result in result_data:
                if not isinstance(result, dict):
                    print(f"Warning: Unexpected result type {type(result)} in window {window_idx}, skipping")
                    continue
                rubric_idx = result.get("rubric_idx")
                if rubric_idx and rubric_idx in updated_states:
                    updated_states[rubric_idx]["justification"] = result.get("justification", "No justification provided")
                    updated_states[rubric_idx]["meetExpectation"] = result.get("meetExpectation", updated_states[rubric_idx]["meetExpectation"])
        else:
            print(f"Warning: Failed to parse LLM response for window {window_idx}, keeping current states")

        return updated_states, window_evaluation_info

    @classmethod
    def _format_window_content(cls, window: List[Message], window_start_idx: int = 0) -> str:
        """
        将窗口内的消息格式化为带全局序号的可读文本。

        格式示例：
            [1] user: 我想查一下订单
            [2] assistant: 好的，请提供订单号。query_order(order_id='12345')

        对于 assistant 消息，如果包含 tool_calls，会将工具调用信息追加到文本末尾，
        格式为 "tool_name(arg1=val1, arg2=val2)"。

        Args:
            window: 当前窗口的消息列表
            window_start_idx: 窗口第一条消息在完整对话中的全局索引（用于计算全局序号）

        Returns:
            str: 格式化后的多行文本
        """
        content_lines = []
        for i, message in enumerate(window):
            role = getattr(message, 'role', 'unknown')
            content = getattr(message, 'content', '')

            full_content = content

            # 如果是 assistant 消息且包含工具调用，将工具信息追加到内容中
            if role == 'assistant' and hasattr(message, 'tool_calls') and message.tool_calls:
                tool_calls_str = []
                for tool_call in message.tool_calls:
                    if hasattr(tool_call, 'name'):
                        tool_name = tool_call.name
                    elif isinstance(tool_call, dict):
                        tool_name = tool_call.get('name', 'unknown_tool')
                    else:
                        tool_name = 'unknown_tool'

                    if hasattr(tool_call, 'arguments'):
                        tool_args = tool_call.arguments
                    elif isinstance(tool_call, dict):
                        tool_args = tool_call.get('arguments', {})
                    else:
                        tool_args = {}

                    if isinstance(tool_args, dict):
                        args_str = ', '.join([f"{k}={repr(v)}" for k, v in tool_args.items()])
                    else:
                        args_str = str(tool_args)
                    tool_calls_str.append(f"{tool_name}({args_str})")

                if tool_calls_str:
                    if full_content:
                        full_content += " " + ".".join(tool_calls_str)
                    else:
                        full_content = ".".join(tool_calls_str)

            global_idx = window_start_idx + i + 1
            content_lines.append(f"[{global_idx}] {role}: {full_content}")

        return "\n".join(content_lines)

    @classmethod
    def _format_current_rubrics(cls, current_states: dict) -> str:
        """
        将当前 rubric 状态格式化为 JSON 字符串，供 LLM 评估时参考。

        输出格式：JSON 数组，每个元素包含 rubric_idx、rubric 文本、justification 和 meetExpectation。
        """
        rubrics_list = []
        for key, state in current_states.items():
            rubrics_list.append({
                "rubric_idx": key,
                "rubric": state["rubric"],
                "justification": state["justification"],
                "meetExpectation": state["meetExpectation"]
            })

        return json.dumps(rubrics_list, ensure_ascii=False, indent=2)

    @classmethod
    def _convert_states_to_checks(cls, final_states: dict) -> List[NLRubricCheck]:
        """
        将最终的 rubric 状态字典转换为 NLRubricCheck 对象列表，用于填充 RewardInfo。
        meetExpectation 为 "not_applicable" 时，met 设为 None 表示不纳入评分。
        """
        checks = []
        for key, state in final_states.items():
            met_value = state["meetExpectation"]
            if met_value == "not_applicable":
                met_value = None
            check = NLRubricCheck(
                nl_rubric=state["rubric"],
                met=met_value,
                justification=state["justification"],
            )
            checks.append(check)

        return checks

    @classmethod
    def calculate_reward_full_traj_rubric(
            cls,
            task: Task,
            full_trajectory: List[Message],
            final_state: dict,
            llm_evaluator: str = None,
            llm_args_evaluator: dict = None,
            language: str = None,
    ) -> RewardInfo:
        """
        【策略二】全轨迹 + rubric 评估。

        与 calculate_reward 的区别：不进行滑动窗口切分，而是将完整对话一次性发送给 LLM 评估。
        适用于对话较短（不超出 LLM 上下文限制）的场景。

        计算流程：
        1. 快速返回检查（无评估标准 / 无 rubric）
        2. 构建环境信息
        3. 初始化 rubric 状态
        4. 调用 _evaluate_trajectory 将完整对话 + rubric 一次性发给 LLM
        5. 汇总：所有 rubric 全部满足 → reward=1.0，否则 → 0.0

        Args:
            task: 任务定义
            full_trajectory: 完整对话消息列表
            final_state: 仿真结束时的状态
            llm_evaluator: 评估用 LLM 模型名
            llm_args_evaluator: LLM 额外参数
            language: prompt 语言

        Returns:
            RewardInfo: 评估结果
        """
        if task.evaluation_criteria is None:
            return RewardInfo(
                reward=1.0,
                nl_rubrics=[],
                info={"note": "No evaluation criteria"},
                reward_breakdown={RewardType.NL_ASSERTION: 1.0},
            )

        evaluation_criteria = task.evaluation_criteria
        if not evaluation_criteria.overall_rubrics:
            return RewardInfo(
                reward=1.0,
                nl_rubrics=[],
                info={"note": "No rubric to evaluate"},
                reward_breakdown={RewardType.NL_ASSERTION: 1.0},
            )

        env_info = {
            "system_time": "",
            "database": []
        }

        if hasattr(task, 'environment') and task.environment:
            time_str = task.environment.get("time", "")
            if time_str:
                weekday = get_weekday(time_str, language)
                env_info["system_time"] = f"{time_str} {weekday or ''}"

        # 注入最终 ride_orders 状态
        final_ride_orders = cls._extract_final_ride_orders(final_state)
        if final_ride_orders:
            env_info["final_ride_orders"] = final_ride_orders

        current_rubric_states = cls._initialize_rubric_states(evaluation_criteria)

        # 动态事件 rubric 自动处理：
        # 未注入动态事件 → [动态事件] 前缀的 rubric 直接设为通过，不送 LLM 评估
        # 已注入动态事件 → 去掉前缀，交 LLM 正常评估
        current_rubric_states = cls._auto_pass_dynamic_rubrics(task, current_rubric_states, full_trajectory)
        # 记录自动通过 / 自动 NA 的 rubric，LLM 评估后强制保留
        auto_passed_idx, auto_na_idx = cls._collect_auto_locked(current_rubric_states)

        # 一次性评估完整轨迹
        current_rubric_states, trajectory_eval_info = cls._evaluate_trajectory(
            env_info, task, full_trajectory, current_rubric_states, llm_evaluator, llm_args_evaluator, language
        )

        # 强制锁定自动通过 / 自动 NA 的 rubric，防止 LLM 误翻
        current_rubric_states = cls._lock_auto(current_rubric_states, auto_passed_idx, auto_na_idx)

        # [v4 暂时禁用] 结构化校验 required_orders
        # current_rubric_states = cls._apply_required_orders_check(
        #     evaluation_criteria, final_ride_orders, current_rubric_states
        # )

        final_nl_rubric_checks = cls._convert_states_to_checks(current_rubric_states)

        # 计算 reward：not_applicable 的 rubric 不纳入评分；全部 N/A 视为任务未完成 → 0.0
        applicable_checks = [r for r in final_nl_rubric_checks if r.met is not None]
        if applicable_checks:
            all_expectations_met = all(result.met for result in applicable_checks)
            rubric_score = sum(1.0 if r.met else 0.0 for r in applicable_checks) / len(applicable_checks)
        else:
            all_expectations_met = False
            rubric_score = 0.0
        reward = 1.0 if all_expectations_met else 0.0

        # 若 evaluation_criteria 中有 enhance_rubrics，在原结果之外独立评估一遍
        enhance_reward_info = None
        enhance_rubrics = getattr(evaluation_criteria, "enhance_rubrics", None)
        if enhance_rubrics:
            print(f"[enhance] Evaluating {len(enhance_rubrics)} enhance_rubrics ...")
            enhance_reward_info = cls._evaluate_enhance_rubrics(
                task, enhance_rubrics, full_trajectory, env_info,
                llm_evaluator=llm_evaluator, llm_args_evaluator=llm_args_evaluator, language=language,
            )

        return RewardInfo(
            reward=reward,
            nl_rubrics=final_nl_rubric_checks,
            reward_breakdown={RewardType.NL_ASSERTION: rubric_score},
            info={"evaluation_method": "full_trajectory_with_rubrics"},
            evaluator_usage=trajectory_eval_info.get("usage"),
            enhance_reward_info=enhance_reward_info,
        )

    @classmethod
    def _evaluate_trajectory(
            cls,
            env_info: dict,
            task: Task,
            trajectory: List[Message],
            current_states: dict,
            llm_evaluator: str = None,
            llm_args_evaluator: dict = None,
            language: str = None,
    ) -> tuple[dict, dict]:
        """
        评估完整对话轨迹（带 rubric），供 calculate_reward_full_traj_rubric 调用。

        将完整对话内容 + 当前 rubric 状态一次性发送给 LLM，获取评估结果并更新 rubric 状态。
        使用 full_trajectory_eval_template prompt 模板。

        Args:
            env_info: 环境信息
            task: 任务定义
            trajectory: 完整对话消息列表
            current_states: 当前 rubric 状态
            llm_evaluator: 评估模型名
            llm_args_evaluator: 模型参数
            language: prompt 语言

        Returns:
            tuple: (updated_states: 更新后的 rubric 状态, trajectory_evaluation_info: 评估详情)
        """
        if llm_evaluator is None:
            llm_evaluator = DEFAULT_LLM_EVALUATOR
        if llm_args_evaluator is None:
            llm_args_evaluator = models[DEFAULT_LLM_EVALUATOR]

        trajectory_content = cls._format_window_content(trajectory)

        current_rubrics_str = cls._format_current_rubrics(current_states)

        prompts = get_prompts(language)
        system_prompt = prompts.full_trajectory_eval_template.format(
            env_info=env_info,
            user_instruction=task.instructions
        )

        user_prompt = f"""
    # Input
    <trajectory_content>
    {trajectory_content}
    </trajectory_content>

    <current_rubrics>
    {current_rubrics_str}
    </current_rubrics>
    """

        messages = [
            SystemMessage(role="system", content=system_prompt),
            UserMessage(role="user", content=user_prompt),
        ]

        assistant_message = generate(
            model=llm_evaluator,
            messages=messages,
            **llm_args_evaluator,
        )

        trajectory_evaluation_info = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "assistant_message_content": assistant_message.content,
            "usage": assistant_message.usage,
        }

        updated_states = copy.deepcopy(current_states)
        result_data = evaluator_extracter(assistant_message.content)

        if result_data:
            for result in result_data:
                if not isinstance(result, dict):
                    print(f"Warning: Unexpected result type {type(result)} in full trajectory eval, skipping")
                    continue
                rubric_idx = result.get("rubric_idx")
                if rubric_idx and rubric_idx in updated_states:
                    updated_states[rubric_idx]["justification"] = result.get("justification",
                                                                             "No justification provided")
                    updated_states[rubric_idx]["meetExpectation"] = result.get("meetExpectation",
                                                                               updated_states[rubric_idx][
                                                                                   "meetExpectation"])
        else:
            print(f"Warning: Failed to parse LLM response for total trajectory, keeping current states")

        return updated_states, trajectory_evaluation_info

    @classmethod
    def calculate_reward_sliding_wo_rubric(
            cls,
            task: Task,
            full_trajectory: List[Message],
            final_state: dict,
            window_size: int = 10,
            overlap: int = 2,
            llm_evaluator: str = None,
            llm_args_evaluator: dict = None,
            language: str = None,
    ) -> RewardInfo:
        """
        【策略三】滑动窗口 + 无 rubric 评估。

        与 calculate_reward 的区别：
        - 不使用预定义的 rubric 列表，而是让 LLM 根据用户指令自行判断任务完成情况
        - 引入 memory 机制：每个窗口评估后 LLM 会输出一段 memory 文本，传递给下一个窗口，
          使 LLM 能够跨窗口积累上下文信息
        - 最终只有一个整体的 meetExpectation 判断，而非逐条 rubric

        计算流程：
        1. 快速返回检查
        2. 构建环境信息
        3. 将对话切分为滑动窗口
        4. 逐窗口评估，传递 memory
        5. 最终 meetExpectation=True → reward=1.0，否则 → 0.0

        Args:
            task: 任务定义
            full_trajectory: 完整对话消息列表
            final_state: 仿真结束时的状态
            window_size: 窗口大小（默认 10）
            overlap: 重叠消息数（默认 2）
            llm_evaluator: 评估模型名
            llm_args_evaluator: LLM 参数
            language: prompt 语言

        Returns:
            RewardInfo: 评估结果，nl_rubrics 中只包含一条整体判定
        """
        if task.evaluation_criteria is None:
            return RewardInfo(
                reward=1.0,
                nl_rubrics=[],
                info={"note": "No evaluation criteria"},
                reward_breakdown={RewardType.NL_ASSERTION: 1.0},
            )

        evaluation_criteria = task.evaluation_criteria
        if not evaluation_criteria.overall_rubrics:
            return RewardInfo(
                reward=1.0,
                nl_rubrics=[],
                info={"note": "No rubric to evaluate"},
                reward_breakdown={RewardType.NL_ASSERTION: 1.0},
            )

        env_info = {
            "system_time": "",
            "database": []
        }

        if hasattr(task, 'environment') and task.environment:
            time_str = task.environment.get("time", "")
            if time_str:
                weekday = get_weekday(time_str, language)
                env_info["system_time"] = f"{time_str} {weekday or ''}"

        windows = cls._create_sliding_windows(full_trajectory, window_size, overlap)
        # 初始评估状态：单一的整体判定（无 rubric 列表）
        current_evaluation = {
            "justification": "Not evaluated yet",
            "meetExpectation": False
        }

        step = window_size - overlap
        window_evaluations = []

        # memory 用于跨窗口传递上下文信息
        memory = ""
        for i, window in enumerate(windows):
            print(f"Processing window {i + 1}/{len(windows)} with {len(window)} messages")
            window_start_idx = i * step
            current_evaluation, window_eval_info, memory = cls._evaluate_window_sliding_wo_rubric(
                env_info, task, memory, current_evaluation, window, i + 1, len(windows), window_start_idx,
                llm_evaluator, llm_args_evaluator, language
            )
            window_evaluations.append(window_eval_info)

        final_nl_rubric_checks = cls._convert_states_to_checks_no_rubric(current_evaluation)

        all_expectations_met = final_nl_rubric_checks.met
        reward = 1.0 if all_expectations_met else 0.0

        evaluator_usage = _sum_usage([w.get("usage") for w in window_evaluations])

        # 若 evaluation_criteria 中有 enhance_rubrics，在原结果之外独立评估一遍
        enhance_reward_info = None
        enhance_rubrics = getattr(evaluation_criteria, "enhance_rubrics", None)
        if enhance_rubrics:
            print(f"[enhance] Evaluating {len(enhance_rubrics)} enhance_rubrics ...")
            enhance_reward_info = cls._evaluate_enhance_rubrics(
                task, enhance_rubrics, full_trajectory, env_info,
                window_size, overlap, llm_evaluator, llm_args_evaluator, language,
            )

        return RewardInfo(
            reward=reward,
            nl_rubrics=[final_nl_rubric_checks],
            info={"evaluation_method": "sliding_window_no_rubrics", "num_windows": len(windows), "window_size": window_size},
            window_evaluations=window_evaluations,
            evaluator_usage=evaluator_usage,
            enhance_reward_info=enhance_reward_info,
        )

    @classmethod
    def _evaluate_window_sliding_wo_rubric(
            cls,
            env_info: dict,
            task: Task,
            memory: str,
            current_evaluation,
            window: List[Message],
            window_idx: int,
            total_windows: int,
            window_start_idx: int = 0,
            llm_evaluator: str = None,
            llm_args_evaluator: dict = None,
            language: str = None,
    ) -> tuple[dict, dict]:
        """
        评估单个窗口（无 rubric 模式），更新整体评估状态和 memory。

        与 _evaluate_window 的区别：
        - 不传入 rubric 列表，而是传入 memory（前序窗口的上下文摘要）和当前整体评估状态
        - LLM 返回更新后的 memory + 整体评估判定
        - 使用 sliding_window_eval_no_rubrics_eval_template prompt 模板

        Args:
            env_info: 环境信息
            task: 任务定义
            memory: 前序窗口传递过来的上下文摘要
            current_evaluation: 当前整体评估状态 {"justification": ..., "meetExpectation": ...}
            window: 当前窗口消息列表
            window_idx: 当前窗口序号
            total_windows: 总窗口数
            window_start_idx: 全局消息索引偏移
            llm_evaluator: 评估模型名
            llm_args_evaluator: 模型参数
            language: prompt 语言

        Returns:
            tuple: (updated_states: 更新后的评估状态, window_evaluation_info: 评估详情, memory: 更新后的 memory)
        """
        if llm_evaluator is None:
            llm_evaluator = DEFAULT_LLM_EVALUATOR
        if llm_args_evaluator is None:
            llm_args_evaluator = models[DEFAULT_LLM_EVALUATOR]

        window_content = cls._format_window_content(window, window_start_idx)

        current_evaluation_str = json.dumps(current_evaluation, ensure_ascii=False, indent=2)

        prompts = get_prompts(language)
        system_prompt = prompts.sliding_window_eval_no_rubrics_eval_template.format(
            env_info=env_info,
            user_instruction=task.instructions,
            window_idx=window_idx,
            total_windows=total_windows
        )

        user_prompt = f"""
# Input
<window_content>
{window_content}
</window_content>
<memory>
{memory}
</memory>
<current_evaluation>
{current_evaluation_str}
</current_evaluation>
"""

        messages = [
            SystemMessage(role="system", content=system_prompt),
            UserMessage(role="user", content=user_prompt),
        ]

        assistant_message = generate(
            model=llm_evaluator,
            messages=messages,
            **llm_args_evaluator,
        )

        window_evaluation_info = {
            "window_idx": window_idx,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "assistant_message_content": assistant_message.content,
            "usage": assistant_message.usage,
        }

        updated_states = current_evaluation
        result_data = evaluator_extracter(assistant_message.content)

        memory = result_data.pop("memory")
        if result_data:
            updated_states = result_data
        else:
            print(f"Warning: Failed to parse LLM response for window {window_idx}")

        return updated_states, window_evaluation_info, memory

    @classmethod
    def _convert_states_to_checks_no_rubric(cls, final_states: dict) -> NLRubricCheck:
        """
        将无 rubric 模式的最终评估状态转换为单个 NLRubricCheck 对象。
        与 _convert_states_to_checks 不同，此方法返回单个对象而非列表。
        """
        check = NLRubricCheck(
            met=final_states["meetExpectation"],
            justification=final_states["justification"]
        )

        return check

    @classmethod
    def calculate_reward_full_traj_12dim(
            cls,
            task: Task,
            full_trajectory: List[Message],
            final_state: dict,
            llm_evaluator: str = None,
            llm_args_evaluator: dict = None,
            language: str = None,
    ) -> RewardInfo:
        """
        【策略五】全轨迹 + 12 维 capability 评估（不依赖 task 自带 rubric）。

        LLM 用通用 12 维框架（A1-A3 / B1-B4 / C1-C5）自己识别本场景激活的维度，逐维判 met。
        reward = sum(met & activated) / sum(activated)；activated=0 时 reward=1.0。

        与策略四（no rubric）的差异：保留维度信号；与策略二（with rubric）的差异：
        不依赖 task 自带 rubric，所有 task 用同一套 12 维标尺，便于横向对比。
        """
        env_info = {"system_time": "", "database": []}
        if hasattr(task, 'environment') and task.environment:
            time_str = task.environment.get("time", "")
            if time_str:
                weekday = get_weekday(time_str, language)
                env_info["system_time"] = f"{time_str} {weekday or ''}"

        result_list, trajectory_eval_info = cls._evaluate_trajectory_full_traj_12dim(
            env_info, task, full_trajectory, llm_evaluator, llm_args_evaluator, language
        )

        # 解析 12 维结果，构造 NLRubricCheck 列表 + 计算 reward
        nl_rubric_checks = []
        activated_count = 0
        met_in_activated = 0
        for item in result_list:
            if not isinstance(item, dict):
                continue
            tag = item.get("capability_tag", "?")
            activated = bool(item.get("activated", False))
            met = bool(item.get("met", False))
            justification = item.get("justification", "") or ""
            # nl_rubric 文本采用 [tag] activated=... met=... 标签格式
            rubric_text = f"[{tag}] activated={activated}"
            nl_rubric_checks.append(NLRubricCheck(
                nl_rubric=rubric_text,
                met=met if activated else True,  # 未激活的填 met=true（占位，不参与 reward）
                justification=justification,
            ))
            if activated:
                activated_count += 1
                if met:
                    met_in_activated += 1

        # reward 计算
        if activated_count == 0:
            reward = 1.0  # 无激活维度，视为 task 不需评估
            note = "no dimension activated"
        else:
            reward = met_in_activated / activated_count
            note = f"{met_in_activated}/{activated_count} activated dims met"

        return RewardInfo(
            reward=reward,
            nl_rubrics=nl_rubric_checks,
            info={
                "evaluation_method": "full_trajectory_12dim",
                "note": note,
                "activated_count": activated_count,
                "met_count": met_in_activated,
            },
            evaluator_usage=trajectory_eval_info.get("usage"),
            reward_breakdown={RewardType.NL_ASSERTION: reward},
        )

    @classmethod
    def _evaluate_trajectory_full_traj_12dim(
            cls,
            env_info: dict,
            task: Task,
            trajectory: List[Message],
            llm_evaluator: str = None,
            llm_args_evaluator: dict = None,
            language: str = None,
    ) -> tuple[list, dict]:
        """评估 12 维 capability 框架；返回 (12-item list, eval_info)。"""
        if llm_evaluator is None:
            llm_evaluator = DEFAULT_LLM_EVALUATOR
        if llm_args_evaluator is None:
            llm_args_evaluator = models[DEFAULT_LLM_EVALUATOR]

        trajectory_content = cls._format_window_content(trajectory)

        prompts = get_prompts(language)
        system_prompt = prompts.full_trajectory_12dim_eval_template.format(
            env_info=env_info,
            user_instruction=task.instructions,
        )

        user_prompt = f"""
    # Input
    <trajectory_content>
    {trajectory_content}
    </trajectory_content>
    """

        messages = [
            SystemMessage(role="system", content=system_prompt),
            UserMessage(role="user", content=user_prompt),
        ]

        assistant_message = generate(
            model=llm_evaluator,
            messages=messages,
            **llm_args_evaluator,
        )

        trajectory_evaluation_info = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "assistant_message_content": assistant_message.content,
            "usage": assistant_message.usage,
        }

        result_data = evaluator_extracter(assistant_message.content)

        # 期望返回是 list[dict]（12 维），如果 LLM 返回 dict 包了一层，尝试取出
        if isinstance(result_data, list):
            result_list = result_data
        elif isinstance(result_data, dict) and "results" in result_data:
            result_list = result_data["results"]
        elif isinstance(result_data, dict):
            # 单 dict 的情况 — 包成 list
            result_list = [result_data]
        else:
            print(f"Warning: 13dim evaluator failed to parse, returning empty")
            result_list = []

        return result_list, trajectory_evaluation_info

    @classmethod
    def calculate_reward_full_traj_wo_rubric(
            cls,
            task: Task,
            full_trajectory: List[Message],
            final_state: dict,
            llm_evaluator: str = None,
            llm_args_evaluator: dict = None,
            language: str = None,
    ) -> RewardInfo:
        """
        【策略四】全轨迹 + 无 rubric 评估。

        最简单的评估方式：将完整对话一次性发给 LLM，不提供 rubric 列表，
        由 LLM 根据用户指令自行判断 agent 是否完成了任务。

        适用于：对话较短 + 不需要细粒度 rubric 评估的场景。

        计算流程：
        1. 快速返回检查
        2. 构建环境信息
        3. 调用 _evaluate_trajectory_full_traj_wo_rubric 一次性评估
        4. meetExpectation=True → reward=1.0，否则 → 0.0

        Args:
            task: 任务定义
            full_trajectory: 完整对话消息列表
            final_state: 仿真结束时的状态
            llm_evaluator: 评估模型名
            llm_args_evaluator: LLM 参数
            language: prompt 语言

        Returns:
            RewardInfo: 评估结果
        """
        if task.evaluation_criteria is None:
            return RewardInfo(
                reward=1.0,
                nl_rubrics=[],
                info={"note": "No evaluation criteria"},
                reward_breakdown={RewardType.NL_ASSERTION: 1.0},
            )

        evaluation_criteria = task.evaluation_criteria
        if not evaluation_criteria.overall_rubrics:
            return RewardInfo(
                reward=1.0,
                nl_rubrics=[],
                info={"note": "No rubric to evaluate"},
                reward_breakdown={RewardType.NL_ASSERTION: 1.0},
            )

        env_info = {
            "system_time": "",
            "database": []
        }

        if hasattr(task, 'environment') and task.environment:
            time_str = task.environment.get("time", "")
            if time_str:
                weekday = get_weekday(time_str, language)
                env_info["system_time"] = f"{time_str} {weekday or ''}"

        final_evaluation, trajectory_eval_info = cls._evaluate_trajectory_full_traj_wo_rubric(
            env_info, task, full_trajectory, llm_evaluator, llm_args_evaluator, language
        )

        final_nl_rubric_checks = cls._convert_states_to_checks_no_rubric(final_evaluation)

        all_expectations_met = final_nl_rubric_checks.met
        reward = 1.0 if all_expectations_met else 0.0

        # 若 evaluation_criteria 中有 enhance_rubrics，在原结果之外独立评估一遍
        enhance_reward_info = None
        enhance_rubrics = getattr(evaluation_criteria, "enhance_rubrics", None)
        if enhance_rubrics:
            print(f"[enhance] Evaluating {len(enhance_rubrics)} enhance_rubrics ...")
            enhance_reward_info = cls._evaluate_enhance_rubrics(
                task, enhance_rubrics, full_trajectory, env_info,
                llm_evaluator=llm_evaluator, llm_args_evaluator=llm_args_evaluator, language=language,
            )

        return RewardInfo(
            reward=reward,
            nl_rubrics=[final_nl_rubric_checks],
            info={"evaluation_method": "full_trajectory_no_rubrics"},
            evaluator_usage=trajectory_eval_info.get("usage"),
            enhance_reward_info=enhance_reward_info,
        )

    @classmethod
    def _evaluate_trajectory_full_traj_wo_rubric(
            cls,
            env_info: dict,
            task: Task,
            trajectory: List[Message],
            llm_evaluator: str = None,
            llm_args_evaluator: dict = None,
            language: str = None,
    ) -> tuple[dict, dict]:
        """
        评估完整对话轨迹（无 rubric），供 calculate_reward_full_traj_wo_rubric 调用。

        将完整对话发送给 LLM，不提供 rubric 列表，LLM 直接返回整体 meetExpectation 判定。
        使用 full_trajectory_no_rubrics_eval_template prompt 模板。

        Args:
            env_info: 环境信息
            task: 任务定义
            trajectory: 完整对话消息列表
            llm_evaluator: 评估模型名
            llm_args_evaluator: 模型参数
            language: prompt 语言

        Returns:
            tuple: (updated_states: 评估结果 dict, trajectory_evaluation_info: 评估详情)
        """
        if llm_evaluator is None:
            llm_evaluator = DEFAULT_LLM_EVALUATOR
        if llm_args_evaluator is None:
            llm_args_evaluator = models[DEFAULT_LLM_EVALUATOR]

        trajectory_content = cls._format_window_content(trajectory)

        prompts = get_prompts(language)
        system_prompt = prompts.full_trajectory_no_rubrics_eval_template.format(
            env_info=env_info,
            user_instruction=task.instructions
        )

        user_prompt = f"""
    # Input
    <trajectory_content>
    {trajectory_content}
    </trajectory_content>
    """

        messages = [
            SystemMessage(role="system", content=system_prompt),
            UserMessage(role="user", content=user_prompt),
        ]

        assistant_message = generate(
            model=llm_evaluator,
            messages=messages,
            **llm_args_evaluator,
        )

        trajectory_evaluation_info = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "assistant_message_content": assistant_message.content,
            "usage": assistant_message.usage,
        }

        result_data = evaluator_extracter(assistant_message.content)

        if result_data and isinstance(result_data, dict):
            updated_states = result_data
        elif result_data and isinstance(result_data, list) and len(result_data) > 0:
            # 如果返回的是列表，取第一个元素作为评估结果
            updated_states = result_data[0]
        else:
            print(f"Warning: Failed to parse LLM response for total trajectory, keeping current states")
            updated_states = {
                "justification": "Not evaluated yet",
                "meetExpectation": False
            }

        return updated_states, trajectory_evaluation_info
