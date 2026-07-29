import json
import math
import re
from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger
from pydantic import BaseModel

from eff.data_model.message import AssistantMessage, ToolMessage, UserMessage
from eff.data_model.simulation import Results


# Per-domain classification of tools as read (retrieval, no side-effect) vs write (state-changing).
# Used by trajectory-efficiency metrics. Conservative: only classify tools we're sure about; unlisted tools
# count as neither read nor write (shown under "uncategorized").
TOOL_CATEGORY: dict[str, dict[str, set[str]]] = {
    "ridehailing": {
        "read": {
            "address_to_longitude_latitude",
            "get_user_all_orders",
            "get_nearby",
            "get_home_address",
            "get_company_address",
            "get_available_car_types",
            "match_drivers",
            "get_ride_order_detail",
            "get_ride_estimate",
            "get_ride_real_time_status",
            "get_history_orders",
            "get_current_location",
            "search_ride_orders",
            "search_pois",
            "get_trip_duration_estimate",
        },
        "write": {
            "select_car_types",
            "create_ride_order",
            "confirm_ride",
            "cancel_ride",
            "create_scheduled_ride",
            "update_ride_destination",
            "add_ride_note",
        },
    },
}


def _canonical_args(arguments: dict) -> str:
    """Canonical JSON for a tool-call arguments dict. Stable under key order."""
    try:
        return json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(sorted(arguments.items())) if isinstance(arguments, dict) else str(arguments)


def _compute_tool_dedup_stats(results: Results) -> dict:
    """跟 trajectory_efficiency_stats.duplicate_tool_calls 不同：
    - 只看查询类工具（写工具不算重复 — 多次下单是 task 本身的需求）
    - 写操作（create_ride_order 等）后清空已见签名（环境状态可能变）
    - STATIC 查询（结果不随时间变）同参重调一律算冗余
    - DYNAMIC 查询（司机/价格/ETA/位置等随时间刷新）按"相邻两次调用的间隔"判定：
      间隔 < DYNAMIC_EXEMPT_SECONDS[tool] 才算冗余（贴脸重查）；够久 = 合理刷新，不算。
      时间戳缺失时不冤枉，按合理刷新处理。（与 tool_dedup.count_short_window_redundancy 同口径）

    这条指标对齐 capability_tag `no-redundant-tool-call`：
    用户重述同一需求时 agent 不应用相同参数重调查询工具。
    """
    from eff.metrics.tool_dedup import (
        STATIC_QUERY_TOOLS, DYNAMIC_QUERY_TOOLS, DYNAMIC_EXEMPT_SECONDS,
        WRITE_TOOLS, _canonical_args as _canon, _parse_ts,
    )

    query_tools = STATIC_QUERY_TOOLS | DYNAMIC_QUERY_TOOLS
    sim_count = len(results.simulations)
    if sim_count == 0:
        return {}

    per_sim = []
    for sim in results.simulations:
        msgs = sim.messages or []
        seen: dict = {}  # sig -> 上次同参调用的 timestamp(datetime|None)；写操作后清空
        total_query = 0
        redundant = 0
        for msg in msgs:
            if not isinstance(msg, AssistantMessage):
                continue
            ts = _parse_ts(getattr(msg, "timestamp", None))
            for tc in msg.tool_calls or []:
                if getattr(tc, "requestor", "assistant") != "assistant":
                    continue
                name = tc.name
                if name in WRITE_TOOLS:
                    # 写操作后环境变了，清空已见签名
                    seen.clear()
                    continue
                if name not in query_tools:
                    continue
                total_query += 1
                sig = (name, _canon(tc.arguments or {}))
                if sig in seen:
                    prev_ts = seen[sig]
                    if name in DYNAMIC_EXEMPT_SECONDS:
                        # 动态查询：仅当与上次同参调用的间隔 < 阈值时才算冗余
                        thr = DYNAMIC_EXEMPT_SECONDS[name]
                        if (prev_ts is not None and ts is not None
                                and (ts - prev_ts).total_seconds() < thr):
                            redundant += 1
                        # 否则视为合理刷新（含时间戳缺失，不冤枉）
                    else:
                        # 静态查询：同参重调即冗余
                        redundant += 1
                seen[sig] = ts  # 始终更新为最近一次（动态按"相邻两次"比间隔）
        per_sim.append({
            "task_id": sim.task_id,
            "total_query_calls": total_query,
            "redundant_calls": redundant,
        })

    total_q = sum(p["total_query_calls"] for p in per_sim)
    total_r = sum(p["redundant_calls"] for p in per_sim)
    sims_with_r = sum(1 for p in per_sim if p["redundant_calls"] > 0)
    top_off = sorted(per_sim, key=lambda p: -p["redundant_calls"])[:5]
    top_off = [(p["task_id"], p["redundant_calls"]) for p in top_off if p["redundant_calls"] > 0]
    return {
        "sim_count": sim_count,
        "total_query_calls": total_q,
        "redundant_calls": total_r,
        "redundant_rate": round(total_r / total_q, 4) if total_q else 0.0,
        "sims_with_redundant": sims_with_r,
        "top_offenders": top_off,
    }


def _compute_trajectory_efficiency_stats(results: Results) -> dict:
    """从对话轨迹中确定性地计算效率指标。

    所有指标都是从 messages 直接派生，无 LLM 判断、无启发式文本匹配，
    确保跨 run 可比、可复现。

    返回 dict：
      agent_turns / user_turns / tool_calls: 总数 + 每 sim 平均
      unique_tools_per_sim: 每 sim 用到的不同工具名数（平均 / 最大 / 最小）
      duplicate_tool_calls: (name, canonical_args) 在同 sim 内重复的次数（首次不计）
      text_only_agent_turns: Assistant 无 tool_calls 的纯文本轮次（诚实命名，不叫"澄清轮次"）
      read_tool_calls / write_tool_calls / uncategorized_tool_calls: 按 domain 分类表聚合
      turns_until_first_write: 首次写类工具调用前的 agent 轮次（-1 表示未调用写类工具）
      tool_calls_by_outcome: {"successful": avg, "failed": avg} —— 按 reward 分组
    """
    from collections import Counter, defaultdict

    sim_count = len(results.simulations)
    if sim_count == 0:
        return {}

    # 按 domain 聚合工具分类（可能有多 domain 混在一个 Results 里）
    read_tools: set[str] = set()
    write_tools: set[str] = set()
    for task in results.tasks:
        cat = TOOL_CATEGORY.get(task.domain)
        if cat:
            read_tools |= cat.get("read", set())
            write_tools |= cat.get("write", set())

    # 为每个 sim 算一套逐项指标，再汇总（平均 / 方差更稳健）
    per_sim: list[dict] = []
    for sim in results.simulations:
        msgs = sim.messages or []
        agent_turns = 0
        user_turns = 0
        text_only_agent = 0
        tool_calls = 0
        read_calls = 0
        write_calls = 0
        uncategorized_calls = 0
        tool_name_set: set[str] = set()
        dedup_counter: Counter = Counter()
        turns_until_first_write = -1  # -1 = 未发生
        saw_first_write = False

        for msg in msgs:
            if isinstance(msg, AssistantMessage):
                agent_turns += 1
                has_tool_calls = bool(msg.tool_calls)
                if not has_tool_calls:
                    text_only_agent += 1
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        # 仅统计 assistant 发起的工具调用（过滤 user 端调用）
                        if getattr(tc, "requestor", "assistant") != "assistant":
                            continue
                        tool_calls += 1
                        tool_name_set.add(tc.name)
                        dedup_counter[(tc.name, _canonical_args(tc.arguments or {}))] += 1
                        if tc.name in read_tools:
                            read_calls += 1
                        elif tc.name in write_tools:
                            write_calls += 1
                            if not saw_first_write:
                                turns_until_first_write = agent_turns - 1  # 本轮之前的 agent 轮次
                                saw_first_write = True
                        else:
                            uncategorized_calls += 1
            elif isinstance(msg, UserMessage):
                user_turns += 1

        duplicate_calls = sum(c - 1 for c in dedup_counter.values() if c > 1)
        reward = sim.reward_info.reward if sim.reward_info else 0.0

        per_sim.append({
            "agent_turns": agent_turns,
            "user_turns": user_turns,
            "text_only_agent_turns": text_only_agent,
            "tool_calls": tool_calls,
            "unique_tools": len(tool_name_set),
            "duplicate_tool_calls": duplicate_calls,
            "read_tool_calls": read_calls,
            "write_tool_calls": write_calls,
            "uncategorized_tool_calls": uncategorized_calls,
            "turns_until_first_write": turns_until_first_write,
            "reward": reward,
            "success": reward == 1.0,
        })

    def _avg(key: str, filter_fn=None) -> float:
        items = [p[key] for p in per_sim if (filter_fn is None or filter_fn(p))]
        return round(sum(items) / len(items), 2) if items else 0.0

    successful_sims = [p for p in per_sim if p["success"]]
    failed_sims = [p for p in per_sim if not p["success"]]

    # turns_until_first_write: 只对真正有写调用的 sim 求平均，否则会被 -1 污染
    twfw_values = [p["turns_until_first_write"] for p in per_sim if p["turns_until_first_write"] >= 0]
    sims_with_write = len(twfw_values)

    return {
        "sim_count": sim_count,
        "avg_agent_turns": _avg("agent_turns"),
        "avg_user_turns": _avg("user_turns"),
        "avg_tool_calls_per_sim": _avg("tool_calls"),
        "avg_unique_tools_per_sim": _avg("unique_tools"),
        "avg_text_only_agent_turns": _avg("text_only_agent_turns"),
        "avg_duplicate_tool_calls": _avg("duplicate_tool_calls"),
        "total_duplicate_tool_calls": sum(p["duplicate_tool_calls"] for p in per_sim),
        "avg_read_tool_calls": _avg("read_tool_calls"),
        "avg_write_tool_calls": _avg("write_tool_calls"),
        "avg_uncategorized_tool_calls": _avg("uncategorized_tool_calls"),
        "sims_with_write_call": sims_with_write,
        "avg_turns_until_first_write": (
            round(sum(twfw_values) / len(twfw_values), 2) if twfw_values else None
        ),
        "avg_tool_calls_successful": _avg("tool_calls", lambda p: p["success"]) if successful_sims else None,
        "avg_tool_calls_failed": _avg("tool_calls", lambda p: not p["success"]) if failed_sims else None,
        "avg_agent_turns_successful": _avg("agent_turns", lambda p: p["success"]) if successful_sims else None,
        "avg_agent_turns_failed": _avg("agent_turns", lambda p: not p["success"]) if failed_sims else None,
    }


def _char_bigrams(s: str) -> set:
    """按 char bigram 分词，对中英混排稳健。短文本退化为单字集合。"""
    s = (s or "").replace(" ", "").replace("\n", "").replace("\t", "")
    if len(s) < 2:
        return set(s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _compute_executive_capability_stats(results: Results) -> dict:
    """报告 Top 3 真失败维度的确定性指标（流程完整性 / 主动性 / 回复节奏）。

    全部从 messages 直接派生，无 LLM 判断、无启发式文本匹配。
    报告 3.1 数据：流程完整性 33.3% + 主动性 29.1% + 信息密度 21.7% = 84% 真失败。

    ── 流程完整性 (process_completeness) ──
    孤儿 create：成功调用 create_ride_order / create_scheduled_ride，但后续没有
    confirm_ride（正常闭环）或 cancel_ride（显式取消）。孤儿 create 数越多 → 流程漏洞越大。
      total_creates: 成功 create 次数
      total_confirms: 成功 confirm 次数
      total_cancels: 成功 cancel 次数
      orphan_creates: 未被 confirm 或 cancel 的孤儿 create 总数
      orphan_create_rate: orphan / total_creates
      sims_with_orphan_create: 至少一个孤儿 create 的 sim 数

    ── 主动性 (proactivity) ──
    （proactive_tool_calls 已在 tool_usage_stats 里，这里换个角度：
    有历史数据 / 家地址 的 user_profile 下，agent 是否真的调了）
      sims_with_history_profile: user_profile 含 history_orders 的 sim 数
      sims_tapped_history: 这些 sim 中真调了 get_history_orders 的
      history_tap_rate: 后者 / 前者
      同理 sims_with_home_profile / sims_tapped_home / home_tap_rate

    ── 增量表达 (incremental_expression) ──
    相邻两轮 agent 回复的 char-bigram Jaccard 相似度。
    高 Jaccard = agent 在重复自己（状态变了但还在说一样的话），反映状态感知 + 增量表达能力。
      avg_consecutive_jaccard: 相邻两条 assistant 文本的 char-bigram Jaccard 平均
      max_consecutive_jaccard: 最大 Jaccard（提示严重复述的 sim）

    ── 工具有效性 (tool_effectiveness) ──
    从工具调用序列确定性推导，度量 agent 是否"调了但没用结果"以及"错了是否盲重试"。
      blind_retry: 同名 + 同参数 再调一次（完全无意义的盲重试）
      adaptive_retry: 同名不同参数 再调（合理的换参重试）
      consecutive_same_read_no_write: 连续调同一个读工具且中间无写操作（结果未被利用的信号）

    ── Token 效率 (token_efficiency) ──
    per-task agent 总 token（input+output）的平均值，跨文件对比时用于"同 task 谁更省"。
    """
    sim_count = len(results.simulations)
    if sim_count == 0:
        return {}

    # 按 domain 聚合工具分类
    read_tools: set[str] = set()
    write_tools: set[str] = set()
    for task in results.tasks:
        cat = TOOL_CATEGORY.get(task.domain)
        if cat:
            read_tools |= cat.get("read", set())
            write_tools |= cat.get("write", set())

    # ═══ 流程完整性 ═══
    total_creates = 0
    total_confirms = 0
    total_cancels = 0
    orphan_creates = 0
    sims_with_orphan = 0

    # ═══ 主动性（基于 user_profile 的条件触发率）═══
    task_profile: dict[str, dict] = {
        t.id: (t.user_scenario.user_profile if t.user_scenario else {})
        for t in results.tasks
    }
    sims_with_hist_profile = sims_tapped_hist = 0
    sims_with_home_profile = sims_tapped_home = 0
    sims_with_company_profile = sims_tapped_company = 0

    # ═══ 增量表达（Jaccard only）═══
    per_sim_jaccards: list[float] = []
    max_jaccard_seen = 0.0

    # ═══ 工具有效性 ═══
    blind_retry = 0
    adaptive_retry = 0
    consecutive_same_read_no_write = 0

    # ═══ Token 效率 ═══
    per_task_tokens: dict[str, list[int]] = {}  # task_id -> [total_tokens per trial]

    # ═══ 决策果断度 ═══
    per_sim_reads_before_first_write: list[int] = []  # only for sims with write
    per_sim_read_write_ratio: list[float] = []  # only for sims with write

    for sim in results.simulations:
        msgs = sim.messages or []

        # -- 流程完整性：扫描订单事件序列 --
        # 只有 create_ride_order 产出的订单是 status=waiting_confirm，需要 confirm_ride 或 cancel_ride 收尾；
        # create_scheduled_ride 产出 status=scheduled，本身即为终态，不算 orphan
        create_positions = []  # (idx, needs_followup) 需跟进的 create 位置
        for i, msg in enumerate(msgs):
            if isinstance(msg, ToolMessage) and msg.requestor == "assistant" and not msg.error:
                if msg.name == "create_ride_order":
                    create_positions.append(i)
                    total_creates += 1
                elif msg.name == "create_scheduled_ride":
                    # 预约单直接终态，只计数不作 orphan 判定
                    total_creates += 1
                elif msg.name == "confirm_ride":
                    total_confirms += 1
                elif msg.name == "cancel_ride":
                    total_cancels += 1

        sim_orphans = 0
        for cpos in create_positions:
            closed = False
            for later in msgs[cpos + 1:]:
                if isinstance(later, ToolMessage) and later.requestor == "assistant" and not later.error:
                    if later.name in ("confirm_ride", "cancel_ride"):
                        closed = True
                        break
            if not closed:
                sim_orphans += 1
        orphan_creates += sim_orphans
        if sim_orphans > 0:
            sims_with_orphan += 1

        # -- 主动性：条件触发率 --
        # 兼容中英文字段名。history_orders 在 ridehailing 场景通常走 environment 侧而非 profile，
        # 所以这里以 task.environment 为主源，profile 里的字段作为兜底。
        profile = task_profile.get(sim.task_id) or {}
        env = next((t.environment for t in results.tasks if t.id == sim.task_id), None) or {}
        history = (
            env.get("history_orders") or env.get("orders")
            or profile.get("history_orders") or profile.get("history")
            or profile.get("打车偏好") or profile.get("历史订单")
        )
        home = (
            profile.get("home_address") or profile.get("home")
            or profile.get("家庭住址") or profile.get("家")
        )
        company = (
            profile.get("company_address") or profile.get("company")
            or profile.get("公司地址") or profile.get("公司")
        )
        tool_names_called = {
            m.name for m in msgs
            if isinstance(m, ToolMessage) and m.requestor == "assistant"
        }
        if history:
            sims_with_hist_profile += 1
            if "get_history_orders" in tool_names_called:
                sims_tapped_hist += 1
        if home:
            sims_with_home_profile += 1
            if "get_home_address" in tool_names_called:
                sims_tapped_home += 1
        if company:
            sims_with_company_profile += 1
            if "get_company_address" in tool_names_called:
                sims_tapped_company += 1

        # -- 增量表达（Jaccard）--
        text_messages: list[str] = []
        for msg in msgs:
            if isinstance(msg, AssistantMessage):
                content = (msg.content or "").strip()
                if content:
                    text_messages.append(content)

        if len(text_messages) >= 2:
            bigram_sets = [_char_bigrams(t) for t in text_messages]
            pair_j = [_jaccard(bigram_sets[i], bigram_sets[i + 1]) for i in range(len(bigram_sets) - 1)]
            sim_avg_j = sum(pair_j) / len(pair_j)
            per_sim_jaccards.append(sim_avg_j)
            max_jaccard_seen = max(max_jaccard_seen, max(pair_j))

        # -- 工具有效性 --
        # 按 AssistantMessage.tool_calls 提取调用序列 + 匹配 ToolMessage 错误状态
        call_seq: list[tuple] = []  # (name, canonical_args, error)
        for msg in msgs:
            if isinstance(msg, AssistantMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    if getattr(tc, "requestor", "assistant") != "assistant":
                        continue
                    call_seq.append((tc.name, _canonical_args(tc.arguments or {}), None))
            elif isinstance(msg, ToolMessage) and msg.requestor == "assistant":
                for i in range(len(call_seq) - 1, -1, -1):
                    if call_seq[i][0] == msg.name and call_seq[i][2] is None:
                        call_seq[i] = (call_seq[i][0], call_seq[i][1], msg.error)
                        break

        prev_call = None
        last_read_name = None
        for name, args, error in call_seq:
            # 重试模式
            if prev_call and prev_call[2] is True:
                if name == prev_call[0]:
                    if args == prev_call[1]:
                        blind_retry += 1
                    else:
                        adaptive_retry += 1
            # 连续同名读无写
            if name in read_tools and error is not True:
                if last_read_name == name:
                    consecutive_same_read_no_write += 1
                last_read_name = name
            elif name in write_tools:
                last_read_name = None
            prev_call = (name, args, error)

        # -- 决策果断度 --
        sim_read_count = 0
        sim_write_count = 0
        sim_reads_before_first_write = 0
        sim_saw_first_write = False
        for name, args, error in call_seq:
            if name in read_tools:
                sim_read_count += 1
                if not sim_saw_first_write:
                    sim_reads_before_first_write += 1
            elif name in write_tools:
                sim_write_count += 1
                sim_saw_first_write = True
        if sim_write_count > 0:
            per_sim_reads_before_first_write.append(sim_reads_before_first_write)
            per_sim_read_write_ratio.append(sim_read_count / sim_write_count)

        # -- Token 效率 --
        sim_tokens = 0
        for msg in msgs:
            if isinstance(msg, AssistantMessage) and msg.usage:
                sim_tokens += msg.usage.get("prompt_tokens", 0) + msg.usage.get("completion_tokens", 0)
        per_task_tokens.setdefault(sim.task_id, []).append(sim_tokens)

    def _avg(xs: list) -> float:
        return round(sum(xs) / len(xs), 2) if xs else 0.0

    return {
        # 流程完整性
        "process_completeness": {
            "total_creates": total_creates,
            "total_confirms": total_confirms,
            "total_cancels": total_cancels,
            "orphan_creates": orphan_creates,
            "orphan_create_rate": round(orphan_creates / total_creates, 4) if total_creates else 0.0,
            "sims_with_orphan_create": sims_with_orphan,
            "sims_with_orphan_rate": round(sims_with_orphan / sim_count, 4) if sim_count else 0.0,
        },
        # 主动性（条件触发率 —— 有 profile 才看是否调用，分母清晰）
        "proactivity": {
            "history": {
                "sims_with_profile": sims_with_hist_profile,
                "sims_tapped": sims_tapped_hist,
                "tap_rate": round(sims_tapped_hist / sims_with_hist_profile, 4) if sims_with_hist_profile else None,
            },
            "home": {
                "sims_with_profile": sims_with_home_profile,
                "sims_tapped": sims_tapped_home,
                "tap_rate": round(sims_tapped_home / sims_with_home_profile, 4) if sims_with_home_profile else None,
            },
            "company": {
                "sims_with_profile": sims_with_company_profile,
                "sims_tapped": sims_tapped_company,
                "tap_rate": round(sims_tapped_company / sims_with_company_profile, 4) if sims_with_company_profile else None,
            },
        },
        # 增量表达（Jaccard）
        "incremental_expression": {
            "avg_consecutive_jaccard": round(sum(per_sim_jaccards) / len(per_sim_jaccards), 4)
                if per_sim_jaccards else 0.0,
            "max_consecutive_jaccard": round(max_jaccard_seen, 4),
            "sims_measured": len(per_sim_jaccards),
        },
        # 工具有效性
        "tool_effectiveness": {
            "blind_retry": blind_retry,
            "adaptive_retry": adaptive_retry,
            "consecutive_same_read_no_write": consecutive_same_read_no_write,
        },
        # 决策果断度
        "decisiveness": {
            "avg_reads_before_first_write": _avg(per_sim_reads_before_first_write),
            "avg_read_write_ratio": _avg(per_sim_read_write_ratio),
            "sims_with_write": len(per_sim_reads_before_first_write),
        },
        # Token 效率（per-task 平均，便于跨文件对比）
        "token_efficiency": {
            "avg_tokens_per_task": round(
                sum(sum(ts) / len(ts) for ts in per_task_tokens.values()) / len(per_task_tokens), 0
            ) if per_task_tokens else 0,
            "per_task": {
                tid: round(sum(ts) / len(ts), 0) for tid, ts in sorted(per_task_tokens.items())
            },
        },
    }


# rubric 文本 → 能力 tag 的启发式映射表
_RUBRIC_TAG_KEYWORDS: dict[str, list[str]] = {
    "constraint-conflict": ["冲突", "trade-off", "权衡", "取舍", "折中", "矛盾", "无法全部满足"],
    "quantitative-comparison": ["价格", "费用", "预估", "对比", "比较", "数据", "具体数据"],
    "eta-constraint": ["时间", "截止", "到达", "准时", "来得及", "赶上", "16:"],
    "proactive-recall": ["主动.*工具调用", "get_history", "get_home", "get_company", "历史.*记录", "主动通过工具"],
    "implicit-constraint": ["备注", "add_ride_note", "行李", "安全座椅", "晕车", "膝盖", "轮椅", "前排"],
    "scheduled-ride": ["预约", "scheduled", "create_scheduled"],
    "cancel-and-rebook": ["取消", "cancel", "重新叫车", "重新匹配"],
    "ambiguity-resolution": ["确认.*哪", "消歧", "候选", "多个.*选择", "列出.*候选"],
    "self-correction": ["承认.*错误", "纠正", "修正", "理解.*偏差"],
    "risk-assessment": ["风险", "概率", "判断.*可行", "紧张.*有可能", "评估.*能否"],
    "plan-and-backtrack": ["备选", "换方案", "不可行", "替代", "主动建议"],
    "temporal-chain": ["反向.*推理", "倒推", "出发时间", "路程时间", "backward"],
    "cost-optimization": ["省钱", "最便宜", "费用.*对比", "总价.*对比", "经济.*方案"],
    "compound-command": ["列出.*需求", "多个需求", "一次性", "5+.*需求", "所有.*需求"],
    "multi-stakeholder": ["两位", "两人", "协调", "折中.*方案", "不同偏好"],
    "implicit-inference": ["推断", "推理.*依据", "推荐.*理由", "弱信号"],
    "mid-dialogue-change": ["更新.*方案", "切换", "改.*目的地", "追加.*途经"],
    "context-carryover": ["引用.*早期", "仍然记得", "保留.*信息", "前后轮.*一致"],
    "multi-stop": ["途经", "多.*目的地", "via", "停靠"],
    "relative-time": ["换算.*时间", "分钟后.*具体时间", "相对时间"],
    "preference-inference": ["历史.*推断", "偏好.*推断", "常用车型"],
    "ambiguous-location": ["厦门站.*北站", "哪个.*站", "多个候选"],
    "policy-boundary": ["代叫", "安全.*风险", "超载", "未成年", "违法"],
    "refusal-required": ["拒绝", "不合理", "说明.*能力范围"],
    "_process_completion": ["创建.*订单", "确认.*订单", "confirmed", "成功.*下单", "创建并确认"],
}


def _match_rubric_to_tags(rubric_text: str) -> list[str]:
    """将一条 rubric 映射到它实际测试的能力 tag（启发式关键词匹配）。"""
    import re
    matched = []
    for tag, keywords in _RUBRIC_TAG_KEYWORDS.items():
        for kw in keywords:
            if re.search(kw, rubric_text):
                matched.append(tag)
                break
    return matched or ["_general"]


def _compute_capability_tag_stats(results: Results) -> dict:
    """按 capability_tags 聚合 pass_rate / avg_reward。

    双层归因：
    1. task_level: 整个 task 的 reward 归因到所有 tag（向后兼容）
    2. rubric_level: 每条 rubric 的 met/not_met 归因到它实际测试的 tag（更准确）

    返回：{tag: {"task_count", "sim_count", "avg_reward", "pass_rate",
                 "rubric_pass_rate", "rubric_total", "rubric_pass"}}
    """
    task_tags: dict[str, list[str]] = {
        t.id: (t.capability_tags or []) for t in results.tasks
    }
    if not any(task_tags.values()):
        return {}

    # Task-level 归因（保留原逻辑）
    agg: dict[str, dict] = {}
    for sim in results.simulations:
        tags = task_tags.get(sim.task_id) or []
        if not tags:
            continue
        reward = sim.reward_info.reward if sim.reward_info else 0.0
        for tag in tags:
            a = agg.setdefault(tag, {"reward_sum": 0.0, "success": 0, "sim_count": 0, "tasks": set()})
            a["reward_sum"] += reward
            a["success"] += 1 if reward == 1.0 else 0
            a["sim_count"] += 1
            a["tasks"].add(sim.task_id)

    # Rubric-level 归因：优先用 task 的 rubric_tag_map（方案 A 显式标注），fallback 到关键词
    task_rubric_tag_maps: dict[str, list] = {}
    for t in results.tasks:
        if t.evaluation_criteria and t.evaluation_criteria.rubric_tag_map:
            task_rubric_tag_maps[t.id] = t.evaluation_criteria.rubric_tag_map
    # 如果 sim 文件中的 task 没有 rubric_tag_map（旧文件），尝试从当前 probe 文件补充
    if not task_rubric_tag_maps:
        try:
            from pathlib import Path
            probe_path = Path("data/eff/domains/ridehailing/tasks_capability_probe.json")
            if probe_path.exists():
                import json as _json
                with open(probe_path) as _f:
                    probe_tasks = _json.load(_f)
                for pt in probe_tasks:
                    rtm = (pt.get("evaluation_criteria") or {}).get("rubric_tag_map")
                    if rtm:
                        task_rubric_tag_maps[pt["id"]] = rtm
        except Exception:
            pass

    rubric_agg: dict[str, dict] = {}
    for sim in results.simulations:
        nrs = sim.reward_info.nl_rubrics if sim.reward_info else []
        if not nrs:
            continue
        rtm = task_rubric_tag_maps.get(sim.task_id)

        # 交集过滤：rubric 标的 tag 必须也是 task 的 capability_tag 才计入
        sim_task_tags = set(task_tags.get(sim.task_id) or [])

        for idx, nr in enumerate(nrs):
            rubric_text = nr.nl_rubric if hasattr(nr, 'nl_rubric') else (nr.get('nl_rubric', '') if isinstance(nr, dict) else '')
            met = nr.met if hasattr(nr, 'met') else (nr.get('met') if isinstance(nr, dict) else None)
            if met is None:
                continue
            if rtm and idx < len(rtm):
                rubric_tags = rtm[idx]
            else:
                rubric_tags = _match_rubric_to_tags(rubric_text)
            # 执行素质 tag（不走交集过滤，全局统计）
            EXEC_QUALITY_TAGS = {"evidence-based-decision", "tool-before-answer", "proactive-tool-use"}
            for rtag in rubric_tags:
                if rtag in EXEC_QUALITY_TAGS or rtag.startswith("_"):
                    # 执行素质 + 通用 tag：全局计入
                    ra = rubric_agg.setdefault(rtag, {"pass": 0, "total": 0})
                    ra["total"] += 1
                    if met:
                        ra["pass"] += 1
                elif rtag in sim_task_tags:
                    # 能力维度 tag：交集过滤
                    ra = rubric_agg.setdefault(rtag, {"pass": 0, "total": 0})
                    ra["total"] += 1
                    if met:
                        ra["pass"] += 1

    out = {}
    for tag, a in sorted(agg.items()):
        n = a["sim_count"]
        ra = rubric_agg.get(tag, {"pass": 0, "total": 0})
        out[tag] = {
            "task_count": len(a["tasks"]),
            "sim_count": n,
            "avg_reward": round(a["reward_sum"] / n, 4) if n else 0.0,
            "pass_rate": round(a["success"] / n, 4) if n else 0.0,
            "rubric_pass_rate": round(ra["pass"] / ra["total"], 4) if ra["total"] else None,
            "rubric_total": ra["total"],
            "rubric_pass": ra["pass"],
        }
    # 加入执行素质 + 通用 tag（不在 task-level agg 里但在 rubric_agg 里）
    for rtag, ra in rubric_agg.items():
        if rtag not in out and ra["total"] > 0:
            out[rtag] = {
                "task_count": 0,
                "sim_count": 0,
                "avg_reward": 0.0,
                "pass_rate": 0.0,
                "rubric_pass_rate": round(ra["pass"] / ra["total"], 4),
                "rubric_total": ra["total"],
                "rubric_pass": ra["pass"],
            }
    return out


def _compute_tool_usage_stats(results: Results) -> dict:
    """从对话轨迹中自动计算工具使用能力指标。

    两个维度：
    1. 工具选择能力：错误率、哪些工具经常出错
    2. 错误恢复能力：出错后是否换方案（vs 重复调同一工具）、最终是否完成任务

    返回 dict 结构：
    {
        "total_tool_calls": int,           # 总工具调用次数
        "total_tool_errors": int,          # 工具调用错误次数
        "tool_error_rate": float,          # 错误率
        "error_tools_distribution": dict,  # 各工具的错误次数 {tool_name: count}
        "avg_tool_calls_per_sim": float,   # 每个 simulation 平均调用次数
        "retry_same_tool_count": int,      # 错误后重复调用同一工具的次数（盲重试）
        "switch_after_error_count": int,   # 错误后换不同工具的次数（好的恢复策略）
        "recovery_success_rate": float,    # 出错的 simulation 中最终仍完成任务的比例
        "sims_with_errors": int,           # 有工具错误的 simulation 数
        "sims_with_errors_succeeded": int, # 有错误但最终成功的 simulation 数
    }
    """
    from collections import Counter, defaultdict

    total_calls = 0
    total_errors = 0
    error_tools = Counter()
    retry_same = 0
    switch_after = 0
    sims_with_errors = 0
    sims_with_errors_succeeded = 0
    sim_count = len(results.simulations)

    for sim in results.simulations:
        msgs = sim.messages or []
        sim_has_error = False
        prev_error_tool = None

        for i, msg in enumerate(msgs):
            if not isinstance(msg, ToolMessage):
                continue
            if msg.requestor != "assistant":
                continue

            total_calls += 1

            if msg.error:
                total_errors += 1
                error_tools[msg.name] += 1
                sim_has_error = True

                # 记录出错的工具名，用于判断后续是否盲重试
                prev_error_tool = msg.name
            else:
                # 当前调用成功：检查上一次是否出错
                if prev_error_tool is not None:
                    if msg.name == prev_error_tool:
                        retry_same += 1  # 出错后又调了同一个工具（可能换了参数，也算重试）
                    else:
                        switch_after += 1  # 出错后换了不同工具（好的恢复策略）
                    prev_error_tool = None

        if sim_has_error:
            sims_with_errors += 1
            if sim.reward_info and sim.reward_info.reward > 0:
                sims_with_errors_succeeded += 1

    recovery_rate = (sims_with_errors_succeeded / sims_with_errors) if sims_with_errors > 0 else 1.0

    # ── 主动性工具调用统计 ──
    # 这些工具 agent 不被用户直接要求但应该主动调用来提升服务质量
    PROACTIVE_TOOLS = {
        "get_history_orders": "查历史订单推断偏好",
        "get_home_address": "查家庭地址",
        "get_company_address": "查公司地址",
    }
    proactive_calls = Counter()
    tool_call_distribution = Counter()

    for sim in results.simulations:
        msgs = sim.messages or []
        for msg in msgs:
            if not isinstance(msg, ToolMessage):
                continue
            if msg.requestor != "assistant":
                continue
            tool_call_distribution[msg.name] += 1
            if msg.name in PROACTIVE_TOOLS:
                proactive_calls[msg.name] += 1

    proactive_stats = {
        "proactive_tool_calls": dict(proactive_calls),
        "proactive_total": sum(proactive_calls.values()),
        "proactive_rate": round(sum(proactive_calls.values()) / sim_count, 2) if sim_count > 0 else 0.0,
        "tool_call_distribution": dict(tool_call_distribution.most_common(15)),
    }

    return {
        "total_tool_calls": total_calls,
        "total_tool_errors": total_errors,
        "tool_error_rate": round(total_errors / total_calls, 4) if total_calls > 0 else 0.0,
        "error_tools_distribution": dict(error_tools.most_common(10)),
        "avg_tool_calls_per_sim": round(total_calls / sim_count, 1) if sim_count > 0 else 0.0,
        "retry_same_tool_count": retry_same,
        "switch_after_error_count": switch_after,
        "recovery_success_rate": round(recovery_rate, 4),
        "sims_with_errors": sims_with_errors,
        "sims_with_errors_succeeded": sims_with_errors_succeeded,
        **proactive_stats,
    }


def is_successful(reward: float) -> bool:
    """
    Check if the reward is successful.
    """
    return reward == 1.0


def _compute_level_metrics(results: Results) -> dict:
    """按 State(L1) / Overall(L2) / Enhance(L3) 三层计算 rubric pass rate 和 task-level binary metrics。

    L1 = expected_states[*].state_rubrics（基础状态校验）
    L2 = overall_rubrics（核心能力评估）
    L3 = enhance_rubrics（增强质量评估）

    Returns:
        {"L1_state": {"avg", "task_pass_rate", "rubric_total", "rubric_pass",
                       "task_count", "pass_hat_ks", "pass_at_ks"}, ...}
    """
    from collections import defaultdict

    # Build per-task state rubric text sets (L1)
    task_state_rubrics: dict[str, set[str]] = {}
    for task in results.tasks:
        state_texts: set[str] = set()
        if task.evaluation_criteria and task.evaluation_criteria.expected_states:
            for es in task.evaluation_criteria.expected_states:
                if es.state_rubrics:
                    state_texts.update(es.state_rubrics)
        task_state_rubrics[task.id] = state_texts

    level_pass = {"L1_state": 0, "L2_overall": 0, "L3_enhance": 0}
    level_total = {"L1_state": 0, "L2_overall": 0, "L3_enhance": 0}

    # task_id -> list of per-trial per-level met values
    task_trial_scores: dict[str, list[dict[str, list[bool]]]] = defaultdict(list)

    for sim in results.simulations:
        state_texts = task_state_rubrics.get(sim.task_id, set())
        trial = {"L1_state": [], "L2_overall": [], "L3_enhance": []}

        # nl_rubrics = state + overall rubrics mixed
        if sim.reward_info and sim.reward_info.nl_rubrics:
            for nr in sim.reward_info.nl_rubrics:
                rubric_text = nr.nl_rubric if hasattr(nr, "nl_rubric") else ""
                met = nr.met if hasattr(nr, "met") else None
                if met is None:
                    continue
                level = "L1_state" if rubric_text in state_texts else "L2_overall"
                trial[level].append(met)
                level_total[level] += 1
                if met:
                    level_pass[level] += 1

        # enhance rubrics -> L3
        if sim.reward_info and sim.reward_info.enhance_reward_info:
            eri = sim.reward_info.enhance_reward_info
            if eri.nl_rubrics:
                for nr in eri.nl_rubrics:
                    met = nr.met if hasattr(nr, "met") else None
                    if met is None:
                        continue
                    trial["L3_enhance"].append(met)
                    level_total["L3_enhance"] += 1
                    if met:
                        level_pass["L3_enhance"] += 1

        task_trial_scores[sim.task_id].append(trial)

    out: dict = {}
    for level in ("L1_state", "L2_overall", "L3_enhance"):
        avg = level_pass[level] / level_total[level] if level_total[level] else 0.0

        # Task-level binary: ALL rubrics of this level passed in a single trial?
        task_binary: dict[str, list[bool]] = {}
        for task_id, trials in task_trial_scores.items():
            binaries = []
            for t in trials:
                scores = t[level]
                if not scores:
                    continue
                binaries.append(all(scores))
            if binaries:
                task_binary[task_id] = binaries

        all_binary = [b for bs in task_binary.values() for b in bs]
        task_pass_rate = sum(all_binary) / len(all_binary) if all_binary else 0.0

        # pass^k / pass@k
        level_pass_hat_ks: dict[int, float] = {}
        level_pass_at_ks: dict[int, float] = {}
        if task_binary:
            min_trials = min(len(v) for v in task_binary.values())
            for k in range(1, min_trials + 1):
                phk_vals = []
                pak_vals = []
                for tid, brs in task_binary.items():
                    n = len(brs)
                    c = sum(brs)
                    phk_vals.append(pass_hat_k(n, c, k))
                    pak_vals.append(pass_at_k(n, c, k))
                level_pass_hat_ks[k] = sum(phk_vals) / len(phk_vals)
                level_pass_at_ks[k] = sum(pak_vals) / len(pak_vals)

        out[level] = {
            "avg": round(avg, 4),
            "task_pass_rate": round(task_pass_rate, 4),
            "rubric_total": level_total[level],
            "rubric_pass": level_pass[level],
            "task_count": len(task_binary),
            "pass_hat_ks": {k: round(v, 4) for k, v in level_pass_hat_ks.items()},
            "pass_at_ks": {k: round(v, 4) for k, v in level_pass_at_ks.items()},
        }
    return out


class AgentMetrics(BaseModel):
    avg_reward: float
    pass_hat_ks: dict[int, float]
    pass_at_n: Optional[dict[int, float]] = None
    average_at_n: Optional[dict[int, float]] = None
    avg_agent_cost: float
    avg_reward_breakdown: Optional[dict] = None
    total_duration: Optional[float] = None
    all_types_metrics: Optional[dict] = None
    token_stats: Optional[dict] = None  # per-role token stats
    enhance_avg_reward: Optional[float] = None
    enhance_avg_reward_excl_proactive: Optional[float] = None  # 排除主动性 rubric 后的 enhance 通过率
    proactive_rubric_pass_rate: Optional[float] = None  # 主动性 rubric 单独通过率
    enhance_avg_reward_breakdown: Optional[dict] = None
    tool_usage_stats: Optional[dict] = None  # 工具使用能力指标
    trajectory_efficiency_stats: Optional[dict] = None  # 轨迹效率指标（从对话确定性推导）
    tool_dedup_stats: Optional[dict] = None  # 查询工具去重（capability `no-redundant-tool-call`，写操作后清空签名）
    capability_tag_stats: Optional[dict] = None  # 按 capability_tags 聚合的 pass_rate / avg_reward
    executive_capability_stats: Optional[dict] = None  # 流程完整性 / 主动性 / 回复节奏（Top 3 真失败）
    partial_credit_avg: Optional[float] = None  # rubric 级别部分得分（过 3/4 条 = 0.75，区别于 binary 的 0/1）
    level_metrics: Optional[dict] = None  # L1/L2/L3 分层指标

    def as_dict(self) -> dict:
        data = {
            "avg_reward": self.avg_reward,
            "avg_agent_cost": self.avg_agent_cost,
        }
        for k, v in self.pass_hat_ks.items():
            data[f"pass_hat_{k}"] = v
        if self.pass_at_n:
            for n, v in self.pass_at_n.items():
                data[f"pass_at_{n}"] = v
        if self.average_at_n:
            for n, v in self.average_at_n.items():
                data[f"average_at_{n}"] = v
        if self.avg_reward_breakdown:
            data["avg_reward_breakdown"] = self.avg_reward_breakdown
        if self.total_duration:
            data["total_duration"] = self.total_duration
        if self.all_types_metrics:
            data["all_types_metrics"] = self.all_types_metrics
        return data


def pass_hat_k(num_trials: int, success_count: int, k: int) -> float:
    """
    Compute the pass^k metric for the given number of trials, success count, and k.
    from https://arxiv.org/pdf/2406.12045
    Args:
        num_trials: The number of trials.
        success_count: The number of successful trials.
        k: The number of trials to consider.
    Returns:
        The pass^k metric.
    """
    if num_trials < k:
        raise ValueError(f"Number of trials {num_trials} is less than k {k}.")
    return math.comb(success_count, k) / math.comb(num_trials, k)


def pass_at_k(num_trials: int, success_count: int, k: int) -> float:
    """
    Compute the pass@k metric for the given number of trials, success count, and k.
    Based on the formula: pass@k = 1 - (n-c choose k) / (n choose k)
    where n is the number of trials and c is the number of successful trials.
    
    Args:
        num_trials: The number of trials (n).
        success_count: The number of successful trials (c).
        k: The number of trials to consider.
    Returns:
        The pass@k metric.
    """
    if num_trials < k:
        return 0.0
    
    if success_count > num_trials:
        return 0.0
    
    if num_trials - success_count >= k:
        # If we have enough unsuccessful trials to choose k
        return 1.0 - (math.comb(num_trials - success_count, k) / math.comb(num_trials, k))
    else:
        # If we don't have enough unsuccessful trials, pass@k = 1
        return 1.0


def average_at_k(rewards: list[float], k: int) -> float:
    if len(rewards) < k:
        return 0.0
    
    if k == 0:
        return 0.0
    
    return sum(rewards) / len(rewards)


def get_metrics_df(results: Results) -> tuple[pd.DataFrame, int]:
    """
    Convert the results to a dataframe and add a column for success.
    Checks that all simulations have the same number of trials.
    Returns the maximum number of trials that can be used for pass^k metrics.
    """
    df = results.to_df()
    df["success"] = df.reward.apply(is_successful)
    if len(df.info_num_trials.unique()) > 1:
        logger.warning(
            f"All simulations must have the same number of trials. Found {df.info_num_trials.unique()}"
        )
    max_k = df.info_num_trials.max()

    task_ids_counts = [(tid, count) for tid, count in df.task_id.value_counts().items()]
    task_ids_counts.sort(key=lambda x: x[1])
    min_k = task_ids_counts[0][1]
    if min_k < max_k:
        logger.warning(
            f"The minimum number of trials for a task is {min_k}, which is less than the expected number of trials {max_k}. Setting max k to {min_k}."
        )
        max_k = min_k
    return df, max_k


def get_tasks_pass_hat_k(results: Results) -> pd.DataFrame:
    """
    Compute the pass^k for each k from 1 to the maximum number of trials.
    """
    df, max_k = get_metrics_df(results)
    dfs = []
    for k in range(1, max_k + 1):
        res = df.groupby("task_id")["success"].apply(
            lambda df: pass_hat_k(len(df), df.sum(), k)
        )
        res.name = f"pass^{k}"
        dfs.append(res)
    df_pass_hat_k = pd.concat(dfs, axis=1)
    return df_pass_hat_k


def prepare_dfs(results: Results) -> tuple[pd.DataFrame, pd.DataFrame]:
    df, max_k = get_metrics_df(results)
    df_pass_hat_k = get_tasks_pass_hat_k(results)
    return df, df_pass_hat_k


def compute_metrics(results: Results) -> AgentMetrics:
    """
    Compute metrics for the agent.
    - average reward
    - pass^k
    - average reward breakdown
    - total duration
    """
    df, df_pass_hat_k = prepare_dfs(results)
    avg_reward = df.reward.mean()
    pass_hat_ks = {}
    for column in df_pass_hat_k.columns:
        if match := re.match(r"pass\^(\d+)", column):
            k = int(match.group(1))
            pass_hat_ks[k] = df_pass_hat_k[column].mean()

    # Calculate pass@k and average@k based on the mathematical formula from the paper
    # pass@k = 1 - E_task [ (n - c choose k) / (n choose k) ]
    pass_at_n = {}
    average_at_n = {}
    num_trials = results.info.num_trials
    
    # Group by task_id to calculate pass@k and average@k
    task_groups = df.groupby("task_id")
    for k in range(1, num_trials + 1):
        pass_at_k_values = []
        average_at_k_values = []
        
        for task_id, task_df in task_groups:
            if len(task_df) >= k:
                n = len(task_df)  # number of trials for this task
                c = task_df["success"].sum()  # number of successful trials
                
                # Calculate pass@k using the helper function
                pass_at_k_value = pass_at_k(n, c, k)
                pass_at_k_values.append(pass_at_k_value)
                
                # Calculate average@k using the helper function
                rewards = task_df["reward"].tolist()
                average_at_k_value = average_at_k(rewards, k)
                average_at_k_values.append(average_at_k_value)
        
        if pass_at_k_values:
            pass_at_n[k] = sum(pass_at_k_values) / len(pass_at_k_values)
        if average_at_k_values:
            average_at_n[k] = sum(average_at_k_values) / len(average_at_k_values)

    avg_agent_cost = df.agent_cost.mean()
    
    # Calculate average reward breakdown
    avg_reward_breakdown = {}
    reward_breakdown_counts = {}
    for sim in results.simulations:
        if sim.reward_info and sim.reward_info.reward_breakdown:
            for reward_type, value in sim.reward_info.reward_breakdown.items():
                if reward_type not in avg_reward_breakdown:
                    avg_reward_breakdown[reward_type] = 0.0
                    reward_breakdown_counts[reward_type] = 0
                avg_reward_breakdown[reward_type] += value
                reward_breakdown_counts[reward_type] += 1

    # Convert to averages
    for reward_type in avg_reward_breakdown:
        if reward_breakdown_counts[reward_type] > 0:
            avg_reward_breakdown[reward_type] /= reward_breakdown_counts[reward_type]

    # Calculate total duration as the time difference between the latest end_time and earliest start_time
    if results.simulations:
        # Parse start_time and end_time strings to datetime objects
        start_times = []
        end_times = []
        for sim in results.simulations:
            try:
                start_times.append(datetime.strptime(sim.start_time, "%Y%m%d_%H%M%S"))
                end_times.append(datetime.strptime(sim.end_time, "%Y%m%d_%H%M%S"))
            except ValueError:
                # Fallback to original duration calculation if time parsing fails
                logger.warning(f"Failed to parse time format for simulation {sim.id}, using original duration calculation")
                total_duration = sum(sim.duration for sim in results.simulations)
                break
        else:
            # If all time parsing succeeded, calculate the time difference
            earliest_start = min(start_times)
            latest_end = max(end_times)
            total_duration = (latest_end - earliest_start).total_seconds()
    else:
        total_duration = 0.0

    # Check if we have all_types evaluation results
    all_types_metrics = {}
    if len(results.simulations) > 0 and results.simulations[0].reward_info and results.simulations[0].reward_info.info:
        first_sim = results.simulations[0]
        if first_sim.reward_info.info.get("evaluation_methods") == ["trajectory"]:
            # We have all_types evaluation, compute metrics for each method

            all_types_metrics = {}
            
            # Compute trajectory metrics
            trajectory_rewards = []
            trajectory_task_ids = []
            for sim in results.simulations:
                if sim.reward_info and sim.reward_info.info and "trajectory_evaluation" in sim.reward_info.info:
                    trajectory_rewards.append(sim.reward_info.info["trajectory_evaluation"]["reward"])
                    trajectory_task_ids.append(sim.task_id)
            
            if trajectory_rewards:
                trajectory_df = pd.DataFrame({
                    "reward": trajectory_rewards,
                    "task_id": trajectory_task_ids
                })
                trajectory_df["success"] = trajectory_df.reward.apply(is_successful)
                
                # Calculate pass_hat_ks for trajectory evaluation using the same logic
                trajectory_pass_hat_ks = {}
                # Get the minimum number of trials across all tasks
                task_counts = trajectory_df.groupby("task_id").size()
                min_trials = task_counts.min()
                max_k = min_trials
                
                for k in range(1, max_k + 1):
                    if min_trials >= k:
                        # Group by task_id and calculate pass^k for each task, then take mean
                        task_pass_ks = trajectory_df.groupby("task_id")["success"].apply(
                            lambda df: pass_hat_k(len(df), df.sum(), k)
                        )
                        trajectory_pass_hat_ks[k] = task_pass_ks.mean()
                
                # Compute trajectory reward breakdown
                trajectory_breakdown = {}
                trajectory_breakdown_counts = {}
                for sim in results.simulations:
                    if sim.reward_info and sim.reward_info.info and "trajectory_evaluation" in sim.reward_info.info:
                        eval_info = sim.reward_info.info["trajectory_evaluation"]
                        if "reward_breakdown" in eval_info and eval_info["reward_breakdown"] is not None:
                            for reward_type, value in eval_info["reward_breakdown"].items():
                                if reward_type not in trajectory_breakdown:
                                    trajectory_breakdown[reward_type] = 0.0
                                    trajectory_breakdown_counts[reward_type] = 0
                                trajectory_breakdown[reward_type] += value
                                trajectory_breakdown_counts[reward_type] += 1

                # Convert to averages
                for reward_type in trajectory_breakdown:
                    if trajectory_breakdown_counts[reward_type] > 0:
                        trajectory_breakdown[reward_type] /= trajectory_breakdown_counts[reward_type]

                # Calculate pass@k and average@k for trajectory evaluation using the same formula
                # pass@k = 1 - E_task [ (n - c choose k) / (n choose k) ]
                trajectory_pass_at_n = {}
                trajectory_average_at_n = {}
                
                trajectory_task_groups = trajectory_df.groupby("task_id")
                for k in range(1, num_trials + 1):
                    trajectory_pass_at_k_values = []
                    trajectory_average_at_k_values = []
                    
                    for task_id, task_df in trajectory_task_groups:
                        if len(task_df) >= k:
                            n = len(task_df)  # number of trials for this task
                            c = task_df["success"].sum()  # number of successful trials
                            
                            # Calculate pass@k using the helper function
                            trajectory_pass_at_k_value = pass_at_k(n, c, k)
                            trajectory_pass_at_k_values.append(trajectory_pass_at_k_value)
                            
                            # Calculate average@k using the helper function
                            rewards = task_df["reward"].tolist()
                            trajectory_average_at_k_value = average_at_k(rewards, k)
                            trajectory_average_at_k_values.append(trajectory_average_at_k_value)
                    
                    if trajectory_pass_at_k_values:
                        trajectory_pass_at_n[k] = sum(trajectory_pass_at_k_values) / len(trajectory_pass_at_k_values)
                    if trajectory_average_at_k_values:
                        trajectory_average_at_n[k] = sum(trajectory_average_at_k_values) / len(trajectory_average_at_k_values)

                all_types_metrics["trajectory"] = {
                    "avg_reward": trajectory_df.reward.mean(),
                    "pass_hat_ks": trajectory_pass_hat_ks,
                    "pass_at_n": trajectory_pass_at_n,
                    "average_at_n": trajectory_average_at_n,
                    "avg_reward_breakdown": trajectory_breakdown if trajectory_breakdown else None,
                }


    # Calculate enhance_reward_info metrics
    # 拆分为三组：proactive（主动性）、其他 enhance、全部 enhance
    PROACTIVE_KEYWORDS = ('历史订单', '常用车型', 'get_history_orders')

    enhance_rewards = []
    enhance_rewards_excl_proactive = []  # 排除主动性 rubric 的 enhance
    proactive_pass = 0
    proactive_total = 0
    enhance_breakdown = {}
    enhance_breakdown_counts = {}

    for sim in results.simulations:
        if sim.reward_info and sim.reward_info.enhance_reward_info:
            eri = sim.reward_info.enhance_reward_info
            enhance_rewards.append(eri.reward)
            if eri.reward_breakdown:
                for reward_type, value in eri.reward_breakdown.items():
                    if reward_type not in enhance_breakdown:
                        enhance_breakdown[reward_type] = 0.0
                        enhance_breakdown_counts[reward_type] = 0
                    enhance_breakdown[reward_type] += value
                    enhance_breakdown_counts[reward_type] += 1

            # 逐条 rubric 拆分统计
            if eri.nl_rubrics:
                other_rubrics = []
                for nr in eri.nl_rubrics:
                    is_proactive = any(kw in nr.nl_rubric for kw in PROACTIVE_KEYWORDS)
                    if is_proactive:
                        proactive_total += 1
                        if nr.met:
                            proactive_pass += 1
                    else:
                        other_rubrics.append(nr)
                # 排除主动性 rubric 后的 enhance 通过率
                if other_rubrics:
                    other_score = sum(1.0 if r.met else 0.0 for r in other_rubrics) / len(other_rubrics)
                    enhance_rewards_excl_proactive.append(other_score)

    enhance_avg_reward = sum(enhance_rewards) / len(enhance_rewards) if enhance_rewards else None
    enhance_avg_reward_excl_proactive = (
        sum(enhance_rewards_excl_proactive) / len(enhance_rewards_excl_proactive)
        if enhance_rewards_excl_proactive else None
    )
    proactive_rubric_pass_rate = (proactive_pass / proactive_total) if proactive_total > 0 else None

    for reward_type in enhance_breakdown:
        if enhance_breakdown_counts[reward_type] > 0:
            enhance_breakdown[reward_type] /= enhance_breakdown_counts[reward_type]

    # Compute per-model token stats
    agent_model = results.info.agent_info.llm
    user_model = results.info.user_info.llm
    token_stats: dict = {}

    def _add_usage(stats: dict, model: str, role: str, input_tokens: int, output_tokens: int):
        key = f"{role} ({model})" if model else role
        if key not in stats:
            stats[key] = {"total_input_tokens": 0, "total_output_tokens": 0, "sim_count": 0}
        stats[key]["total_input_tokens"] += input_tokens
        stats[key]["total_output_tokens"] += output_tokens
        stats[key]["sim_count"] += 1

    evaluator_model = getattr(results.info, "llm_evaluator", None) or results.info.agent_info.llm

    for sim in results.simulations:
        agent_input = agent_output = 0
        user_input = user_output = 0
        for msg in sim.messages:
            if isinstance(msg, AssistantMessage) and msg.usage:
                agent_input += msg.usage.get("prompt_tokens", 0)
                agent_output += msg.usage.get("completion_tokens", 0)
            elif isinstance(msg, UserMessage) and msg.usage:
                user_input += msg.usage.get("prompt_tokens", 0)
                user_output += msg.usage.get("completion_tokens", 0)
        _add_usage(token_stats, agent_model, "agent", agent_input, agent_output)
        if user_model:
            _add_usage(token_stats, user_model, "user", user_input, user_output)

        if sim.reward_info and sim.reward_info.evaluator_usage:
            eu = sim.reward_info.evaluator_usage
            _add_usage(token_stats, evaluator_model, "evaluator",
                       eu.get("prompt_tokens", 0), eu.get("completion_tokens", 0))

    for stats in token_stats.values():
        n = stats.pop("sim_count")
        stats["avg_input_tokens"] = stats["total_input_tokens"] / n if n else 0.0
        stats["avg_output_tokens"] = stats["total_output_tokens"] / n if n else 0.0

    # ── 工具使用能力指标 ──
    # 从对话轨迹中自动计算，不需要 LLM 评估
    tool_usage_stats = _compute_tool_usage_stats(results)

    # ── 轨迹效率指标 + 能力标签聚合 + 执行能力三件套 + L1/L2/L3 分层 ──
    trajectory_efficiency_stats = _compute_trajectory_efficiency_stats(results)
    tool_dedup_stats = _compute_tool_dedup_stats(results)
    capability_tag_stats = _compute_capability_tag_stats(results)
    executive_capability_stats = _compute_executive_capability_stats(results)
    level_metrics = _compute_level_metrics(results)

    # Partial credit: rubric 级别部分得分
    partial_scores = []
    for sim in results.simulations:
        if sim.reward_info and sim.reward_info.nl_rubrics:
            nrs = sim.reward_info.nl_rubrics
            met = sum(1 for r in nrs if r.met)
            partial_scores.append(met / len(nrs) if nrs else 0.0)
        elif sim.reward_info:
            partial_scores.append(sim.reward_info.reward)
    partial_credit_avg = sum(partial_scores) / len(partial_scores) if partial_scores else None

    return AgentMetrics(
        avg_reward=avg_reward,
        pass_hat_ks=pass_hat_ks,
        pass_at_n=pass_at_n,
        average_at_n=average_at_n,
        avg_agent_cost=avg_agent_cost,
        avg_reward_breakdown=avg_reward_breakdown if avg_reward_breakdown else None,
        total_duration=total_duration,
        all_types_metrics=all_types_metrics if all_types_metrics else None,
        token_stats=token_stats if token_stats else None,
        enhance_avg_reward=enhance_avg_reward,
        enhance_avg_reward_excl_proactive=enhance_avg_reward_excl_proactive,
        proactive_rubric_pass_rate=proactive_rubric_pass_rate,
        enhance_avg_reward_breakdown=enhance_breakdown if enhance_breakdown else None,
        tool_usage_stats=tool_usage_stats,
        trajectory_efficiency_stats=trajectory_efficiency_stats if trajectory_efficiency_stats else None,
        tool_dedup_stats=tool_dedup_stats if tool_dedup_stats else None,
        capability_tag_stats=capability_tag_stats if capability_tag_stats else None,
        executive_capability_stats=executive_capability_stats if executive_capability_stats else None,
        partial_credit_avg=partial_credit_avg,
        level_metrics=level_metrics if level_metrics else None,
    )


def display_metrics(metrics: AgentMetrics) -> None:
    print(f"🏆 Average reward: {metrics.avg_reward}")
    if metrics.partial_credit_avg is not None:
        print(f"📊 Partial credit avg: {metrics.partial_credit_avg:.4f}  (rubric-level, vs binary {metrics.avg_reward:.4f})")

    # ── L1/L2/L3 分层指标 ──
    if metrics.level_metrics:
        LEVEL_LABELS = {
            "L1_state": "L1 State（基础状态）",
            "L2_overall": "L2 Overall（核心能力）",
            "L3_enhance": "L3 Enhance（增强质量）",
        }
        print(f"\n📊 L1/L2/L3 Rubric Pass Rate:")
        print(f"  {'Level':<30s} {'avg':>8s} {'task_pass%':>10s} {'rubrics':>10s}")
        for level_key in ("L1_state", "L2_overall", "L3_enhance"):
            lm = metrics.level_metrics.get(level_key)
            if not lm:
                continue
            label = LEVEL_LABELS[level_key]
            avg_pct = f"{lm['avg']*100:.1f}%"
            tp_pct = f"{lm['task_pass_rate']*100:.1f}%"
            rubrics = f"{lm['rubric_pass']}/{lm['rubric_total']}"
            print(f"  {label:<30s} {avg_pct:>8s} {tp_pct:>10s} {rubrics:>10s}")
            if lm.get("pass_hat_ks"):
                parts = [f"pass^{k}={v:.4f}" for k, v in lm["pass_hat_ks"].items()]
                print(f"    {' | '.join(parts)}")
            if lm.get("pass_at_ks"):
                parts = [f"pass@{k}={v:.4f}" for k, v in lm["pass_at_ks"].items()]
                print(f"    {' | '.join(parts)}")

    print("📈 Pass^k")
    for k, pass_hat_k in metrics.pass_hat_ks.items():
        print(f"  k={k}: {pass_hat_k:.4f}")

    # Display pass@k and average@k metrics
    if metrics.pass_at_n:
        print("📈 Pass@K")
        for k, pass_at_k_value in metrics.pass_at_n.items():
            print(f"  k={k}: {pass_at_k_value:.4f}")

    if metrics.average_at_n:
        print("📈 Average@K")
        for k, average_at_k_value in metrics.average_at_n.items():
            print(f"  k={k}: {average_at_k_value:.4f}")

    print(f"💰 Average agent cost: {metrics.avg_agent_cost}")
    
    # Display reward breakdown averages
    if metrics.avg_reward_breakdown:
        print("\n📊 Average Reward Breakdown:")
        for reward_type, avg_value in metrics.avg_reward_breakdown.items():
            print(f"  {reward_type}: {avg_value:.4f}")

    # Display total duration
    if metrics.total_duration:
        print(f"\n⏱️ Total Duration: {metrics.total_duration/60:.2f}min")

    # Display token stats
    if metrics.token_stats:
        print("\n🔢 Token Usage:")
        for role_model, stats in metrics.token_stats.items():
            print(f"  [{role_model}]")
            print(f"    Total Input:  {stats['total_input_tokens']:,}")
            print(f"    Total Output: {stats['total_output_tokens']:,}")
            print(f"    Avg Input/sim:  {stats['avg_input_tokens']:.1f}")
            print(f"    Avg Output/sim: {stats['avg_output_tokens']:.1f}")

    # Display enhance metrics
    if metrics.enhance_avg_reward is not None:
        print(f"\n✨ Enhance Average Reward: {metrics.enhance_avg_reward:.4f}")
        if metrics.enhance_avg_reward_excl_proactive is not None:
            print(f"✨ Enhance (excl proactive): {metrics.enhance_avg_reward_excl_proactive:.4f}")
        if metrics.proactive_rubric_pass_rate is not None:
            print(f"🧠 Proactive Rubric Pass Rate: {metrics.proactive_rubric_pass_rate:.4f}")
        if metrics.enhance_avg_reward_breakdown:
            print("✨ Enhance Reward Breakdown:")
            for reward_type, avg_value in metrics.enhance_avg_reward_breakdown.items():
                print(f"  {reward_type}: {avg_value:.4f}")

    # Display all_types metrics if available
    if metrics.all_types_metrics:
        print("\n🔄 All Evaluation Types Results:")
        for eval_type, eval_metrics in metrics.all_types_metrics.items():
            print(f"  {eval_type.upper()}:")
            if "avg_reward" in eval_metrics:
                print(f"    Average reward: {eval_metrics['avg_reward']:.4f}")

            # Display reward breakdown for this evaluation type
            if eval_metrics.get("avg_reward_breakdown"):
                print("    Reward Breakdown:")
                for reward_type, avg_value in eval_metrics["avg_reward_breakdown"].items():
                    print(f"      {reward_type}: {avg_value:.4f}")

            if "pass_hat_ks" in eval_metrics:
                print("    Pass^k:")
                for k, pass_hat_k in eval_metrics["pass_hat_ks"].items():
                    print(f"      k={k}: {pass_hat_k:.4f}")

            if "pass_at_n" in eval_metrics:
                print("    Pass@K:")
                for k, pass_at_k_value in eval_metrics["pass_at_n"].items():
                    print(f"      k={k}: {pass_at_k_value:.4f}")

            if "average_at_n" in eval_metrics:
                print("    Average@K:")
                for k, average_at_k_value in eval_metrics["average_at_n"].items():
                    print(f"      k={k}: {average_at_k_value:.4f}")


    # Display tool usage stats
    if metrics.tool_usage_stats:
        ts = metrics.tool_usage_stats
        print(f"\n🔧 Tool Usage Stats:")
        print(f"  Total calls: {ts['total_tool_calls']}, Errors: {ts['total_tool_errors']} ({ts['tool_error_rate']:.1%})")
        print(f"  Avg calls/sim: {ts['avg_tool_calls_per_sim']}")
        if ts['total_tool_errors'] > 0:
            print(f"  Error distribution: {ts['error_tools_distribution']}")
            print(f"  After error → retry same tool: {ts['retry_same_tool_count']}, switch tool: {ts['switch_after_error_count']}")
            print(f"  Sims with errors: {ts['sims_with_errors']}, still succeeded: {ts['sims_with_errors_succeeded']} ({ts['recovery_success_rate']:.1%})")
        # 主动性工具
        print(f"\n🧠 Proactive Tool Usage (agent 自主性):")
        print(f"  Total proactive calls: {ts.get('proactive_total', 0)} (avg {ts.get('proactive_rate', 0)}/sim)")
        for tool, count in (ts.get('proactive_tool_calls') or {}).items():
            print(f"    {tool}: {count}")
        if ts.get('proactive_total', 0) == 0:
            print(f"  ⚠️ Agent 未主动调用任何上下文工具（get_history_orders/get_home_address/get_company_address）")

    # Display trajectory efficiency stats
    if metrics.trajectory_efficiency_stats:
        te = metrics.trajectory_efficiency_stats
        print(f"\n📐 Trajectory Efficiency (deterministic from messages):")
        print(f"  Avg agent turns: {te['avg_agent_turns']}  |  user turns: {te['avg_user_turns']}  |  text-only agent turns: {te['avg_text_only_agent_turns']}")
        print(f"  Avg tool calls/sim: {te['avg_tool_calls_per_sim']}  (unique tools: {te['avg_unique_tools_per_sim']})")
        print(f"  Read/Write/Uncategorized: {te['avg_read_tool_calls']} / {te['avg_write_tool_calls']} / {te['avg_uncategorized_tool_calls']}")
        print(f"  Duplicate calls (same name+args): total={te['total_duplicate_tool_calls']}, avg/sim={te['avg_duplicate_tool_calls']}")
        if te.get('avg_turns_until_first_write') is not None:
            print(f"  Avg agent turns before first write-tool call: {te['avg_turns_until_first_write']}  (sims w/ write: {te['sims_with_write_call']}/{te['sim_count']})")
        else:
            print(f"  No write-tool calls observed (sims w/ write: 0/{te['sim_count']})")
        if te.get('avg_tool_calls_successful') is not None and te.get('avg_tool_calls_failed') is not None:
            print(f"  Tool calls by outcome — success: {te['avg_tool_calls_successful']}, fail: {te['avg_tool_calls_failed']}")
        if te.get('avg_agent_turns_successful') is not None and te.get('avg_agent_turns_failed') is not None:
            print(f"  Agent turns  by outcome — success: {te['avg_agent_turns_successful']}, fail: {te['avg_agent_turns_failed']}")

    # Display tool dedup stats (capability `no-redundant-tool-call`)
    if metrics.tool_dedup_stats:
        td = metrics.tool_dedup_stats
        print(f"\n🔁 Tool Dedup (capability: no-redundant-tool-call):")
        print(f"  Query calls: {td['total_query_calls']}  |  Redundant: {td['redundant_calls']} ({td['redundant_rate']*100:.1f}%)")
        print(f"  Sims with ≥1 redundant: {td['sims_with_redundant']}/{td['sim_count']}")
        if td.get('top_offenders'):
            top_str = ', '.join(f"{tid[:30]}×{n}" for tid, n in td['top_offenders'][:3])
            print(f"  Top offenders: {top_str}")
        else:
            print(f"  ✓ 无重复调用（上下文意识满分）")

    # Display executive-capability (Top 3 failure dims — deterministic)
    if metrics.executive_capability_stats:
        ec = metrics.executive_capability_stats
        pc = ec["process_completeness"]
        pr = ec["proactivity"]
        print(f"\n🎯 Executive Capability (deterministic signals):")
        print(f"  [Process Completeness]")
        print(f"    creates={pc['total_creates']}  confirms={pc['total_confirms']}  cancels={pc['total_cancels']}")
        print(f"    orphan creates (no confirm/cancel after): {pc['orphan_creates']}  (rate={pc['orphan_create_rate']:.1%})")
        print(f"    sims with ≥1 orphan create: {pc['sims_with_orphan_create']}  ({pc['sims_with_orphan_rate']:.1%})")
        print(f"  [Proactivity — conditional tap rate: sims where profile has X, agent actually called get_X]")
        for field in ("history", "home", "company"):
            f = pr[field]
            if f["sims_with_profile"] > 0:
                print(f"    {field:<8s}: {f['sims_tapped']}/{f['sims_with_profile']}  (rate={f['tap_rate']:.1%})")
            else:
                print(f"    {field:<8s}: no sims have this profile")
        ie = ec["incremental_expression"]
        print(f"  [Incremental Expression — lower Jaccard = less repetition = better]")
        print(f"    avg consecutive-reply Jaccard: {ie['avg_consecutive_jaccard']}  (max={ie['max_consecutive_jaccard']})  (n_sims={ie['sims_measured']})")
        te = ec["tool_effectiveness"]
        print(f"  [Tool Effectiveness]")
        print(f"    blind retry (same name+args after error): {te['blind_retry']}")
        print(f"    adaptive retry (same name, diff args after error): {te['adaptive_retry']}")
        print(f"    consecutive same-read without write: {te['consecutive_same_read_no_write']}")
        dc = ec["decisiveness"]
        print(f"  [Decisiveness — fewer reads before first write = more decisive]")
        print(f"    avg reads before first write: {dc['avg_reads_before_first_write']}  (lower=more decisive)")
        print(f"    avg read/write ratio: {dc['avg_read_write_ratio']}  (lower=more efficient)")
        tk = ec["token_efficiency"]
        print(f"  [Token Efficiency]")
        print(f"    avg agent tokens per task (per trial): {tk['avg_tokens_per_task']:,.0f}")

    # Display capability-tag breakdown (dual: task-level + rubric-level)
    if metrics.capability_tag_stats:
        # 按 Cooper 文档 v5 定义的 5 组分层输出（内部按 rubric n 降序）
        CAPABILITY_GROUPS = [
            ("A 意图理解 Intent",       ["implicit-constraint","proactive-recall","preference-inference",
                                        "ambiguous-location","relative-time"]),
            ("B 多约束组合 Multi-constraint",
                                       ["eta-constraint","multi-param","scheduled-ride","multi-stop","price-ceiling"]),
            ("C 状态追踪 State-tracking",
                                       ["cancel-and-rebook","mid-dialogue-change","context-carryover","long-horizon"]),
            ("D 高阶推理 Reasoning",    ["constraint-conflict","cost-optimization","quantitative-comparison",
                                        "ambiguity-resolution","temporal-chain","implicit-inference",
                                        "plan-and-backtrack","compound-command","multi-stakeholder",
                                        "cascading-recovery","multi-objective-tradeoff","risk-assessment",
                                        "self-correction"]),
            ("F 安全 Safety",           ["refusal-required","policy-boundary"]),
        ]
        # 置信度阈值：rubric 样本 < 5 视为不可信，标 ⚠
        RELIABILITY_THRESHOLD = 5

        def _rubric_reliability_flag(n: int) -> str:
            return "⚠" if n < RELIABILITY_THRESHOLD else " "

        print(f"\n🏷️  Capability Tag Breakdown（按 5 组分层；⚠ = rubric 样本 < {RELIABILITY_THRESHOLD}，信号不可信）:")
        grand_pass = 0
        grand_total = 0
        for group_name, group_tags in CAPABILITY_GROUPS:
            print(f"\n  ── {group_name} ──")
            print(f"    {'tag':<28s} {'task_pass%':>10s} {'rubric_pass%':>12s} {'Δ':>6s}  {'rubrics':>8s}")
            group_rp = 0
            group_rt = 0
            for tag in group_tags:
                s = metrics.capability_tag_stats.get(tag)
                if not s:
                    # tag 在词表但跑这批数据没命中，打 - 提示
                    print(f"    {tag:<28s}  {'-':>9s} {'-':>12s} {'':>6s}  n={0:>5d} ⚠")
                    continue
                tp = s['pass_rate'] * 100
                rp = s.get('rubric_pass_rate')
                rp_str = f"{rp*100:.0f}%" if rp is not None else "n/a"
                delta = f"{(rp-s['pass_rate'])*100:+.0f}pp" if rp is not None else ""
                rt = s.get('rubric_total', 0)
                rpass = s.get('rubric_pass', 0)
                flag = _rubric_reliability_flag(rt)
                print(f"    {tag:<28s} {tp:>9.0f}% {rp_str:>12s} {delta:>6s}  n={rt:>5d} {flag}")
                if rt > 0:
                    group_rp += rpass
                    group_rt += rt
            gp = group_rp/group_rt*100 if group_rt else 0
            print(f"    {'── 组汇总 rubric% ──':<28s} {'':>10s} {gp:>11.0f}% {'':>6s}  n={group_rt:>5d}")
            grand_pass += group_rp
            grand_total += group_rt
        if grand_total:
            print(f"\n  ══ 全局加权 rubric%（能力维度合计）: {grand_pass/grand_total*100:.0f}%  n={grand_total} ══")

        # 3 个执行素质
        exec_tags = {
            "evidence-based-decision": "决策有数据依据（vs 凭空猜测）",
            "tool-before-answer": "先调工具再回答（vs 编造数据）",
            "proactive-tool-use": "主动获取额外信息（vs 被动等用户说）",
        }
        print(f"\n📋 Execution Quality (3 执行素质):")
        for tag, desc in exec_tags.items():
            s = metrics.capability_tag_stats.get(tag)
            if s:
                rp = s.get('rubric_pass_rate')
                rt = s.get('rubric_total', 0)
                rpass = s.get('rubric_pass', 0)
                if rt > 0:
                    print(f"  {tag:<28s}  {rpass}/{rt} ({rp*100:.0f}%)  {desc}")

        # 流程完成
        for tag_key in ("_process_completion",):
            s = metrics.capability_tag_stats.get(tag_key)
            if s:
                rp = s.get('rubric_pass_rate')
                rt = s.get('rubric_total', 0)
                rpass = s.get('rubric_pass', 0)
                if rt > 0:
                    print(f"\n  _process_completion (下单确认): {rpass}/{rt} ({rp*100:.0f}%)")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    args = parser.parse_args()
    results = Results.load(Path(args.results))
    metrics = compute_metrics(results)
    display_metrics(metrics)
