"""Detector for redundant tool calls in agent conversations.

Capability：`no-redundant-tool-call`（参见 capability_tags.py）。

判定：同一 simulation 内，agent 对**查询类工具**使用相同参数调用 >1 次，
且两次调用之间只有 user 消息（即没有改变环境状态的写操作），就计为重复。

写操作（create_ride_order / cancel_ride / update_ride_destination 等）后再调查询不算重复，
因为环境状态可能变了。

DYNAMIC 查询工具（match_drivers / 司机分布、价格、ETA、实时状态、位置）结果随时间刷新：
count_short_window_redundancy 对它们按"两次调用的时间间隔"豁免（≥ DYNAMIC_EXEMPT_SECONDS
[tool] 即合理刷新），而非按对话轮数——因为不同模型每轮耗时不同，轮数不是可比的时间代理。
STATIC 查询工具结果不随时间变，同参重调一律算冗余。
"""
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple


# 动态状态查询：长对话期间结果可能变化，重复调用属合理刷新（不算冗余）
# - match_drivers: 司机分布随时变化
# - weather: 天气变化
# - get_ride_real_time_status: 名字就是 real_time
# - get_ride_estimate: 价格随路况波动
# - get_trip_duration_estimate: 路况波动
# - get_current_location: 用户位置可变
DYNAMIC_QUERY_TOOLS = {
    "match_drivers", "weather", "get_ride_real_time_status",
    "get_ride_estimate", "get_trip_duration_estimate", "get_current_location",
}

# 动态查询工具的"刷新豁免时长"（秒）。两次同参调用的（模拟）时间间隔 ≥ 阈值
# → 期间世界确实变了，属合理刷新，不算冗余；< 阈值 → 数据还没来得及变就贴脸重调 = 冗余。
#
# 为何用时间而非轮数：agent 侧 sim_time = task 起始时间 + 现实已过时长（见 llm_agent
# 的时间流动），所以两条消息时间戳之差 ≈ agent 感知到的 sim_time 流逝。用时间天然抵消
# "不同模型每轮耗时不同 → 轮数不可比"的问题。
#
# 阈值标定（基于 lc_09 长上下文实测，6 文件 73 个 dynamic 同参重调）：
#   - 1 轮（相邻 user 间隔）中位数 11s；1 个 agent turn 中位数 9s；故"3 轮"≈ 30s。
#   - 重调间隔呈双峰：7~21s（1~2 条 user 消息之隔，贴脸重查）vs 51s+（闲聊十几轮后
#     回来重新咨询，合理刷新）。阈值落在 21↔51s 的空档，精准只抓前者。
#   - 按各量变化速度分档：变化越快阈值越小（更早允许重查）。
#       realtime(15) < 路况/司机/价格(30) < 位置(60) < 天气(600)
#
# 注：键集合必须等于 DYNAMIC_QUERY_TOOLS（下方 assert 守护，改一处要同步另一处）。
DYNAMIC_EXEMPT_SECONDS = {
    "match_drivers": 30,               # 司机分布，分钟级；≈3 轮
    "get_trip_duration_estimate": 30,  # 路况波动，分钟级；≈3 轮
    "get_ride_estimate": 30,           # 价格随路况，分钟级；≈3 轮
    "get_ride_real_time_status": 15,   # 在途订单真实时变，~1-2 轮重刷合理
    "get_current_location": 60,        # 用户位置更慢（常静止），≈5-6 轮才算真移动
    "weather": 600,                    # 天气分钟级，单 task 内几乎不变
}
assert set(DYNAMIC_EXEMPT_SECONDS) == DYNAMIC_QUERY_TOOLS, (
    "DYNAMIC_EXEMPT_SECONDS 必须覆盖且仅覆盖 DYNAMIC_QUERY_TOOLS"
)

_TS_FORMATS = ("%Y%m%d_%H%M%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M")

# 静态查询：结果短期内不变，重复调用 = 没用上上次结果 = 冗余
STATIC_QUERY_TOOLS = {
    "search_pois", "address_to_longitude_latitude", "get_available_car_types",
    "select_car_types", "get_history_orders", "get_home_address",
    "get_company_address", "get_user_all_orders", "get_flight_info",
    "get_ride_order_detail", "nearby_search",
}

# 兼容旧接口（不再使用，但保留避免外部依赖断裂）
QUERY_TOOLS = DYNAMIC_QUERY_TOOLS | STATIC_QUERY_TOOLS

WRITE_TOOLS = {
    "create_ride_order", "create_scheduled_ride", "cancel_ride",
    "update_ride_destination", "confirm_ride", "add_ride_note",
}


def _canonical_args(args) -> str:
    """工具参数 → canonical 字符串（用于签名比对）。"""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return args
    try:
        return json.dumps(args, sort_keys=True, ensure_ascii=False)
    except Exception:
        return str(args)


def _parse_ts(s) -> Optional[datetime]:
    """消息时间戳字符串 → datetime（兼容多种格式），无法解析返回 None。"""
    if not s or not isinstance(s, str):
        return None
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _elapsed_seconds(ts_a, ts_b) -> Optional[float]:
    """两条消息时间戳之间的秒数（b - a）。任一不可解析时返回 None。"""
    ta, tb = _parse_ts(ts_a), _parse_ts(ts_b)
    if ta is None or tb is None:
        return None
    return (tb - ta).total_seconds()


def count_redundant_calls(messages: List[Dict]) -> Dict:
    """扫描一段 messages，返回重复统计。

    Returns:
      {
        'total_query_calls': int,            # 查询类工具总调用次数
        'redundant_calls':   int,            # 被判定为重复的调用次数（n-1 per signature）
        'redundant_rate':    float,          # redundant / total
        'redundant_details': [               # 每条重复的详情
          {'tool': str, 'args_canonical': str, 'first_idx': int, 'repeat_idx': int}
        ],
      }
    """
    # 维护"当前可见的查询签名"集合；遇到写操作就清空（环境状态变了）
    seen: Dict[Tuple[str, str], int] = {}  # sig -> first message index
    redundant_details: List[Dict] = []
    total_calls = 0
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "assistant":
            for tc in m.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                name = tc.get("name")
                if name in WRITE_TOOLS:
                    # 写操作 → 环境状态变化，清空已见签名
                    seen.clear()
                    continue
                if name not in STATIC_QUERY_TOOLS:
                    # 只对静态查询算冗余；动态查询（match_drivers/weather/real_time 等）
                    # 在长对话期间重新调用是合理刷新，不计冗余
                    continue
                sig = (name, _canonical_args(tc.get("arguments")))
                total_calls += 1
                if sig in seen:
                    redundant_details.append({
                        "tool": name,
                        "args_canonical": sig[1],
                        "first_idx": seen[sig],
                        "repeat_idx": i,
                    })
                else:
                    seen[sig] = i
    redundant = len(redundant_details)
    return {
        "total_query_calls": total_calls,
        "redundant_calls": redundant,
        "redundant_rate": (redundant / total_calls) if total_calls else 0.0,
        "redundant_details": redundant_details,
    }


def count_short_window_redundancy(messages: List[Dict], k: int = 2) -> Dict:
    """同参重调冗余检测（capability no-redundant-tool-call / B2 真重复）：
    同一工具相同参数被调用 ≥2 次，按工具类型判定是否冗余。

    判定（写操作出现在两次调用之间 → 环境已变，恒不算冗余）：
      - STATIC 工具：结果不随时间变，同参重调即冗余（无时间/轮数豁免）。
      - DYNAMIC 工具：结果随时间刷新，按"两次调用的（模拟）时间间隔"判定——
        间隔 ≥ DYNAMIC_EXEMPT_SECONDS[tool] → 合理刷新，豁免；否则贴脸重调 = 冗余。
        时间间隔取消息时间戳之差（≈ agent 感知的 sim_time 流逝），从而消除
        "不同模型每轮耗时不同 → 轮数不可比"的问题。时间戳缺失时退回轮数窗口
        （user 消息数 ≤ k）。

    与 count_redundant_calls 的区别：
      - 含 DYNAMIC_QUERY_TOOLS（贴脸重调动态查询也是浪费），但给时间豁免
      - 写操作仅当出现在 a→b 之间时阻断这一对（不全局清空 seen）

    Returns:
      {
        'total_query_calls': int,    # STATIC ∪ DYNAMIC 的总查询次数
        'redundant_calls':   int,    # 冗余对数
        'redundant_rate':    float,
        'redundant_details': [{'tool', 'args_canonical', 'first_idx', 'repeat_idx',
                               'user_msgs_between', 'seconds_between'}],
      }
    """
    relevant = STATIC_QUERY_TOOLS | DYNAMIC_QUERY_TOOLS
    # 收集每个签名出现的位置
    sig_to_idx: Dict[Tuple[str, str], List[int]] = {}
    total_calls = 0
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            name = tc.get("name") or (tc.get("function") or {}).get("name")
            args = tc.get("arguments") or (tc.get("function") or {}).get("arguments")
            if name not in relevant:
                continue
            total_calls += 1
            sig = (name, _canonical_args(args))
            sig_to_idx.setdefault(sig, []).append(i)

    redundant_details: List[Dict] = []
    for sig, indices in sig_to_idx.items():
        tool = sig[0]
        for a, b in zip(indices, indices[1:]):
            # 数 (a, b) 之间的 user 消息 + 检测写操作
            user_cnt = 0
            has_write = False
            for j in range(a + 1, b):
                mid = messages[j]
                if mid.get("role") == "user":
                    user_cnt += 1
                elif mid.get("role") == "assistant":
                    for tc in mid.get("tool_calls") or []:
                        if not isinstance(tc, dict):
                            continue
                        nm = tc.get("name") or (tc.get("function") or {}).get("name")
                        if nm in WRITE_TOOLS:
                            has_write = True
                            break
                if has_write:
                    break
            if has_write:
                continue  # 期间有写操作，环境已变，不算冗余

            seconds = _elapsed_seconds(messages[a].get("timestamp"),
                                       messages[b].get("timestamp"))
            if tool in DYNAMIC_EXEMPT_SECONDS:
                threshold = DYNAMIC_EXEMPT_SECONDS[tool]
                if seconds is None:
                    # 无时间戳 → 退回轮数窗口（窗口外视为合理刷新，豁免）
                    if user_cnt > k:
                        continue
                elif seconds >= threshold:
                    continue  # 时间足够，世界已刷新，豁免
                # 否则：间隔 < 阈值，贴脸重调 → 冗余
            # STATIC：同参重调即冗余（写操作已在上面排除）

            redundant_details.append({
                "tool": tool,
                "args_canonical": sig[1],
                "first_idx": a,
                "repeat_idx": b,
                "user_msgs_between": user_cnt,
                "seconds_between": seconds,
            })

    redundant = len(redundant_details)
    return {
        "total_query_calls": total_calls,
        "redundant_calls": redundant,
        "redundant_rate": (redundant / total_calls) if total_calls else 0.0,
        "redundant_details": redundant_details,
    }


def summarize_simulations(simulations: List[Dict]) -> Dict:
    """汇总一组 simulation 的重复调用统计。

    Returns: {
      'total_query_calls', 'redundant_calls', 'redundant_rate',
      'sims_with_redundant', 'total_sims',
      'top_offenders': [(task_id, trial, redundant_count), ...]  # 按 redundant 数排序
    }
    """
    total = 0
    redundant = 0
    sims_with = 0
    per_sim: List[Tuple[str, int, int]] = []
    for s in simulations:
        r = count_redundant_calls(s.get("messages") or [])
        total += r["total_query_calls"]
        redundant += r["redundant_calls"]
        if r["redundant_calls"] > 0:
            sims_with += 1
        per_sim.append((s.get("task_id"), s.get("trial"), r["redundant_calls"]))
    return {
        "total_query_calls": total,
        "redundant_calls": redundant,
        "redundant_rate": (redundant / total) if total else 0.0,
        "sims_with_redundant": sims_with,
        "total_sims": len(simulations),
        "top_offenders": sorted(per_sim, key=lambda x: -x[2])[:5],
    }
