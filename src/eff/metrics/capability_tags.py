"""Capability-tag schema for RideWay tasks.

受控词表（controlled vocabulary）。每个 task 可以挂若干 tag，
metrics 层按 tag 聚合 pass_rate / avg_reward，暴露模型在不同能力维度上的差异。

设计原则：
- 独立于现有 Task 结构：`capability_tags` 是 Optional 字段，老 task JSON 完全兼容。
- 词表有边界：不允许任意字符串，通过 `validate_tags()` 校验。
- 标签正交：一个 task 可带多个 tag，每个 tag 反映一个独立的能力维度。
- 可扩展：新增 tag 只需改本文件，不影响任何运行时代码。

两层结构：
- 输入侧（task 复杂度画像）：A/B/C/F 组
- 输出侧（agent 行为画像）：通过 metrics 层的轨迹指标单独度量，不在本词表

动态事件不作为独立能力维度，改为复杂度修饰符（参 dynamic_events 字段），
指标层做"有扰动 vs 无扰动"分层对比，暴露鲁棒性 Δ。
"""

from typing import Dict, List, Set


# ─────────────────────────────────────────────────────────────
# A. 意图理解（Intent understanding）
# ─────────────────────────────────────────────────────────────
INTENT_TAGS: Dict[str, str] = {
    "ambiguous-location": "用户给出模糊地点（如「最近的火车站」「我们常去那家店」），agent 需消歧",
    "relative-time": "相对时间表述（如「半小时后」「现在」），需要换算成绝对时间",
    "implicit-constraint": "隐含约束（带娃→儿童座椅、带行李→后备箱空间）未显式说出",
    "preference-inference": "需要从历史订单 / 用户画像推断偏好（常用车型、常去地址）",
    "proactive-recall": "用户省略具体信息（「回家」「老地方」「就按推荐」），agent 需主动调 get_home_address / get_history_orders",
}

# ─────────────────────────────────────────────────────────────
# B. 多约束组合（Multi-constraint）
# ─────────────────────────────────────────────────────────────
MULTI_CONSTRAINT_TAGS: Dict[str, str] = {
    "multi-stop": "多目的地 / 途经点",
    "multi-param": "车型 + 价格 + 时间 + 路线等多参数组合",
    "price-ceiling": "显式价格上限",
    "eta-constraint": "到达时间约束（必须 X 点前到）",
    "scheduled-ride": "预约用车（非即时单）",
}

# ─────────────────────────────────────────────────────────────
# C. 状态追踪（State tracking）
# ─────────────────────────────────────────────────────────────
STATE_TRACKING_TAGS: Dict[str, str] = {
    "mid-dialogue-change": "对话中途用户改变主意（换地址 / 换车型 / 加需求）",
    "cancel-and-rebook": "先取消后重新下单",
    "context-carryover": "后续轮次需用到前面轮次给出的信息（跨轮状态）",
    "long-horizon": "一次对话内包含多次行程或多个独立流程",
    "no-redundant-tool-call": "用户重述同一需求（语义等价、无新信息）时，agent 不应用相同参数重调同一查询工具，应复用上文已有结果。失败模式：重复 match_drivers / get_home_address / get_flight_info 等。检测器：metrics.tool_dedup.count_redundant_calls()",
}

# ─────────────────────────────────────────────────────────────
# D. 高阶推理与决策（Higher-order reasoning）
# 源自 failure-analysis report + 现有 complex task 观察
# ─────────────────────────────────────────────────────────────
REASONING_TAGS: Dict[str, str] = {
    "proactive-tool-use": "agent 需在用户未明确要求的情况下主动调用工具获取信息（如主动查路况、查溢价、查天气）以提升服务质量",
    "constraint-conflict": "约束间存在冲突（「又便宜又要舒适」「必须 X 点到 + 排除便宜品牌」），agent 需 trade-off",
    "compound-command": "一口气多命令并行（5 个需求同时说），agent 需一次性转为参数而非串行追问",
    "implicit-inference": "弱信号推断（「3 件行李」→大车；「不能倒」→平稳驾驶备注）��需把隐性信号转为 action",
    "quantitative-comparison": "需要对多个方案分别调工具获取数据，然后做数值比较决策（如比价、比时间）",
    "plan-and-backtrack": "规划→评估可行性→发现不可行→主动切换备选方案，需要 if-then 推理",
    "cascading-recovery": "连续故障（司机取消+工具出错+…），每步失败都改变后续可行路径，需全局 re-plan",
    "ambiguity-resolution": "工具返回多结果或用户表述有歧义，需推理消歧或主动向用户确认",
    "cost-optimization": "多方案费用比较 + 可行性验证（如拼车 vs 独享 vs 多辆车），需数值推理+约束检查",
    "temporal-chain": "多步反向时间推理（航班→到达→出发→叫车→预约时间），需 backward chaining",
    "multi-objective-tradeoff": "4+ 个目标且无法全部满足（便宜+舒适+准时+特定品牌），需 trade-off 排序",
    "multi-stakeholder": "多个利益方有冲突需求（家人偏好不同、同事拼车诉求不同），需协调而非单选",
    "risk-assessment": "在不确定下做决策（时间紧但有可能赶上、信号弱但有线索），需评估风险并明示概率",
    "self-correction": "起初做出假设/选择，后续发现错误（用户纠正、工具返回不符）需主动修正而非辩护",
}

# ─────────────────────────────────────────────────────────────
# F. 安全 / 策略边界（Safety & policy）
# ─────────────────────────────────────────────────────────────
SAFETY_TAGS: Dict[str, str] = {
    "policy-boundary": "请求可能触及策略边界（代叫、超范围、违规上车点）",
    "refusal-required": "正确行为是拒绝或引导",
}


# 合并所有 tag —— 受控词表
ALL_CAPABILITY_TAGS: Dict[str, str] = {
    **INTENT_TAGS,
    **MULTI_CONSTRAINT_TAGS,
    **STATE_TRACKING_TAGS,
    **REASONING_TAGS,
    **SAFETY_TAGS,
}

# 每个 tag 归属的分组（用于聚合展示 / 雷达图分组）
TAG_GROUPS: Dict[str, str] = {
    **{k: "intent" for k in INTENT_TAGS},
    **{k: "multi_constraint" for k in MULTI_CONSTRAINT_TAGS},
    **{k: "state_tracking" for k in STATE_TRACKING_TAGS},
    **{k: "reasoning" for k in REASONING_TAGS},
    **{k: "safety" for k in SAFETY_TAGS},
}


def validate_tags(tags: List[str]) -> List[str]:
    """返回不在受控词表中的 tag 列表（空 list 表示全部合法）。"""
    allowed: Set[str] = set(ALL_CAPABILITY_TAGS.keys())
    return [t for t in tags if t not in allowed]


def describe_tag(tag: str) -> str:
    """返回 tag 的人类可读描述，未知 tag 返回空字符串。"""
    return ALL_CAPABILITY_TAGS.get(tag, "")
