#!/usr/bin/env python3
"""
Generate a self-contained HTML visualization from a RideWay simulation JSON file.

Usage:
    python3 visualize_simulation.py [--task-ids id1,id2,...] file1.json [file2.json ...] output.html
Examples:
    python3 visualize_simulation.py data/simulations/gpt-5.json view_simulations/gpt-5.html
    python3 visualize_simulation.py --task-ids I0716001,A0714002 data/simulations/gpt-5.json view_simulations/gpt-5.html
    python3 visualize_simulation.py data/simulations/a.json data/simulations/b.json view_simulations/a_vs_b.html
"""
import json
import random
import sys
import html as html_lib
from collections import defaultdict
from pathlib import Path

# ── CLI ──────────────────────────────────────────────────────────────────────
# Usage: python3 visualize_simulation.py [--task-ids id1,id2,...] file1.json [file2.json ...] output.html
args = sys.argv[1:]

# Extract --task-ids flag (comma-separated) before positional args
FILTER_TASK_IDS: list[str] | None = None
if "--task-ids" in args:
    idx = args.index("--task-ids")
    if idx + 1 < len(args):
        FILTER_TASK_IDS = [t.strip() for t in args[idx + 1].split(",") if t.strip()]
    args = args[:idx] + args[idx + 2:]

if len(args) >= 2 and args[-1].endswith('.html'):
    INPUT_FILES = args[:-1]
    OUTPUT_FILE = args[-1]
elif len(args) == 1 and not args[0].endswith('.html'):
    INPUT_FILES = [args[0]]
    OUTPUT_FILE = "view_simulations/output.html"
else:
    INPUT_FILES = ["data/simulations/gemini_v2.json"]
    OUTPUT_FILE = "ridehailing/ridehailing_simulation_visualization.html"

# ── Load & merge ──────────────────────────────────────────────────────────────
simulations: list[dict] = []
tasks_by_id: dict[str, dict] = {}
model_entries: list[dict] = []   # one entry per input file

for fpath in INPUT_FILES:
    with open(fpath, encoding="utf-8") as f:
        data = json.load(f)
    file_info   = data.get("info", {})
    agent_llm   = file_info.get("agent_info", {}).get("llm", Path(fpath).stem)
    user_llm    = file_info.get("user_info", {}).get("llm", "Unknown")
    domain_name = file_info.get("environment_info", {}).get("domain_name", "Unknown")
    num_trials  = file_info.get("num_trials", "-")
    model_entries.append({"agent_llm": agent_llm, "user_llm": user_llm,
                          "domain_name": domain_name, "num_trials": num_trials,
                          "file": Path(fpath).stem})
    for sim in data.get("simulations", []):
        sim = dict(sim)
        sim["_agent_llm"] = agent_llm
        simulations.append(sim)
    for task in data.get("tasks", []):
        tasks_by_id.setdefault(task["id"], task)

tasks_list  = list(tasks_by_id.values())
all_models  = list(dict.fromkeys(e["agent_llm"] for e in model_entries))  # ordered, unique
multi_model = len(all_models) > 1

# ── Metrics ──────────────────────────────────────────────────────────────────
# Overall (all models pooled per task)
task_rewards: dict[str, list[float]] = defaultdict(list)
for sim in simulations:
    reward = sim.get("reward_info", {}).get("reward", 0.0) or 0.0
    task_rewards[sim["task_id"]].append(reward)

all_task_ids = sorted(task_rewards.keys())
task_avg     = {tid: sum(v) / len(v)           for tid, v in task_rewards.items()}
pass_at_k    = {tid: 1 if max(v) >= 1.0 else 0 for tid, v in task_rewards.items()}
pass_all_k   = {tid: 1 if min(v) >= 1.0 else 0 for tid, v in task_rewards.items()}

overall_avg_score  = sum(task_avg.values())    / len(task_avg)    if task_avg    else 0.0
overall_pass_at_k  = sum(pass_at_k.values())   / len(pass_at_k)   if pass_at_k  else 0.0
overall_pass_all_k = sum(pass_all_k.values())  / len(pass_all_k)  if pass_all_k else 0.0

# Enhance metrics (only present when enhance_rubrics were evaluated)
task_enhance_rewards: dict[str, list[float]] = defaultdict(list)
for sim in simulations:
    eri = (sim.get("reward_info") or {}).get("enhance_reward_info") or {}
    if eri:
        task_enhance_rewards[sim["task_id"]].append(eri.get("reward", 0.0) or 0.0)

has_enhance          = bool(task_enhance_rewards)
enhance_task_avg     = {tid: sum(v)/len(v)             for tid, v in task_enhance_rewards.items()}
enhance_pass_at_k    = {tid: 1 if max(v) >= 1.0 else 0 for tid, v in task_enhance_rewards.items()}
enhance_pass_all_k   = {tid: 1 if min(v) >= 1.0 else 0 for tid, v in task_enhance_rewards.items()}
# Use same denominator as main metrics (all tasks), so tasks without enhance data count as 0
_n_all = len(all_task_ids)
overall_enhance_avg    = sum(enhance_task_avg.values())   / _n_all if _n_all else 0.0
overall_enhance_pat_k  = sum(enhance_pass_at_k.values())  / _n_all if _n_all else 0.0
overall_enhance_pall_k = sum(enhance_pass_all_k.values()) / _n_all if _n_all else 0.0

# ── 拆分 enhance：主动性 rubric vs 其他 ──
PROACTIVE_KW = ('历史订单', '常用车型', 'get_history_orders')
_proactive_pass = 0
_proactive_total = 0

# 按 task 聚合排除主动性后的 enhance 分数（用于计算 pass@k / pass^k）
task_enhance_excl_rewards: dict[str, list[float]] = defaultdict(list)

for sim in simulations:
    eri = (sim.get("reward_info") or {}).get("enhance_reward_info") or {}
    nl = eri.get("nl_rubrics") or []
    if not nl:
        continue
    other_rubrics = []
    for nr in nl:
        rubric_text = nr.get("nl_rubric", "")
        is_proactive = any(kw in rubric_text for kw in PROACTIVE_KW)
        if is_proactive:
            _proactive_total += 1
            if nr.get("met"):
                _proactive_pass += 1
        else:
            other_rubrics.append(nr)
    if other_rubrics:
        # 每条 rubric 的通过率作为该 sim 的 enhance 分数
        score = sum(1.0 if r.get("met") else 0.0 for r in other_rubrics) / len(other_rubrics)
        # 全通过才算 1.0，否则 0.0（与原 enhance reward 一致的一票否决制）
        all_pass = 1.0 if all(r.get("met") for r in other_rubrics) else 0.0
        task_enhance_excl_rewards[sim["task_id"]].append(all_pass)

# 排除主动性后的 enhance 指标
_excl_task_avg  = {tid: sum(v)/len(v) for tid, v in task_enhance_excl_rewards.items()}
_excl_pass_at_k = {tid: 1 if max(v) >= 1.0 else 0 for tid, v in task_enhance_excl_rewards.items()}
_excl_pass_all_k = {tid: 1 if min(v) >= 1.0 else 0 for tid, v in task_enhance_excl_rewards.items()}

overall_enhance_excl_proactive   = sum(_excl_task_avg.values()) / _n_all if _n_all else 0.0
overall_enhance_excl_pat_k       = sum(_excl_pass_at_k.values()) / _n_all if _n_all else 0.0
overall_enhance_excl_pall_k      = sum(_excl_pass_all_k.values()) / _n_all if _n_all else 0.0

overall_proactive_pass_rate = (
    _proactive_pass / _proactive_total if _proactive_total > 0 else None
)

# ── 综合指标：基础 + enhance（排除主动性）合并 ──
# 每个 sim 的综合分 = 基础 rubrics 全通过 AND enhance（排除主动性）全通过 → 1.0，否则 0.0
task_combined_rewards: dict[str, list[float]] = defaultdict(list)
for sim in simulations:
    base_reward = (sim.get("reward_info") or {}).get("reward", 0.0) or 0.0
    base_pass = base_reward >= 1.0

    # enhance（排除主动性）是否全通过
    eri = (sim.get("reward_info") or {}).get("enhance_reward_info") or {}
    enl = eri.get("nl_rubrics") or []
    other_rubrics = [nr for nr in enl if not any(kw in nr.get("nl_rubric", "") for kw in PROACTIVE_KW)]
    if other_rubrics:
        enhance_pass = all(r.get("met") for r in other_rubrics)
    else:
        enhance_pass = True  # 无 enhance rubric 视为通过

    combined = 1.0 if (base_pass and enhance_pass) else 0.0
    task_combined_rewards[sim["task_id"]].append(combined)

_comb_task_avg   = {tid: sum(v)/len(v) for tid, v in task_combined_rewards.items()}
_comb_pass_at_k  = {tid: 1 if max(v) >= 1.0 else 0 for tid, v in task_combined_rewards.items()}
_comb_pass_all_k = {tid: 1 if min(v) >= 1.0 else 0 for tid, v in task_combined_rewards.items()}

overall_combined_avg    = sum(_comb_task_avg.values()) / _n_all if _n_all else 0.0
overall_combined_pat_k  = sum(_comb_pass_at_k.values()) / _n_all if _n_all else 0.0
overall_combined_pall_k = sum(_comb_pass_all_k.values()) / _n_all if _n_all else 0.0

# Per-model metrics
model_task_rewards: dict[str, dict[str, list[float]]] = {m: defaultdict(list) for m in all_models}
for sim in simulations:
    reward = sim.get("reward_info", {}).get("reward", 0.0) or 0.0
    model_task_rewards[sim["_agent_llm"]][sim["task_id"]].append(reward)

def _model_overall(m_tr: dict[str, list[float]]) -> tuple[float, float, float]:
    if not m_tr:
        return 0.0, 0.0, 0.0
    t_avg  = {tid: sum(v)/len(v)           for tid, v in m_tr.items()}
    t_pat  = {tid: 1 if max(v) >= 1.0 else 0 for tid, v in m_tr.items()}
    t_pall = {tid: 1 if min(v) >= 1.0 else 0 for tid, v in m_tr.items()}
    return (sum(t_avg.values())/len(t_avg),
            sum(t_pat.values())/len(t_pat),
            sum(t_pall.values())/len(t_pall))

model_metrics: dict[str, tuple[float, float, float]] = {
    m: _model_overall(model_task_rewards[m]) for m in all_models
}

# ── Determine which tasks to show ────────────────────────────────────────────
tasks_with_sims = sorted(tid for tid in all_task_ids if task_rewards[tid])

if FILTER_TASK_IDS:
    # Only show explicitly requested task IDs (preserve user-specified order)
    sample_task_ids = [tid for tid in FILTER_TASK_IDS if tid in task_rewards or tid in tasks_by_id]
    unknown = [tid for tid in FILTER_TASK_IDS if tid not in task_rewards and tid not in tasks_by_id]
    if unknown:
        print(f"Warning: unknown task IDs ignored: {unknown}")
else:
    # Default: show all tasks that have simulations
    sample_task_ids = tasks_with_sims

# For each sampled task, collect ALL sims per model (sorted by trial)
sample_sims: dict[str, dict[str, list[dict]]] = {}
for tid in sample_task_ids:
    sample_sims[tid] = {}
    for model in all_models:
        model_sims = [s for s in simulations if s["task_id"] == tid and s["_agent_llm"] == model]
        model_sims.sort(key=lambda s: s.get("trial", 0))
        sample_sims[tid][model] = model_sims

# ── Tool name → Chinese ───────────────────────────────────────────────────────
TOOL_CN_NAMES: dict[str, str] = {
    # Ridehailing 打车
    "select_car_types":                 "选择车型",
    "add_ride_note":                    "添加行程备注",
    "create_ride_reminder":             "创建用车提醒",
    "create_scheduled_ride":            "创建预约单",
    "modify_ride_destination":          "修改目的地",
    "get_flight_info":                  "查询航班信息",
    "get_history_orders":               "查询历史订单",
    "get_home_address":                 "查询家庭地址",
    "get_company_address":              "查询公司地址",
    "get_current_location":             "查询当前位置",
    "get_ride_real_time_status":        "查询行程实时状态",
    "get_trip_duration_estimate":       "预估行程时长",
    "nearby_search":                    "附近地点搜索",
    "get_available_car_types":          "获取可用车型",
    "match_drivers":                    "匹配司机",
    "create_ride_order":                "创建打车订单",
    "confirm_ride":                     "确认打车",
    "cancel_ride":                      "取消打车",
    "get_ride_order_detail":            "查询打车订单详情",
    "get_ride_estimate":                "获取打车预估费用",
    "search_ride_orders":               "搜索打车订单",
}

def tool_cn(fn: str) -> str:
    return TOOL_CN_NAMES.get(fn, fn)

# ── Model color palette ───────────────────────────────────────────────────────
MODEL_COLORS = [
    ("rgba(123,104,238,0.15)", "#c8b8ff", "rgba(123,104,238,0.5)"),   # purple
    ("rgba(0,212,255,0.12)",   "#00d4ff", "rgba(0,212,255,0.5)"),     # cyan
    ("rgba(255,107,157,0.12)", "#ff6b9d", "rgba(255,107,157,0.5)"),   # pink
    ("rgba(93,232,138,0.12)",  "#5de88a", "rgba(93,232,138,0.5)"),    # green
    ("rgba(240,192,64,0.12)",  "#f0c040", "rgba(240,192,64,0.5)"),    # yellow
]

def model_color(idx: int) -> tuple[str, str, str]:
    return MODEL_COLORS[idx % len(MODEL_COLORS)]

# ── HTML helpers ─────────────────────────────────────────────────────────────
def esc(s: str) -> str:
    return html_lib.escape(str(s or ""))

def fmt_role(role: str) -> str:
    if role == "assistant":
        return "助手"
    if role == "user":
        return "用户"
    if role == "tool":
        return "工具"
    return role

def role_class(role: str) -> str:
    return {"assistant": "msg-assistant", "user": "msg-user", "tool": "msg-tool"}.get(role, "msg-other")

def render_message(msg: dict) -> str:
    role    = msg.get("role", "")
    content = msg.get("content") or ""
    tcs     = msg.get("tool_calls") or []
    parts   = []
    if content:
        parts.append(f"<div class='msg-content'>{esc(content)}</div>")
    for tc in tcs:
        fn   = tc.get("name") or tc.get("function", {}).get("name", "")
        args = tc.get("arguments") or tc.get("function", {}).get("arguments", {})
        if isinstance(args, str):
            try:   args = json.loads(args)
            except Exception: pass
        args_str = json.dumps(args, ensure_ascii=False, indent=2) if isinstance(args, dict) else str(args)
        cn = tool_cn(fn)
        cn_label = f" <span class='tool-cn'>（{esc(cn)}）</span>" if cn != fn else ""
        parts.append(
            f"<div class='tool-call'>"
            f"<span class='tool-name'>🔧 {esc(fn)}</span>{cn_label}"
            f"<pre class='tool-args'>{esc(args_str)}</pre>"
            f"</div>"
        )
    body = "\n".join(parts) or "<em class='empty'>(空)</em>"
    turn = msg.get("turn_idx", "")
    usage = msg.get("usage") or {}
    tokens = f"{usage.get('prompt_tokens',0)}+{usage.get('completion_tokens',0)}" if usage else ""
    meta = f"<span class='msg-meta'>turn {turn}" + (f" · {tokens} tok" if tokens else "") + "</span>"
    return (
        f"<div class='msg {role_class(role)}'>"
        f"<div class='msg-header'><span class='msg-role'>{fmt_role(role)}</span>{meta}</div>"
        f"{body}"
        f"</div>"
    )

def render_rubric(r: dict) -> str:
    met = r.get("met")
    # 三态：True=通过(绿✓) / False=失败(红✗) / None or "not_applicable"=N/A(黄—)
    if met is True:
        icon, cls = "✓", "rubric-pass"
    elif met is False:
        icon, cls = "✗", "rubric-fail"
    else:  # None / "not_applicable" / 其他
        icon, cls = "—", "rubric-na"
    just = r.get("justification", "")
    return (
        f"<div class='rubric {cls}'>"
        f"<div class='rubric-header'><span class='rubric-icon'>{icon}</span>"
        f"<span class='rubric-text'>{esc(r.get('nl_rubric',''))}</span></div>"
        + (f"<div class='rubric-just'>{esc(just)}</div>" if just else "")
        + "</div>"
    )


def render_model_trial_block(sims: list[dict], task_idx: int, model_idx: int,
                             model_label: str | None, color_idx: int) -> str:
    """Render all trials for one model, with clickable trial tabs when >1 trial."""
    bg, fg, border = model_color(color_idx)
    model_header = (
        f"<div class='model-header' style='background:{bg};border:1px solid {border};color:{fg};'>"
        f"🤖 {esc(model_label)}</div>"
    ) if model_label else ""

    if not sims:
        return model_header + "<p class='empty-note'>暂无仿真数据</p>"

    if len(sims) == 1:
        # Single trial — no tabs needed
        return model_header + _render_trial_content(sims[0])

    # Multiple trials — render tab bar + panels
    tab_bar_parts = []
    panel_parts   = []
    for ti, sim in enumerate(sims):
        reward = sim.get("reward_info", {}).get("reward", 0.0) or 0.0
        trial  = sim.get("trial", ti)
        icon   = " ✓" if reward >= 1.0 else " ✗"
        active = " active" if ti == 0 else ""
        tab_bar_parts.append(
            f"<button class='trial-tab{active}' "
            f"data-task='{task_idx}' data-model='{model_idx}' "
            f"onclick='showTrial({task_idx},{model_idx},{ti},event)'>"
            f"Trial {trial}{icon}</button>"
        )
        panel_parts.append(
            f"<div class='trial-panel{active}' id='trp-{task_idx}-{model_idx}-{ti}'>"
            + _render_trial_content(sim)
            + "</div>"
        )

    tab_bar = f"<div class='trial-tabs'>{''.join(tab_bar_parts)}</div>"
    return model_header + tab_bar + "".join(panel_parts)


def _render_trial_content(sim: dict) -> str:
    """Render the conversation + rubrics for one trial (no model header)."""
    messages    = sim.get("messages", [])
    reward_info = sim.get("reward_info", {})
    reward      = reward_info.get("reward", 0.0)
    nl_rubrics  = reward_info.get("nl_rubrics") or []
    duration    = sim.get("duration", 0)
    trial       = sim.get("trial", "")

    rubrics_html    = "".join(render_rubric(r) for r in nl_rubrics)
    msgs_html       = "".join(render_message(m) for m in messages)
    reward_cls      = "reward-pass" if reward >= 1.0 else "reward-fail"
    tool_call_count = sum(len(m.get("tool_calls") or []) for m in messages)

    # enhance_reward_info section
    enhance_html = ""
    enhance_info = reward_info.get("enhance_reward_info") or {}
    if enhance_info:
        enhance_reward   = enhance_info.get("reward", 0.0) or 0.0
        enhance_rubrics  = enhance_info.get("nl_rubrics") or []
        enhance_cls      = "reward-pass" if enhance_reward >= 1.0 else "reward-fail"
        enhance_rubrics_html = "".join(render_rubric(r) for r in enhance_rubrics)
        enhance_html = (
            f"<div class='section-title' style='margin-top:12px'>✨ 增强评测结果"
            f"  <span class='{enhance_cls}'>得分 {enhance_reward:.2f}</span>"
            f"</div>"
            f"<div class='rubrics'>{enhance_rubrics_html if enhance_rubrics_html else '<p class=\"empty-note\">暂无增强评测细则</p>'}</div>"
        )

    return (
        f"<div class='section-title'>🗨️ 对话记录"
        f"  <span class='badge'>trial {trial} · {len(messages)} 轮 · {int(duration)}s · 🔧 {tool_call_count} 次工具调用</span>"
        f"</div>"
        f"<div class='chat-window'>{msgs_html}</div>"
        f"<div class='section-title' style='margin-top:12px'>📋 评测结果"
        f"  <span class='{reward_cls}'>得分 {reward:.2f}</span>"
        f"</div>"
        f"<div class='rubrics'>{rubrics_html if rubrics_html else '<p class=\"empty-note\">暂无评测细则</p>'}</div>"
        f"{enhance_html}"
    )


def render_task_panel(tid: str, idx: int) -> str:
    task            = tasks_by_id.get(tid, {})
    task_model_sims = sample_sims.get(tid, {})

    instructions  = task.get("instructions", "")
    initial_utt   = task.get("initial_user_utterance", "") or ""
    user_profile  = task.get("user_scenario", {}).get("user_profile", {})
    domain        = task.get("domain", "")
    trial_rewards = task_rewards.get(tid, [])
    avg           = task_avg.get(tid, float("nan"))
    pat_k         = pass_at_k.get(tid, 0)
    pall_k        = pass_all_k.get(tid, 0)

    # user profile table
    profile_rows = "".join(
        f"<tr><td class='p-key'>{esc(k)}</td><td>{esc(str(v))}</td></tr>"
        for k, v in user_profile.items()
    )
    profile_html = f"<table class='profile-table'>{profile_rows}</table>" if profile_rows else ""

    # score pills — per-model when multi_model, per-trial otherwise
    if multi_model:
        pills = " ".join(
            f"<span class='pill model-pill' style='background:{model_color(i)[0]};color:{model_color(i)[1]};border:1px solid {model_color(i)[2]};'>"
            f"{esc(m)}: "
            + (f"{sum(model_task_rewards[m][tid])/len(model_task_rewards[m][tid]):.2f}" if model_task_rewards[m].get(tid) else "—")
            + "</span>"
            for i, m in enumerate(all_models)
        )
    else:
        pills = " ".join(
            f"<span class='pill {'pill-pass' if r >= 1.0 else 'pill-fail'}'>trial{i}: {r:.0f}</span>"
            for i, r in enumerate(trial_rewards)
        )

    # conversation blocks — multi-model 用 tab 切换；单 model 直接渲染
    if multi_model and len(all_models) > 1:
        tab_bar = []
        panels = []
        for i, m in enumerate(all_models):
            sims = task_model_sims.get(m) or []
            # tab 标签含模型名 + 平均 reward
            rewards = [s.get("reward_info", {}).get("reward", 0.0) or 0.0 for s in sims]
            avg_r = (sum(rewards) / len(rewards)) if rewards else 0.0
            bg, fg, border = model_color(i)
            active = " active" if i == 0 else ""
            tab_bar.append(
                f"<button class='model-tab{active}' "
                f"style='background:{bg};color:{fg};border-color:{border};' "
                f"onclick='showModelTab({idx},{i},event)'>"
                f"🤖 {esc(m)} <span class='model-tab-score'>({avg_r:.2f})</span></button>"
            )
            panels.append(
                f"<div class='model-tab-panel{active}' id='mtp-{idx}-{i}'>"
                + render_model_trial_block(sims, idx, i, None, i)
                + "</div>"
            )
        conv_blocks = (
            f"<div class='model-tab-bar'>{''.join(tab_bar)}</div>"
            + "".join(panels)
        )
    else:
        conv_blocks = "".join(
            render_model_trial_block(task_model_sims.get(m) or [], idx, i, m if multi_model else None, i)
            for i, m in enumerate(all_models)
        )

    return f"""
<div class='task-panel' id='panel-{idx}' data-idx='{idx}'>
  <div class='task-meta-bar'>
    <span class='task-id-badge'>Task {esc(tid)}</span>
    <span class='domain-badge'>{esc(domain)}</span>
    {f"<span class='avg-score'>均分 {avg:.2f}</span>" if trial_rewards and not multi_model else ""}
    {f"<span class='patak'>pass@k {pat_k}</span>" if trial_rewards and not multi_model else ""}
    {f"<span class='pallk'>pass^k {pall_k}</span>" if trial_rewards and not multi_model else ""}
    {pills}
  </div>

  <div class='two-col'>
    <div class='col-left'>
      <div class='section-title'>📝 任务指令</div>
      <div class='instructions-box'>{esc(instructions)}</div>
      {f'<div class="section-title" style="margin-top:14px">💬 用户开场白</div><div class="initial-utt-box">{esc(initial_utt)}</div>' if initial_utt else ''}
      {f'<div class="section-title" style="margin-top:14px">👤 用户画像</div>{profile_html}' if profile_html else ''}
    </div>
    <div class='col-right'>
      {conv_blocks}
    </div>
  </div>
</div>"""

# ── Summary table ─────────────────────────────────────────────────────────────
def render_summary() -> str:
    if not all_task_ids:
        return "<p class='empty-note'>暂无数据</p>"
    n = len(all_task_ids)

    # ── per-model metric cards (shown when multi-model) ──
    model_cards_html = ""
    if multi_model:
        cards = ""
        for i, m in enumerate(all_models):
            bg, fg, border = model_color(i)
            m_avg, m_pat, m_pall = model_metrics[m]
            m_tasks = len(model_task_rewards[m])
            cards += f"""
<div class='metric-card model-metric-card' style='border-color:{border};'>
  <div class='model-card-name' style='color:{fg};'>{esc(m)}</div>
  <div class='model-card-row'><span class='model-card-label'>任务数</span><span>{m_tasks}</span></div>
  <div class='model-card-row'><span class='model-card-label'>平均分</span>
    <span class='{"metric-green" if m_avg >= 0.5 else "metric-red"}'>{m_avg:.3f}</span></div>
  <div class='model-card-row'><span class='model-card-label'>pass@k</span>
    <span class='{"metric-green" if m_pat >= 0.5 else "metric-red"}'>{m_pat:.3f}</span></div>
  <div class='model-card-row'><span class='model-card-label'>pass^k</span>
    <span class='{"metric-green" if m_pall >= 0.5 else "metric-red"}'>{m_pall:.3f}</span></div>
</div>"""
        model_cards_html = f"<div class='model-cards-grid'>{cards}</div>"

    # ── summary table ──
    # Columns: Task ID | (per-model avg cols when multi) | overall avg | pass@k | pass^k
    if multi_model:
        model_th = "".join(
            f"<th style='color:{model_color(i)[1]}'>{esc(m)}<br><small>均分</small></th>"
            for i, m in enumerate(all_models)
        )
        thead = f"<tr><th>Task ID</th>{model_th}<th>pass@k</th><th>pass^k</th></tr>"
    else:
        thead = "<tr><th>Task ID</th><th>各 Trial 得分</th><th>平均分</th><th>pass@k</th><th>pass^k</th></tr>"

    rows = ""
    for tid in sorted(all_task_ids):
        rewards = task_rewards[tid]
        avg  = task_avg[tid]
        pat  = pass_at_k[tid]
        pall = pass_all_k[tid]
        if multi_model:
            model_tds = ""
            for i, m in enumerate(all_models):
                m_rewards = model_task_rewards[m].get(tid, [])
                if m_rewards:
                    m_avg = sum(m_rewards) / len(m_rewards)
                    cls = "t-green" if m_avg >= 1.0 else ""
                    model_tds += f"<td class='t-num {cls}'>{m_avg:.2f}</td>"
                else:
                    model_tds += "<td class='t-num' style='color:#555d7a'>—</td>"
            rows += (
                f"<tr>"
                f"<td class='t-id'>{esc(tid)}</td>"
                f"{model_tds}"
                f"<td class='t-num {'t-green' if pat else ''}'>{pat}</td>"
                f"<td class='t-num {'t-green' if pall else ''}'>{pall}</td>"
                f"</tr>"
            )
        else:
            pills = " ".join(
                f"<span class='pill-sm {'pill-pass' if r >= 1.0 else 'pill-fail'}'>{r:.0f}</span>"
                for r in rewards
            )
            rows += (
                f"<tr>"
                f"<td class='t-id'>{esc(tid)}</td>"
                f"<td class='t-rewards'>{pills}</td>"
                f"<td class='t-num'>{avg:.2f}</td>"
                f"<td class='t-num {'t-green' if pat else ''}'>{pat}</td>"
                f"<td class='t-num {'t-green' if pall else ''}'>{pall}</td>"
                f"</tr>"
            )

    if multi_model:
        model_footer_tds = "".join(
            f"<td class='t-num'><b>{model_metrics[m][0]:.2f}</b></td>"
            for m in all_models
        )
        footer = (
            f"<tr class='summary-footer'><td><b>均值</b></td>{model_footer_tds}"
            f"<td class='t-num {'t-green' if overall_pass_at_k > 0 else ''}'><b>{overall_pass_at_k:.2f}</b></td>"
            f"<td class='t-num {'t-green' if overall_pass_all_k > 0 else ''}'><b>{overall_pass_all_k:.2f}</b></td></tr>"
        )
    else:
        footer = (
            f"<tr class='summary-footer'>"
            f"<td><b>总计 / 均值</b></td>"
            f"<td></td>"
            f"<td class='t-num'><b>{overall_avg_score:.2f}</b></td>"
            f"<td class='t-num {'t-green' if overall_pass_at_k > 0 else ''}'><b>{overall_pass_at_k:.2f}</b></td>"
            f"<td class='t-num {'t-green' if overall_pass_all_k > 0 else ''}'><b>{overall_pass_all_k:.2f}</b></td>"
            f"</tr>"
        )

    table_html = f"""
<table class='summary-table'>
  <thead>{thead}</thead>
  <tbody>{rows}{footer}</tbody>
</table>"""

    # ── overall metric cards ──
    overall_cards = f"""
<div class='metric-cards'>
  <div class='metric-card'>
    <div class='metric-label'>任务数</div>
    <div class='metric-value'>{n}</div>
  </div>
  <div class='metric-card'>
    <div class='metric-label'>平均得分</div>
    <div class='metric-value {"metric-green" if overall_avg_score >= 0.5 else "metric-red"}'>{overall_avg_score:.3f}</div>
  </div>
  <div class='metric-card'>
    <div class='metric-label'>pass@k</div>
    <div class='metric-value {"metric-green" if overall_pass_at_k >= 0.5 else "metric-red"}'>{overall_pass_at_k:.3f}</div>
    <div class='metric-sub'>至少1次成功</div>
  </div>
  <div class='metric-card'>
    <div class='metric-label'>pass^k</div>
    <div class='metric-value {"metric-green" if overall_pass_all_k >= 0.5 else "metric-red"}'>{overall_pass_all_k:.3f}</div>
    <div class='metric-sub'>全部成功</div>
  </div>
</div>"""

    enhance_cards = ""
    if has_enhance:
        proactive_card = ""
        if overall_proactive_pass_rate is not None:
            proactive_card = f"""
  <div class='metric-card' style='border-color:rgba(234,179,8,0.4)'>
    <div class='metric-label' style='color:#eab308'>🧠 主动性 rubric</div>
    <div class='metric-value {"metric-green" if overall_proactive_pass_rate >= 0.3 else "metric-red"}'>{overall_proactive_pass_rate:.3f}</div>
    <div class='metric-sub'>历史订单推荐通过率</div>
  </div>"""

        enhance_cards = f"""
<div class='metric-cards' style='margin-top:12px'>
  <div class='metric-card' style='border-color:rgba(16,185,129,0.4)'>
    <div class='metric-label' style='color:#10b981'>📊 综合平均 (基础+Enhance)</div>
    <div class='metric-value {"metric-green" if overall_combined_avg >= 0.5 else "metric-red"}'>{overall_combined_avg:.3f}</div>
    <div class='metric-sub'>排除主动性rubric</div>
  </div>
  <div class='metric-card' style='border-color:rgba(16,185,129,0.4)'>
    <div class='metric-label' style='color:#10b981'>📊 综合 pass@k</div>
    <div class='metric-value {"metric-green" if overall_combined_pat_k >= 0.5 else "metric-red"}'>{overall_combined_pat_k:.3f}</div>
    <div class='metric-sub'>基础+Enhance至少1次全通过</div>
  </div>
  <div class='metric-card' style='border-color:rgba(16,185,129,0.4)'>
    <div class='metric-label' style='color:#10b981'>📊 综合 pass^k</div>
    <div class='metric-value {"metric-green" if overall_combined_pall_k >= 0.5 else "metric-red"}'>{overall_combined_pall_k:.3f}</div>
    <div class='metric-sub'>基础+Enhance全部全通过</div>
  </div>{proactive_card}
</div>"""

    return model_cards_html + table_html + overall_cards + enhance_cards

# ── Build HTML ────────────────────────────────────────────────────────────────
# Meta info: show all models when multi
if multi_model:
    agents_str  = " · ".join(e["agent_llm"]   for e in model_entries)
    users_str   = " · ".join(dict.fromkeys(e["user_llm"]    for e in model_entries))
    domains_str = " · ".join(dict.fromkeys(e["domain_name"] for e in model_entries))
    trials_str  = " · ".join(dict.fromkeys(str(e["num_trials"]) for e in model_entries))
else:
    e = model_entries[0]
    agents_str  = e["agent_llm"]
    users_str   = e["user_llm"]
    domains_str = e["domain_name"]
    trials_str  = str(e["num_trials"])

tab_buttons = "\n".join(
    f"<button class='task-tab{' active' if i == 0 else ''}' onclick='showTask({i})'>任务 {tid}</button>"
    for i, tid in enumerate(sample_task_ids)
)
task_panels = "\n".join(render_task_panel(tid, i) for i, tid in enumerate(sample_task_ids))

# ── Page title: derive from input filenames to identify dataset + experiment ──
# Single file → use full stem (e.g. "v6_im_r1_dsv31")
# Multi-file → longest common token prefix as dataset id + model suffixes
#   (e.g. ["v6_round17_airport_dsv31", "..._gpt52", "..._qw36p"]
#    → "v6_round17_airport · dsv31 · gpt52 · qw36p")
input_stems = [e["file"] for e in model_entries]


def _common_token_prefix(stems: list[str]) -> str:
    if not stems:
        return ""
    parts_list = [s.split("_") for s in stems]
    common: list[str] = []
    for tokens in zip(*parts_list):
        if all(t == tokens[0] for t in tokens):
            common.append(tokens[0])
        else:
            break
    return "_".join(common)


if len(input_stems) == 1:
    page_title = input_stems[0]
else:
    dataset_label = _common_token_prefix(input_stems) or Path(OUTPUT_FILE).stem
    suffixes = []
    for s in input_stems:
        sfx = s[len(dataset_label):].lstrip("_") if s.startswith(dataset_label) else s
        suffixes.append(sfx or s)
    uniq_suffixes = list(dict.fromkeys(suffixes))
    page_title = f"{dataset_label} · {' · '.join(uniq_suffixes)}" if uniq_suffixes else dataset_label

page_subtitle = f"{len(simulations)} 仿真 · {len(all_task_ids)} 任务 · {len(model_entries)} 模型"

HTML = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{esc(page_title)}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, 'PingFang SC', 'Microsoft YaHei', sans-serif;
        background:#0a0e27; color:#e0e0e0; min-height:100vh; }}

/* ── Header ── */
.header {{ text-align:center; padding:36px 20px 20px;
           background:linear-gradient(135deg,#0a0e27 0%,#1a1f4e 100%); }}
.header h1 {{ font-size:1.9em;
              background:linear-gradient(90deg,#00d4ff,#7b68ee,#ff6b9d);
              -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:6px; }}
.header .subtitle {{ color:#8892b0; font-size:0.88em; margin-bottom:16px; }}
.meta-bar {{ display:flex; justify-content:center; gap:10px; flex-wrap:wrap; margin-top:10px; }}
.meta-item {{ background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1);
              border-radius:8px; padding:6px 14px; font-size:0.8em; }}
.meta-label {{ color:#7b68ee; font-weight:600; margin-right:4px; }}

/* ── Nav ── */
.section-nav {{ display:flex; justify-content:center; gap:6px; padding:14px 20px;
                position:sticky; top:0; z-index:100;
                background:rgba(10,14,39,0.96); backdrop-filter:blur(10px);
                border-bottom:1px solid rgba(255,255,255,0.06); flex-wrap:wrap; }}
.nav-btn {{ padding:7px 20px; border-radius:18px; border:1px solid rgba(123,104,238,0.3);
            background:transparent; color:#8892b0; cursor:pointer; font-size:0.82em;
            transition:all 0.25s; }}
.nav-btn:hover, .nav-btn.active {{ background:rgba(123,104,238,0.2); color:#fff;
                                    border-color:#7b68ee; }}

/* ── Container ── */
.container {{ max-width:1300px; margin:0 auto; padding:28px 20px; }}
.section {{ display:none; animation:fadeIn 0.35s ease; }}
.section.active {{ display:block; }}
@keyframes fadeIn {{ from {{ opacity:0; transform:translateY(14px); }} to {{ opacity:1; transform:translateY(0); }} }}

.section-heading {{ font-size:1.2em; font-weight:700; margin-bottom:18px;
                    padding-bottom:10px; border-bottom:1px solid rgba(255,255,255,0.08); }}

/* ── Task tabs ── */
.task-tabs {{ display:flex; gap:8px; margin-bottom:20px; flex-wrap:wrap; }}
.task-tab {{ padding:8px 22px; border-radius:20px; border:1px solid rgba(255,255,255,0.1);
             background:transparent; color:#8892b0; cursor:pointer; font-size:0.85em;
             transition:all 0.25s; }}
.task-tab.active {{ color:#fff; border-color:#7b68ee; background:rgba(123,104,238,0.2); }}
.task-panel {{ display:none; }}
.task-panel.active {{ display:block; animation:fadeIn 0.3s ease; }}

/* ── Trial tabs (inside a task panel, per model) ── */
.trial-tabs {{ display:flex; gap:6px; margin-bottom:10px; flex-wrap:wrap; }}
.trial-tab {{ padding:5px 14px; border-radius:14px; border:1px solid rgba(255,255,255,0.1);
              background:transparent; color:#8892b0; cursor:pointer; font-size:0.8em;
              transition:all 0.2s; }}
.trial-tab:hover {{ background:rgba(0,212,255,0.1); color:#cdd6f4; border-color:rgba(0,212,255,0.3); }}
.trial-tab.active {{ background:rgba(0,212,255,0.18); color:#00d4ff; border-color:#00d4ff; font-weight:700; }}
.trial-panel {{ display:none; }}
.trial-panel.active {{ display:block; animation:fadeIn 0.25s ease; }}

/* ── Task meta bar ── */
.task-meta-bar {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center;
                  margin-bottom:16px; padding:10px 14px;
                  background:rgba(255,255,255,0.03);
                  border:1px solid rgba(255,255,255,0.07); border-radius:10px; }}
.task-id-badge {{ background:rgba(123,104,238,0.3); color:#c8b8ff;
                  padding:3px 10px; border-radius:12px; font-size:0.8em; font-weight:700; }}
.domain-badge {{ background:rgba(0,212,255,0.15); color:#00d4ff;
                 padding:3px 10px; border-radius:12px; font-size:0.8em; }}
.avg-score {{ color:#f0c040; font-size:0.82em; font-weight:600; }}
.patak {{ color:#5de88a; font-size:0.82em; }}
.pallk {{ color:#ff9f5a; font-size:0.82em; }}
.pill, .pill-sm {{ display:inline-block; padding:2px 8px; border-radius:10px;
                   font-size:0.78em; font-weight:600; }}
.pill-pass {{ background:rgba(93,232,138,0.2); color:#5de88a;
              border:1px solid rgba(93,232,138,0.3); }}
.pill-fail {{ background:rgba(255,100,100,0.2); color:#ff7070;
              border:1px solid rgba(255,100,100,0.3); }}
.pill-sm {{ padding:1px 6px; font-size:0.74em; }}

/* ── Two-col layout ── */
.two-col {{ display:grid; grid-template-columns:320px 1fr; gap:18px; }}
@media (max-width:900px) {{ .two-col {{ grid-template-columns:1fr; }} }}
.col-left, .col-right {{ display:flex; flex-direction:column; gap:14px; }}

/* ── Section title ── */
.section-title {{ font-size:0.9em; font-weight:700; color:#8892b0;
                  text-transform:uppercase; letter-spacing:0.8px;
                  margin-bottom:8px; display:flex; align-items:center; gap:8px; }}
.badge {{ font-size:0.8em; font-weight:400; color:#7b68ee;
          background:rgba(123,104,238,0.15); padding:2px 9px; border-radius:10px;
          text-transform:none; letter-spacing:0; }}
.reward-pass {{ color:#5de88a; font-weight:700; }}
.reward-fail {{ color:#ff7070; font-weight:700; }}

/* ── Instructions ── */
.instructions-box {{ background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08);
                     border-radius:10px; padding:14px; font-size:0.88em;
                     line-height:1.7; color:#cdd6f4; }}

/* ── Initial user utterance ── */
.initial-utt-box {{ background:rgba(0,212,255,0.07); border:1px solid rgba(0,212,255,0.25);
                    border-left:3px solid #00d4ff; border-radius:8px; padding:10px 14px;
                    font-size:0.88em; line-height:1.6; color:#cdd6f4; font-style:italic; }}
.initial-utt-box::before {{ content:'"'; color:#00d4ff; font-size:1.4em; margin-right:4px; }}
.initial-utt-box::after {{ content:'"'; color:#00d4ff; font-size:1.4em; margin-left:4px; }}

/* ── Model tabs (multi-model HTML) ── */
.model-tab-bar {{ display:flex; gap:8px; flex-wrap:wrap; margin-bottom:14px;
                  padding-bottom:10px; border-bottom:1px solid rgba(255,255,255,0.08); }}
.model-tab {{ padding:8px 16px; border-radius:8px; border:1px solid;
              background:rgba(255,255,255,0.04); color:#cdd6f4;
              cursor:pointer; font-size:0.92em; font-weight:600;
              transition:all 0.15s; }}
.model-tab:hover {{ filter:brightness(1.15); }}
.model-tab.active {{ filter:brightness(1.25); box-shadow:0 0 0 2px rgba(255,255,255,0.18); }}
.model-tab-score {{ opacity:0.8; font-weight:400; font-size:0.85em; }}
.model-tab-panel {{ display:none; }}
.model-tab-panel.active {{ display:block; animation:fadeIn 0.25s ease; }}

/* ── Profile table ── */
.profile-table {{ width:100%; border-collapse:collapse; font-size:0.82em; }}
.profile-table td {{ padding:5px 8px; border-bottom:1px solid rgba(255,255,255,0.05); }}
.p-key {{ color:#7b68ee; font-weight:600; white-space:nowrap; width:90px; }}

/* ── Chat window ── */
.chat-window {{ display:flex; flex-direction:column; gap:10px;
                max-height:520px; overflow-y:auto; padding:12px;
                background:rgba(255,255,255,0.02);
                border:1px solid rgba(255,255,255,0.07); border-radius:10px; }}
.chat-window::-webkit-scrollbar {{ width:5px; }}
.chat-window::-webkit-scrollbar-thumb {{ background:rgba(123,104,238,0.4); border-radius:3px; }}

.msg {{ border-radius:10px; padding:10px 13px; max-width:92%; font-size:0.86em;
        line-height:1.6; }}
.msg-assistant {{ background:rgba(123,104,238,0.12); border:1px solid rgba(123,104,238,0.25);
                  align-self:flex-start; }}
.msg-user {{ background:rgba(0,212,255,0.08); border:1px solid rgba(0,212,255,0.2);
             align-self:flex-end; }}
.msg-tool {{ background:rgba(255,200,80,0.08); border:1px solid rgba(255,200,80,0.2);
             align-self:flex-start; font-size:0.82em; }}
.msg-header {{ display:flex; justify-content:space-between; align-items:center;
               margin-bottom:5px; }}
.msg-role {{ font-weight:700; font-size:0.8em; color:#8892b0; text-transform:uppercase; letter-spacing:0.5px; }}
.msg-meta {{ font-size:0.75em; color:#555d7a; }}
.msg-content {{ white-space:pre-wrap; word-break:break-word; }}
.tool-call {{ background:rgba(0,0,0,0.2); border-radius:7px; padding:8px 10px; margin-top:6px; }}
.tool-name {{ color:#f0c040; font-weight:700; font-size:0.85em; }}
.tool-cn {{ color:#a8b2d0; font-weight:400; font-size:0.85em; }}
.tool-args {{ color:#a8b2d0; font-size:0.78em; margin-top:5px; white-space:pre-wrap;
              word-break:break-word; max-height:120px; overflow:auto; }}
.empty {{ color:#555d7a; font-style:italic; }}

/* ── Rubrics ── */
.rubrics {{ display:flex; flex-direction:column; gap:8px; }}
.rubric {{ padding:10px 13px; border-radius:9px; border:1px solid; font-size:0.86em; }}
.rubric-pass {{ background:rgba(93,232,138,0.07); border-color:rgba(93,232,138,0.25); }}
.rubric-fail {{ background:rgba(255,100,100,0.07); border-color:rgba(255,100,100,0.25); }}
.rubric-na   {{ background:rgba(245,200,80,0.10); border-color:rgba(245,200,80,0.40); }}
.rubric-header {{ display:flex; align-items:flex-start; gap:8px; }}
.rubric-icon {{ font-weight:800; font-size:1em; flex-shrink:0; }}
.rubric-pass .rubric-icon {{ color:#5de88a; }}
.rubric-fail .rubric-icon {{ color:#ff7070; }}
.rubric-na   .rubric-icon {{ color:#f5c850; }}
.rubric-text {{ line-height:1.5; }}
.rubric-just {{ margin-top:7px; margin-left:24px; color:#8892b0;
                font-size:0.82em; line-height:1.5; }}

/* ── Summary ── */
.summary-table {{ width:100%; border-collapse:collapse; font-size:0.87em; margin-bottom:24px; }}
.summary-table th {{ background:rgba(255,255,255,0.05); padding:10px 12px;
                     text-align:left; font-weight:700; color:#8892b0; font-size:0.8em;
                     text-transform:uppercase; letter-spacing:0.6px;
                     border-bottom:2px solid rgba(123,104,238,0.3); }}
.summary-table td {{ padding:9px 12px; border-bottom:1px solid rgba(255,255,255,0.05); }}
.t-id {{ font-family:monospace; color:#c8b8ff; }}
.t-num {{ text-align:center; }}
.t-green {{ color:#5de88a; font-weight:700; }}
.summary-footer {{ background:rgba(123,104,238,0.08); font-weight:600; }}

.metric-cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:14px; }}
.metric-card {{ background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08);
                border-radius:12px; padding:18px; text-align:center; }}
.metric-label {{ font-size:0.8em; color:#8892b0; text-transform:uppercase;
                 letter-spacing:0.7px; margin-bottom:8px; }}
.metric-value {{ font-size:2em; font-weight:800; }}
.metric-green {{ color:#5de88a; }}
.metric-red {{ color:#ff7070; }}
.metric-sub {{ font-size:0.75em; color:#555d7a; margin-top:4px; }}

.empty-note {{ color:#555d7a; font-style:italic; font-size:0.88em; padding:16px; text-align:center; }}

/* ── Multi-model ── */
.model-header {{ font-size:0.88em; font-weight:700; padding:6px 14px;
                 border-radius:8px; margin-bottom:10px; margin-top:4px; }}
.model-sep {{ border-top:1px dashed rgba(255,255,255,0.1); margin:20px 0; }}
.model-pill {{ font-size:0.78em; padding:2px 10px; border-radius:10px; font-weight:600; }}
.model-cards-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
                     gap:14px; margin-bottom:24px; }}
.model-metric-card {{ background:rgba(255,255,255,0.03); border-radius:12px;
                      padding:16px; border-width:1px; border-style:solid; }}
.model-card-name {{ font-size:0.82em; font-weight:700; margin-bottom:10px;
                    white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.model-card-row {{ display:flex; justify-content:space-between; font-size:0.82em;
                   padding:3px 0; border-bottom:1px solid rgba(255,255,255,0.05); }}
.model-card-label {{ color:#8892b0; }}
</style>
</head>
<body>

<div class="header">
  <h1>{esc(page_title)}</h1>
  <p class="subtitle">{esc(page_subtitle)}</p>
  <div class="meta-bar">
    <div class="meta-item"><span class="meta-label">Agent LLM</span>{esc(agents_str)}</div>
    <div class="meta-item"><span class="meta-label">User LLM</span>{esc(users_str)}</div>
    <div class="meta-item"><span class="meta-label">Domain</span>{esc(domains_str)}</div>
    <div class="meta-item"><span class="meta-label">Trials</span>{esc(trials_str)}</div>
    <div class="meta-item"><span class="meta-label">仿真数</span>{len(simulations)}</div>
    <div class="meta-item"><span class="meta-label">任务数</span>{len(all_task_ids)}</div>
  </div>
</div>

<div class="section-nav">
  <button class="nav-btn active" onclick="showSection('tasks')">📋 任务示例</button>
  <button class="nav-btn"        onclick="showSection('summary')">📊 汇总结果</button>
</div>

<div class="container">

  <!-- ── Tasks Section ─────────────────────────────── -->
  <div class="section active" id="sec-tasks">
    <div class="section-heading">{"指定任务" if FILTER_TASK_IDS else "全部任务"} · 共 {len(sample_task_ids)} 个 · 完整对话与评测展示</div>
    <div class="task-tabs">
      {tab_buttons}
    </div>
    {task_panels}
  </div>

  <!-- ── Summary Section ──────────────────────────── -->
  <div class="section" id="sec-summary">
    <div class="section-heading">汇总评测结果</div>
    {render_summary()}
  </div>

</div>

<script>
function showSection(name) {{
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('sec-' + name).classList.add('active');
  event.target.classList.add('active');
}}

function showTask(idx) {{
  document.querySelectorAll('.task-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.task-tab').forEach(b => b.classList.remove('active'));
  document.getElementById('panel-' + idx).classList.add('active');
  document.querySelectorAll('.task-tab')[idx].classList.add('active');
}}

function showTrial(taskIdx, modelIdx, trialIdx, ev) {{
  ev.stopPropagation();
  const prefix = 'trp-' + taskIdx + '-' + modelIdx + '-';
  // hide all trial panels for this task+model
  document.querySelectorAll('[id^="' + prefix + '"]').forEach(p => p.classList.remove('active'));
  // deactivate all trial tabs for this task+model
  document.querySelectorAll('.trial-tab[data-task="' + taskIdx + '"][data-model="' + modelIdx + '"]')
          .forEach(b => b.classList.remove('active'));
  document.getElementById(prefix + trialIdx).classList.add('active');
  ev.target.classList.add('active');
}}

function showModelTab(taskIdx, modelIdx, ev) {{
  ev.stopPropagation();
  const prefix = 'mtp-' + taskIdx + '-';
  // hide all model panels for this task
  document.querySelectorAll('[id^="' + prefix + '"]').forEach(p => p.classList.remove('active'));
  // deactivate sibling model-tab buttons
  const tabBar = ev.currentTarget.parentElement;
  tabBar.querySelectorAll('.model-tab').forEach(b => b.classList.remove('active'));
  document.getElementById(prefix + modelIdx).classList.add('active');
  ev.currentTarget.classList.add('active');
}}

// Activate first task panel
document.addEventListener('DOMContentLoaded', () => {{
  const first = document.querySelector('.task-panel');
  if (first) first.classList.add('active');
}});
</script>
</body>
</html>"""

# ── Write ─────────────────────────────────────────────────────────────────────
out = Path(OUTPUT_FILE)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(HTML, encoding="utf-8")
print(f"✓ Written to {out.resolve()}")
print(f"  simulations : {len(simulations)}")
print(f"  tasks shown : {sample_task_ids}")
print(f"  avg score   : {overall_avg_score:.3f}")
print(f"  pass@k      : {overall_pass_at_k:.3f}")
print(f"  pass^k      : {overall_pass_all_k:.3f}")
if has_enhance:
    print(f"  📊 综合avg (基础+Enhance) : {overall_combined_avg:.3f}")
    print(f"  📊 综合pass@k            : {overall_combined_pat_k:.3f}")
    print(f"  📊 综合pass^k            : {overall_combined_pall_k:.3f}")
    if overall_proactive_pass_rate is not None:
        print(f"  🧠 主动性 rubric          : {overall_proactive_pass_rate:.3f}")
