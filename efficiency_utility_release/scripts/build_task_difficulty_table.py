#!/usr/bin/env python3
"""Task difficulty reference: SR, pass@3, pass^3 for all 58 benchmark tasks
(44 rh44 + 14 custom14), pooled across all 24 tier-pool models.

Success is decided by the finalized 3-LLM judge panel (hy3-preview + Seed 2.0
Lite + Grok 4.3): each judge's verdict is a 3-pass majority over its base/_e2/_e3
re-evaluations of the same trajectory, and a trajectory passes on a 2-of-3
majority across the three judges. This matches the panel used to gate every
BT/IAA/listwise pair (see RESULTS_ablation.md header) rather than the original
single deepseek-v4-pro evaluator.

Per task: SR is pooled over all 72 trials (24 models x 3 trials). pass@3/pass^3
are computed per model (n=3 trials each) then averaged across the 24 models.

Outputs RESULTS_task_difficulty.md (markdown table, sorted hardest->easiest)
and prints the same table to stdout.

Needs the full RideWay simulation corpus (same tier-2 dependency as
build_model_leaderboard.py) -- not self-contained in this bundle.

Usage:
    python3 build_task_difficulty_table.py --sim-dir /path/to/data/simulations \
        --tasks-dir /path/to/data/eff/domains/ridehailing
    (defaults point at the full RideWay checkout's standard locations; see --help)
"""
import argparse, json, os, math
from collections import defaultdict

TIERS = {
    'opus46': 'A', 'gemini35flash': 'A', 'sonnet5': 'A', 'gpt55nothink': 'A',
    'dsv4pro': 'B', 'qw37max': 'B', 'qw37plus': 'B', 'glm52': 'B', 'kimik26': 'B', 'qw36p': 'B', 'mimo': 'B',
    'qw35ba3b': 'C', 'qw3627b': 'C', 'mistralsmall': 'C', 'llama3370b': 'C', 'gemma431b': 'C', 'gemma426b': 'C', 'nemotron3nano30b': 'C',
    'qw359b': 'D', 'granite418b': 'D', 'ministral14b': 'D', 'ministral8b': 'D', 'ministral3b': 'D', 'llama318b': 'D',
}

MODELS = sorted(TIERS.keys())
JUDGES = ['hyeval', 'seed2liteeval', 'grok43eval']   # hy3-preview, Seed 2.0 Lite, Grok 4.3


def pass_hat_k(n, c, k):
    if n < k:
        return float('nan')
    return math.comb(c, k) / math.comb(n, k)


def pass_at_k(n, c, k):
    if n < k:
        return float('nan')
    return 1 - math.comb(n - c, k) / math.comb(n, k)


def judge_verdicts(sim_dir, judge, prefix, model):
    """3-pass majority verdict per (task, trial) for one judge.

    Reads the judge's base/_e2/_e3 re-evaluation files and returns
    {(task, trial): bool} where the judge passes a trajectory iff a strict
    majority of its available passes scored reward > 0.
    """
    passes = []
    for suffix in ['', '_e2', '_e3']:
        rm = {}
        p = f"{sim_dir}/{judge}_{prefix}_{model}{suffix}"
        if os.path.exists(p):
            for s in json.load(open(p))['simulations']:
                rm[(s['task_id'], s.get('trial'))] = s['reward_info']['reward']
        passes.append(rm)
    keys = set().union(*[set(rm) for rm in passes]) if passes else set()
    verdict = {}
    for k in keys:
        vals = [rm[k] for rm in passes if k in rm]
        pos = sum(1 for v in vals if v and v > 0)
        verdict[k] = len(vals) >= 2 and pos * 2 > len(vals)   # strict majority of available passes
    return verdict


def compute_rows(sim_dir):
    results = {}
    for prefix in ['rh44v2', 'c14v6b']:
        for m in MODELS:
            jv = {j: judge_verdicts(sim_dir, j, prefix, m) for j in JUDGES}
            p = f"{sim_dir}/{prefix}_{m}"
            if not os.path.exists(p):
                continue
            sims = json.load(open(p))['simulations']
            by_task = defaultdict(list)
            for s in sims:
                key = (s['task_id'], s['trial'])
                judge_votes = sum(1 for j in JUDGES if jv[j].get(key, False))
                panel_success = judge_votes >= 2       # 2-of-3 majority across judges
                by_task[s['task_id']].append(panel_success)
            for tid, lst in by_task.items():
                results.setdefault(tid, []).append(lst)

    rows = []
    for tid, model_lists in sorted(results.items()):
        all_trials = [x for lst in model_lists for x in lst]
        sr = sum(all_trials) / len(all_trials) * 100
        pak_vals = []; phk_vals = []
        for lst in model_lists:
            n = len(lst); c = sum(lst)
            if n == 3:
                pak_vals.append(pass_at_k(n, c, 3)); phk_vals.append(pass_hat_k(n, c, 3))
        pak = sum(pak_vals) / len(pak_vals) if pak_vals else float('nan')
        phk = sum(phk_vals) / len(phk_vals) if phk_vals else float('nan')
        rows.append((tid, len(all_trials), sr, pak, phk))
    rows.sort(key=lambda r: r[2])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim-dir', default='../../data/simulations')
    ap.add_argument('--tasks-dir', default='../../data/eff/domains/ridehailing')
    args = ap.parse_args()

    rows = compute_rows(args.sim_dir)

    c14_tasks = {t['id'] for t in json.load(open(os.path.join(args.tasks_dir, 'tasks_custom14_all.json')))}

    def domain(tid):
        return 'custom14' if tid in c14_tasks else 'rh44'

    lines = [
        "# Task difficulty reference — all 58 benchmark tasks",
        "",
        "SR, pass@3, pass^3 pooled across all 24 tier-pool models (72 trials/task: 24 models x 3",
        "trials each). Success is decided by the finalized 3-LLM judge panel (hy3-preview + Seed",
        "2.0 Lite + Grok 4.3): each judge is a 3-pass majority over its base/_e2/_e3 re-evaluations,",
        "and a trajectory passes on a 2-of-3 majority across the three judges — the same panel that",
        "gates every BT/IAA/listwise pair (see RESULTS_ablation.md). Sorted hardest -> easiest by",
        "pooled SR.",
        "`pass@3` = P(at least 1 of 3 trials succeeds, per model, averaged across models).",
        "`pass^3` = P(all 3 trials succeed, per model, averaged across models).",
        "",
        "Reproduce: `python3 build_task_difficulty_table.py`.",
        "",
        f"44 rh44 tasks + 14 custom14 tasks = {len(rows)} total, all with full 72-trial coverage.",
        "",
        "| Task | Domain | n | SR% | pass@3 | pass^3 |",
        "|---|---|---|---|---|---|",
    ]
    for tid, n, sr, pak, phk in rows:
        lines.append(f"| {tid} | {domain(tid)} | {n} | {sr:.1f} | {pak:.3f} | {phk:.3f} |")

    open('RESULTS_task_difficulty.md', 'w').write('\n'.join(lines))
    print(f"wrote RESULTS_task_difficulty.md, {len(rows)} rows")

    print(f"\n{'task':50s} {'dom':10s} {'n':5s} {'SR%':7s} {'pass@3':8s} {'pass^3':8s}")
    for tid, n, sr, pak, phk in rows:
        print(f"{tid[:48]:50s} {domain(tid):10s} {n:5d} {sr:6.1f}% {pak:8.3f} {phk:8.3f}")


if __name__ == '__main__':
    main()
