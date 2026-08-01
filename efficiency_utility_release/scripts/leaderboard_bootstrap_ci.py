#!/usr/bin/env python3
"""Task-level (cluster) bootstrap CI on mean U_custom14 per model.

This is the script behind the 90% CI column in Table `tab:model-leaderboard`
in the paper. Resamples the 14 custom14 tasks with replacement (not
individual trials), pooling all trials of each resampled task on every
draw -- this respects the within-task correlation between a model's three
attempts on the same task, rather than treating all 42 trials as
independent, which would understate the true interval.

Needs the full RideWay simulation corpus (same tier-2 dependency as
build_model_leaderboard.py) -- not self-contained in this bundle. Precomputed
output (both 90% and 95%) is in ../data/RESULTS_leaderboard_ci.md; verified 
against the full corpus that every model's CI reproduces
Table `tab:model-leaderboard`'s published column exactly.

Usage:
    python3 leaderboard_bootstrap_ci.py --sim-dir /path/to/data/simulations \
        --tasks-dir /path/to/data/eff/domains/ridehailing
    (defaults point at the full RideWay checkout's standard locations; see --help)
"""
import argparse, json, os, random
from collections import defaultdict

MODELS = ['dsv4pro', 'gemma426b', 'gemma431b', 'glm52', 'granite418b', 'kimik26',
          'llama3370b', 'llama318b', 'mimo', 'ministral14b', 'ministral3b', 'ministral8b',
          'mistralsmall', 'nemotron3nano30b', 'qw359b', 'qw35ba3b', 'qw3627b', 'qw36p', 'qw37max',
          'qw37plus', 'gpt55nothink', 'opus46', 'gemini35flash', 'sonnet5']
JUDGES = ['hyeval', 'seed2liteeval', 'grok43eval']   # hy3-preview, Seed 2.0 Lite, Grok 4.3
LAMBDA_TOOL = 0.8838
LAMBDA_TURN = 0.7829
PREFIX = 'c14v6b'


def judge_verdicts(sim_dir, judge, model):
    passes = []
    for suffix in ['', '_e2', '_e3']:
        rm = {}
        p = os.path.join(sim_dir, f"{judge}_{PREFIX}_{model}{suffix}")
        if os.path.exists(p):
            for s in json.load(open(p))['simulations']:
                rm[(s['task_id'], s.get('trial'))] = s['reward_info']['reward']
        passes.append(rm)
    keys = set().union(*[set(rm) for rm in passes]) if passes else set()
    verdict = {}
    for k in keys:
        vals = [rm[k] for rm in passes if k in rm]
        pos = sum(1 for v in vals if v and v > 0)
        verdict[k] = len(vals) >= 2 and pos * 2 > len(vals)
    return verdict


def load_floor(tasks_dir):
    tasks = json.load(open(os.path.join(tasks_dir, 'tasks_custom14_all.json')))
    return {t['id']: (t['turn_human'], t['tool_human']) for t in tasks}


def per_task_records(sim_dir, model, floor):
    """Returns {task_id: [(success, U), ...]} for this model's 42 custom14 trials."""
    p = os.path.join(sim_dir, f"{PREFIX}_{model}")
    by_task = defaultdict(list)
    if not os.path.exists(p):
        return by_task
    jv = {j: judge_verdicts(sim_dir, j, model) for j in JUDGES}
    sims = json.load(open(p))['simulations']
    for s in sims:
        tid, trial = s['task_id'], s['trial']
        votes = sum(1 for j in JUDGES if jv[j].get((tid, trial), False))
        success = votes >= 2
        msgs = s['messages']
        tool = sum(len(x.get('tool_calls') or []) for x in msgs if x.get('role') == 'assistant')
        turn = sum(1 for x in msgs if x.get('role') == 'assistant'
                   and x.get('content') != '###STOP###' and not x.get('tool_calls'))
        tmin, tlmin = floor.get(tid, (turn, tool))
        etool = max(0, tool - tlmin)
        eturn = max(0, turn - tmin)
        u = (LAMBDA_TOOL ** etool) * (LAMBDA_TURN ** eturn) if success else 0.0
        by_task[tid].append((success, u))
    return by_task


def mean_u_pooled(by_task, task_ids):
    trials = [x for tid in task_ids for x in by_task.get(tid, [])]
    if not trials:
        return float('nan')
    return sum(u for _, u in trials) / len(trials)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim-dir', default='../../data/simulations')
    ap.add_argument('--tasks-dir', default='../../data/eff/domains/ridehailing')
    ap.add_argument('--out', default=None, help='optional path to write JSON results')
    ap.add_argument('-B', type=int, default=5000)
    ap.add_argument('--ci', type=float, default=90.0, help='CI level as a percentage, e.g. 90 or 95')
    args = ap.parse_args()

    floor = load_floor(args.tasks_dir)
    random.seed(0)
    lo_p = (100 - args.ci) / 2
    hi_p = 100 - lo_p

    rows = []
    for model in MODELS:
        by_task = per_task_records(args.sim_dir, model, floor)
        task_ids = sorted(by_task.keys())
        if len(task_ids) < 14:
            print(f"WARNING: {model} only has {len(task_ids)}/14 custom14 tasks")
        point = mean_u_pooled(by_task, task_ids)
        boots = []
        for _ in range(args.B):
            samp = [task_ids[random.randrange(len(task_ids))] for _ in range(len(task_ids))]
            boots.append(mean_u_pooled(by_task, samp))
        boots.sort()
        lo = boots[int(lo_p / 100 * args.B)]
        hi = boots[int(hi_p / 100 * args.B)]
        rows.append((model, point, lo, hi))

    rows.sort(key=lambda r: -r[1])
    print(f"{'rank':4s} {'model':16s} {'U_c14':>8s} {f'{args.ci:.0f}% CI':>18s} {'width':>7s}")
    for i, (model, point, lo, hi) in enumerate(rows, 1):
        print(f"{i:4d} {model:16s} {point:8.3f} [{lo:.3f}, {hi:.3f}]  {hi - lo:6.3f}")

    print(f"\n=== Adjacent-rank {args.ci:.0f}% CI overlap ===")
    overlap_count = 0
    for i in range(len(rows) - 1):
        _, p1, lo1, hi1 = rows[i]
        _, p2, lo2, hi2 = rows[i + 1]
        overlaps = not (hi2 < lo1)
        if overlaps:
            overlap_count += 1
        print(f"  rank {i+1} vs {i+2}: {'OVERLAP' if overlaps else 'separated'}")
    print(f"\n{overlap_count}/{len(rows)-1} adjacent pairs have overlapping {args.ci:.0f}% CIs")

    if args.out:
        json.dump([{'model': m, 'U_c14': p, 'ci_lo': lo, 'ci_hi': hi} for m, p, lo, hi in rows],
                   open(args.out, 'w'), indent=2)
        print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
