#!/usr/bin/env python3
"""Bootstrap CI on the exp(-C) vs. 1/(1+C) ranking-equivalence claim in
Appendix G ("Scope of the ranking-equivalence claim"): recomputes every
model's mean custom14 score under both functional forms and reports the
Spearman correlation between the two resulting 24-model orderings, with a
task-level bootstrap CI (same methodology as leaderboard_bootstrap_ci.py).

Needs the full RideWay simulation corpus (same tier-2 dependency as
build_model_leaderboard.py / leaderboard_bootstrap_ci.py) -- not
self-contained in this bundle.

Usage:
    python3 functional_form_ranking_ci.py --sim-dir /path/to/data/simulations \
        --tasks-dir /path/to/data/eff/domains/ridehailing
"""
import argparse, json, os, random
from collections import defaultdict

MODELS = ['dsv4pro', 'gemma426b', 'gemma431b', 'glm52', 'granite418b', 'kimik26',
          'llama3370b', 'llama318b', 'mimo', 'ministral14b', 'ministral3b', 'ministral8b',
          'mistralsmall', 'nemotron3nano30b', 'qw359b', 'qw35ba3b', 'qw3627b', 'qw36p', 'qw37max',
          'qw37plus', 'gpt55nothink', 'opus46', 'gemini35flash', 'sonnet5']
JUDGES = ['hyeval', 'seed2liteeval', 'grok43eval']
A_FIT, B_FIT = 0.1235, 0.2448
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
    """{task_id: [(u_exp, u_inv), ...]} for this model's custom14 trials."""
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
        c = A_FIT * etool + B_FIT * eturn
        u_exp = pow(2.718281828459045, -c) if success else 0.0
        u_inv = 1.0 / (1.0 + c) if success else 0.0
        by_task[tid].append((u_exp, u_inv))
    return by_task


def mean_pooled(by_task, task_ids, idx):
    trials = [x[idx] for tid in task_ids for x in by_task.get(tid, [])]
    return sum(trials) / len(trials) if trials else float('nan')


def spearman(vals_a, vals_b):
    def rank(v):
        idx = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for ri, i in enumerate(idx):
            r[i] = ri + 1
        return r
    ra, rb = rank(vals_a), rank(vals_b)
    n = len(ra)
    ma = sum(ra) / n; mb = sum(rb) / n
    num = sum((a - ma) * (b - mb) for a, b in zip(ra, rb))
    da = sum((a - ma) ** 2 for a in ra) ** 0.5
    db = sum((b - mb) ** 2 for b in rb) ** 0.5
    return num / (da * db) if da > 0 and db > 0 else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim-dir', default='../../data/simulations')
    ap.add_argument('--tasks-dir', default='../../data/eff/domains/ridehailing')
    ap.add_argument('-B', type=int, default=5000)
    ap.add_argument('--ci', type=float, default=90.0, help='CI level as a percentage, e.g. 90 or 95')
    args = ap.parse_args()
    lo_p, hi_p = (100 - args.ci) / 2, 100 - (100 - args.ci) / 2

    floor = load_floor(args.tasks_dir)
    by_model = {m: per_task_records(args.sim_dir, m, floor) for m in MODELS}
    task_ids = sorted({tid for bt in by_model.values() for tid in bt})
    print(f"{len(task_ids)} custom14 tasks, {len(MODELS)} models")

    u_exp = [mean_pooled(by_model[m], task_ids, 0) for m in MODELS]
    u_inv = [mean_pooled(by_model[m], task_ids, 1) for m in MODELS]
    pt = spearman(u_exp, u_inv)
    print(f"Point estimate: Spearman(exp(-C), 1/(1+C)) = {pt:.3f}")

    random.seed(0)
    vals = []
    for _ in range(args.B):
        samp = [task_ids[random.randrange(len(task_ids))] for _ in range(len(task_ids))]
        ue = [mean_pooled(by_model[m], samp, 0) for m in MODELS]
        ui = [mean_pooled(by_model[m], samp, 1) for m in MODELS]
        v = spearman(ue, ui)
        if v == v:
            vals.append(v)
    vals.sort()
    lo, hi = vals[int(lo_p / 100 * len(vals))], vals[int(hi_p / 100 * len(vals))]
    print(f"{args.ci:.0f}% task-level bootstrap CI (B={args.B}): [{lo:.3f}, {hi:.3f}]")


if __name__ == '__main__':
    main()
