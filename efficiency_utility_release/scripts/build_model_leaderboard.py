#!/usr/bin/env python3
"""Per-model leaderboard for all 24 pool models: SR / pass@3 / pass^3 on the
44-task rh44v2 calibration split, on the pooled 58-task suite (44 rh44v2 + 14
custom14), and on the 14-task custom14 held-out split -- plus mean Efficiency
Utility U, reported on the custom14 held-out split only (U's validity as a
calibrated preference-generalization score rests on the held-out evidence; see
RESULTS_ablation.md / RESULTS_lex_baseline.md).

Success is decided by the finalized 3-LLM judge panel (hy3-preview + Seed 2.0
Lite + Grok 4.3): each judge's verdict is a 3-pass majority over its base/_e2/_e3
re-evaluations of the same trajectory, and a trajectory passes on a 2-of-3
majority across the three judges. This is the same panel that gates every
BT/IAA/listwise pair (see RESULTS_ablation.md in the parent bt_calibration dir).

U(tau) = success(tau) * lambda_tool^etool(tau) * lambda_turn^eturn(tau), with the
final calibrated fit lambda_tool=0.8978, lambda_turn=0.6320 (a=0.1078, b=0.4588;
see fit_bt.py run on bt_pairs_train.json / bt_votes_train.json in this bundle).
etool/eturn are excess tool calls / excess turns over each task's human-annotated
reference floor (median of 3 annotators), floored at 0.

Per model, per scope: SR = mean success over that scope's trials (n_tasks x 3
trials). pass@3 and pass^3 are computed per task from that model's 3 trials,
then averaged over the scope's tasks. Mean U (custom14 scope only) is the mean
of U(tau) over that scope's 42 trials (0 for failed trials).

Usage:
    python3 build_model_leaderboard.py --sim-dir /path/to/data/simulations \
        --tasks-dir /path/to/data/vita/domains/ridehailing
    (defaults point at the full research checkout's standard locations; see --help)
"""
import argparse, json, math, os
from collections import defaultdict

MODELS = ['dsv4pro', 'gemma426b', 'gemma431b', 'glm52', 'granite418b', 'kimik26',
          'llama3370b', 'llama318b', 'mimo', 'ministral14b', 'ministral3b', 'ministral8b',
          'mistralsmall', 'nemotron3nano30b', 'qw359b', 'qw35ba3b', 'qw3627b', 'qw36p', 'qw37max',
          'qw37plus', 'gpt55nothink', 'opus46', 'gemini35flash', 'sonnet5']
JUDGES = ['hyeval', 'seed2liteeval', 'grok43eval']   # hy3-preview, Seed 2.0 Lite, Grok 4.3
LAMBDA_TOOL = 0.8978
LAMBDA_TURN = 0.6320
PREFIXES = ['rh44v2', 'c14v6b']   # 44-task calibration + 14-task held-out simulation-file prefixes


def pass_hat_k(n, c, k):
    if n < k:
        return float('nan')
    return math.comb(c, k) / math.comb(n, k)


def pass_at_k(n, c, k):
    if n < k:
        return float('nan')
    return 1 - math.comb(n - c, k) / math.comb(n, k)


def judge_verdicts(sim_dir, judge, prefix, model):
    """3-pass majority verdict per (task, trial) for one judge."""
    passes = []
    for suffix in ['', '_e2', '_e3']:
        rm = {}
        p = os.path.join(sim_dir, f"{judge}_{prefix}_{model}{suffix}")
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


def load_human_floor(tasks_dir):
    rh44 = json.load(open(os.path.join(tasks_dir, 'tasks_rh44_all.json')))
    c14 = json.load(open(os.path.join(tasks_dir, 'tasks_custom14_all.json')))
    return {t['id']: (t['turn_human'], t['tool_human']) for t in rh44 + c14 if 'turn_human' in t}


def scope_metrics(by_task, with_u=False):
    all_trials = [x for lst in by_task.values() for x in lst]
    sr = sum(1 for succ, _ in all_trials if succ) / len(all_trials) * 100

    pak_vals, phk_vals = [], []
    for tid, lst in by_task.items():
        n = len(lst); c = sum(1 for succ, _ in lst if succ)
        if n == 3:
            pak_vals.append(pass_at_k(n, c, 3)); phk_vals.append(pass_hat_k(n, c, 3))
    pak = sum(pak_vals) / len(pak_vals) if pak_vals else float('nan')
    phk = sum(phk_vals) / len(phk_vals) if phk_vals else float('nan')

    out = {'SR': sr, 'pass@3': pak, 'pass^3': phk}
    if with_u:
        out['mean_U'] = sum(u for _, u in all_trials) / len(all_trials)
    return out


def compute_model_row(sim_dir, tasks_dir, model, floor):
    by_task_domain = {}   # prefix -> {task_id: [(success, U), ...]}
    for prefix in PREFIXES:
        p = os.path.join(sim_dir, f"{prefix}_{model}")
        by_task = defaultdict(list)
        if os.path.exists(p):
            jv = {j: judge_verdicts(sim_dir, j, prefix, model) for j in JUDGES}
            sims = json.load(open(p))['simulations']
            for s in sims:
                tid, trial = s['task_id'], s['trial']
                judge_votes = sum(1 for j in JUDGES if jv[j].get((tid, trial), False))
                success = judge_votes >= 2
                msgs = s['messages']
                tool = sum(len(x.get('tool_calls') or []) for x in msgs if x.get('role') == 'assistant')
                turn = sum(1 for x in msgs if x.get('role') == 'assistant'
                           and x.get('content') != '###STOP###' and not x.get('tool_calls'))
                tmin, tlmin = floor.get(tid, (turn, tool))
                etool = max(0, tool - tlmin)
                eturn = max(0, turn - tmin)
                u = (LAMBDA_TOOL ** etool) * (LAMBDA_TURN ** eturn) if success else 0.0
                by_task[tid].append((success, u))
        by_task_domain[prefix] = by_task

    by_task_all58 = defaultdict(list)
    for by_task in by_task_domain.values():
        for tid, lst in by_task.items():
            by_task_all58[tid].extend(lst)

    return {
        'model': model,
        'rh44v2': scope_metrics(by_task_domain['rh44v2']),
        'all58': scope_metrics(by_task_all58),
        'custom14': scope_metrics(by_task_domain['c14v6b'], with_u=True),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim-dir', default='../../data/simulations')
    ap.add_argument('--tasks-dir', default='../../data/vita/domains/ridehailing')
    ap.add_argument('--out', default='RESULTS_leaderboard.md')
    args = ap.parse_args()

    floor = load_human_floor(args.tasks_dir)
    rows = [compute_model_row(args.sim_dir, args.tasks_dir, m, floor) for m in MODELS]
    rows.sort(key=lambda r: -r['custom14']['mean_U'])

    lines = [
        "# Model leaderboard — 24 pool models",
        "",
        "SR / pass@3 / pass^3 use the finalized 3-judge panel (hy3-preview + Seed 2.0 Lite +",
        "Grok 4.3, 2-of-3 majority, each judge itself a 3-pass internal majority), reported on",
        "three scopes: the 44-task rh44v2 calibration split, the pooled 58-task suite, and the",
        "14-task custom14 held-out split. Mean U (custom14 only) uses the final 320-pair",
        f"Bradley-Terry fit, lambda_tool={LAMBDA_TOOL}, lambda_turn={LAMBDA_TURN} (U=0 on failed",
        "trials). Sorted by custom14 mean U descending.",
        "",
        "Reproduce: `python3 build_model_leaderboard.py`.",
        "",
        "| Rank | Model | SR%(rh44) | p@3(rh44) | p^3(rh44) | SR%(58) | p@3(58) | p^3(58) | "
        "SR%(c14) | p@3(c14) | p^3(c14) | U(c14) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        a, b, c = r['rh44v2'], r['all58'], r['custom14']
        lines.append(
            f"| {i} | {r['model']} | {a['SR']:.1f} | {a['pass@3']:.3f} | {a['pass^3']:.3f} | "
            f"{b['SR']:.1f} | {b['pass@3']:.3f} | {b['pass^3']:.3f} | "
            f"{c['SR']:.1f} | {c['pass@3']:.3f} | {c['pass^3']:.3f} | {c['mean_U']:.3f} |")

    open(args.out, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
