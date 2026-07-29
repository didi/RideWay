#!/usr/bin/env python3
"""Score the blind listwise efficiency-ranking study.

Within each task, compare the human efficiency ranking (4 buckets) against the
ranking implied by U (and by tool-only / turn-only baselines), using tie-aware
Kendall tau_b. Aggregate the per-task tau_b values across tasks for the top-line
result + a bootstrap test that mean tau_b > 0.

Human buckets: '1' 最省(most efficient) .. '4' 很浪费(least efficient).
We map to an efficiency score where HIGHER = more efficient, so it lines up with
U (higher U = more efficient):  1->4, 2->3, 3->2, 4->1.

Usage:
    python3 score_listwise_eff.py --votes listwise_eff_votes.json
"""
import argparse, json, math, random
from collections import defaultdict

BUCKET_EFF = {'1': 4.0, '2': 3.0, '3': 2.0, '4': 1.0}  # higher = more efficient


def U(it, lt, lo):
    et = max(0.0, it['eturn']); eo = max(0.0, it['etool'])
    return (lt ** et) * (lo ** eo)


def kendall_tau_b(xs, ys):
    n = len(xs); C = D = Tx = Ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = xs[i] - xs[j]; b = ys[i] - ys[j]
            if a == 0 and b == 0:
                continue
            if a == 0:
                Tx += 1; continue
            if b == 0:
                Ty += 1; continue
            if (a > 0) == (b > 0):
                C += 1
            else:
                D += 1
    denom = math.sqrt((C + D + Tx) * (C + D + Ty))
    return (C - D) / denom if denom else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='listwise_eff_manifest.json')
    ap.add_argument('--votes', required=True)
    ap.add_argument('--lambda-turn', type=float, default=0.6594)
    ap.add_argument('--lambda-tool', type=float, default=0.8948)
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    votes = json.load(open(args.votes))
    lt, lo = args.lambda_turn, args.lambda_tool

    by_task = defaultdict(list)
    for it in manifest:
        g = votes.get(it['annotation_id'])
        if g not in BUCKET_EFF:
            continue
        by_task[it['task_id']].append({
            'human': BUCKET_EFF[g],
            'U': U(it, lt, lo),
            'tool_eff': -it['etool'],   # higher = fewer excess tools = more efficient
            'turn_eff': -it['eturn'],
            'etool': it['etool'], 'eturn': it['eturn'],
        })

    print(f"lambda_turn={lt} lambda_tool={lo}")
    print(f"tasks with >=2 labeled trajectories: "
          f"{sum(1 for v in by_task.values() if len(v) >= 2)}\n")

    def per_task(keyfn, label):
        taus = []
        rows = []
        for tid, rs in sorted(by_task.items()):
            if len(rs) < 2:
                continue
            tau = kendall_tau_b([r['human'] for r in rs], [keyfn(r) for r in rs])
            if tau == tau:  # not nan
                taus.append(tau)
                rows.append((tid, len(rs), tau))
        mean = sum(taus) / len(taus) if taus else float('nan')
        med = sorted(taus)[len(taus) // 2] if taus else float('nan')
        # bootstrap 90% CI on the mean of per-task taus
        random.seed(0)
        boots = []
        for _ in range(5000):
            samp = [random.choice(taus) for _ in taus]
            boots.append(sum(samp) / len(samp))
        boots.sort()
        lo_ci = boots[int(0.05 * len(boots))]
        hi_ci = boots[int(0.95 * len(boots))]
        frac_pos = sum(1 for t in taus if t > 0) / len(taus) if taus else float('nan')
        print(f"=== {label} ===")
        print(f"  per-task tau_b: mean={mean:+.3f}  median={med:+.3f}  "
              f"90% CI [{lo_ci:+.3f}, {hi_ci:+.3f}]  ({sum(1 for t in taus if t>0)}/{len(taus)} tasks positive)")
        for tid, n, tau in rows:
            print(f"    {tid[:44]:46s} n={n} tau_b={tau:+.3f}")
        print()
        return taus

    per_task(lambda r: r['U'], 'U (efficiency metric)')
    per_task(lambda r: r['tool_eff'], 'tool-only baseline (fewer tool calls better)')
    per_task(lambda r: r['turn_eff'], 'turn-only baseline (fewer turns better)')
    # lexicographic turn-first: turn dominates the ranking; tool only breaks exact turn ties.
    # Implemented as a single sort key (-eturn*BIG_M - etool) fed through the same tau_b
    # machinery -- BIG_M=1000 exceeds max etool (18) so it never lets tool flip a turn ordering.
    per_task(lambda r: -r['eturn'] * 1000 - r['etool'], 'lex-turn-first (turn primary, tool tiebreak)')
    # mirror image: tool dominates the ranking; turn only breaks exact tool ties.
    per_task(lambda r: -r['etool'] * 1000 - r['eturn'], 'lex-tool-first (tool primary, turn tiebreak)')


if __name__ == '__main__':
    main()
