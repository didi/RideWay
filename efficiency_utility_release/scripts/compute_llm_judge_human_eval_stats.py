#!/usr/bin/env python3
"""Human evaluation of the LLM-as-a-judge success gate.

Every trajectory in this benchmark is scored success/fail by a 3-judge LLM
panel (hy3-preview + Seed 2.0 Lite + Grok 4.3), each judge itself a 3-pass
internal majority -- see RESULTS_leaderboard.md and HOW_TO_REPRODUCE.md #5.
This script checks that panel's own reliability: a human annotator, blind to
the panel's verdict and to model identity, relabels a random sample of
trajectories success/fail, given exactly the same three inputs the panel's
own prompt receives (the rubric, the user's original scenario, and the final
system state) so the comparison is apples-to-apples.

Sampling: a random sample of trajectories from the full judge-panel output
pool (all 24 models x both domains).

Usage:
    python3 compute_llm_judge_human_eval_stats.py \
        --manifest ../data/llm_judge_human_eval_manifest.json \
        --votes ../data/llm_judge_human_eval_votes.json
"""
import argparse, json, random


def cohen_kappa(pairs):
    n = len(pairs)
    po = sum(1 for a, b in pairs if a == b) / n
    cats = set(a for a, _ in pairs) | set(b for _, b in pairs)
    pe = 0.0
    for c in cats:
        p1 = sum(1 for a, _ in pairs if a == c) / n
        p2 = sum(1 for _, b in pairs if b == c) / n
        pe += p1 * p2
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1 - pe)


def bootstrap_ci(pairs, statfn, B=10000, seed=0, lo_p=5, hi_p=95):
    rng = random.Random(seed)
    n = len(pairs)
    boots = []
    for _ in range(B):
        samp = [pairs[rng.randrange(n)] for _ in range(n)]
        try:
            boots.append(statfn(samp))
        except ZeroDivisionError:
            continue
    boots.sort()
    return boots[int(lo_p / 100 * len(boots))], boots[int(hi_p / 100 * len(boots))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='../data/llm_judge_human_eval_manifest.json')
    ap.add_argument('--votes', default='../data/llm_judge_human_eval_votes.json')
    ap.add_argument('-B', type=int, default=10000)
    ap.add_argument('--ci', type=float, default=90.0, help='CI level as a percentage, e.g. 90 or 95')
    args = ap.parse_args()
    lo_p, hi_p = (100 - args.ci) / 2, 100 - (100 - args.ci) / 2

    manifest = {it['annotation_id']: it for it in json.load(open(args.manifest))}
    human = json.load(open(args.votes))

    pairs, meta = [], []
    for aid, h in human.items():
        if aid not in manifest:
            continue
        m = manifest[aid]
        panel = 'success' if m['panel_success'] else 'fail'
        pairs.append((h, panel))
        meta.append(m)

    n = len(pairs)
    if n == 0:
        print("No overlapping items between votes and manifest.")
        return

    agree = sum(1 for a, b in pairs if a == b) / n
    kappa = cohen_kappa(pairs)
    lo_a, hi_a = bootstrap_ci(pairs, lambda ps: sum(1 for a, b in ps if a == b) / len(ps), B=args.B, lo_p=lo_p, hi_p=hi_p)
    lo_k, hi_k = bootstrap_ci(pairs, cohen_kappa, B=args.B, lo_p=lo_p, hi_p=hi_p)

    print(f"n = {n} trajectories, blind human relabel vs. LLM judge panel verdict\n")
    print(f"Raw agreement:  {agree*100:.1f}%  {args.ci:.0f}% CI [{lo_a*100:.1f}%, {hi_a*100:.1f}%]")
    print(f"Cohen's kappa:  {kappa:.3f}   {args.ci:.0f}% CI [{lo_k:.3f}, {hi_k:.3f}]")

    fp = sum(1 for (a, b) in pairs if b == 'success' and a == 'fail')
    fn = sum(1 for (a, b) in pairs if b == 'fail' and a == 'success')
    n_panel_success = sum(1 for _, b in pairs if b == 'success')
    n_panel_fail = n - n_panel_success
    print(f"\nPanel said success, human disagreed (potential false positive): "
          f"{fp}/{n_panel_success} ({fp/n_panel_success*100:.1f}%)" if n_panel_success else "")
    print(f"Panel said fail, human disagreed (potential false negative):    "
          f"{fn}/{n_panel_fail} ({fn/n_panel_fail*100:.1f}%)" if n_panel_fail else "")

    print("\n=== By panel decision confidence ===")
    for is_border in (True, False):
        sub = [(a, b) for (a, b), m in zip(pairs, meta) if m['borderline'] == is_border]
        if not sub:
            continue
        a_ = sum(1 for x, y in sub if x == y) / len(sub)
        label = 'borderline (2-1 vote somewhere)' if is_border else 'clear (unanimous)'
        print(f"  {label:32s} n={len(sub):3d}  agreement={a_*100:.1f}%")

    print("\n=== By domain ===")
    for dom in sorted(set(m['domain'] for m in meta)):
        sub = [(a, b) for (a, b), m in zip(pairs, meta) if m['domain'] == dom]
        a_ = sum(1 for x, y in sub if x == y) / len(sub)
        print(f"  {dom:10s} n={len(sub):3d}  agreement={a_*100:.1f}%")

    print("\n=== By model tier ===")
    for tier in sorted(set(m['tier'] for m in meta)):
        sub = [(a, b) for (a, b), m in zip(pairs, meta) if m['tier'] == tier]
        a_ = sum(1 for x, y in sub if x == y) / len(sub)
        print(f"  Tier {tier}  n={len(sub):3d}  agreement={a_*100:.1f}%")


if __name__ == '__main__':
    main()
