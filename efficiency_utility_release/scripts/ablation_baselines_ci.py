#!/usr/bin/env python3
"""Baseline comparison table with per-cell bootstrap CIs + paired significance.

Compares U (both axes) against degenerate baselines tool-only (b=0) and
turn-only (a=0) on the held-out custom14 test set. Reports, for each metric:
coverage, accuracy-on-answered, tau_b-on-answered, tau_b-full (abstain=neutral),
each with a 90% bootstrap CI. Then paired bootstrap (same resample for both
methods, not separately-computed marginal CIs) on accuracy-on-answered AND
tau_b(full): U - baseline.

Usage:
    python3 ablation_baselines_ci.py \
        --manifest ../data/bt_manifest_FINAL_test.json \
        --votes ../data/bt_votes_FINAL_test.json
"""
import argparse, json, math, random

A_FIT, B_FIT = 0.1235, 0.2448   # 320-pair training fit (lambda_tool=0.8838, lambda_turn=0.7829)


def sig(z):
    return 1 / (1 + math.exp(max(-60, min(60, -z))))


def load_recs(mf, vf):
    m = json.load(open(mf)); v = json.load(open(vf)); idx = {x['pair_id']: x for x in m}
    recs = []
    for pid, who in v.items():
        if who == 'tie' or pid not in idx:
            continue
        x = idx[pid]
        recs.append((x['d_etool'], x['d_eturn'], 1 if who == 'A' else -1))
    return recs


def pred(rec, method):
    dt, dn, td = rec
    if method == 'U':
        return (1 if sig(-A_FIT * dt - B_FIT * dn) > 0.5 else -1), td
    if method == 'tool':
        return (0 if dt == 0 else (1 if dt < 0 else -1)), td
    if method == 'turn':
        return (0 if dn == 0 else (1 if dn < 0 else -1)), td
    if method == 'lex-turn':
        # lexicographic: turn excess decides unless exactly tied, then tool breaks the tie
        if dn != 0:
            return (1 if dn < 0 else -1), td
        return (0 if dt == 0 else (1 if dt < 0 else -1)), td
    if method == 'lex-tool':
        # mirror image: tool excess decides unless exactly tied, then turn breaks the tie
        if dt != 0:
            return (1 if dt < 0 else -1), td
        return (0 if dn == 0 else (1 if dn < 0 else -1)), td


def stats(sample, method):
    conc = disc = abst = 0
    for r in sample:
        pd, td = pred(r, method)
        if pd == 0:
            abst += 1; continue
        if pd == td:
            conc += 1
        else:
            disc += 1
    n = conc + disc; tot = len(sample)
    cov = n / tot
    acc = conc / n if n else float('nan')
    tau_ans = (conc - disc) / n if n else float('nan')
    tau_full = (conc - disc) / tot
    return cov, acc, tau_ans, tau_full


def ci(recs, method, idx, B, lo_p=5, hi_p=95):
    vals = []
    for _ in range(B):
        s = [random.choice(recs) for _ in recs]
        vals.append(stats(s, method)[idx])
    vals = [x for x in vals if x == x]; vals.sort()
    return vals[int(lo_p / 100 * len(vals))], vals[int(hi_p / 100 * len(vals))]


def paired(recs, mA, mB, B, idx=3, lo_p=5, hi_p=95):
    """Paired bootstrap on stats(...)[idx] (default tau_b(full)); pass idx=1 for
    accuracy-on-answered. Same resample scores both methods so within-draw
    correlation is preserved, rather than diffing two separate marginal CIs."""
    diffs = []
    for _ in range(B):
        s = [random.choice(recs) for _ in recs]
        vA, vB = stats(s, mA)[idx], stats(s, mB)[idx]
        if vA != vA or vB != vB:   # skip resamples where a baseline has 0 answered pairs
            continue
        diffs.append(vA - vB)
    diffs.sort()
    n = len(diffs)
    pt = stats(recs, mA)[idx] - stats(recs, mB)[idx]
    return pt, diffs[int(lo_p / 100 * n)], diffs[int(hi_p / 100 * n)], sum(1 for d in diffs if d > 0) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='../data/bt_manifest_FINAL_test.json')
    ap.add_argument('--votes', default='../data/bt_votes_FINAL_test.json')
    ap.add_argument('-B', type=int, default=10000)
    ap.add_argument('--ci', type=float, default=90.0, help='CI level as a percentage, e.g. 90 or 95')
    args = ap.parse_args()
    lo_p, hi_p = (100 - args.ci) / 2, 100 - (100 - args.ci) / 2

    random.seed(0)
    recs = load_recs(args.manifest, args.votes)
    labels = ['coverage', 'acc(answered)', 'tau_b(answered)', 'tau_b(full)']
    print(f"n = {len(recs)} directional test pairs · {args.ci:.0f}% bootstrap CI (B={args.B})\n")
    print(f"{'metric':14s}" + ''.join(f"{l:>24s}" for l in labels))
    for method, name in [('U', 'U (both axes)'), ('tool', 'tool-only'), ('turn', 'turn-only'), ('lex-turn', 'lex-turn-first'), ('lex-tool', 'lex-tool-first')]:
        pt = stats(recs, method)
        cells = []
        for i in range(4):
            lo, hi = ci(recs, method, i, args.B, lo_p=lo_p, hi_p=hi_p)
            cells.append(f"{pt[i]:+.3f} [{lo:+.2f},{hi:+.2f}]")
        print(f"{name:14s}" + ''.join(f"{c:>24s}" for c in cells))
    print("\nPaired bootstrap, U - baseline (same resample scores both methods):")
    for mB, nm in [('turn', 'turn-only'), ('tool', 'tool-only'), ('lex-turn', 'lex-turn-first'), ('lex-tool', 'lex-tool-first')]:
        pa, loa, hia, fa = paired(recs, 'U', mB, args.B, idx=1, lo_p=lo_p, hi_p=hi_p)
        pt, lo, hi, f = paired(recs, 'U', mB, args.B, idx=3, lo_p=lo_p, hi_p=hi_p)
        print(f"  U - {nm:10s}: dacc={pa:+.3f} {args.ci:.0f}% CI [{loa:+.3f},{hia:+.3f}] P(U>base)={fa*100:.1f}%   "
              f"dtau_b={pt:+.3f} {args.ci:.0f}% CI [{lo:+.3f},{hi:+.3f}] P(U>base)={f*100:.1f}%")


if __name__ == '__main__':
    main()
