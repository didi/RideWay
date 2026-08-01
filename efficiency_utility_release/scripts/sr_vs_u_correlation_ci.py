#!/usr/bin/env python3
"""Bootstrap CI on the SR-vs-U Spearman correlation: does SR_58 predict U_custom14, overall and among
the strongest models?

Resamples the 24 models with replacement and recomputes Spearman's rho each
draw. The "top 12" resample pool is the fixed 12 models with the highest
U_custom14 in the real (unresampled) data -- this tests how much the
correlation estimate on that specific subgroup would vary under resampling,
not a re-ranked top-12 per draw.

Usage:
    python3 sr_vs_u_correlation_ci.py --leaderboard ../data/RESULTS_leaderboard.md --ci 90
"""
import argparse, random, re


def spearman(rows):
    sr = [r[1] for r in rows]; u = [r[2] for r in rows]

    def rank(v):
        idx = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for ri, i in enumerate(idx):
            r[i] = ri + 1
        return r

    rsr, ru = rank(sr), rank(u)
    n = len(rsr)
    msr = sum(rsr) / n; mu = sum(ru) / n
    num = sum((a - msr) * (b - mu) for a, b in zip(rsr, ru))
    dsr = sum((a - msr) ** 2 for a in rsr) ** 0.5
    du = sum((b - mu) ** 2 for b in ru) ** 0.5
    return num / (dsr * du) if dsr > 0 and du > 0 else float('nan')


def boot_ci(pop, B, lo_p, hi_p, seed=0):
    rng = random.Random(seed)
    n = len(pop)
    vals = []
    for _ in range(B):
        samp = [pop[rng.randrange(n)] for _ in range(n)]
        v = spearman(samp)
        if v == v:
            vals.append(v)
    vals.sort()
    return vals[int(lo_p / 100 * len(vals))], vals[int(hi_p / 100 * len(vals))]


def load_rows(path):
    """Parse the (Model, SR_58, U_c14) columns out of RESULTS_leaderboard.md's table."""
    rows = []
    for line in open(path):
        m = re.match(r'\|\s*\d+\s*\|\s*(\S+)\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|'
                     r'\s*([\d.]+)\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|'
                     r'\s*[\d.]+\s*\|\s*([\d.]+)\s*\|', line)
        if m:
            rows.append((m.group(1), float(m.group(2)), float(m.group(3))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--leaderboard', default='../data/RESULTS_leaderboard.md')
    ap.add_argument('-B', type=int, default=10000)
    ap.add_argument('--ci', type=float, default=90.0, help='CI level as a percentage, e.g. 90 or 95')
    args = ap.parse_args()
    lo_p, hi_p = (100 - args.ci) / 2, 100 - (100 - args.ci) / 2

    rows = load_rows(args.leaderboard)
    print(f"n = {len(rows)} models (SR_58, U_custom14)\n")

    pt_all = spearman(rows)
    lo, hi = boot_ci(rows, args.B, lo_p, hi_p)
    print(f"All {len(rows)}:  rho={pt_all:+.3f}  {args.ci:.0f}% CI [{lo:+.3f},{hi:+.3f}]")

    top12 = sorted(rows, key=lambda r: -r[2])[:12]
    pt_top = spearman(top12)
    lo, hi = boot_ci(top12, args.B, lo_p, hi_p)
    print(f"Top 12:   rho={pt_top:+.3f}  {args.ci:.0f}% CI [{lo:+.3f},{hi:+.3f}]")


if __name__ == '__main__':
    main()
