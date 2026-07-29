#!/usr/bin/env python3
"""Full IAA statistics suite for the tool-axis (and later control) 2-annotator studies.

Reports: raw agreement, Cohen's kappa, linear-weighted kappa (ordinal A>tie>B),
Gwet's AC1 (robust to skewed marginals), Krippendorff's alpha (nominal + ordinal),
marginal tables, and Krippendorff's alpha with U included as a third "rater"
(if alpha barely drops when U joins, U is statistically indistinguishable from
another human annotator on these pairs).

All statistics come with 90% bootstrap CIs (resampling items).

Usage:
    python3 compute_iaa_stats.py \
        --manifest iaa_c14_tool_axis_subset_manifest.json \
        --a1 iaa_c14_tool_axis_annotator1_votes.json \
        --a2 iaa_c14_tool_axis_subset_annotator2_votes.json
"""
import argparse, json, math, random

CATS = ['A', 'tie', 'B']          # ordered: A > tie > B (preference direction)
SCORE = {'A': 0.0, 'tie': 1.0, 'B': 2.0}

A_FIT, B_FIT = 0.1111, 0.4164     # 325-pair training fit


def u_pred(x):
    z = -A_FIT * x['d_etool'] - B_FIT * x['d_eturn']
    return 'A' if z > 0 else ('B' if z < 0 else 'tie')


def raw_agreement(pairs):
    return sum(1 for a, b in pairs if a == b) / len(pairs)


def cohen_kappa(pairs, weighted=False):
    n = len(pairs)
    po_num = 0.0
    m1 = {c: 0 for c in CATS}; m2 = {c: 0 for c in CATS}
    for a, b in pairs:
        w = 1.0 - abs(SCORE[a] - SCORE[b]) / (len(CATS) - 1) if weighted else (1.0 if a == b else 0.0)
        po_num += w
        m1[a] += 1; m2[b] += 1
    po = po_num / n
    pe = 0.0
    for ca in CATS:
        for cb in CATS:
            w = 1.0 - abs(SCORE[ca] - SCORE[cb]) / (len(CATS) - 1) if weighted else (1.0 if ca == cb else 0.0)
            pe += w * (m1[ca] / n) * (m2[cb] / n)
    return (po - pe) / (1 - pe) if pe < 1 else float('nan')


def gwet_ac1(pairs):
    n = len(pairs)
    po = raw_agreement(pairs)
    # average marginal proportion per category across both raters
    pi = {}
    for c in CATS:
        pi[c] = (sum(1 for a, _ in pairs if a == c) + sum(1 for _, b in pairs if b == c)) / (2 * n)
    pe = sum(pi[c] * (1 - pi[c]) for c in CATS) / (len(CATS) - 1)
    return (po - pe) / (1 - pe) if pe < 1 else float('nan')


def kripp_alpha(columns, ordinal=False):
    """columns: list of per-rater dicts item->label (missing items allowed)."""
    items = set()
    for col in columns:
        items |= set(col)
    # coincidence matrix over pairable values
    coinc = {(a, b): 0.0 for a in CATS for b in CATS}
    n_total = 0.0
    for it in items:
        vals = [col[it] for col in columns if it in col]
        m = len(vals)
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i != j:
                    coinc[(vals[i], vals[j])] += 1.0 / (m - 1)
        n_total += m
    if n_total <= 1:
        return float('nan')
    marg = {c: sum(coinc[(c, b)] for b in CATS) for c in CATS}

    def delta(a, b):
        if not ordinal:
            return 0.0 if a == b else 1.0
        return (SCORE[a] - SCORE[b]) ** 2

    do = sum(coinc[(a, b)] * delta(a, b) for a in CATS for b in CATS) / n_total
    de = sum(marg[a] * marg[b] * delta(a, b) for a in CATS for b in CATS) / (n_total * (n_total - 1))
    return 1 - do / de if de > 0 else float('nan')


def boot_ci(fn, items, B=10000, seed=0):
    rng = random.Random(seed)
    vals = []
    for _ in range(B):
        samp = [items[rng.randrange(len(items))] for _ in items]
        v = fn(samp)
        if v == v:
            vals.append(v)
    vals.sort()
    return vals[int(0.05 * len(vals))], vals[int(0.95 * len(vals))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='iaa_c14_tool_axis_subset_manifest.json')
    ap.add_argument('--a1', default='iaa_c14_tool_axis_annotator1_votes.json')
    ap.add_argument('--a2', default='iaa_c14_tool_axis_subset_annotator2_votes.json')
    ap.add_argument('-B', type=int, default=10000)
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    idx = {x['pair_id']: x for x in manifest}
    a1 = json.load(open(args.a1))
    a2 = json.load(open(args.a2))

    common = [pid for pid in a2 if pid in a1 and pid in idx]
    triples = [(pid, a1[pid], a2[pid], u_pred(idx[pid])) for pid in common]
    pairs = [(v1, v2) for _, v1, v2, _ in triples]
    n = len(pairs)

    print(f"n = {n} pairs labeled by both annotators\n")

    print("Marginals:")
    for name, col in [('annotator1', [v1 for _, v1, _, _ in triples]),
                      ('annotator2', [v2 for _, _, v2, _ in triples]),
                      ('U         ', [u for _, _, _, u in triples])]:
        counts = {c: col.count(c) for c in CATS}
        print(f"  {name}: {counts}")
    print()

    stats = [
        ('raw agreement', lambda ps: raw_agreement(ps)),
        ("Cohen's kappa", lambda ps: cohen_kappa(ps, weighted=False)),
        ('linear-weighted kappa', lambda ps: cohen_kappa(ps, weighted=True)),
        ("Gwet's AC1", lambda ps: gwet_ac1(ps)),
    ]
    print(f"{'statistic':24s} {'value':8s} {'90% CI':18s}")
    for name, fn in stats:
        pt = fn(pairs)
        lo, hi = boot_ci(fn, pairs, B=args.B)
        print(f"{name:24s} {pt:+.3f}   [{lo:+.3f}, {hi:+.3f}]")

    # Krippendorff's alpha: humans only vs humans+U (item-resampled bootstrap)
    def alpha_from_triples(ts, cols, ordinal):
        columns = []
        for ci in cols:
            columns.append({i: t[ci] for i, t in enumerate(ts)})
        return kripp_alpha(columns, ordinal=ordinal)

    print()
    print(f"{'Krippendorff alpha':38s} {'value':8s} {'90% CI':18s}")
    for label, cols, ordinal in [
        ('humans only, nominal', (1, 2), False),
        ('humans only, ordinal', (1, 2), True),
        ('humans + U as 3rd rater, nominal', (1, 2, 3), False),
        ('humans + U as 3rd rater, ordinal', (1, 2, 3), True),
    ]:
        fn = lambda ts, c=cols, o=ordinal: alpha_from_triples(ts, c, o)
        pt = fn(triples)
        lo, hi = boot_ci(fn, triples, B=args.B)
        print(f"{label:38s} {pt:+.3f}   [{lo:+.3f}, {hi:+.3f}]")

    print("\nNote: U never votes 'tie' (it always has a directional preference when")
    print("d_etool != 0), which depresses its alignment with human tie votes; the")
    print("humans+U alpha is therefore a conservative read of U's human-likeness.")


if __name__ == '__main__':
    main()
