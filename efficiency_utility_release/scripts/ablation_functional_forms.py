#!/usr/bin/env python3
"""Functional-form ablation: fit alternative BT efficiency models on the training
set, evaluate held-out. Answers 'is the two-weight linear-in-excess form right?'

Forms (BT logit theta, per-side excess with transform f):
  two-weight linear (current):  theta = -a*(etool_A-etool_B) - b*(eturn_A-eturn_B)
  single-combined-step (a=b):   theta = -c*((etool_A+eturn_A) - (etool_B+eturn_B))
  saturating log / sqrt:        f = log(1+x) / sqrt(x) applied per side before diff

Ties are split into balanced win/loss rows (same as fit_bt.py). Held-out eval on
custom14 directional pairs: accuracy-on-answered and tau_b(full), 90% boot CI.

Usage: python3 ablation_functional_forms.py
"""
import json, math, random

F = {'linear': lambda x: x, 'log1p': lambda x: math.log(1 + x), 'sqrt': lambda x: math.sqrt(x)}


def sig(z):
    return 1 / (1 + math.exp(max(-60, min(60, -z))))


def load_rows(mf, vf, f):
    m = json.load(open(mf)); v = json.load(open(vf)); idx = {x['pair_id']: x for x in m}
    rows = []
    for pid, who in v.items():
        if pid not in idx:
            continue
        x = idx[pid]
        dtf = f(x['A']['etool']) - f(x['B']['etool'])
        dnf = f(x['A']['eturn']) - f(x['B']['eturn'])
        if who == 'A':
            rows.append((dtf, dnf, 1))
        elif who == 'B':
            rows.append((dtf, dnf, 0))
        elif who == 'tie':
            rows.append((dtf, dnf, 1)); rows.append((dtf, dnf, 0))
    return rows


def fit(rows, single, l2=0.3, lr=0.05, iters=10000):
    a = b = 0.1; n = len(rows)
    for _ in range(iters):
        ga = gb = 0.0
        for dt, dn, y in rows:
            if single:
                p = sig(-a * (dt + dn)); ga += (p - y) * (-(dt + dn))
            else:
                p = sig(-a * dt - b * dn); e = p - y
                ga += e * (-dt); gb += e * (-dn)
        if single:
            a -= lr * (ga / n + l2 * a / n); a = max(1e-6, a); b = a
        else:
            a -= lr * (ga / n + l2 * a / n); b -= lr * (gb / n + l2 * b / n)
            a = max(1e-6, a); b = max(1e-6, b)
    return a, b


def test_recs(f):
    m = json.load(open('bt_manifest_FINAL_test.json')); v = json.load(open('bt_votes_FINAL_test.json'))
    idx = {x['pair_id']: x for x in m}; recs = []
    for pid, who in v.items():
        if who == 'tie' or pid not in idx:
            continue
        x = idx[pid]
        recs.append((f(x['A']['etool']) - f(x['B']['etool']),
                     f(x['A']['eturn']) - f(x['B']['eturn']),
                     1 if who == 'A' else -1))
    return recs


def stats(sample, a, b, single):
    conc = disc = 0
    for dt, dn, td in sample:
        z = -a * (dt + dn) if single else -a * dt - b * dn
        pd = 1 if z > 0 else (-1 if z < 0 else 0)
        if pd == 0:
            continue
        if pd == td:
            conc += 1
        else:
            disc += 1
    tot = len(sample); n = conc + disc
    return (conc / n if n else float('nan')), (conc - disc) / tot


def ci(recs, a, b, single, B=6000):
    accs = []; taus = []
    for _ in range(B):
        s = [random.choice(recs) for _ in recs]
        ac, ta = stats(s, a, b, single); accs.append(ac); taus.append(ta)
    accs = [x for x in accs if x == x]; accs.sort(); taus.sort()
    return ((accs[int(0.05 * len(accs))], accs[int(0.95 * len(accs))]),
            (taus[int(0.05 * len(taus))], taus[int(0.95 * len(taus))]))


def main():
    random.seed(0)
    print("Functional-form ablation — fit on 320 training pairs, eval on 75 held-out custom14 pairs\n")
    print(f"{'form':28s} {'params':24s} {'acc(ans)':22s} {'tau_full':22s}")
    for name, ft, single in [
        ('two-weight linear (current)', 'linear', False),
        ('single-combined-step (a=b)', 'linear', True),
        ('two-weight saturating log', 'log1p', False),
        ('two-weight saturating sqrt', 'sqrt', False),
    ]:
        f = F[ft]
        a, b = fit(load_rows('bt_manifest_FINAL_train.json', 'bt_votes_FINAL_train.json', f), single)
        recs = test_recs(f)
        ac, ta = stats(recs, a, b, single)
        (alo, ahi), (tlo, thi) = ci(recs, a, b, single)
        pstr = f"c={a:.3f}" if single else f"a={a:.3f} b={b:.3f} a/b={a / b:.2f}"
        print(f"{name:28s} {pstr:24s} {ac:+.3f} [{alo:+.2f},{ahi:+.2f}]   {ta:+.3f} [{tlo:+.2f},{thi:+.2f}]")


if __name__ == '__main__':
    main()
