#!/usr/bin/env python3
"""Fit lambda_a / lambda_b from Bradley-Terry pairwise votes.

Model (same-task same-success pairs, so success term cancels):
    theta = -a * excess_tool - b * excess_turn
    P(A > B) = sigmoid(theta_A - theta_B)

Fitted a,b  ->  lambda_a = exp(-a),  lambda_b = exp(-b)

Usage:
    python3 fit_bt.py bt_manifest.json bt_votes.json
    python3 fit_bt.py bt_manifest.json --demo 0.07 0.12
"""
import json, sys, math, random

def sigmoid(z): return 1.0 / (1.0 + math.exp(max(-60, min(60, -z))))

def fit(rows, l2=0.3, lr=0.05, iters=10000):
    a = b = 0.1
    n = len(rows)
    for it in range(iters):
        ga = gb = 0.0
        for dt, dn, y in rows:
            p = sigmoid(-a * dt - b * dn)
            e = p - y
            ga += e * (-dt); gb += e * (-dn)
        a -= lr * (ga / n + l2 * a / n)
        b -= lr * (gb / n + l2 * b / n)
        a = max(1e-6, a); b = max(1e-6, b)
    return a, b

def loglik(rows, a, b):
    s = 0.0
    for dt, dn, y in rows:
        p = min(max(sigmoid(-a * dt - b * dn), 1e-9), 1 - 1e-9)
        s += y * math.log(p) + (1 - y) * math.log(1 - p)
    return s

def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    manifest = json.load(open(sys.argv[1]))
    demo = '--demo' in sys.argv

    rows = []
    if demo:
        i = sys.argv.index('--demo')
        ta, tb = float(sys.argv[i + 1]), float(sys.argv[i + 2])
        random.seed(42)
        for x in manifest:
            p = sigmoid(-ta * x['d_etool'] - tb * x['d_eturn'])
            rows.append((x['d_etool'], x['d_eturn'], 1 if random.random() < p else 0))
        print(f"[demo] true a={ta} b={tb}  "
              f"→  lambda_a={math.exp(-ta):.3f}  lambda_b={math.exp(-tb):.3f}")
    else:
        votes = json.load(open(sys.argv[2]))
        idx = {x['pair_id']: x for x in manifest}
        ties = 0
        for pid, who in votes.items():
            if pid not in idx: continue
            x = idx[pid]; dt = x['d_etool']; dn = x['d_eturn']
            if who == 'A':   rows.append((dt, dn, 1))
            elif who == 'B': rows.append((dt, dn, 0))
            elif who == 'tie':
                rows.append((dt, dn, 1)); rows.append((dt, dn, 0)); ties += 1
        print(f"loaded {len(votes)} votes ({ties} ties split) → {len(rows)} labels")

    if not rows:
        print("no usable votes"); return

    a, b = fit(rows)
    la, lb = math.exp(-a), math.exp(-b)
    print()
    print(f"  a (tool penalty rate) = {a:.4f}  →  lambda_a = {la:.4f}")
    print(f"  b (turn penalty rate) = {b:.4f}  →  lambda_b = {lb:.4f}")
    print(f"  a / b ratio           = {a/b:.2f}  ({'工具更严' if a>b else '打扰更严' if b>a else '等权'})")
    print()
    print(f"  解读：多出 10 次超额工具  ≈ 损失成功率 {(1-la**10)*100:.1f}%")
    print(f"        多出 10 次超额轮次  ≈ 损失成功率 {(1-lb**10)*100:.1f}%")
    print(f"  log-likelihood = {loglik(rows, a, b):.2f}")

if __name__ == '__main__':
    main()
