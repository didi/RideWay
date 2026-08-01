#!/usr/bin/env python3
"""Floor-perturbation robustness check.

Human reference-effort floors (tool_human, turn_human) are the only channel
through which human annotation enters etool/eturn. This perturbs every task's
floor by the same delta (in {-2,-1,0,+1,+2} tool/turn units, clamped at 0),
recomputes etool/eturn for every trajectory from the raw tool/turn counts
embedded in the manifests, refits a/b on the perturbed training pairs, and
re-evaluates held-out accuracy and tau_b under that refit -- checking whether
the calibration and its validation depend on the floor being exactly right,
or are stable under a plausible amount of floor mis-annotation.

Usage:
    python3 ablation_floor_robustness.py \
        --manifest-train ../data/bt_manifest_FINAL_train.json \
        --votes-train ../data/bt_votes_FINAL_train.json \
        --manifest-test ../data/bt_manifest_FINAL_test.json \
        --votes-test ../data/bt_votes_FINAL_test.json \
        --tasks-rh44 ../data/tasks_rh44_all.json \
        --tasks-c14 ../data/tasks_custom14_all.json
"""
import argparse, json, math, random

DELTAS = [-2, -1, 0, 1, 2]


def sigmoid(z):
    return 1 / (1 + math.exp(max(-60, min(60, -z))))


def fit(rows, l2=0.3, lr=0.05, iters=12000):
    a = b = 0.1
    n = len(rows)
    for _ in range(iters):
        ga = gb = 0.0
        for dt, dn, y in rows:
            p = sigmoid(-a * dt - b * dn)
            e = p - y
            ga += e * (-dt)
            gb += e * (-dn)
        a -= lr * (ga / n + l2 * a / n)
        b -= lr * (gb / n + l2 * b / n)
        a = max(1e-6, a)
        b = max(1e-6, b)
    return a, b


def votes_to_rows(pairs_dn_dt_who):
    rows = []
    for dt, dn, who in pairs_dn_dt_who:
        if who == 'A':
            rows.append((dt, dn, 1))
        elif who == 'B':
            rows.append((dt, dn, 0))
        else:
            rows.append((dt, dn, 1))
            rows.append((dt, dn, 0))
    return rows


def load_floors(tasks_paths):
    floors = {}
    for p in tasks_paths:
        for t in json.load(open(p)):
            if 'tool_human' in t:
                floors[t['id']] = (t['tool_human'], t['turn_human'])
    return floors


def usable_records(manifest, votes, floors):
    """(d_etool, d_eturn, who, raw_tool_A, raw_turn_A, raw_tool_B, raw_turn_B, tid)
    for every voted pair that has raw tool/turn counts on both sides."""
    idx = {x['pair_id']: x for x in manifest}
    out = []
    for pid, who in votes.items():
        if pid not in idx:
            continue
        x = idx[pid]
        A, B = x['A'], x['B']
        if 'tool' not in A or 'tool' not in B:
            continue
        if x['tid'] not in floors:
            continue
        out.append((who, A['tool'], A['turn'], B['tool'], B['turn'], x['tid']))
    return out


def excess(tool, turn, tool_h, turn_h, d_tool, d_turn):
    tool_h2 = max(0, tool_h + d_tool)
    turn_h2 = max(0, turn_h + d_turn)
    return max(0, tool - tool_h2), max(0, turn - turn_h2)


def stats_at_delta(train_recs, test_recs, floors, d_tool, d_turn):
    train_rows = []
    for who, tA, nA, tB, nB, tid in train_recs:
        tool_h, turn_h = floors[tid]
        etA, enA = excess(tA, nA, tool_h, turn_h, d_tool, d_turn)
        etB, enB = excess(tB, nB, tool_h, turn_h, d_tool, d_turn)
        train_rows.append((etA - etB, enA - enB, who))
    a, b = fit(votes_to_rows(train_rows))

    conc = disc = abst = 0
    for who, tA, nA, tB, nB, tid in test_recs:
        if who == 'tie':
            continue
        tool_h, turn_h = floors[tid]
        etA, enA = excess(tA, nA, tool_h, turn_h, d_tool, d_turn)
        etB, enB = excess(tB, nB, tool_h, turn_h, d_tool, d_turn)
        dt, dn = etA - etB, enA - enB
        z = -a * dt - b * dn
        pred = 1 if z > 0 else (0 if z < 0 else -1)
        y = 1 if who == 'A' else 0
        if pred == -1:
            abst += 1
        elif pred == y:
            conc += 1
        else:
            disc += 1
    n = conc + disc
    tot = n + abst
    acc = conc / n if n else float('nan')
    tau_full = (conc - disc) / tot if tot else float('nan')
    return a, b, acc, tau_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest-train', default='../data/bt_manifest_FINAL_train.json')
    ap.add_argument('--votes-train', default='../data/bt_votes_FINAL_train.json')
    ap.add_argument('--manifest-test', default='../data/bt_manifest_FINAL_test.json')
    ap.add_argument('--votes-test', default='../data/bt_votes_FINAL_test.json')
    ap.add_argument('--tasks-rh44', default='../data/tasks_rh44_all.json')
    ap.add_argument('--tasks-c14', default='../data/tasks_custom14_all.json')
    args = ap.parse_args()

    floors = load_floors([args.tasks_rh44, args.tasks_c14])
    train_manifest = json.load(open(args.manifest_train))
    train_votes = json.load(open(args.votes_train))
    test_manifest = json.load(open(args.manifest_test))
    test_votes = json.load(open(args.votes_test))

    train_recs = usable_records(train_manifest, train_votes, floors)
    test_recs = usable_records(test_manifest, test_votes, floors)
    print(f"train: {len(train_recs)} usable voted pairs (of {len(train_votes)} votes)")
    print(f"test:  {len(test_recs)} usable voted pairs (of {len(test_votes)} votes)\n")

    print(f"{'delta':>6s} {'a':>8s} {'b':>8s} {'a/b':>7s} {'held-out acc':>13s} {'tau_b(full)':>12s}")
    accs, taus = [], []
    baseline = None
    for d in DELTAS:
        a, b, acc, tau = stats_at_delta(train_recs, test_recs, floors, d, d)
        marker = '  <- baseline' if d == 0 else ''
        print(f"{d:+6d} {a:8.4f} {b:8.4f} {a/b:7.2f} {acc:13.3f} {tau:12.3f}{marker}")
        if d == 0:
            baseline = (acc, tau)
        accs.append(acc)
        taus.append(tau)

    print(f"\nAcross delta in {DELTAS}: accuracy in [{min(accs):.3f}, {max(accs):.3f}], "
          f"tau_b(full) in [{min(taus):.3f}, {max(taus):.3f}]")
    print(f"Baseline (delta=0): accuracy={baseline[0]:.3f}, tau_b(full)={baseline[1]:.3f}")


if __name__ == '__main__':
    main()
