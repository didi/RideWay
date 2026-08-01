#!/usr/bin/env python3
"""Check that the IAA pair pools are correctly classified by reference-effort
floor: primary tool-axis pairs should have d_eturn==0 exactly ("pure tool
axis"); the control-set pairs should have d_eturn!=0 ("turn+mixed axis").
Recomputes each pair's d_eturn from its raw tool/turn counts and the current
task floors, reports how many pairs classify as expected, plus human-human
agreement (raw + Cohen's kappa) on each resulting slice.

Usage: python3 check_iaa_pool_validity.py
"""
import json

TASKS_RH44 = '../data/tasks_rh44_all.json'
TASKS_C14 = '../data/tasks_custom14_all.json'


def cohen_kappa(pairs):
    n = len(pairs)
    po = sum(1 for a, b in pairs if a == b) / n
    cats = set(a for a, _ in pairs) | set(b for _, b in pairs)
    pe = sum((sum(1 for a, _ in pairs if a == c) / n) * (sum(1 for _, b in pairs if b == c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float('nan')


def report(label, ids, a1, a2):
    pairs = [(a1[pid], a2[pid]) for pid in ids if pid in a1 and pid in a2]
    if not pairs:
        print(f'{label:20s} n=0')
        return
    agree = sum(1 for a, b in pairs if a == b) / len(pairs)
    k = cohen_kappa(pairs)
    print(f'{label:20s} n={len(pairs):3d}  raw_agreement={agree*100:.1f}%  kappa={k:.3f}')


def main():
    tasks = json.load(open(TASKS_RH44)) + json.load(open(TASKS_C14))
    floors = {t['id']: (t['tool_human'], t['turn_human']) for t in tasks}

    print('=== Primary tool-axis IAA (verified pure-tool under the reference-effort floors) ===')
    m = json.load(open('../data/iaa_c14_tool_axis_FINAL_manifest.json'))
    a1 = json.load(open('../data/iaa_c14_tool_axis_FINAL_annotator1_votes.json'))
    a2 = json.load(open('../data/iaa_c14_tool_axis_FINAL_annotator2_votes.json'))
    still_pure = []
    for x in m:
        tool_h, turn_h = floors[x['tid']]
        d_eturn_new = max(0, x['A']['turn'] - turn_h) - max(0, x['B']['turn'] - turn_h)
        if d_eturn_new == 0:
            still_pure.append(x['pair_id'])
    report('still-pure', still_pure, a1, a2)

    print()
    print('=== IAA control set (verified turn-differing under the reference-effort floors) ===')
    m2 = json.load(open('../data/iaa_control_manifest.json'))
    b1 = json.load(open('../data/iaa_control_annotator1_votes.json'))
    b2 = json.load(open('../data/iaa_control_annotator2_votes.json'))
    still_differ = []
    for x in m2:
        tool_h, turn_h = floors[x['tid']]
        d_eturn_new = max(0, x['A']['turn'] - turn_h) - max(0, x['B']['turn'] - turn_h)
        if d_eturn_new != 0:
            still_differ.append(x['pair_id'])
    report('still-valid', still_differ, b1, b2)


if __name__ == '__main__':
    main()
