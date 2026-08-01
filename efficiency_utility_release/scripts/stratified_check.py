import json, math, argparse

def sig(z): return 1/(1+math.exp(max(-60,min(60,-z))))

ap = argparse.ArgumentParser()
ap.add_argument('--manifest'); ap.add_argument('--votes')
ap.add_argument('--a', type=float); ap.add_argument('--b', type=float)
ap.add_argument('--label', default='')
args = ap.parse_args()

m = json.load(open(args.manifest)); v = json.load(open(args.votes))
idx = {x['pair_id']: x for x in m}
turn_differ, pure_tool, all_pairs = [], [], []
n_ties = 0
for pid, who in v.items():
    if pid not in idx: continue
    x = idx[pid]
    dt, dn = x['d_etool'], x['d_eturn']
    if who == 'tie':
        n_ties += 1; continue
    td = 1 if who=='A' else -1
    pred = 1 if sig(-args.a*dt-args.b*dn) > 0.5 else -1
    correct = (pred==td)
    all_pairs.append(correct)
    (turn_differ if dn!=0 else pure_tool).append(correct)

print(f'--- {args.label} ---')
print(f'n_votes={len(v)} ties={n_ties} directional_pairs={len(all_pairs)}')
print(f'overall accuracy: {sum(all_pairs)/len(all_pairs)*100:.1f}%')
if turn_differ:
    print(f'turn-differing: n={len(turn_differ)} acc={sum(turn_differ)/len(turn_differ)*100:.1f}%')
if pure_tool:
    print(f'pure-tool:      n={len(pure_tool)} acc={sum(pure_tool)/len(pure_tool)*100:.1f}%')
