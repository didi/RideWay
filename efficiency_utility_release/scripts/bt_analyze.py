#!/usr/bin/env python3
"""Full BT analysis on labeled pairs: point fit + bootstrap CIs + leave-one-task-out
concordance vs naive baselines (pass@k, tool-only, turn-only).

This is the script behind the a, b, and a/b bootstrap CIs reported in Sec 5.2 of
the paper (default --ci 90 reproduces a=[0.09,0.16], b=[0.19,0.34], a/b=[0.33,0.71];
pass --ci 95 to confirm a vs b stays disjoint at the stricter level too).

Pass --unconstrained to remove the non-negativity clip on every gradient step --
this is the script behind Appendix G's claim that removing the clip "reproduces
the unconstrained point estimate and the confidence intervals on a and b" (both
constrained and unconstrained give the identical point estimate a=0.1235,
b=0.2448 on the current corrected data, with a>0 in 100% of bootstrap resamples
even with the clip off).

Pass --skip-loto to skip section [3] (leave-one-task-out concordance) -- an
internal sanity check, not a number reported in the paper, and by far the most
expensive part of a run at B>=1000.

Usage: python3 bt_analyze.py bt_manifest.json bt_votes.json [B] [--ci 90|95] [--unconstrained] [--skip-loto]
"""
import json, sys, math, random
from collections import defaultdict, Counter
random.seed(0)

_ci_level = 90.0
if '--ci' in sys.argv:
    i = sys.argv.index('--ci')
    _ci_level = float(sys.argv[i + 1])
    del sys.argv[i:i + 2]
_lo_p, _hi_p = (100 - _ci_level) / 2, 100 - (100 - _ci_level) / 2

_unconstrained = '--unconstrained' in sys.argv
if _unconstrained:
    sys.argv.remove('--unconstrained')
_skip_loto = '--skip-loto' in sys.argv
if _skip_loto:
    sys.argv.remove('--skip-loto')

def sigmoid(z): return 1.0/(1.0+math.exp(max(-60,min(60,-z))))

def fit(rows, l2=0.3, lr=0.05, iters=12000, clip=not _unconstrained):
    a=b=0.1; n=len(rows)
    for _ in range(iters):
        ga=gb=0.0
        for dt,dn,y in rows:
            p=sigmoid(-a*dt-b*dn); e=p-y; ga+=e*(-dt); gb+=e*(-dn)
        a-=lr*(ga/n+l2*a/n); b-=lr*(gb/n+l2*b/n)
        if clip:
            a=max(1e-6,a); b=max(1e-6,b)
    return a,b

def votes_to_rows(items):
    rows=[]
    for dt,dn,who in items:
        if who=='A': rows.append((dt,dn,1))
        elif who=='B': rows.append((dt,dn,0))
        else: rows.append((dt,dn,1)); rows.append((dt,dn,0))
    return rows

man=json.load(open(sys.argv[1])); votes=json.load(open(sys.argv[2]))
idx={x['pair_id']:x for x in man}
# items: (d_etool, d_eturn, who, tid, axis)
items=[(idx[p]['d_etool'], idx[p]['d_eturn'], who, idx[p]['tid'], idx[p]['axis'])
       for p,who in votes.items() if p in idx]
print(f"{len(items)} votes  | ties={sum(1 for x in items if x[2]=='tie')}  "
      f"directional={sum(1 for x in items if x[2]!='tie')}  "
      f"| {'UNCONSTRAINED (clip off)' if _unconstrained else 'constrained (clip on)'}")

# ---------- 1. point fit ----------
a0,b0=fit(votes_to_rows([(d,n,w) for d,n,w,_,_ in items]))
print(f"\n[1] POINT FIT")
print(f"  a={a0:.4f} -> lambda_a={math.exp(-a0):.4f}   b={b0:.4f} -> lambda_b={math.exp(-b0):.4f}   a/b={a0/b0:.2f}")
print(f"  每多10次超额工具 ~ 损失 {(1-math.exp(-a0)**10)*100:.0f}% ; 每多10次超额轮次 ~ 损失 {(1-math.exp(-b0)**10)*100:.0f}%")

# ---------- 2. bootstrap CIs ----------
As=[];Bs=[];R=[];Rba=[];bgt=0;a_pos=0;B=int(sys.argv[3]) if len(sys.argv) > 3 else 2000;n=len(items)
for _ in range(B):
    samp=[items[random.randrange(n)] for _ in range(n)]
    a,b=fit(votes_to_rows([(d,nn,w) for d,nn,w,_,_ in samp]))
    As.append(a);Bs.append(b)
    R.append(a/b if b>1e-6 else float('nan'))
    Rba.append(b/a if a>1e-6 else float('nan'))
    if b>a: bgt+=1
    if a>0: a_pos+=1
def pc(x,p): x=sorted([v for v in x if v==v]); return x[max(0,min(len(x)-1,int(p/100*len(x))))]
lo,hi=_lo_p,_hi_p
print(f"\n[2] BOOTSTRAP (B={B}, {_ci_level:.0f}% CI)")
print(f"  a   {_ci_level:.0f}%CI [{pc(As,lo):.4f}, {pc(As,hi):.4f}]  -> lambda_a [{math.exp(-pc(As,hi)):.3f}, {math.exp(-pc(As,lo)):.3f}]")
print(f"  b   {_ci_level:.0f}%CI [{pc(Bs,lo):.4f}, {pc(Bs,hi):.4f}]  -> lambda_b [{math.exp(-pc(Bs,hi)):.3f}, {math.exp(-pc(Bs,lo)):.3f}]")
print(f"  a/b {_ci_level:.0f}%CI [{pc(R,lo):.2f}, {pc(R,hi):.2f}]   median {pc(R,50):.2f}")
print(f"  b/a {_ci_level:.0f}%CI [{pc(Rba,lo):.2f}, {pc(Rba,hi):.2f}]   median {pc(Rba,50):.2f}")
print(f"  P(turn penalized > tool | data) = {bgt/B*100:.0f}%")
print(f"  a significantly > 0 ({lo:.1f}%ile>0)? {'YES' if pc(As,lo)>1e-3 else 'NO'}    b>0? {'YES' if pc(Bs,lo)>1e-3 else 'NO'}")
print(f"  a vs b disjoint at {_ci_level:.0f}%? {'YES' if pc(As,hi) < pc(Bs,lo) else 'NO'}")
if _unconstrained:
    print(f"  a>0 in {a_pos}/{B} = {a_pos/B*100:.1f}% of resamples (even with the non-negativity clip off)")

if _skip_loto:
    sys.exit(0)

# ---------- 3. leave-one-task-out concordance ----------
tasks=sorted({t for _,_,_,t,_ in items})
def predict_acc(predfn):
    cor=tot=0.0
    by_ax=defaultdict(lambda:[0.0,0.0])
    for ho in tasks:
        train=[(d,n,w) for d,n,w,t,_ in items if t!=ho]
        a,b=fit(votes_to_rows(train))
        for d,n,w,t,ax in items:
            if t!=ho or w=='tie': continue
            y=1 if w=='A' else 0
            p=predfn(d,n,a,b)
            c=0.5 if p==0.5 else (1.0 if (p>0.5)==(y==1) else 0.0)
            cor+=c; tot+=1; by_ax[ax][0]+=c; by_ax[ax][1]+=1
    return cor/tot, {k:v[0]/v[1] for k,v in by_ax.items()}

preds={
 'pass@k (coin)': lambda d,n,a,b: 0.5,
 'tool-only'    : lambda d,n,a,b: 0.5 if d==0 else (1.0 if d<0 else 0.0),
 'turn-only'    : lambda d,n,a,b: 0.5 if n==0 else (1.0 if n<0 else 0.0),
 'U (BT both)'  : lambda d,n,a,b: sigmoid(-a*d-b*n),
}
print(f"\n[3] LEAVE-ONE-TASK-OUT CONCORDANCE  ({len(tasks)} tasks, eval on {sum(1 for x in items if x[2]!='tie')} directional votes)")
print(f"  {'predictor':14s} {'overall':>8s} {'tool-axis':>10s} {'turn-axis':>10s}")
for name,fn in preds.items():
    acc,ax=predict_acc(fn)
    print(f"  {name:14s} {acc*100:7.0f}% {ax.get('tool',0)*100:9.0f}% {ax.get('turn',0)*100:9.0f}%")
