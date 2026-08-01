# Human evaluation of the LLM-as-a-judge success gate

Every trajectory's success/fail label in this benchmark comes from a 3-judge LLM panel
(hy3-preview + Seed 2.0 Lite + Grok 4.3, each judge itself a 3-pass internal majority,
2-of-3 judges required for the panel to pass a trajectory — the same panel used
throughout RESULTS_leaderboard.md and RESULTS_task_difficulty.md). This study checks
that panel's own reliability against a blind human relabel.

**Sampling**: a random sample of 75 trajectories drawn from the full panel-output pool
across all 24 models and both domains (rh44 + custom14).

**What the human annotator sees**: exactly the three inputs the judge panel's own prompt
receives, so the comparison is apples-to-apples — the task's grading rubric, the user's
original scenario instructions, and the final system state (system time + the ride order
created, if any), alongside the full trajectory. The annotator never sees model identity,
tier, or the panel's own verdict.

Reproduce: `python3 compute_llm_judge_human_eval_stats.py`.

## Headline numbers (n=75)

| Statistic | Value | 90% CI |
|---|---|---|
| Raw agreement (human vs. panel) | 90.7% | [85.3%, 96.0%] |
| Cohen's kappa | 0.813 | [0.70, 0.92] |
| False positive rate (panel said success, human disagreed) | 5.4% (2/37) | — |
| False negative rate (panel said fail, human disagreed) | 13.2% (5/38) | — |

kappa=0.813 is in the "almost perfect" Landis–Koch band (>0.80), well clear of the 0.60
"substantial agreement" threshold used elsewhere in this bundle (compare
RESULTS_iaa's human-human kappa=0.639 on the tool-axis preference task).

## Breakdown by model tier

| Tier | n | Agreement |
|---|---|---|
| A | 13 | 100.0% |
| B | 17 | 100.0% |
| C | 22 | 81.8% |
| D | 23 | 87.0% |

All 7 disagreements across the full sample come from Tier C/D models: Tier A+B is a
combined 30/30 (100%), Tier C+D is a combined 39/45 (86.7%). The panel is essentially
perfectly reliable on strong-model trajectories — exactly where the leaderboard's top
ranks and the paper's headline claims live — with a characterizable soft spot on
weaker-tier trajectories, which are harder to grade correctly in both directions (the
7 disagreements split 2 false-positive / 5 false-negative, not one-directional).

## Breakdown by panel decision confidence

| Panel confidence | n | Agreement |
|---|---|---|
| Borderline (2-1 vote somewhere) | 14 | 78.6% |
| Clear (unanimous 3-0) | 61 | 93.4% |

As expected, the panel's own internal disagreement (a 2-1 split among the three judges)
is itself predictive of when a human is more likely to disagree with the panel's final
call.
