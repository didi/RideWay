# RideWay Efficiency Utility — release bundle

This bundle packages the Efficiency Utility (U) metric's calibration data,
Bradley-Terry fit/analysis scripts, and headline results.

```
efficiency_utility_release/
  data/
    tasks_rh44_all.json / tasks_custom14_all.json   the 58 RideWay task definitions (44+14), including each task's tool_human/turn_human reference-effort floors
    bt_pairs_train.json          320 labeled BT calibration pairs, full trajectories embedded
    bt_pairs_test.json           80 labeled BT held-out pairs, full trajectories embedded
    listwise_labels.json         80 listwise-tier-labeled trajectories (10 tasks), full trajectories embedded
    iaa_pairs.json                36 originally-sampled tool-axis IAA pairs with full trajectories embedded (qualitative/browsing copy; 23 of these are strictly pure-tool under the reference-effort floors and are what compute_iaa_stats.py actually scores -- see iaa_c14_tool_axis_FINAL_* below)
    example_trajectories.json     15 hand-picked example trajectories (5 high-eff / 5 low-eff / 5 failed), qualitative -- no U score
    taxonomy_pairs.json           11 tool-axis discordant pairs (U vs. human disagreement), full trajectories + error-taxonomy classification
    RESULTS_leaderboard.md        24-model leaderboard: SR/pass@3/pass^3 (rh44, pooled-58, custom14) + mean U (custom14)
    RESULTS_leaderboard_ci.md     per-model 90%/95% task-level bootstrap CI on custom14 mean U (Table tab:model-leaderboard's CI column)
    RESULTS_task_difficulty.md    per-task SR/pass@3/pass^3 across all 58 tasks, pooled over 24 models (72 trials/task)
    bt_manifest_FINAL_{train,test}.json / bt_votes_FINAL_{train,test}.json   raw manifest+vote format (no embedded trajectories), 320/80 voted pairs -- feed these directly into fit_bt.py / bt_analyze.py / ablation_*.py
    iaa_c14_tool_axis_FINAL_*.json                                          raw IAA manifest+votes format, 23 pure-tool pairs -- feed into compute_iaa_stats.py
    iaa_control_manifest.json / iaa_control_annotator{1,2}_votes.json       17 dual-annotated turn+mixed-axis pairs (secondary IAA check, outside the pure-tool axis) -- feed into compute_iaa_stats.py
    listwise_eff_manifest.json / listwise_eff_votes.json                    raw listwise format, 80 labeled trajectories -- feed into score_listwise_eff.py
    llm_judge_human_eval_manifest.json / llm_judge_human_eval_votes.json    75 blind-relabeled trajectories vs. LLM judge panel verdict -- feed into compute_llm_judge_human_eval_stats.py
    RESULTS_llm_judge_human_eval.md   human evaluation of the LLM-as-a-judge success gate: raw agreement, Cohen's kappa, tier breakdown
  scripts/
    fit_bt.py                    core Bradley-Terry fit (the number everything else derives from)
    bt_analyze.py                fuller version: point fit + bootstrap CI + leave-one-task-out concordance (--ci 90|95, --unconstrained, --skip-loto)
    ablation_baselines_ci.py     held-out test table: U vs tool-only/turn-only/lex-turn-first/lex-tool-first, plus paired bootstrap on accuracy and tau_b (--ci 90|95)
    ablation_functional_forms.py functional-form ablation (linear vs single-weight vs log/sqrt saturating)
    ablation_floor_robustness.py refits a/b and re-evaluates held-out accuracy/tau_b under perturbed reference-effort floors (delta in {-2,-1,+1,+2})
    stratified_check.py          turn-differing vs. pure-tool accuracy breakdown (Table tab:tool-axis-stats)
    compute_iaa_stats.py         inter-annotator agreement statistics (kappa, Gwet's AC1, Krippendorff's alpha) (--ci 90|95)
    check_iaa_pool_validity.py   re-derives which IAA pairs are correctly classified under the reference-effort floors
    score_listwise_eff.py        listwise ranking validation (tie-aware Kendall tau_b) (--ci 90|95)
    compute_llm_judge_human_eval_stats.py  human evaluation of the LLM-as-a-judge success gate itself (raw agreement, Cohen's kappa, bootstrap CI, tier/domain/confidence breakdown) (--ci 90|95)
    sr_vs_u_correlation_ci.py        SR-vs-U Spearman correlation + bootstrap CI (Section 7.7) (--ci 90|95)
    build_model_leaderboard.py       regenerates RESULTS_leaderboard.md  [needs full sim corpus, see below]
    build_task_difficulty_table.py   regenerates RESULTS_task_difficulty.md  [needs full sim corpus, see below]
    leaderboard_bootstrap_ci.py      per-model 90%/95% bootstrap CI on custom14 mean U (Appendix E table)  [needs full sim corpus, see below]
    functional_form_ranking_ci.py    exp(-C) vs. 1/(1+C) ranking-equivalence check + bootstrap CI (Appendix G)  [needs full sim corpus, see below]
```

## What's self-contained vs. what needs the full repo

Two different tiers of reproducibility, by design:

- **Self-contained** (`fit_bt.py`, `bt_analyze.py`, `ablation_baselines_ci.py`,
  `ablation_functional_forms.py`, `ablation_floor_robustness.py`,
  `stratified_check.py`, `compute_iaa_stats.py`,
  `check_iaa_pool_validity.py`, `score_listwise_eff.py`,
  `compute_llm_judge_human_eval_stats.py`, `sr_vs_u_correlation_ci.py`):
  these only need the small manifest/vote JSON files already in `data/`. Run them
  straight from `scripts/` as shown below -- no external data needed. All the
  bootstrap-CI scripts in this group take an optional `--ci` flag (default 90,
  pass `--ci 95` for the stricter interval).
- **Needs the full RideWay simulation corpus** (`build_model_leaderboard.py`,
  `build_task_difficulty_table.py`, `leaderboard_bootstrap_ci.py`,
  `functional_form_ranking_ci.py`): these read
  every model's raw trajectory dump (`data/simulations/<domain>_<model>`, one
  file per model, hundreds of MB total) plus the per-judge re-evaluation
  files. That corpus is not part of this bundle -- run these from within a
  full RideWay checkout, pointing `--sim-dir`/`--tasks-dir` at
  `data/simulations` / `data/eff/domains/ridehailing` (all three scripts take
  the same two flags with the same defaults, so any working directory two
  levels below the checkout root works without overriding them).
  `RESULTS_leaderboard.md` and `RESULTS_task_difficulty.md` in `data/` are
  `build_model_leaderboard.py`/`build_task_difficulty_table.py`'s precomputed
  output, included so the point-estimate numbers are inspectable without
  needing the corpus. `RESULTS_leaderboard.md` does not include a CI column,
  but `data/RESULTS_leaderboard_ci.md` does (90% and 95%, `leaderboard_bootstrap_ci.py`'s
  precomputed output) -- every one of the 24 models' CI bounds there reproduces
  Appendix E's Table `tab:model-leaderboard` column exactly (verified against
  the full corpus).

Every `trajectory` field embedded in `bt_pairs_*.json` / `listwise_labels.json`
/ `iaa_pairs.json` / `example_trajectories.json` / `taxonomy_pairs.json` was
also built from that same full simulation corpus, using internal export
tooling that isn't included here -- what's shipped for those files is the
finished data, not the scaffolding that assembled it.

## Reproducing every number

### 1. The Bradley-Terry fit (lambda_tool, lambda_turn)

```
cd scripts
python3 fit_bt.py ../data/bt_manifest_FINAL_train.json ../data/bt_votes_FINAL_train.json
```
Expected: `a=0.1235 -> lambda_tool=0.8838`, `b=0.2448 -> lambda_turn=0.7829`,
`a/b=0.50` (one excess turn costs ~2.0x one excess tool call). This is the
number `U(tau) = success(tau) * lambda_tool^etool(tau) * lambda_turn^eturn(tau)`
is built from -- every other script below either consumes it or re-derives it.

For the fuller version (bootstrap CI + leave-one-task-out concordance vs
naive baselines -- the LOTO section is an internal sanity check, not a number
reported in the paper, and by far the most expensive part of a run; pass
`--skip-loto` to skip straight to the bootstrap CI, which is what most callers
actually want):
```
python3 bt_analyze.py ../data/bt_manifest_FINAL_train.json ../data/bt_votes_FINAL_train.json 2500 --skip-loto
python3 bt_analyze.py ../data/bt_manifest_FINAL_train.json ../data/bt_votes_FINAL_train.json 2500 --ci 95 --skip-loto
```
Expected 90% bootstrap intervals (B=2500, App. "Bootstrap
resample counts"): `a` [0.09, 0.16], `b` [0.19, 0.34], `a/b` [0.34, 0.71]
(median 0.50), and `b/a` [1.40, 2.96] (median 2.01). At 95% (`--ci 95`): `a`
[0.08, 0.17], `b` [0.18, 0.36], `a/b` [0.32, 0.77], `b/a` [1.31, 3.17] -- `a`
and `b` stay disjoint at 95% too, so the asymmetry claim isn't an artifact of
the 90% cutoff.

Pass `--unconstrained` to remove the non-negativity clip on every gradient
step -- this is the script behind Appendix G's claim that removing the clip
reproduces the same point estimate and CI:
```
python3 bt_analyze.py ../data/bt_manifest_FINAL_train.json ../data/bt_votes_FINAL_train.json 2500 --unconstrained --skip-loto
```
Expected: point estimate identical to the constrained fit (a=0.1235,
b=0.2448), with `a>0` in 100% of resamples even with the clip off.

For the floor-perturbation robustness check (does the calibration depend on
getting the human reference-effort floor exactly right?):
```
python3 ablation_floor_robustness.py
```
Expected: perturbing every task's floor by delta in {-2,-1,+1,+2} tool/turn
units and refitting from scratch changes held-out accuracy only within
[0.787, 0.813] and tau_b(full) only within [0.573, 0.627], both at or above
the unperturbed baseline (0.787, 0.573).

### 2. Held-out test-set validation (accuracy, tau_b, baseline comparison)

```
python3 ablation_baselines_ci.py --manifest ../data/bt_manifest_FINAL_test.json \
    --votes ../data/bt_votes_FINAL_test.json -B 10000
```
Expected headline row: `U (both axes)` accuracy 0.787, tau_b 0.573, full
coverage, on the 75 directional pairs among 80 held-out custom14 labels (5
ties). Also reports `tool-only`, `turn-only`, `lex-turn-first`, and
`lex-tool-first` for comparison -- the two lexicographic baselines are the
sharpest test of whether a hand-designed ordering rule could replace the
calibrated weights (it can't: lex-tool-first scores 0.427/-0.147, well below U).
Pass `--ci 95` for the 95% interval; `-B` controls resample count (paper uses
10,000 here). The script also reports a paired bootstrap (`U - baseline`, same
resample scores both methods) on accuracy and tau_b -- this is the number
behind the "paired bootstrap never separates U from lex-turn-first" claim in
Section 7.3 (dacc CI [-0.07,+0.04], dtau_b CI [-0.13,+0.08], P(U>base)=28%).

For the stratified turn-differing-vs-pure-tool breakdown (Table
`tab:tool-axis-stats`):
```
python3 stratified_check.py --manifest ../data/bt_manifest_FINAL_test.json \
    --votes ../data/bt_votes_FINAL_test.json --a 0.1235 --b 0.2448
```
Expected: turn-differing n=53 accuracy 90.6%, pure-tool n=22 accuracy 50.0%
(chance level) -- the metric's accuracy collapses on pairs where tool calls
are the only difference, which is exactly what Section 7 argues: the model
can't tell useful verification calls from wasteful ones at the per-call level.

For the functional-form ablation (does the log-linear form actually help over
a single combined weight, or saturating alternatives?):
```
python3 ablation_functional_forms.py
```
(no flags -- hardcoded to `../data/bt_manifest_FINAL_{train,test}.json` /
`../data/bt_votes_FINAL_{train,test}.json`.)

### 3. Inter-annotator agreement (is the tool axis's weak spot a metric failure or a noisy ground truth?)

```
python3 compute_iaa_stats.py --manifest ../data/iaa_c14_tool_axis_FINAL_manifest.json \
    --a1 ../data/iaa_c14_tool_axis_FINAL_annotator1_votes.json \
    --a2 ../data/iaa_c14_tool_axis_FINAL_annotator2_votes.json
```
Expected: n=23 pairs, raw agreement 0.783, Cohen's kappa 0.658 (crosses the
Landis-Koch "substantial" threshold at 0.60) -- two independent humans, both
with full rubric context, still disagree on roughly 1-in-5 verified
pure-tool-axis pairs. Efficiency Utility's agreement with either annotator
sits below the human-human ceiling on this hardest slice, but not implausibly
so given that ceiling. Pass `--ci 95` for the 95% interval.

The script also prints a same-slice three-way comparison (U vs A1, U vs A2,
A1 vs A2, all on the identical n=23 pairs -- the source of Table
`tab:iaa-stats`'s U--A1/U--A2 kappa rows, 0.179 and 0.377): raw agreement is
52.2% (U vs A1) and 60.9% (U vs A2), both well below the 78.3% human-human
ceiling on the same pairs -- a same-slice comparison, not the cross-slice
"U's overall accuracy looks similar to human agreement" read that mixing
different slices could invite.

To confirm the shipped manifest's 23 pairs are all pure-tool (identical
excess-turn count on both sides):
```
python3 check_iaa_pool_validity.py
```
recomputes each pair's `d_eturn` directly from `tasks_rh44_all.json` /
`tasks_custom14_all.json`'s `tool_human`/`turn_human` floors. The same script
also confirms the control set below is all turn-differing.

**Control set (turn+mixed-axis pairs, not pure-tool)**: the same two
annotators also relabeled a set of 17 held-out pairs that differ in turns as
well as tool calls, shipped as `iaa_control_manifest.json` /
`iaa_control_annotator1_votes.json` / `iaa_control_annotator2_votes.json`:
```
python3 compute_iaa_stats.py --manifest ../data/iaa_control_manifest.json \
    --a1 ../data/iaa_control_annotator1_votes.json \
    --a2 ../data/iaa_control_annotator2_votes.json
```
Expected: n=17, raw agreement 100.0%, Cohen's kappa=1.000.

### 4. Listwise validation (harder ranking task: sort 8-trajectory slates into 4 efficiency tiers)

```
python3 score_listwise_eff.py --manifest ../data/listwise_eff_manifest.json \
    --votes ../data/listwise_eff_votes.json
```
Expected: mean per-task Kendall tau_b = 0.488 (median 0.553) across 10 tasks,
90% CI [0.391, 0.582], positive on all 10 tasks. Also reports tool-only,
turn-only, lex-turn-first, and lex-tool-first for the same comparison as #2
(turn-only scores positively on 6 of the 8 tasks it can be evaluated on --
negative on 2 of 8). Pass `--ci 95` for the 95% interval. `--manifest` and
`--votes` both default to these same paths, so the flags above are optional.

### 5. Model leaderboard (SR, pass@3, pass^3, mean U across 24 models)

Precomputed in `data/RESULTS_leaderboard.md`. SR/pass@3/pass^3 are reported
on three scopes -- the 44-task rh44 calibration split, the pooled 58-task
suite, and the 14-task custom14 held-out split; mean U is reported on
custom14 only (U's validity as a calibrated, generalizing preference score
rests on the held-out evidence in #2 and #4 above, so headline U numbers are
scoped there). `RESULTS_leaderboard.md` also has the `SR%(c14)` column shown
in the paper's Table `tab:model-leaderboard` (mean U's own scope, not the
58-task SR reported elsewhere in the same table) -- since $\text{U} =
\text{SR} \times \text{cond-U}$ (U=0 on failures), this lets you derive
`cond_U = U(c14) / (SR%(c14)/100)` per model, which is what Appendix E's
diagnostic decomposition paragraph reports (e.g. gpt55nothink vs opus46 tie on
SR%(c14)=88.1 but differ 11 points in cond_U). `cond_U` is a diagnostic
decomposition only, not an alternative ranking metric -- it's never used for
model comparison elsewhere in this bundle. The e_tool=0/e_turn=0 proportions
cited in the same paragraph (28.8%/34.3% of 496 unique successful
trajectories) come from deduplicating the `A`/`B` sides of
`bt_manifest_FINAL_{train,test}.json` by `(tid, model, trial)` and checking
each side's embedded `etool`/`eturn` fields -- fully self-contained, no
simulation corpus needed.

Success everywhere in this bundle is the finalized 3-judge panel: hy3-preview
+ Seed 2.0 Lite + Grok 4.3, each judge itself a 3-pass internal majority
(base + 2 re-evaluations), 2-of-3 judges required for the panel to pass a
trajectory. This is NOT the original single deepseek-v4-pro evaluator score
still visible in each trajectory's own `reward_info.reward` field within the
embedded trajectory records (that field is kept for reference/debugging, not
as the success signal).

For the per-model 90% (or 95%) bootstrap CI on custom14 mean U used in
Appendix E's leaderboard table (task-level/cluster bootstrap -- resamples
whole tasks with replacement, pooling all trials of each resampled task, to
respect the within-task correlation between a model's three attempts on the
same task):
```
python3 leaderboard_bootstrap_ci.py --sim-dir /path/to/data/simulations \
    --tasks-dir /path/to/data/eff/domains/ridehailing --ci 90
```
Needs the full simulation corpus (tier-2 dependency, see below) -- not
self-contained in this bundle. Precomputed 90%/95% output for all 24 models is
in `data/RESULTS_leaderboard_ci.md`; verified against the full
corpus that every model's CI reproduces Table `tab:model-leaderboard`'s
column exactly (e.g. `gpt55nothink` 90% CI [0.571,0.826] rounds to the
published [0.57,0.83]).

For the SR-vs-U Spearman correlation reported in Section 7.7 ("What Efficiency
Reveals"), self-contained (only needs `RESULTS_leaderboard.md`):
```
python3 sr_vs_u_correlation_ci.py --ci 90
```
Expected: all 24 models rho=+0.785, matching the paper exactly, 90% CI
[+0.52,+0.93]; top 12 by U rho=-0.161, matching the paper exactly, 90%
CI [-0.66,+0.42] -- comfortably straddles zero, supporting the paper's "not
itself significant at this sample size" reading.

For the functional-form ranking-equivalence check in Appendix G (does ranking
by mean U change if `exp(-C)` is swapped for `1/(1+C)`?):
```
python3 functional_form_ranking_ci.py --sim-dir /path/to/data/simulations \
    --tasks-dir /path/to/data/eff/domains/ridehailing --ci 90
```
Needs the full simulation corpus. Expected: Spearman(exp(-C), 1/(1+C)) = 0.989
across all 24 models, 90% task-level bootstrap CI [0.983,0.998].

### 6. Task-difficulty ledger (SR/pass@3/pass^3 per task, pooled over 24 models)

Precomputed in `data/RESULTS_task_difficulty.md`: all 58 tasks, 72 trials each
(24 models x 3 trials), sorted hardest to easiest by pooled success rate.
Needs the full simulation corpus to regenerate from scratch (same tier-2
dependency as #5).

### 7. Tool-axis discordant-pair taxonomy (does U penalize wasted calls, or useful verification work?)

Precomputed in `data/taxonomy_pairs.json`: the 11 held-out pairs (of 22 pure
tool-axis directional test pairs -- see `stratified_check.py` in #2 above)
where U disagrees with the human vote. For each pair, the human-preferred
(higher-tool-count) side is classified as `VERIFY` (majority of the extra
calls are verification/comparison/preference-preservation work), `ERROR`
(majority are malformed/failed calls), or `OTHER`. Result: 11/11
VERIFY-dominant, 0/11 ERROR-dominant, 0/11 OTHER -- none of the discordant
tool-axis cases is ambiguous. The classification counts (`winner_calls`,
`winner_errors`, `winner_verify_calls`, `winner_verify_pct`, `classification`)
come from a manual taxonomy pass (VERIFY-tool whitelist:
`get_trip_duration_estimate`, `address_to_longitude_latitude`,
`nearby_search`, `add_ride_note`, `get_history_orders`, `get_ride_estimate`,
`get_home_address`, `get_company_address`; ERROR = majority of extra calls are
malformed/failed); this file adds the full trajectory content for both sides
of each pair so the classification can be checked against the actual
transcripts.

### 8. Human evaluation of the LLM-as-a-judge success gate (is the panel itself reliable?)

```
python3 compute_llm_judge_human_eval_stats.py \
    --manifest ../data/llm_judge_human_eval_manifest.json \
    --votes ../data/llm_judge_human_eval_votes.json
```
Expected: n=75 trajectories, randomly sampled from the full panel-output pool across all
24 models and both domains, raw agreement 90.7% (90% CI [85.3%, 96.0%]), Cohen's kappa
0.813 (90% CI [0.70, 0.92]) -- "almost perfect" agreement between a blind human relabel
and the 3-judge panel used everywhere else in this bundle. The human annotator sees
exactly what the panel's own prompt sees -- rubric, user scenario, and final system
state -- so the comparison is apples-to-apples; see `RESULTS_llm_judge_human_eval.md`
for the full breakdown. Pass `--ci 95` for the 95% interval (agreement [84.0%, 96.0%],
kappa [0.675, 0.920]).

Precomputed in `data/RESULTS_llm_judge_human_eval.md`: all 7 disagreements (2 false
positive, 5 false negative) come from Tier C/D models -- Tier A/B trajectories are a
combined 30/30 (100% agreement), Tier C/D a combined 39/45 (86.7%). The panel is
essentially fully reliable on the strong-model trajectories that drive the leaderboard's
top ranks, with a characterizable, non-directional soft spot on weaker-tier
trajectories.

## Data file schemas

**`bt_pairs_{train,test}.json`** -- list of pair records:
`{pair_id, split, axis, task_id, d_etool, d_eturn, label, A: {model, trial,
tool, turn, etool, eturn, trajectory}, B: {...same...}}`. `label` is `"A"`,
`"B"`, or `"tie"` -- which side the blind human annotator judged more
efficient. `trajectory` is `{task_id, trial, termination_reason, reward_info:
{reward, nl_rubrics}, messages: [...]}`.

**`listwise_labels.json`** -- list grouped by task:
`{task_id, n_items, items: [{annotation_id, anon_slot, model, trial, tool,
turn, etool, eturn, human_tier, human_tier_meaning, trajectory}]}`.
`human_tier` is `"1"`-`"4"` (1 = most efficient .. 4 = least efficient), the
blind human annotator's bucket assignment for that trajectory within its
task's 8-trajectory slate.

**`iaa_pairs.json`** -- list of pair records:
`{pair_id, task_id, axis, d_etool, d_eturn, annotator1_label,
annotator2_label, agree, source, orig_pair_id, A: {...}, B: {...}}`.
`annotator1_label` is the original BT test-set vote (gold); `annotator2_label`
is a fresh, independent, blind relabel of the same pair.

**`example_trajectories.json`** -- list of 15 records:
`{bucket, task_id, model, trial, why, tool, turn, etool, eturn, trajectory}`.
`bucket` is `successful_high_efficiency` (5), `successful_low_efficiency` (5),
or `failed` (5, spanning 5 distinct failure categories: incomplete multi-leg
trip, wrong order-type/timing, unresolved ambiguity, preference-constraint
violation, suboptimal route ordering -- see each `why` field). This file is a
qualitative demonstration: `tool`/`turn`/`etool`/`eturn` are shown as
observable counts, but the numeric Efficiency Utility score U is deliberately
not included -- "high efficiency" / "low efficiency" / "failed" are meant to
be read off the transcript and rubric checklist, not a computed number. All
15 examples use 15 different models. Every `successful_*` entry's own stored
`reward_info` shows `reward=1.0` with 100% of its rubrics met (verified, not
just panel-passing) -- see the note on evaluators just above.

**`taxonomy_pairs.json`** -- list of 15 records:
`{pair_id, task_id, human_preferred_side, winner_calls, winner_errors,
winner_verify_calls, winner_verify_pct, classification, human_preferred: {...},
U_preferred: {...}}`. `human_preferred` is the side the blind annotator voted
for (also the higher-tool-count side, since these are exactly the pairs where
U ranks the other side higher); `U_preferred` is the side Efficiency Utility
ranked higher. Both carry the same `{model, trial, tool, turn, etool, eturn,
trajectory}` shape as the BT pair files above.
