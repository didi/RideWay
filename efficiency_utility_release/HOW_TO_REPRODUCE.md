# RideWay Efficiency Utility — release bundle

This bundle packages the Efficiency Utility (U) metric's calibration data,
Bradley-Terry fit/analysis scripts, and headline results.

```
dd_release/
  data/
    bt_pairs_train.json          320 labeled BT calibration pairs, full trajectories embedded
    bt_pairs_test.json           80 labeled BT held-out pairs, full trajectories embedded
    listwise_labels.json         80 listwise-tier-labeled trajectories (10 tasks), full trajectories embedded
    iaa_pairs.json                36 dual-annotated tool-axis IAA pairs, full trajectories embedded
    example_trajectories.json     15 hand-picked example trajectories (5 high-eff / 5 low-eff / 5 failed), qualitative -- no U score
    taxonomy_pairs.json           15 tool-axis discordant pairs (U vs. human disagreement), full trajectories + error-taxonomy classification
    RESULTS_leaderboard.md        24-model leaderboard: SR/pass@3/pass^3 (rh44v2, pooled-58, custom14) + mean U (custom14)
    RESULTS_task_difficulty.md    per-task SR/pass@3/pass^3 across all 58 tasks, pooled over 24 models (72 trials/task)
    bt_manifest_FINAL_{train,test}.json / bt_votes_FINAL_{train,test}.json   raw manifest+vote format (no embedded trajectories) -- feed these directly into fit_bt.py / bt_analyze.py / ablation_*.py
    iaa_c14_tool_axis_FINAL_*.json                                          raw IAA manifest+votes format -- feed into compute_iaa_stats.py
    listwise_eff_manifest.json / listwise_eff_votes.json                    raw listwise format -- feed into score_listwise_eff.py
  scripts/
    fit_bt.py                    core Bradley-Terry fit (the number everything else derives from)
    bt_analyze.py                fuller version: point fit + bootstrap CI + leave-one-task-out concordance
    ablation_baselines_ci.py     held-out test table: U vs tool-only/turn-only/lex-turn-first/lex-tool-first
    ablation_functional_forms.py functional-form ablation (linear vs single-weight vs log/sqrt saturating)
    compute_iaa_stats.py         inter-annotator agreement statistics (kappa, Gwet's AC1, Krippendorff's alpha)
    score_listwise_eff.py        listwise ranking validation (tie-aware Kendall tau_b)
    build_model_leaderboard.py       regenerates RESULTS_leaderboard.md  [needs full sim corpus, see below]
    build_task_difficulty_table.py   regenerates RESULTS_task_difficulty.md  [needs full sim corpus, see below]
    bt_manifest_FINAL_{train,test}.json / bt_votes_FINAL_{train,test}.json  local copies, needed because ablation_functional_forms.py hardcodes these bare filenames
```

## What's self-contained vs. what needs the full repo

Two different tiers of reproducibility, by design:

- **Self-contained** (`fit_bt.py`, `bt_analyze.py`, `ablation_baselines_ci.py`,
  `ablation_functional_forms.py`, `compute_iaa_stats.py`, `score_listwise_eff.py`):
  these only need the small manifest/vote JSON files already in `data/` (or
  `scripts/`, for the one script that hardcodes bare filenames). Run them
  straight from `scripts/` as shown below -- no external data needed.
- **Needs the full ddbench simulation corpus** (`build_model_leaderboard.py`,
  `build_task_difficulty_table.py`): these read every model's raw trajectory
  dump (`data/simulations/<domain>_<model>`, one file per model, hundreds of
  MB total) plus the per-judge re-evaluation files. That corpus is not part
  of this bundle -- run these from within a full ddbench checkout, in
  `bt_calibration/`, pointing `--sim-dir`/`--tasks-dir` (for
  `build_model_leaderboard.py`) at `data/simulations` /
  `data/vita/domains/ridehailing`, or from `bt_calibration/` directly for
  `build_task_difficulty_table.py` (its paths are hardcoded relative to that
  location). `RESULTS_leaderboard.md` and `RESULTS_task_difficulty.md` in
  `data/` are these scripts' precomputed output, included so the numbers are
  inspectable without needing the corpus.

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
Expected: `a=0.1078 -> lambda_tool=0.8978`, `b=0.4588 -> lambda_turn=0.6320`,
`a/b=0.24` (one excess turn costs ~4.3x one excess tool call). This is the
number `U(tau) = success(tau) * lambda_tool^etool(tau) * lambda_turn^eturn(tau)`
is built from -- every other script below either consumes it or re-derives it.

For the fuller version (bootstrap CI + leave-one-task-out concordance vs
naive baselines):
```
python3 bt_analyze.py ../data/bt_manifest_FINAL_train.json ../data/bt_votes_FINAL_train.json
```

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

Note: the script's built-in fit constant is from an earlier calibration pass
and differs slightly from the final fit above; this does not change any
reported result.

For the functional-form ablation (does the log-linear form actually help over
a single combined weight, or saturating alternatives?):
```
python3 ablation_functional_forms.py
```
(no flags -- hardcoded to the same FINAL train/test files, which is why copies
of `bt_manifest_FINAL_{train,test}.json` / `bt_votes_FINAL_{train,test}.json`
also live directly in `scripts/`, not just `data/`.)

### 3. Inter-annotator agreement (is the tool axis's weak spot a metric failure or a noisy ground truth?)

```
python3 compute_iaa_stats.py --manifest ../data/iaa_c14_tool_axis_FINAL_manifest.json \
    --a1 ../data/iaa_c14_tool_axis_FINAL_annotator1_votes.json \
    --a2 ../data/iaa_c14_tool_axis_FINAL_annotator2_votes.json
```
Expected: raw agreement 0.778, Cohen's kappa 0.639 (crosses the Landis-Koch
"substantial" threshold at 0.60) -- two independent humans, both with full
rubric context, still disagree on 1-in-5 pure-tool-axis pairs. U's 55.6-63.9%
accuracy on these same pairs sits below the human ceiling, but not
implausibly so given that ceiling.

### 4. Listwise validation (harder ranking task: sort 8-trajectory slates into 4 efficiency tiers)

```
python3 score_listwise_eff.py --manifest ../data/listwise_eff_manifest.json \
    --votes ../data/listwise_eff_votes.json --lambda-tool 0.8978 --lambda-turn 0.6320
```
Expected: mean per-task Kendall tau_b = 0.534 (median 0.566) across 10 tasks,
90% CI [0.409, 0.649], positive on all 10 tasks. Also reports tool-only,
turn-only, lex-turn-first, and lex-tool-first for the same comparison as #2.

Note: pass `--lambda-tool`/`--lambda-turn` explicitly -- the script's built-in
defaults are from an earlier calibration pass.

### 5. Model leaderboard (SR, pass@3, pass^3, mean U across 24 models)

Precomputed in `data/RESULTS_leaderboard.md`. SR/pass@3/pass^3 are reported
on three scopes -- the 44-task rh44v2 calibration split, the pooled 58-task
suite, and the 14-task custom14 held-out split; mean U is reported on
custom14 only (U's validity as a calibrated, generalizing preference score
rests on the held-out evidence in #2 and #4 above, so headline U numbers are
scoped there).

Success everywhere in this bundle is the finalized 3-judge panel: hy3-preview
+ Seed 2.0 Lite + Grok 4.3, each judge itself a 3-pass internal majority
(base + 2 re-evaluations), 2-of-3 judges required for the panel to pass a
trajectory. This is NOT the original single deepseek-v4-pro evaluator score
still visible in each trajectory's own `reward_info.reward` field within the
embedded trajectory records (that field is kept for reference/debugging, not
as the success signal).

### 6. Task-difficulty ledger (SR/pass@3/pass^3 per task, pooled over 24 models)

Precomputed in `data/RESULTS_task_difficulty.md`: all 58 tasks, 72 trials each
(24 models x 3 trials), sorted hardest to easiest by pooled success rate.
Needs the full simulation corpus to regenerate from scratch (same tier-2
dependency as #5).

### 7. Tool-axis discordant-pair taxonomy (does U penalize wasted calls, or useful verification work?)

Precomputed in `data/taxonomy_pairs.json`: the 15 held-out pairs (of 33 pure
tool-axis directional test pairs) where U disagrees with the human vote.
For each pair, the human-preferred (higher-tool-count) side is classified as
`VERIFY` (majority of the extra calls are verification/comparison/
preference-preservation work), `ERROR` (majority are malformed/failed calls),
or `OTHER`. Result: 14/15 VERIFY-dominant, 0/15 ERROR-dominant, 1/15 OTHER.
The classification counts (`winner_calls`, `winner_errors`,
`winner_verify_calls`, `winner_verify_pct`, `classification`) come from a
manual taxonomy pass (VERIFY-tool whitelist: `get_trip_duration_estimate`,
`address_to_longitude_latitude`, `nearby_search`, `add_ride_note`,
`get_history_orders`, `get_ride_estimate`, `get_home_address`,
`get_company_address`; ERROR = majority of extra calls are
malformed/failed); this file adds the full trajectory content for both sides
of each pair so the classification can be checked against the actual
transcripts.

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
