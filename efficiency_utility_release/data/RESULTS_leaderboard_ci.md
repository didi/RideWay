# Model leaderboard per-model bootstrap CI on custom14 mean U

Task-level (cluster) bootstrap: resamples the 14 custom14 tasks with replacement
(not individual trials), pooling all trials of each resampled task on every draw --
this respects the within-task correlation between a model's three attempts on the
same task. B=5000 resamples. This is the script behind the 90% CI column in Table
`tab:model-leaderboard` in the paper; the 95% column is additionally reported here.

Reproduce: `python3 leaderboard_bootstrap_ci.py --sim-dir /path/to/data/simulations
--tasks-dir /path/to/data/eff/domains/ridehailing --ci 90` (or `--ci 95`).

| Rank | Model | U(c14) | 90% CI | 95% CI |
|---|---|---|---|---|
| 1 | gpt55nothink | 0.705 | [0.571, 0.826] | [0.540, 0.845] |
| 2 | qw37max | 0.661 | [0.558, 0.765] | [0.541, 0.781] |
| 3 | qw3627b | 0.638 | [0.479, 0.790] | [0.450, 0.812] |
| 4 | sonnet5 | 0.637 | [0.492, 0.765] | [0.464, 0.785] |
| 5 | gemma431b | 0.628 | [0.480, 0.768] | [0.449, 0.794] |
| 6 | qw37plus | 0.624 | [0.497, 0.746] | [0.472, 0.768] |
| 7 | glm52 | 0.620 | [0.498, 0.735] | [0.477, 0.754] |
| 8 | qw36p | 0.612 | [0.495, 0.728] | [0.470, 0.748] |
| 9 | opus46 | 0.606 | [0.498, 0.707] | [0.476, 0.724] |
| 10 | mimo | 0.590 | [0.443, 0.727] | [0.417, 0.749] |
| 11 | kimik26 | 0.563 | [0.407, 0.711] | [0.378, 0.736] |
| 12 | dsv4pro | 0.543 | [0.425, 0.657] | [0.404, 0.676] |
| 13 | gemma426b | 0.493 | [0.352, 0.636] | [0.322, 0.663] |
| 14 | gemini35flash | 0.443 | [0.332, 0.541] | [0.311, 0.557] |
| 15 | qw359b | 0.431 | [0.309, 0.547] | [0.285, 0.570] |
| 16 | qw35ba3b | 0.323 | [0.197, 0.458] | [0.175, 0.482] |
| 17 | ministral8b | 0.311 | [0.174, 0.459] | [0.150, 0.491] |
| 18 | mistralsmall | 0.291 | [0.156, 0.441] | [0.134, 0.470] |
| 19 | ministral14b | 0.270 | [0.132, 0.426] | [0.108, 0.456] |
| 20 | ministral3b | 0.229 | [0.116, 0.350] | [0.097, 0.377] |
| 21 | granite418b | 0.090 | [0.000, 0.223] | [0.000, 0.246] |
| 22 | llama3370b | 0.051 | [0.000, 0.135] | [0.000, 0.146] |
| 23 | llama318b | 0.003 | [0.000, 0.009] | [0.000, 0.009] |
| 24 | nemotron3nano30b | 0.000 | [0.000, 0.000] | [0.000, 0.000] |
