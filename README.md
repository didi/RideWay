# RideWay: Benchmarking Efficient Task Completion for Tool-Using Language Agents

## 📖 Introduction

**RideWay** is a benchmark for evaluating LLM agents on multi-turn, tool-using conversations in a realistic ride-hailing environment. Each task simulates a conversation between an **agent** (the LLM under test) and a **user simulator** (another LLM), grounded in a database-backed environment with **26 API tools** (POI search, driver matching, order creation/modification, flight lookup, ride reminders, etc.). Trajectories are scored by a **rubric-based trajectory evaluator** (a third LLM).

The benchmark contains **58 Chinese ride-hailing tasks**:

- **44 scenario-derived tasks** (`tasks_rh44_all.json`) — constructed from abstracted ride-hailing interaction patterns, covering immediate/scheduled rides, airport pickups with flight constraints, multi-stop trips, order modifications, and compound temporal constraints. Rubrics are calibrated against human pairwise judgments.
- **14 custom long-horizon tasks** (`tasks_custom14_all.json`) — long-context tasks with personas, shifting intent, chit-chat distractors, and preference constraints.

Tasks require agents to reason over temporal and spatial constraints, proactively clarify ambiguous instructions, track shifting user intent, and use tools efficiently across long conversations.

The Efficiency Utility metric and its human-preference calibration artifacts are packaged under [`efficiency_utility_release/`](efficiency_utility_release/). Start with [`efficiency_utility_release/HOW_TO_REPRODUCE.md`](efficiency_utility_release/HOW_TO_REPRODUCE.md) for the metric definition, Bradley--Terry calibration workflow, pairwise/listwise preference data, validation scripts, and commands for reproducing the reported efficiency results.

## 🚀 Getting Started

### 1. Requirements

- Python **3.10+**
- API access to at least one LLM provider (any OpenAI-compatible endpoint, Azure OpenAI, Google Gemini, or Anthropic Claude)

### 2. Installation

```bash
git clone <this-repo>
cd RideWay

# recommended: use a virtual environment
python3 -m venv venv
source venv/bin/activate

pip install -e .
```

This installs the `eff` CLI. Verify with:

```bash
eff --help
```

### 3. Configure Models

Model endpoints are defined in [`src/eff/models.yaml`](src/eff/models.yaml). The shipped file is a template covering all four supported backends — fill in your own keys:

```yaml
models:
  # OpenAI-compatible endpoint (default backend) — works for DeepSeek, DashScope/Qwen,
  # vLLM, or any /chat/completions server
  - name: deepseek-chat
    model_id: deepseek-chat            # optional: overrides `name` in the request body
    base_url: https://api.deepseek.com/chat/completions
    headers:
      Authorization: "Bearer YOUR_API_KEY"
      Content-Type: "application/json"

  # Azure OpenAI — `name` must match your Azure deployment name
  - name: gpt-5.2
    backend: azure
    azure_endpoint: "https://YOUR_RESOURCE.openai.azure.com"
    api_key: "YOUR_API_KEY"
    api_version: "2025-01-01-preview"

  # Google Gemini (generateContent API)
  - name: gemini-2.5-flash
    backend: gemini
    base_url: https://generativelanguage.googleapis.com/v1beta/models
    headers:
      x-goog-api-key: "YOUR_API_KEY"

  # Anthropic Claude (Messages API)
  - name: claude-sonnet-4-5
    backend: claude
    base_url: https://api.anthropic.com/v1/messages
    headers:
      x-api-key: "YOUR_API_KEY"
      anthropic-version: "2023-06-01"
```

Three ways to provide the config (highest priority last):

1. Edit `src/eff/models.yaml` directly.
2. Create `src/eff/models.local.yaml` — it is **gitignored** and loaded automatically when present. Recommended for real keys.
3. Set the `EFF_MODEL_CONFIG_PATH` environment variable to any config file path.

On startup the CLI prints `Available models: [...]` so you can confirm what was loaded. Optional per-model settings: `temperature`, `max_tokens`, `enable_think: true` (reasoning mode, mechanism auto-detected per model family; can also be switched on per run with `--enable-think`).

### 4. Smoke Test (one task)

Run a single task first to verify everything is wired up:

```bash
eff run \
  --agent-llm deepseek-chat \
  --user-llm deepseek-chat \
  --evaluator-llm deepseek-chat \
  --task-ids custom_015_shanghai_mcdonalds_rain_nosmoking \
  --save-to smoke_test
```

You should see the agent↔user conversation being simulated, followed by the evaluator scoring the trajectory. The result lands in `data/simulations/smoke_test.json`.

### 5. Full Benchmark Run

```bash
eff run \
  --agent-llm <model-under-test> \
  --user-llm <user-sim-model> \
  --evaluator-llm <evaluator-model> \
  --num-trials 3 \
  --max-concurrency 6 \
  --save-to my_run
```

Notes:

- **All three roles need a model.** The agent is what you are benchmarking; the user simulator and evaluator should be fixed to the same models across all agent runs for comparability.
- **Resume:** if `data/simulations/my_run.json` already exists, completed task ids are skipped automatically — re-running the same command continues an interrupted run.
- **Multiple trials** (`--num-trials 3`) are recommended: single-trial scores are noisy on interactive tasks.
- Other options: `--max-steps` (default 300), `--num-tasks N` (first N tasks only), `--enable-think`, `--evaluation-type` (default `trajectory`, see `eff run --help` for the alternatives).

### What You Need to Run

A run needs exactly three pieces of information, all pointing at model names defined in your `models.yaml`:

| Role | Flag | What it does | Required? |
|---|---|---|---|
| Agent | `--agent-llm` | The model being benchmarked; receives the 26 tools and drives the conversation | Yes (defaults to `deepseek-chat`) |
| User simulator | `--user-llm` | Plays the passenger following the task's persona and hidden instructions | Yes (defaults to `deepseek-chat`) |
| Evaluator | `--evaluator-llm` | Scores the finished trajectory against the task rubrics | Yes (defaults to `deepseek-chat`) |

The three roles are independent — any mix of providers works (e.g. Claude agent + DeepSeek user sim + GPT evaluator). For fair comparison across agents, **fix the user simulator and evaluator models** and only vary `--agent-llm`.

### Configuration Reference (`eff run`)

| Flag | Default | Description |
|---|---|---|
| `--agent-llm` / `--user-llm` / `--evaluator-llm` | `deepseek-chat` | Model name (from `models.yaml`) for each role |
| `--agent-llm-args` / `--user-llm-args` / `--evaluator-llm-args` | — | JSON dict of extra per-call LLM parameters, e.g. `'{"temperature": 0.5}'` |
| `--task-ids id1 id2 ...` | all 58 | Run only the listed task ids |
| `--num-tasks N` | all | Run only the first N tasks |
| `--task-file path.json` | built-in 44+14 | Point at a custom task JSON file instead of the shipped task sets |
| `--num-trials N` | 1 | Trials per task (3 recommended; interactive scores are noisy) |
| `--max-concurrency N` | 1 | Parallel simulations |
| `--max-steps N` | 300 | Hard cap on conversation steps per simulation |
| `--max-errors N` | 10 | Abort a simulation after N tool-call errors |
| `--save-to name` | auto timestamped | Output file `data/simulations/<name>.json`; existing file → resume |
| `--re-evaluate-file path` | — | Re-score an existing simulation file instead of simulating (only needs `--evaluator-llm`) |
| `--re-run` | off | With `--re-evaluate-file` + `--task-ids`: re-simulate those tasks and merge |
| `--evaluation-type` | `trajectory` | `trajectory` (sliding-window + rubrics), `trajectory_full_traj_rubric`, `trajectory_sliding_wo_rubric`, `trajectory_full_traj_wo_rubric` |
| `--enable-think` | off | Turn on the agent model's reasoning/thinking mode (mechanism auto-detected per model family) |
| `--dynamic-events 0/1` | 0 | Inject mid-conversation dynamic events (e.g. system notifications) for tasks that define them |
| `--language` | `chinese` | Prompt language (`chinese`/`english`); task data itself is Chinese |
| `--seed` | 300 | Random seed |
| `--csv-output path` | — | Append run summary rows to a CSV |
| `--log-level` | DEBUG | Logger verbosity |

Agent/user implementation variants: `--agent llm_agent` (default) or `llm_solo_agent`; `--user user_simulator` (default) or `dummy_user`.

### 6. Inspect Results

```bash
# Terminal summary of a run
eff view --file data/simulations/my_run.json

# Metrics: avg_reward, pass^k, pass@k, per-role token stats
python -m eff.metrics.agent_metrics --results data/simulations/my_run.json

# Re-evaluate an existing run with a different evaluator (no re-simulation)
eff run --re-evaluate-file data/simulations/my_run.json \
  --evaluator-llm <another-model> --save-to my_run_reeval

# Self-contained HTML visualization; pass multiple files for side-by-side model comparison
python3 visualize_simulation.py data/simulations/my_run.json view_simulations/my_run.html
python3 visualize_simulation.py data/simulations/a.json data/simulations/b.json view_simulations/a_vs_b.html
```

## 🏗️ Architecture

```
eff run (CLI)
  └── run_domain() [run.py]
        └── run_tasks() — async, concurrent
              └── run_task() — single simulation loop
                    ├── Orchestrator — drives agent↔user turns
                    ├── LLMAgent — calls agent LLM with tools
                    ├── UserSimulator — calls user LLM
                    └── evaluate_simulation() → rubric-based trajectory evaluator
```

Repository layout:

```
src/eff/                  core package (CLI, orchestrator, agent, user sim, evaluator, metrics)
src/eff/domains/ridehailing/   domain tools (26 APIs) + data models
src/eff/prompts/          system prompts and evaluator templates (zh/en)
data/eff/domains/ridehailing/  the 44 + 14 task definitions
visualize_simulation.py   standalone HTML report generator
```

Tasks are written in Chinese and the default `--language` is `chinese` (prompts also ship with English variants; task data is Chinese-only).

See [`PROJECT.md`](PROJECT.md) for detailed technical documentation (end-to-end flow, task JSON format, evaluation modes, adding new domains).

## 🔒 Data and Privacy

RideWay is evaluated entirely in a simulated environment. The released task records, database states, and trajectories are benchmark fixtures rather than production ride records. Names, phone numbers, vehicle plates, driver and order identifiers, home/work addresses, preferences, and trip histories appearing in the data are synthetic or fictional and are not intended to identify or describe real individuals. Public place names and POIs may refer to real-world locations solely to support realistic spatial reasoning.

The repository does not intentionally include raw production logs, user account data, real device-location records, credentials, or direct personal identifiers. Model-generated trajectories may nevertheless contain inaccurate, outdated, or incidental references to real entities and must not be used as factual travel, safety, or dispatch guidance. See [`DATA_CARD.md`](DATA_CARD.md) for provenance, intended use, limitations, and reporting guidance.

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.

## 🤗 Acknowledgement

The overall engineering framework of this project references [tau-bench](https://github.com/sierra-research/tau-bench) / [tau2-bench](https://github.com/sierra-research/tau2-bench) and [VitaBench](https://github.com/meituan-longcat/vitabench) ([paper](https://arxiv.org/abs/2509.26490)): the agent / user-simulator / evaluator simulation loop design follows their architecture. The ride-hailing domain, environment toolkit, task sets, efficiency metrics, and evaluation calibration are new in this work. We greatly appreciate their contributions to the agent community.
