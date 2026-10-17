# dash interface (`forge.dash`)

Owner: dash. A read-only web page over the files the other components write. It never writes into their dirs.

## Run

On the rig: `forge/dash/deploy.sh` from the laptop. It tars `forge/dash` into `~/forge`, then (re)starts
`~/venvs/vllm-bolt/bin/python -m forge.dash --root ~/forge --port 8790`. The pid and log go in `~/forge/state/dash/`.

- `http://100.68.107.11:8790/`: the real files under `~/forge`.
- `http://100.68.107.11:8790/?src=fixture`: fake demo data, rewritten at startup under `~/forge/dash_fixture`,
  kept live by a small appender, and labelled `[fixture]` everywhere.
- `/api/snapshot[?src=fixture]`: the whole page state as JSON. `/api/stream` is SSE (`event: snapshot`, pushed
  on change, with a ping every 15 s). `/api/health`.
- Keys 1-8 zoom a panel; Esc goes back. `?nostream=1` renders once (for headless screenshots).

## What it reads (relative to FORGE_ROOT, and also under `state/`)

| panel | files | fields used |
|---|---|---|
| leaderboard | `forge/recipes/*.yaml` (catalog: `id`, `variants[].name/metric`, `hardware.fits`, `model`), `missions/*/row.json`, `mission_end.row`, `results/rows/*.json`, `results/boxes/*/rows/*.json`, `results/leaderboard.json[l]`, `fleet/queue.json` | `recipe_id`, `gpu_class`, `baseline`, `best`, `delta_pct`, `change_summary`, `done.status` (win → verified, no_win → no-gain), `gate_verdicts[].gate_id/verdict/pairs` |
| live missions | `missions/<id>/spec.json`, `mission.json`, `events.jsonl` | spec `recipe.id`, `machine.name/gpu_class`, `time_box_min`, `budget_usd`; events `t`, `kind`, `agent`/`worker`/`by`, `role`, `tool`, `lane`, `cost_usd` |
| gates | `gate/ledger.jsonl`, `results/boxes/<machine>/gate/ledger.jsonl`, the mission events `gate_submitted` and `gate_result` | latest row per `gate_id`: `status`, `verdict`, `delta_pct`, `lossless_pass`, `pairs`, `kind`, `machine`, `mission_id`, `recipe_id`, `detail` |
| decisions | events `decision` (`decision`, `by`, `plan`, `evidence`), `ask_human` | |
| alerts | events `detector` (`signal`, `detail`, `infra`), `trigger`, `tool_bug`, `error`; `fleet/alerts.jsonl` (`severity`, `machine`, `detector`, `message`) | |
| Version N+1 | `n1/log.jsonl` | one version per `version`/`new_hash`: `status`/`verdict` (promote/reject), `parent`/`old_hash`, `per_recipe`/`unseen` (`before`/`old`, `after`/`new` as delta %), `paired_delta`, `reason` |
| Shared Brain | `brain/facts.jsonl` | `fact_id` (later lines update earlier ones), `claim`, `status`, `recipe_id`, `lane`, `mission_id`, `t` |
| spend | `llm_call.cost_usd` by `role`; `fleet/machines.json`, `fleet/machines/*.json`, `fleet/spend.jsonl` | `usd_per_hour`, `started_at`, `returned_at`/`status`, `spent_usd` |

Missing files give empty panels. Half-written JSON is retried on the next poll. JSONL is tailed incrementally:
only appended bytes are parsed, and a file that shrinks is re-read. A log over 8 MB is read from its last 8 MB only.
