# agents interface (`forge.agents`, `forge.tools`)

Owner: agents. Everything here is importable from the `forge` package; nothing reads outside it.

## Run a mission

```python
from forge.agents.mission import run_mission
row = run_mission(spec)            # blocking; one thread per team member inside
```
CLI (on the rig): `~/venvs/vllm-bolt/bin/python -m forge.agents run --spec mission.json [--profile mock|local|crusoe]`

`spec` (JSON-able dict):

| key | meaning |
|---|---|
| `mission_id` | unique id; state goes to `<FORGE_STATE>/missions/<id>/` |
| `recipe` | the recipe dict as loaded from `forge/recipes/*.yaml`: needs `id`, `model`, `serve.args` (list or flag dict), optional `serve.env`, `serve.source`, `serve.served_model_name`, `workload` (`concurrency`, `max_tokens`, `metric`, `bench_prompts`, optional `chat`), optional `python` |
| `machine` | dict: `name`, `forge_root`, `python`, `gpu_class`, plus whatever `forge.fleet.executor.executor_for(machine)` needs |
| `python` | venv python on the target (overrides recipe/machine) |
| `bench_prompts` | path on the target of the visible bench prompts JSONL (`{"prompt": ...}` or `{"messages": [...]}` per line); default `recipe.workload.bench_prompts` |
| `site_packages` | optional; else asked from `python` |
| `time_box_min` (75), `budget_usd` (15) | hard limits |
| `team` | `{"workers": 3, "lead": {"tier": "ultra", "context_cap": 48000}, "worker": {"tier": "super", "context_cap": 64000}, ...}` |
| `goal` | free text; "at least N%" sets the min gain in the done criteria |
| `require_confirm` (true), `min_gain_pct` | done criteria inputs |
| `playbook` | playbook hash (Version N+1) or null for base |
| `profile` | `crusoe` (default) / `local` / `mock` |
| `gate_grace_min` (0) | wait this long after the time box for open gates before writing the row |
| `bench_vram_gb` (90) | VRAM asked for each quick-bench job |
| `brain_path` | shared fact store (default `<FORGE_STATE>/brain/facts.jsonl`) |

Returned row (also written to `missions/<id>/row.json`):
`{mission_id, recipe_id, machine, gpu_class, metric, baseline, best, delta_pct, gate_verdicts[], change_summary,
evidence_ids[], done{status: win|no_win|pending, met, criteria_sha256, open_gates}, quick_bench_best, n_benches,
n_gate_requests, cost_usd, llm{calls, cost_usd, prompt_tokens, cached_tokens, completion_tokens, cache_hit, by_role},
playbook, ended, errors[], detector{signal: count}, minutes}`.
`baseline` is the quick-bench reference value; `best`/`delta_pct` come only from the gate ledger (0 if no win).
`forge.agents.mission.audit_mission(mission_id, gate)` re-audits later when gates were still open.

Spec from the catalog (on the rig): `python -m forge.agents mission --recipe bolt-74b-lilicorr.c1 --machine rig
--minutes 45 --budget 6 --workers 2 --id <mid> --out state/specs/<mid>.json`; or detached in one go:
`bash scripts/agents/launch_mission.sh <recipe> <mission_id> [minutes] [budget] [workers]` (log in
`state/<mid>.out`). Stop a mission with SIGTERM/SIGINT: it ends cleanly and still writes `row.json`.
Other commands: `probe` (Crusoe ids, tool calls, caching), `selftest` (mock mission, no GPU, no network),
`answer <mission> <question_id> <text>` (human inbox).

Models (Crusoe serves no Nemotron Ultra): all roles default to `nvidia/NVIDIA-Nemotron-3.5-Super-VL` with
automatic fallback to `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B` (see `forge/agents/config.py`; overridable with
`FORGE_ULTRA_MODEL` / `FORGE_SUPER_MODEL`). Prices per model id from Crusoe's `/v1/models`.

## Files per mission (for dash): `<FORGE_STATE>/missions/<id>/`

All append-only JSONL, one object per line, each with `t` (unix seconds).
- `events.jsonl`: `{t, mission_id, kind, ...}`. Kinds: `mission_start` (team, models, playbook), `done_locked`
  (sha256, criteria), `llm_call` (agent, role, prompt_tokens, cached_tokens, completion_tokens, cost_usd, text),
  `tool_call` (agent, tool, args, ok, cls, error, result), `tool_bug`, `detector` (agent, signal, detail, infra),
  `trigger` (trigger_id, trigger_kind, summary, infra), `decision` (by, decision, trigger_ids, evidence, plan),
  `worker_task` (worker, lane, task), `worker_report`, `gate_request`, `gate_review`, `gate_submitted`,
  `gate_result` (gate_id, verdict, delta_pct), `compaction`, `ask_human`, `human_answer`, `error`,
  `mission_ending` (reason), `mission_end` (row).
- `usage.jsonl`: one row per LLM call: `{mission_id, role, model, tier, prompt_tokens, cached_tokens,
  completion_tokens, cost_usd, latency_s}`.
- `decisions.jsonl`, `notes.jsonl`, `benches.jsonl` (quick benches), `gate_requests.jsonl`.
- `inbox.jsonl`: questions `{question_id, by, question}`. **Humans answer by appending**
  `{"question_id": "...", "answer": "...", "to": "<agent, optional>"}` to `answers.jsonl` in the same dir.
- `done.lock.json`: `{criteria, sha256}`; `spec.json`; `row.json`.

## Executor (fleet provides `forge.fleet.executor.executor_for(machine) -> Executor`)

`forge/tools/executor.py` has the Protocol and `LocalExecutor`:
`run(cmd, *, cwd, env, timeout_s, kind='work'|'bench', vram_gb, gpus, mission_id, label) -> ExecResult(rc, stdout,
stderr, elapsed_s, infra_error)`, `read_text`, `write_text`, `exists`, `list_files(root)`, `listdir(path)`,
`remove(path)`, `workspace(mission_id) -> abs path`.
`kind='bench'` (or `gpus>0`) must go through core's scheduler for a GPU grant (queue while pending), with
`CUDA_VISIBLE_DEVICES` set for the job. `infra_error` means it could not run at all (not the command's fault).

## Quick bench (`forge.tools.benchjob`, runs on the target inside the grant)

`python -m forge.tools.benchjob --spec spec.json` prints `BENCH_RESULT {...}`. Starts `vllm serve` from the recipe
args + candidate (config delta / env / bundle), runs the visible prompts, reports `decode_tok_s` / `output_tok_s`.
Reference cached per GPU uuid + recipe + prompt-set hash in `<forge_root>/state/refcache/`.

## Bundles (`forge.tools.bundle`)

Dir with `manifest.json` `{config_delta, env, overlays: {"vllm/...py": "overlays/vllm/...py"}, kernels: [...],
drafter, recipe_id, note}` + files. Hash = `hash_dir(dir)` (sha256 over relative paths + bytes, skips `.hook/`).
`serve_command(python, model, serve_args, bundle_dir, port, ...) -> (argv, env)` builds the server command the same
way for bench and (if core wants it) gate; overlays apply through a generated `sitecustomize` import hook on
PYTHONPATH.

## Gate (core's `forge.gate`; adapter `forge.tools.gateclient.CoreGate`)

Called only after the Verifier approves:
`forge.gate.request(recipe_id=, bundle_dir=, mission_id=, machine=, requester=, claim=) -> {gate_id, ...}`,
`forge.gate.status(gate_id) -> row`, `forge.gate.ledger(machine=) -> [rows]`. Rows are normalized to
`{gate_id, mission_id, recipe_id, bundle_hash, kind: candidate|confirm|reference, status, verdict: win|loss|lossy|
invalid|error, delta_pct, evidence_id, confirmed}`. The done auditor reads only these rows.

## Shared Brain

`<FORGE_STATE>/brain/facts.jsonl`: `{fact_id, claim, lane, status: measured|refuted|blocked|open, evidence,
recipe_id, gpu_class, mission_id, by, t}`. Written only by agents.

## Version N+1 (`forge.agents.mechanic`)

See the module docstring; fleet schedules waves and passes `playbook` in each spec.
- `propose(wave_mission_ids, base_hash) -> new_hash | None` (Mechanic, Ultra, from trace stats)
- `compare(old_hash, new_hash, rows) -> {verdict: promote|reject, per_recipe, reason}`; logs to
  `<FORGE_STATE>/n1/log.jsonl`
- `current() -> hash` of the promoted playbook.
