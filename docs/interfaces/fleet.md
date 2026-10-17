# fleet interface (`forge.fleet`, `forge.recipes`, `bin/forge`)

Owner: fleet. Everything is importable from the `forge` package.

## Recipes (`forge.recipes`)

One YAML per base recipe in `forge/recipes/*.yaml`: NVIDIA's exact published `vllm serve` args (`serve.args`,
model positional and host/port removed), `serve.source` (the page), `serve.quoted` (the text as published),
`vllm.env`, `vllm.run_image` (the image the fleet runs), `chat` (reasoning off), `hardware.fits`
(`rtxpro6000_95g`, `rtx5090_32g`), optional `hardware.adapt.<gpu_class>` (a documented minimal change needed for
the published command to load on that GPU class; it applies to the baseline and every candidate alike) and
`variants` (concurrency 1 / 8 / 16; a variant exists only when the published `--max-num-seqs` allows it).

```python
from forge.recipes import load_all, get, fits, load_prompts
get("nano-9b-v2:c8", gpu_class="rtx5090_32g") -> dict
```
Expanded recipe keys: `id` (`<base>:<variant>`), `base_id`, `variant`, `model`, `drafter`, `serve {args,
published_args, env, adapted, served_model_name?, source, quoted}`, `workload {concurrency, max_tokens: 512,
temperature: 0, metric: decode_tok_s|output_tok_s, bench_prompts, chat}`, `hardware`, `gpu_class`,
`recipe_sha256`.

Visible bench prompts: `forge/recipes/prompts/public_v1.jsonl` (24 lines, `{"id","kind","messages"}`), relative to
FORGE_ROOT. The gate's hidden pool is disjoint and lives elsewhere.

## Machines (`forge.fleet.machines`)

`all_machines() -> {name: machine}`, `get_machine(name)`, `save_machine(m)`, `python_for(m, recipe)`.
Static: `forge/fleet/machines.yaml` (the rig). Rented boxes: `<FORGE_STATE>/fleet/machines/<name>.json`
(`FORGE_STATE` defaults to `$FORGE_ROOT/state`), synced to the rig on every change.

Machine dict: `name`, `hostname`, `building` (landlord), `ssh {host, user, port}`, `forge_root`, `gpu_class`,
`gpus {timing, work}`, `python`, `python_for {base_id: python}`, `recipes` (base ids it can serve),
`env` (exported for every command: HF_HOME, LANDLORD_URL, ...), `results_dir`; boxes also `kind: vast`,
`rental_id`, `instance`, `usd_per_hour`, `image`.

## Executor (`forge.fleet.executor`)

`executor_for(machine | name) -> FleetExecutor`. Implements the `forge.tools.executor.Executor` protocol:

- `run(cmd, *, cwd, env, timeout_s=600, kind='work', vram_gb=None, gpus=0, mission_id='', label='') -> ExecResult`
  - `cmd` is a str (run by bash) or argv list. Local when the machine's `hostname` is this host, else ssh.
  - `kind in ('bench','serve','gate')` or `gpus>0`: wrapped on the target in
    `python -m forge.fleet.gpujob`, which takes a grant from that machine's `forge.sched` daemon (queues while
    pending, never refused for capacity), runs with the grant's env (`CUDA_VISIBLE_DEVICES`,
    `LANDLORD_LEASE_ID`, `FORGE_PORT_LO/HI`, `FORGE_GRANT_ID`), releases on exit. The ssh call allows
    `timeout_s` + 1 h of queueing.
  - `infra_error` is set when it could not run at all: ssh down (rc 255), scheduler down / impossible grant
    (gpujob exits 199 with `FORGE_INFRA: <reason>`). Escalate those; they are not the agent's fault.
- `read_text`, `read_bytes`, `write_text`, `write_bytes(path, data, mode=None)` (atomic), `exists`,
  `list_files(root)` (relative, recursive), `listdir`, `remove`, `workspace(mission_id)`
  (`$FORGE_ROOT/state/missions/<id>/ws`, created).
- Jobs (detached, survive the ssh session, own process group):
  `start_job(job_id, cmd, cwd=, env=)`, `job_status(job_id) -> {state: running|exited|lost, rc, pid}`,
  `read_log(job_id, offset) -> (text, new_offset)`, `stream_log(job_id, follow=True)` (generator),
  `tail(job_id, n)`, `stop_job(job_id)` (TERM then KILL its group; only jobs it started).
  Files: `$FORGE_ROOT/state/jobs/<job_id>/{run.sh,log,pid,rc}`.
- Transfer: `put_tree(local_dir, remote_dir)`, `get_tree(remote_path, local_dir)` (tar.gz over the transport).

## Results off the boxes

Each box runs `forge.fleet.pusher`: every row/artifact written under the box's `$FORGE_ROOT/results/` is copied to
the rig's `~/forge/results/boxes/<machine>/` as it appears. Mission rows: `results/rows/<mission_id>.json`.

## Controller and CLI (`bin/forge`, `forge.fleet.controller`)

`forge doctor | deploy <machine> | rent | missions | run-fleet | status | leaderboard | stop`.
Fleet state for the dash: `<FORGE_STATE>/fleet/machines.json` (snapshot), `fleet/queue.json`,
`fleet/spend.jsonl`, `fleet/alerts.jsonl`; leaderboard in `results/leaderboard.json` + `.md`, rows in
`results/rows/*.json`.

## Gate on a remote target (`forge.fleet.remotegate`)

`gate_for(machine)` returns core's `CoreGate` when the target is this machine and `RemoteGate(machine)` otherwise.
RemoteGate calls `forge.gate.request/status/ledger` ON the target through the executor (one gate and ledger per
machine). Same signatures and normalized rows as CoreGate; `bundle_dir` is a path on the target.

## Recipe check (`forge.fleet.recipecheck`)

`python -m forge.fleet.recipecheck --recipe <id> --gpu-class <c>` (run via the executor with `kind="serve"`):
starts NVIDIA's published server for the recipe in its recipe-files dir, waits for health, sends one chat request,
stops, prints `RECIPE_CHECK {ok, startup_s, completion_tokens, ...}`. Verified Oct 17: Bolt+LiLiCorr on a Vast
RTX PRO 6000 (startup 201 s), Super-VL+LiLiCorr with the 95 GB adapt (startup 498 s).

## Boxes

Vast instances are docker containers (PID 1 is not systemd; Landlock ABI 8 is available). Each box runs, started
by `forge.fleet.boxsetup` with nohup: the `forge.sched` daemon (LANDLORD_URL=http://127.0.0.1:7788/mcp through
landlord's reverse tunnel, building `vast-<instance>`) and the `forge.gate.service`. Recipe files (reasoning-parser
plugins) are in `$FORGE_ROOT/state/recipe_files/<base_id>/`. Setup progress: `$FORGE_ROOT/state/box_status.json`.
