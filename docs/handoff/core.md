# core handoff (forge.sandbox, forge.sched, forge.gate), 12:45 Oct 17

API details: `docs/interfaces/core.md`. Tests: `~/venvs/vllm-bolt/bin/python -m unittest tests.core.test_sched
tests.core.test_sandbox tests.core.test_gate tests.core.test_daemon` (Linux only; fcntl/Landlock).

## Verified for real vs only with mocks

| What | Real (rig) | Mock only |
|---|---|---|
| Landlock file/TCP/signal/abstract-socket rules, clean env, CPU pin, setsid child reaped by scope | yes (tests.core.test_sandbox) | |
| GPU device-node grant: granted GPU 2 shows as CUDA 0, tiny tensor under a lease | yes | |
| Scheduler daemon + real landlord: gate grant rents GPUs 0+1, extends, moves out on release | yes (smoke runs) | |
| Keeper survives landlord outage, persists pending move-outs, SIGTERM moves out all | daemon test vs fake MCP (real process) | landlord outage only faked |
| Gate reference-vs-reference on Bolt c1, reps=2, sandboxed gate-started servers | yes: `g10171232-b0e287` | |
| Gate candidate win / confirmation / lossy / loss verdicts, auto reference controls, dedupe, retention | | yes (tests.core.test_gate, fake servers) |
| Candidate bundle through the real gate (serve_command, overlay hook, bundle copy + hash) | **no** | partly |
| Gate service loop (`forge.gate.service`, unit forge-gate) | **no** (only `service once <id>`) | loop logic via records |

## Measured (rig, Bolt 74B + LiLiCorr, c1, 512 tokens, 8 hidden prompts per rep)

- Reference gate, reps=2: **7.7 min** wall (4 servers). Server startup 54-84 s with the warm ref cache;
  first cold start several minutes (flashinfer/b12x JIT). Expect reps=4 ~ 15 min, a candidate gate ~ 17-20 min
  (cold candidate cache on its first server), confirmation ~ 8 min.
- Null pairs: 265.3 vs 258.9 tok/s (-2.4%), 263.7 vs 262.4 (-0.5%). Fixed-length outputs: 4096/4096 tokens.
- Ref-vs-ref lossless band: probe top-1 agree 0.90, flip rate 0.9-1.0%, KL 0.024, max regret 4.75;
  decode flip rate 0.6-0.8%, lp gap 0.021-0.024, max regret 1.75. (Identical servers disagree this much
  across restarts; worth a look: kernel autotune nondeterminism?)
- Run dir 544 KB. Ref compile cache `~/forge/gate/cache/ref/bolt-74b-lilicorr_c1` = 1.7 GB.

## Gate rules as implemented (`forge/gate/runner.py`, `stats.py`, `lossless.py`)

- Sides: A = recipe exactly as published; B = recipe + bundle via `forge.tools.bundle.serve_command` on a
  private copy of the bundle (hash re-checked against submit, symlinks refused, bundle env cannot set
  FORGE_/LANDLORD_/HF_/cache/PATH/HOME vars).
- Reps: pair r is AB when r even, BA when odd. Each server: started by the gate in its own systemd scope +
  Landlock (ro venv/HF/bundle; rw only its job dir + its cache; TCP only its grant port block), warmed on one
  prompt per stratum (128 tokens), timed once over rep r's fixed hidden slice (no prompt repeats in a server
  lifetime), then teacher-forces the probes and (if the other side ran earlier in the pair) the other side's
  timed outputs.
- Timing: `/v1/completions`, token-id prompts, temperature 0, `logprobs=1`, `ignore_eos` (fixed length).
  c=1: sum(tokens)/sum(per-request e2e time); c>1: tokens / pass wall. Tokens = gate re-tokenized count;
  must match the server's returned token list within 1% else invalid.
- Win (candidate): every pair log-gain > 0 and mean > t(1-alpha, df=N_null) * s_null / sqrt(n), alpha 0.001,
  s_null = RMS of reference-pair log-gains in the per-recipe history (`gate/history/<recipe_key>.json`).
  Fewer than 8 null pairs -> `invalid`; the service auto-queues reference controls (reps 4) first.
- Win queues a confirmation: 2 pairs on fresh confirm slices, alpha 0.05; sets `confirmed` on the parent row.
- Lossless (both directions of decode + probes) against the null band: top-1 agree >= band min - 0.01;
  KL <= max(3x band, 1e-3); flip rates <= max(2x band, 0.2%/0.5%); decode max regret <= max(2x band, 1 nat);
  logprob gap <= max(3x band, 0.02). Near-tie floor 0.25 nats. Any fail -> `lossy`.
- Machine checks: no foreign compute process on gate GPUs and gate cores mean load < 30% before each server
  (6 retries x 10 s), foreign-process watch during the timed pass, lock verify, venv content hash, recipe
  hash, pool hash recorded in verdict.json.
- Hidden pool: `FORGE_ROOT/gate_hidden/pool_v1.json` (mode 700) from curated templates in `pool.py` and a
  secret seed created on first use (`gate_hidden/seed`). 15 strata (5 task families x 3 lengths), 12 per
  stratum. Visible bench prompts (`forge/recipes/prompts/public_v1.jsonl`) are not checked for overlap yet;
  the sources differ.

## How to call and lock

```bash
cd ~/forge; P=~/venvs/vllm-bolt/bin/python; export FORGE_ROOT=~/forge
$P -m forge.sched.daemon ensure                         # running now as unit forge-sched
$P -m forge.gate.lock write --name main                 # freeze; rehearsals: --name rehearsal + FORGE_GATE_LOCK=rehearsal
$P -m forge.gate.service ensure                         # the queue worker (unit forge-gate); not started yet
$P -c 'import forge.gate as g; print(g.reference("bolt-74b-lilicorr:c1", reps=4))'   # optional manual control
$P -m forge.gate.service once <gate_id>                 # run one request in the foreground
```
From Python: `forge.gate.request(recipe_id=, bundle_dir=, mission_id=, machine=, requester=, claim=)`,
`status(gate_id)`, `ledger(...)`, `wait(gate_id)`. Ledger: `FORGE_ROOT/gate/ledger.jsonl`, latest row per
gate_id wins; `pairs` is already the list of per-pair log-gains (dash ask). Recipe ids: `<id>:<variant>`;
`<id>.<variant>` is accepted too.

## Known gaps / TODO, priority order

1. **No systemd on Vast boxes** (docker, PID 1 = sh): `forge.sandbox.launch` uses `systemd-run --user --scope`
   and fails there; so do both `ensure` commands (use nohup). Needs a fallback: setsid + process-group kill +
   track descendants via /proc (cgroup-less), chosen when `systemctl --user is-system-running` fails.
   Until then the gate cannot run on boxes. On boxes the pool/seed is created on first use (don't copy the
   rig's), and `lock write --name main` is needed.
2. Run one real candidate gate (a trivial config-delta bundle) end to end, and start the service loop.
3. Null history now holds the 2 rehearsal pairs for `bolt-74b-lilicorr:c1` (made under lock `rehearsal`);
   history is not keyed by lock. Delete `~/forge/gate/history/*.json` before real runs if that matters.
4. Probe band: identical servers agree only 90% on top-1 over the synthetic probe text; consider natural-text
   probes, or teacher-forcing reference outputs, to tighten the band.
5. Residuals: `/dev/shm` is shared rw (candidate server could leave data for agents); cpusets not delegated
   (jobs can re-pin onto gate cores); hidden-pool overlap check vs public_v1 not done; server startup
   measured in 30 s steps (health poll stalls), harmless.
6. Agents' sandboxed jobs must never get `~/forge` itself as a ro path (gate/ and gate_hidden/ live there).

## File:function map

- `forge/sandbox/landlock.py:apply` Landlock syscalls. `forge/sandbox/__init__.py:launch, build_rules,
  build_env, gpu_nodes, Job.{wait,stop,tail,cleanup}`. `forge/sandbox/enter.py:run` (in-scope entry).
- `forge/sched/landlord.py:LandlordClient.{call,rent,extend,move_out}` MCP over HTTP.
  `forge/sched/keeper.py:LeaseKeeper.{rent,release,tick,close}`. `forge/sched/core.py:Scheduler.{request,
  tick,_place,_lease,release,renew,shutdown}`. `forge/sched/daemon.py:serve, ensure, _building_by_ip`.
  `forge/sched/client.py:SchedClient.{acquire,request,status}`, `Grant`. `forge/sched/machine.py:load`.
- `forge/gate/__init__.py:request, reference, status, ledger, wait`. `forge/gate/records.py:submit,
  next_request, finish, resolve` (queue + ledger). `forge/gate/service.py:run_one, RealCtx, Service, ensure,
  main`. `forge/gate/runner.py:run, timed_pass, teacher_force, History, record_reference`.
  `forge/gate/server.py:GateServer, GpuWatch`. `forge/gate/stats.py:paired_test, t_ppf`.
  `forge/gate/lossless.py:probe_compare, decode_regret, band, judge`. `forge/gate/pool.py:build, Slices`.
  `forge/gate/integrity.py:hash_tree, venv_hash`. `forge/gate/lock.py:write, verify`.

Leases held by core: none (smoke leases moved out; scheduler daemon shows 0 grants, 0 leases).
