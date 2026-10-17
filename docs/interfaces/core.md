# core interface (`forge.sandbox`, `forge.sched`, `forge.gate`)

Owner: core. Everything runs on the target machine with the recipe venv python (`~/venvs/vllm-bolt/bin/python`
on the rig). `FORGE_ROOT` defaults to `~/forge`.

## FORGE_ROOT layout (on each machine)

| path | who | notes |
|---|---|---|
| `forge/` | deploy | the package (read by trusted processes only; sandboxed jobs never need it) |
| `machine.json` | fleet (optional) | `forge.sched.machine.Machine` fields; else auto-detected |
| `sched/` | scheduler daemon | `grants.json`, `leases.json`, `daemon.lock` |
| `jobs/<id>/` | sandbox | per job: `home/ tmp/ ws/` (private, writable), `log`, `spec.json` |
| `gate/` | gate service | ledger, queue, history, runs, caches, locks (mode 700, never granted to jobs) |
| `gate_hidden/` | gate | hidden prompt pool (mode 700, never granted to jobs) |

Never grant a sandboxed job read access to `FORGE_ROOT` itself, only to the dirs it needs.

## forge.sandbox

```python
from forge.sandbox import Spec, launch
job = launch(Spec(cmd=[py, "x.py"], job_dir=f"{root}/jobs/j1",
                  ro=[venv_dir, hf_home, workspace_src], rw=[extra_writable],
                  gpus=grant.gpus, ports=tuple(grant.ports), cpus=grant.cpus,
                  env={"LANDLORD_LEASE_ID": grant.env["LANDLORD_LEASE_ID"], "HF_HOME": hf_home, "HF_HUB_OFFLINE": "1"},
                  path_prepend=[f"{venv_dir}/bin"]))
rc = job.wait(timeout=600)   # on timeout: SIGTERM the whole scope, SIGKILL after 15 s
job.tail(4000); job.stop(); job.cleanup(keep_log=True)
```

- Landlock (ABI >= 6 required). Read-only: system dirs + `ro`. Writable: `<job_dir>/{home,tmp,ws}` + `rw`.
  `ro` inside `rw` raises `SandboxError` (it would be writable). Everything else (home dir, FORGE_ROOT, /tmp,
  other jobs) is denied.
- TCP: bind and connect only inside `ports` (inclusive). `connect_extra=[...]` adds outbound ports;
  `connect_any=True` leaves outbound unrestricted. No `ports` = no TCP at all. Signals and abstract unix sockets
  are scoped to the job.
- GPUs: only the granted device nodes are visible; the job sees them as CUDA 0..n-1 (`CUDA_VISIBLE_DEVICES` is set
  to `0..n-1`). No `gpus` = no GPU.
- Env is rebuilt from scratch: PATH, HOME/TMPDIR (private), XDG caches, inductor/triton caches under HOME,
  `FORGE_PORT_LO/HI`, plus `env`. Nothing from the caller leaks in.
- Each job is a `systemd-run --user --scope` unit `forge-<name>`; stop/wait kills setsid children too.
- `cpus` pins via sched_setaffinity (cpuset is not delegated on the rig, so a job could re-pin itself).

## forge.sched (one daemon per machine)

Start: `python -m forge.sched.daemon ensure` (systemd --user unit `forge-sched`, 127.0.0.1:7791).

```python
from forge.sched.client import SchedClient
sc = SchedClient()
g = sc.acquire("bench", owner="mission-7/worker-1", purpose="quick bench of Bolt c1 with a config delta",
               minutes=30, where="any", on_wait=lambda g: print(g.wait_reason))   # blocks while queued
# g.gpus [0], g.gpu_uuids, g.ports [20000, 20099], g.cpus, g.env (CUDA_VISIBLE_DEVICES physical,
# LANDLORD_LEASE_ID, FORGE_LEASE_IDS, FORGE_PORT_LO/HI, FORGE_GRANT_ID), g.leases {gpu: lease_id}
g.renew(60); g.release("outcome text")      # or: with sc.acquire(...) as g: ...
sc.request(...)  # non-blocking: state queued|leasing|active with wait_reason
sc.status()      # machine, grants (queue order), recent, keeper {leases, pending_out, last_error}
```

| kind | placement |
|---|---|
| `gate` | all timing GPUs, exclusive. While a gate waits, nothing new starts on timing GPUs. Gate cores. |
| `serve`, `bench` | one GPU. `where`: `any` (timing first, then work), `timing`, `work`. Exclusive on a timing GPU; VRAM-accounted on a work GPU (`vram_gb`, default the whole card). |
| `work` | a work GPU (a timing GPU only if the machine has no work GPU), needs `vram_gb`, `pct` (default 50). |

- Requests queue with a human-readable `wait_reason`; they fail only when they can never fit.
- Each grant holds one landlord lease per GPU (purpose = `forge <kind> for <owner>: <purpose>`). The keeper
  extends while the grant is active, moves out on release, retries move-outs while landlord is unreachable
  (persisted in `sched/leases.json`), and moves out everything on SIGTERM/SIGINT/SIGHUP/exit.
- A grant ends when released, when its holder pid exits (the client registers `os.getpid()`), or at its
  deadline (`minutes` after it started; extend with `renew`).

## forge.gate

```python
import forge.gate as gate
r = gate.request(recipe_id="bolt-74b-lilicorr:c1", bundle_dir="/abs/bundle", mission_id="m7",
                 machine="squaredcube1-ubuntu", requester="m7/verifier", claim="...")
# -> row {gate_id, status: queued, kind: candidate, ...}; identical (recipe, bundle_hash, kind) requests dedupe
gate.status(gate_id)   # -> row
gate.ledger(machine=None, recipe_id=None, mission_id=None)   # -> [rows], latest per gate_id
gate.wait(gate_id, timeout=None)
```

Row: `{gate_id, mission_id, recipe_id, machine, bundle_hash, kind: candidate|confirm|reference, status:
queued|running|done|error, verdict: win|loss|lossy|invalid|error|None, delta_pct, evidence_id, confirmed, t,
detail}`. A candidate `win` queues its own `confirm` gate (2 pairs, fresh prompts, alpha 0.05); `confirmed`
becomes true on the candidate row when it passes. `recipe_id` is `<recipe>` or `<recipe>:<variant>`
(variant name from the recipe's `variants`, default the first). Bundles use `forge.tools.bundle` (manifest +
`serve_command`).

The gate service runs as `python -m forge.gate.service ensure` (unit `forge-gate`). Rows are agent-safe:
no prompts, no server logs. `python -m forge.gate.service once <gate_id>` runs one queued request in the
foreground (rehearsals). `gate.reference(recipe_id, reps=4)` queues a reference control by hand; the service
also queues them itself until a recipe has 8 null pairs, and candidate rows before that are `invalid`.

How a run works (for the dash and the auditor; agent-visible text should not repeat this):
- Grant `gate` from the scheduler (all timing GPUs, gate cores); servers run on the first timing GPU(s)
  (`hardware.tensor_parallel`). Reference = recipe `serve.args` + `vllm.env` + `serve.env`; candidate =
  `forge.tools.bundle.serve_command` over a private copy of the bundle (hash re-checked, symlinks refused;
  bundle env cannot set FORGE_/LANDLORD_/HF_/cache/PATH/HOME vars).
- Caches: reference servers share a warm, fixed-path cache per recipe (`gate/cache/ref/<key>`); candidate
  servers get `gate/cache/cand`, wiped at the start of each gate. Servers get the venv's CUDA toolkit
  (`site-packages/nvidia/cu*/`) as CUDA_HOME, since cold caches JIT-compile kernels.
- Warmup: one prompt per stratum (every prompt-length class) before timing, so lazy kernel tuning never
  lands in the timed pass.
- Timing: `/v1/completions` with chat-templated token ids (`apply_chat_template` returns a BatchEncoding:
  take `["input_ids"]`), temperature 0, `logprobs=1`, fixed-length outputs (`ignore_eos`, recipe workload
  `gate_ignore_eos: false` turns it off); per-request end-to-end time; tokens = gate re-tokenized count,
  must match the server's returned token list within 1%.
- Locks: `python -m forge.gate.lock write --name main|rehearsal`; a request runs under `FORGE_GATE_LOCK`
  (default `main`) and is `invalid` if that lock does not verify. The lock covers forge/gate, forge/sandbox,
  forge/sched and forge/tools/bundle.py.

## Machine detection

Without `FORGE_ROOT/machine.json`: GPUs whose power limit is below their default limit are work GPUs
(rig: GPU 2), the rest timing; gate cores = first quarter of physical cores + SMT siblings (rig: 0-7,32-39);
the landlord building is the one whose tailscale IP is ours (rig: `squaredcube1-ubuntu`, hostname
`squaredcuberig`).
