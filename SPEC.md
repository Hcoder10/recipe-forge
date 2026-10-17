# recipe-forge: build spec (written at 10:00 on Oct 17; all code written today)

**Goal.** Fleets of Nemotron agents, running on Crusoe Managed Inference, make **verified, lossless speedups to
NVIDIA's published serving recipes for Nemotron models** across many GPUs in parallel. The output is a
leaderboard: recipe × GPU, NVIDIA's baseline vs the best gate-verified result, the change, and its proof.

Hackathon themes:
- **Done Means Done:** each mission defines and verifies its own "done".
- **Version N+1:** the playbook improves between mission waves, and recipes not yet run are the held-out tasks.

## 0. Hard rules (every builder)

- **New code only.** Don't read, copy or import any other project's code. The only allowed outside reading is the
  builders' lessons file the lead gives you, and only for lessons; never copy from it into agent-visible text.
- **NO-LEAK.** Agent-visible text (prompts, tool descriptions, manual, mission briefs, error messages) describes
  capabilities and procedure, never tactics or "try X". No speed findings from humans or Claude, ever.
- **GPUs:** never start GPU work without a landlord lease (MCP at http://100.112.46.57:7788/mcp, JSON-RPC with
  an mcp-session-id header). Never kill a process you didn't start. No sudo unless the lead says so.
- **Rig storage:** the rig's root disk is nearly full (about 18 GB). The external SSD is fragile, so read it in place
  and never do bulk writes. Keep state small, and retention is on by default.
- **Integration tests run in the real process** (a lesson: module-name collisions passed the unit tests). Use
  unique top-level package names only: everything is under the `forge` package.
- **"Done" means deployed and verified on the target machine,** not only edited on the laptop.
- Push to GitHub often: small commits with clear messages.

## 1. Topology

- **Laptop** (Windows): the repo (canonical), the landlord coordinator, and the fleet controller CLI (`forge`).
  **It must stay awake.**
- **Rig** `cube@100.68.107.11` (Ubuntu): 3× RTX PRO 6000 Blackwell Max-Q. GPUs 0 and 1 are **timing**. GPU 2 is
  power-capped, so it's a **work** GPU only (never timing).
  - venvs: ~/venvs/vllm-bolt, ~/venvs/vllm-svl. HF cache: ~/hf/hub (Bolt, Super-VL and their LiLiCorr drafters).
  - **The mission runner (the LLM loops) runs here**, in ~/forge, and holds the Crusoe key (~/forge_secrets.env,
    mode 600).
  - The rig has no git: deploy is tar over ssh (`forge deploy <machine>`).
- **Vast boxes,** rented through landlord: RTX PRO 6000 (95 GB) for Bolt and Super-VL, RTX 5090 (32 GB) for the
  small Nemotrons. Each box runs its own **executor plus gate**. Results are pushed off the box as they happen,
  because boxes are destroyed on return.

## 2. Core concepts

- **Recipe** (`forge/recipes/*.yaml`):
  - `id`, `model` (HF id), `drafter` (optional), `engine: vllm`
  - `serve`: NVIDIA's exact published args, cited with a source URL
  - `workload`: prompt set, concurrency, max_tokens, temperature 0
  - `metric`: decode_tok_s for concurrency 1, output_tok_s for throughput
  - `hardware`: which GPU classes it fits
- **Machine:** name, ssh, gpus {timing, work}, FORGE_ROOT, python/venv per recipe, landlord building name.
- **Mission:** recipe × machine × GPU(s), plus a time box (default 75 min), an LLM budget and a team config. It
  ends with a **row**: baseline, best, delta, gate verdicts, change summary and evidence ids.
- **Bundle:** a content-addressed candidate: config delta, source overlays (diff), kernel files, drafter dir, env
  clone, system settings. Its hash covers **contents** (never just mtimes).
- **Gate:** one per machine. The only judge of a win.

## 3. Components and owners

### core (owner: `core`): `forge/gate`, `forge/sandbox`, `forge/sched`

**Sandbox:** Landlock (bwrap is blocked by apparmor; kernel 7.0, ABI 8). Each job gets:
- file rules
- a TCP port range
- signal and abstract-socket scoping
- GPU device nodes per grant (a granted card shows up as index 0)
- **a systemd --user scope** (it reaps setsid children)
- CPU-core sets (gate cores separate from agent cores)

A read-only directory nested inside a writable one is still writable, so lay out dirs to avoid that.

**Scheduler and leases:** one per machine.
- Kinds: gate (exclusive on all timing GPUs), serve, bench, work.
- VRAM accounting on work GPUs.
- Queue (don't refuse) while a lease is pending.
- A lease keeper that extends and moves out leases, survives the landlord being briefly unreachable, and always
  moves out on exit.

**Gate:**
- **Timing:** paired AB/BA reps on a fixed, stratified, hidden prompt set (the hidden pool lives on internal disk
  and is never readable by agents).
  - Win: every pair positive and the mean log-gain > t·s/√n, at α = 0.001, from per-recipe reference-pair
    history.
  - Champion: a light confirmation gate (2 pairs, a fresh prompt set, α_confirm = 0.05).
- **Each server** is started by the gate itself, in a private scope, with a private ws, tmp, shm and compile cache.
  The reference cache is read-only and warm.
- **Timing measurement:** one timed pass, never repeat a prompt per server, end-to-end time over **the gate's own
  re-tokenized count** with 1% tolerance.
- **Lossless:** teacher-forced prompt_logprobs on hidden probes, plus logprobs on the timed outputs: top-1
  agreement, KL and max regret against the reference's own run-to-run band, with a near-tie floor.
- **Machine and integrity checks:** a quiet-machine check before each run, one gate per machine, venv hashes, a
  content hash of the bundle, and retention (< 20 MB per gate run).
- **Setup:** 2 reference controls per recipe (so there are ≥ 8 null pairs) before candidate gates count.
- Locked by sha256 (`forge gate lock`). Rehearsals use their own lock.

### agents (owner: `agents`): `forge/agents`, `forge/tools`

**Team per mission:** Lead (Ultra; plans, decides), 2-3 Workers (Super), Verifier (Ultra; approves gate requests).

**LLM client:** Crusoe, OpenAI-compatible, native tool calls, args coerced to the schema, no raw tracebacks to
agents. Calls are logged with tokens, cached tokens and $.

**Prompt layout:** cache-stable. Fixed prefix: role + tools + MANUAL + **mission brief** + the **experiment
protocol**. Then a slowly refreshed facts view, then the history appended at the end. Volatile numbers go only in
the last message.

**Context caps:** Worker 64k, Lead 48k. Compact in chunks, keeping the newest turns plus a summary of what was
learned.

**Experiment protocol (procedure only):**
1. claim + predict
2. measure the reference (or reuse it from facts)
3. make one change
4. measure the delta against the same-GPU reference
5. record a fact
6. request the gate only if the delta clearly clears the Verifier's bar

**Tools:** small, with high-level ones first:
- `profile`, `bench(config_delta)`, `bench(bundle)`
- `edit`, `diff`, `make_bundle`, `kernel_test`, `train_drafter` (where feasible)
- `request_gate`, `note`, `fact_add`, `decide`, `ask_human`
- raw `sh` and `py` last (sandboxed)

They run on the target machine through the executor.

**Done Means Done:**
- The mission's done criteria are compiled from the recipe and goal and locked at mission start.
- An auditor checks them against the ledger only.
- A drift and loop detector: repeat_call, repeat_failure, read_streak, no_progress, prediction_miss, tool_bug,
  env_blocked. Agent faults: nudge, then Lead. Infra faults: escalate at once.
- A forced decision log: continue | replan | stop | ask_human, with evidence.

**Shared Brain:** facts (claim, recipe, lane, status, evidence) shared across missions, so one model's
discoveries carry to the next.

**Version N+1:** after each mission wave, a Mechanic (Ultra) proposes a playbook or prompt revision from trace
stats. It's kept only if the **next unseen recipes** do better than the previous version on a paired comparison
(missions per version, per-recipe deltas). Every promotion or rejection is logged.

### fleet (owner: `fleet`): `forge/fleet`, `forge/recipes`, `bin/forge`

**Recipe catalog** from NVIDIA's model cards and NVIDIA recipe pages, with exact serve args and source URLs:
- Bolt 74B-A9B and Super-VL 120B NVFP4 (local)
- Nemotron Nano 9B v2, Nemotron 3 Nano 30B-A3B, Nemotron 3.5 Lightning 30B-A3B, plus any other current Nemotron
  NVIDIA publishes vLLM recipes for (download on the boxes)
- single-stream and throughput variants

**Machines:**
- **Vast provisioning:** search and rent through landlord, then set up the box: a vLLM image or venv that runs the
  recipe, model download, the executor and gate install, ssh access from the rig.
- **Getting results off a box:** push rows and artifacts to the rig as they happen.
- **Verified rentals:** replace a box that doesn't come up within 10 min.

**Executor:** a small daemon or ssh command surface on each box that runs tool calls from the rig's mission runner.

**Fleet controller:** a queue of missions → free machines. Starts, monitors and collects; writes leaderboard
rows; sends the next wave.

**CLI:** `forge doctor | deploy | rent | missions | run-fleet | status | leaderboard | stop`.

### dash (owner: `dash`): `forge/dash`

A live web page on the rig (FastAPI + SSE, no CDN), showing:
- the **leaderboard** (recipe × GPU: baseline, best, delta, gate proof)
- live missions (team, current step, lane, $)
- gate queue and verdicts
- decision feed and alerts
- Version N+1 history
- spend (Crusoe and Vast)

Projector-readable.

## 4. Milestones

- **M1 (target 11:30):** one mission end to end on the rig, with Bolt on GPU 0: agents → bench → (gate request
  if the bar is cleared) → row.
- **M2 (12:15):** 4-6 Vast boxes provisioned; fleet running missions across recipes.
- **M3 (12:15-16:30):** waves of missions; N+1 between waves; leaderboard rows pushed.
- **M4 (16:30-16:45):** final rows, README, repo made public, submission.
