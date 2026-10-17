# recipe-forge

Fleets of NVIDIA Nemotron agents, served on Crusoe, work to produce verified, lossless speedups of NVIDIA's vLLM serving recipes for Nemotron models across many GPUs. The central idea is **recursive Version N+1**: each mission leaves structured evidence that the next mission can use, so the harness learns across architectures instead of starting over. Lead, Workers and Verifier all use **Nemotron Super-VL**, with a **Super-120B fallback**. Built for the Recursive Agents Hackathon and its Best Use of Nemotron prize, recipe-forge combines autonomous experimentation, a live operator demo and an independent gate whose evidence determines what can be claimed.

## Results

**TBD: no performance results are claimed in this table.** Populate each row only with attributable reference measurements and gate evidence for the same recipe, GPU and workload.

| Recipe | GPU | NVIDIA published/reference tok/s | Verified tok/s | Delta | Gate id |
|---|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | TBD | TBD | TBD | TBD | TBD |

A published recipe is a cited serving configuration, not necessarily a published throughput number. Label locally measured references explicitly; disclose any hardware adaptation applied to both reference and candidate. Use gate-run measurements for verified tok/s: the mission summary's `best` can be derived from a quick-bench baseline and a gate delta, so it is not itself an absolute gate measurement.

## Recursive Version N+1

The append-only win library in [forge/agents/winlib.py](forge/agents/winlib.py) records **WHAT** changed, **WHERE** it was tested (architecture traits, GPU and workload), **HOW** it was applied (configuration, environment, bundle identity and changed files), and the outcome with evidence ids. It preserves unsuccessful observations as well as promising ones, distinguishing quick-bench outcomes from gate verdicts.

New missions receive relevant entries selected by architecture, layer mix, quantization, drafter, GPU and concurrency. Reuse is recorded with the new experiment's evidence; a previous outcome still needs verification on the new target. Current automatic reuse matching covers configuration/environment changes, not proof of code-edit reuse. This is the mechanism for improving the harness across Nemotron architectures; cross-architecture improvement remains a claim to demonstrate with results. A separate Mechanic workflow proposes versioned playbooks and records promotion/rejection comparisons on unseen recipes.

## How it works

- **Mission:** a recipe, target machine/GPU, time box and LLM budget. Done criteria are compiled from the recipe and goal, then hash-locked at startup. Calls, experiments, decisions, usage and the final row are recorded.
- **Team:** the Lead assigns work and records evidence-backed `continue`, `replan`, `stop` or `ask_human` decisions. Workers inspect, edit, measure and freeze candidates. The Verifier reviews evidence for the exact bundle before submitting it to the gate. Prompt prefixes remain stable; facts refresh and older context is compacted.
- **Tools:** `profile`, `bench`; `read`, `search`, `ls`, `glob`, `edit`, `apply_patch`, `diff`, `revert`; `make_bundle`, `kernel_test`; `request_gate`, `gate_requests`, `review_gate`, `gate_status`; `assign`, `team_status`, `report`, `decide`, `note`, `fact_add`, `facts`, `ask_human`; sandboxed `sh` and `py`. Tools are restricted by role. `profile` reports the target configuration and environment; it is not a kernel profiler.
- **Fleet:** the controller deploys over SSH, queues missions across registered machines, provisions landlord-managed rentals, monitors jobs and collects results. Each target has an executor and gate. Results must be copied off rented disks while runs proceed.
- **Scheduler:** GPU jobs queue for grants backed by landlord leases. Gate jobs reserve timing GPUs exclusively; work-GPU allocation accounts for VRAM. Lease renewal and release are managed centrally. Never start GPU work without a lease or pre-empt another tenant.
- **Sandbox:** Linux Landlock restricts files, TCP ports, signals and abstract sockets; jobs receive granted GPU devices and a rebuilt environment. Job scopes handle process cleanup. Hidden gate data stays outside agent workspaces.
- **TUI:** `python -m forge.tui` shows mission activity, leaderboard and human inbox. `/missions`, `/mission <id>`, `/leaderboard`, `/inbox` and `/help` navigate the demo. Fixture mode is explicitly synthetic; it is not evidence.

### Done Means Done

**The gate is the only judge.** It starts its own reference and candidate servers, checks bundle and code integrity, and measures hidden prompts with paired **AB/BA timing at α=0.001**. Every pair must be positive and the mean log-gain must clear a threshold calibrated from reference-versus-reference history. Timed outputs use fixed length by default, and the gate independently re-tokenizes output to check token counts. Candidate wins trigger confirmation on fresh prompt slices.

The **lossless logprob band** checks teacher-forced probes and timed outputs against reference run-to-run variation, using top-1 agreement, KL, token regret and logprob gaps with explicit tolerances. The done auditor reads the gate ledger, never an agent's declaration; confirmation is required by default. The detector catches repeated calls/failures, reading loops, stalled progress and prediction misses. Infrastructure failures escalate immediately to the Lead and human inbox.

```text
Operator / TUI             Fleet controller
      |                         |
      +------> Mission runner <-+---- recipe + target + budget
                   |
             Crusoe Nemotron
          Lead / Workers / Verifier
                   |
          tools -> target executor
                   |
          scheduler -> landlord leases
                   |
             Linux sandbox
                   |
       visible bench -> frozen bundle
                              |
                       Verifier review
                              |
                   independent hidden gate
                    AB/BA + lossless band
                              |
                       ledger -> done audit
                              |
                win library + reuse records
                              |
                    next mission (N+1)
```

## Quickstart

Run from the repository root with **Python 3.12**. Real execution needs a configured Linux target, the recipe's vLLM environment and model files, SSH access, a reachable landlord, and Crusoe credentials supplied privately through `CRUSOE_API_KEY` or the runner's protected `~/forge_secrets.env`. Never put credentials in mission specs, notes or logs. Machine configuration lives in `forge/fleet/machines.yaml`; see the [fleet interface](docs/interfaces/fleet.md) and [core setup](docs/interfaces/core.md) before a real run.

Inspect the commands and run the mock mission without GPUs or network:

```bash
python bin/forge --help
python -m forge.agents --help
python -m forge.agents selftest
```

On a configured Linux target with a working systemd user session, select the recipe's Python environment and initialize services:

```bash
export FORGE_ROOT="$PWD"
export FORGE_STATE="$FORGE_ROOT/state"
python -m forge.sched.daemon ensure
python -m forge.gate.lock write --name main
python -m forge.gate.service ensure
python -m forge.agents probe --profile crusoe
```

Run one mission on the configured rig (the limits below are example operator budgets):

```bash
python -m forge.agents mission --recipe bolt-74b-lilicorr:c1 --machine rig \
  --minutes 45 --budget 6 --workers 2 --id demo --out state/specs/demo.json
python -m forge.agents run --spec state/specs/demo.json
```

Alternatively, dispatch through the controller; keep it running and use another terminal for status and the TUI:

```bash
python bin/forge deploy rig
python bin/forge doctor
python bin/forge missions add bolt-74b-lilicorr:c1 rig --wave 1
python bin/forge run-fleet
# In another terminal:
python bin/forge status
python bin/forge leaderboard
# On the Linux runner:
python -m forge.tui
```

Deploy before writing the gate lock when setting up a fresh target. A deployment that changes locked code requires a deliberate new lock before judging candidates. `python bin/forge stop <mission_id>` stops a mission started by that controller. For a synthetic UI walkthrough, use `python -m forge.tui --fixture`.

The runner and TUI share append-only files under `<state_root>/missions/<id>/`:

| File | Row contract |
|---|---|
| `control.jsonl` | `{"t": float, "by": "operator", "action": "pause"\|"resume"\|"stop", "reason": str}` |
| `operator_notes.jsonl` | `{"t": float, "by": "operator", "to": "lead"\|"worker-N"\|"all", "text": str}` |

These are schema sketches; append valid JSON objects with a numeric Unix timestamp and one selected action/recipient. Pause holds new agent steps; it does not cancel an in-flight tool call or freeze the mission time box. Keep operator notes procedural and free of speed findings or result-bearing tuning advice.

Tests use the standard library, not pytest. On Linux:

```bash
python -m unittest tests.agents.test_units tests.agents.test_mission_mock
python -m unittest tests.core.test_sched tests.core.test_sandbox tests.core.test_gate tests.core.test_daemon
```

## Honesty and disclosure

Agents receive a human-written **STARTER TECHNIQUES** list of generic, standard levers with no expected results in [forge/agents/playbook.py](forge/agents/playbook.py). Beyond that disclosed starting list, optimization discoveries are made by Nemotron agents through experiments. Humans supply the harness, recipes and operating procedure, not speed findings or “try X gives Y” advice. Only gate-verified wins are claimed. Mock missions and fixture displays demonstrate plumbing and presentation, not serving performance.

## Limitations

- “Lossless” means passing the implemented statistical logprob band on sampled probes and outputs. It is not a proof of bit-identical output or universal equivalence.
- Results are specific to model, recipe, GPU, environment and workload. Quick-bench gains and library reuse do not establish a win on another target.
- Linux isolation and service support are prerequisites; container hosts need compatible process cleanup and daemon startup. A configured fleet is not proof that every target has passed an end-to-end candidate gate.
- The hidden pool is finite and template-based. Reference calibration, confirmation and quiet-machine requirements can leave missions pending or without a verified win.
- The hackathon handoffs distinguish real infrastructure checks from mock-only verdict paths. This README's TBD table makes no claim of completed fleet-wide validation or demonstrated N+1 improvement.

Built during the hackathon on **Oct 17 2026**.
