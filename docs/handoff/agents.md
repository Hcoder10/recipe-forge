# Handoff: agents (`forge/agents`, `forge/tools`)

Written 13:45 Oct 17 by the agents builder. The lead owns this code from here on. The interface is in
`docs/interfaces/agents.md`.

## Real vs mock

Proven for real on the rig:
- **m1b-bolt-c1:** a real Crusoe team on Bolt c1, with core's scheduler grants, landlord leases through gpujob,
  sandboxed quick benches and a row.
- **Results:** 29 min, 10 benches. The Lead's recorded stop decision was "no candidate clears the bar". Cost
  $0.61, 78% cache hit.
- **Infra:** sandboxed `sh`/`py`, and a sandboxed GPU job through the scheduler (it saw 1 device and had a lease).
- **Crusoe:** `probe` showed native tool calls and caching.

Only with mocks so far:
- the Verifier approve → `forge.gate.request` → gate watcher → `gate_result` trigger → audit-win path
  (`tests/agents/test_mission_mock.py`)
- the Mechanic `propose` / `compare` (unit tests)
- `kernel_test` from inside a mission (only the sbxrun+gpujob smoke test ran)
- RemoteGate for box missions (switched to it at 13:45, not yet run)

## Run a mission (rig)

```
cd ~/forge && bash scripts/agents/launch_mission.sh bolt-74b-lilicorr:c1 <mission_id> 45 6 2
# spec only:  ~/venvs/vllm-bolt/bin/python -m forge.agents mission --recipe <id> --machine rig --minutes 45 \
#             --budget 6 --workers 2 --id <mid> --out state/specs/<mid>.json
# then:       ~/venvs/vllm-bolt/bin/python -m forge.agents run --spec state/specs/<mid>.json
# fleet:      runs the same `run --spec` with FORGE_ROOT/FORGE_STATE/PYTHONPATH set and ~/forge_secrets.env sourced
```
- **Outputs:** `~/forge/state/missions/<mid>/` holds `events.jsonl`, `row.json`, `usage.jsonl`, `benches.jsonl`,
  `decisions.jsonl`, `inbox.jsonl` and `answers.jsonl`; the log is `state/<mid>.out`.
- **Stop:** SIGTERM (or SIGINT) ends the mission cleanly and still writes the row.
- **Other commands:** `python -m forge.agents probe | selftest | answer <mid> <qid> "<text>"`.

## Crusoe models (probe 12:20)

- Served Nemotron models: 3.5-Super-VL, 3-Super-120B-A12B, 3.5-Lightning-30B-A3B, 3-Nano-30B-A3B and
  Nano-Omni-Reasoning. There is no Ultra.
- **Tool calls:**
  - Super-120B: correct native tool calls, about 1.0 s per tiny call, cached tokens reported (1152 of 1474 on
    the repeat call).
  - Super-VL: correct native tool calls, about 0.7–0.9 s, cached = 0 in the probe.
- **Prices per M tokens (in / cached / out):** Super-120B 0.30/0.15/2.40; Super-VL 0.10/0.08/0.40;
  Lightning 0.05/0.03/0.20.
- **Measured cost in m1b (all roles on Super-VL):** about $0.002 per call at a 20k-token prompt with ~90% cached.
  Mission total $0.61 for 258 calls.
- **Mapping:** the lead's config decides it (`forge/agents/config.py`, owned by the lead). Last approved: Lead,
  Verifier and Mechanic on Super-120B, Workers on Super-VL. The code currently has all roles on Super-VL with
  fallback to Super-120B.

## Known bugs / TODOs, in priority order

1. **The gate path has never run live.** The row now has `gate_path`, which says explicitly when nothing was
   requested or sent to the gate. Watch the first mission that clears the bar.
2. **`noise_pct` / `verdict_hint` don't reach the agents.** `benchjob.py` computes them, but `t_bench` in
   `forge/tools/builtin.py` (lead-owned) does not copy them into the tool result. The playbook text (Lead and
   Verifier) now tells agents to read both fields. Add them to the `res` dict in `t_bench`.
3. **The quick-bench reference noise is about 4–5% across server restarts** on the same GPU (312 vs 298 on GPU 2).
   A 3% `verifier_bar_pct` sits inside that noise. The A/B/A confirm in benchjob helps; consider raising the bar or
   requiring a confirmed bench.
4. **`operator_notes.jsonl` (TUI):** the lead said they are wiring it. The runner's own human channel is
   `answers.jsonl`, read in `MissionRunner.run`.
5. **The Lead calls `team_status` in bursts** (3 in 2 s). It's cheap but noisy; consider rate-limiting it in
   `_lead_loop`.
6. **Bench-server `/tmp/fz*` dirs from `_Sandboxed`** (lead's short-tmp change) are not removed in `stop()`.
7. **`quick_bench_best` in the row ignores benches whose `delta_pct` is None** (bundle benches without a candidate).

## Where things live

| piece | file:function |
|---|---|
| LLM client, retries, metering, IPv4-first HTTP | `forge/agents/llm.py:LLMClient.chat`, `_connect_v4_first`, `Meter` |
| models, prices, team caps | `forge/agents/config.py` (lead-owned) |
| mission state, gate queue, team, runner loops, row | `forge/agents/mission.py:MissionState`, `GateQueue`, `Team`, `MissionRunner.run/_lead_loop/_worker_loop/_verifier_loop/row`, `normalize_recipe`, `audit_mission` |
| one agent: prompt layout, caps, compaction, step | `forge/agents/agent.py:Agent.step/compact/messages` |
| prompt texts (brief, layout, summarizer) | `forge/agents/prompts.py:mission_brief/fixed_prefix/status_block` |
| role texts, protocol, knobs, versions | `forge/agents/playbook.py:DEFAULT/load/save/content_hash` |
| done criteria lock + auditor | `forge/agents/done.py:compile_criteria/lock/load_locked/audit` |
| triggers, forced decide, inbox, notes | `forge/agents/decisions.py:Decisions/Inbox/Notes` |
| drift / loop detector | `forge/agents/detector.py:Detector.observe/observe_prediction` |
| shared brain | `forge/agents/brain.py:Brain.add/view/prompt_view` |
| Version N+1 | `forge/agents/mechanic.py:current/trace_stats/propose/compare/score/leak_check` |
| mock brain | `forge/agents/mock.py:MockBrain` |
| CLI | `forge/agents/__main__.py` (run, probe, selftest, mission, answer) |
| tool registry, error classes, arg aliases, MANUAL | `forge/tools/registry.py:Registry.dispatch/manual` |
| arg coercion | `forge/tools/schema.py:coerce` |
| tools | `forge/tools/builtin.py` (lead-owned now) |
| quick-bench job (on target) | `forge/tools/benchjob.py:main/serve_and_measure/_Sandboxed` |
| bench runners | `forge/tools/bench.py:RemoteBenchRunner/MockBenchRunner` |
| bundles, serve command, overlay hook | `forge/tools/bundle.py:hash_dir/hash_files/serve_command/merged_args` |
| gate adapter, mock gate | `forge/tools/gateclient.py:CoreGate/MockGate/normalize_row` (box missions: `forge.fleet.remotegate.gate_for`) |
| sandboxed sh/py/kernel_test | `forge/tools/sbxrun.py:main` |
| executor protocol, local executor | `forge/tools/executor.py` |
