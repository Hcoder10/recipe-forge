"""Fake FORGE_ROOT for demoing every dashboard panel before real runs exist.

Everything written here is fake: the root gets a FIXTURE marker (the page shows a [fixture] banner) and
every free-text field starts with "[fixture]". The change texts are deliberately content-free placeholders.
Mission files follow docs/interfaces/agents.md, so the fixture exercises the same parsing as real runs.

    python -m forge.dash.fixtures --out ~/forge/dash_fixture            # (re)write, timestamps relative to now
    python -m forge.dash.fixtures --out ~/forge/dash_fixture --live     # then keep appending events
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

FX = "[fixture]"
GPU_BIG, GPU_SMALL = "RTX PRO 6000", "RTX 5090"
RECIPES = [
    ("bolt-74b-a9b-c1", "nvidia/Nemotron-Bolt-74B-A9B-NVFP4", "decode_tok_s", [GPU_BIG]),
    ("bolt-74b-a9b-tput", "nvidia/Nemotron-Bolt-74B-A9B-NVFP4", "output_tok_s", [GPU_BIG]),
    ("super-vl-120b-c1", "nvidia/Nemotron-Super-VL-120B-NVFP4", "decode_tok_s", [GPU_BIG]),
    ("nano-9b-v2-c1", "nvidia/NVIDIA-Nemotron-Nano-9B-v2", "decode_tok_s", [GPU_SMALL, GPU_BIG]),
    ("nano-9b-v2-tput", "nvidia/NVIDIA-Nemotron-Nano-9B-v2", "output_tok_s", [GPU_SMALL]),
    ("nemotron3-nano-30b-a3b-c1", "nvidia/Nemotron-3-Nano-30B-A3B", "decode_tok_s", [GPU_SMALL, GPU_BIG]),
    ("lightning-30b-a3b-c1", "nvidia/Nemotron-3.5-Lightning-30B-A3B", "decode_tok_s", [GPU_SMALL]),
]
MODEL = {r[0]: r[1] for r in RECIPES}
MACHINES = [("rig", GPU_BIG, None), ("vast-3101", GPU_BIG, 1.02), ("vast-3102", GPU_BIG, 0.98),
            ("vast-3207", GPU_SMALL, 0.41), ("vast-3208", GPU_SMALL, 0.39)]
TOOLS = ["profile", "bench", "edit", "diff", "make_bundle", "kernel_test", "request_gate", "note", "fact_add",
         "sh", "py"]
LANES = ["config", "kernel", "drafter", "scheduler", "system"]
ROLE_PRICE = {"lead": 0.012, "worker": 0.004, "verifier": 0.010, "mechanic": 0.015}
DETECTORS = [("repeat_call", "agent", "nudge"), ("read_streak", "agent", "nudge"),
             ("no_progress", "agent", "lead"), ("prediction_miss", "agent", "nudge"),
             ("tool_bug", "infra", "escalate"), ("env_blocked", "infra", "escalate")]


def _w(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1)


def _a(path: str, recs, mode: str = "a") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode, encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def _wl(path: str, recs) -> None:
    _a(path, recs, "w")


def _team(i: int) -> list[dict]:
    t = [{"name": "lead", "role": "lead", "model": "nemotron-ultra"},
         {"name": "w1", "role": "worker", "model": "nemotron-super"},
         {"name": "w2", "role": "worker", "model": "nemotron-super"}]
    if i % 2 == 0:
        t.append({"name": "w3", "role": "worker", "model": "nemotron-super"})
    t.append({"name": "verifier", "role": "verifier", "model": "nemotron-ultra"})
    return t


def _llm(rng, ts, agent) -> dict:
    p = rng.randint(6000, 40000)
    c = int(p * rng.uniform(0.5, 0.95))
    return {"t": ts, "kind": "llm_call", "agent": agent["name"], "role": agent["role"], "model": agent["model"],
            "prompt_tokens": p, "cached_tokens": c, "completion_tokens": rng.randint(80, 900),
            "cost_usd": round(ROLE_PRICE[agent["role"]] * rng.uniform(0.4, 1.6), 5)}


def _step(rng, ts, agent) -> dict:
    return {"t": ts, "kind": "tool_call", "agent": agent["name"], "role": agent["role"], "tool": rng.choice(TOOLS),
            "ok": rng.random() > 0.1}


def _decision(rng, ts, text) -> dict:
    return {"t": ts, "kind": "decision", "by": "lead", "role": "lead",
            "decision": rng.choice(["continue", "continue", "continue", "replan", "stop", "ask_human"]),
            "plan": f"{FX} {text}", "evidence": [f"f{rng.randint(1, 40)}"]}


def _detector(rng, ts, agent) -> dict:
    d, fault, act = rng.choice(DETECTORS)
    return {"t": ts, "kind": "detector", "agent": agent["name"], "signal": d, "infra": fault == "infra",
            "action": act, "detail": f"{FX} {d} on {agent['name']}"}


def _gate_result(rng, ts, gid, win) -> dict:
    return {"t": ts, "kind": "gate_result", "gate_id": gid, "verdict": "win" if win else rng.choice(["loss", "lossy"]),
            "delta_pct": round(rng.uniform(3, 12) if win else rng.uniform(-2, 1.5), 2)}


def _verdict(rng, ts, vid, machine, recipe, win) -> dict:
    """One line of a gate's verdicts.jsonl."""
    n = 6 if win else rng.choice([4, 6])
    gains = [round(rng.uniform(0.02, 0.09) if win else rng.uniform(-0.03, 0.04), 4) for _ in range(n)]
    return {"ts": ts, "verdict_id": vid, "machine": machine, "recipe": recipe, "kind": "candidate",
            "bundle_hash": f"{rng.getrandbits(64):016x}", "pairs": n, "pair_gains": gains,
            "mean_log_gain": round(sum(gains) / n, 4), "threshold": 0.018, "win": win,
            "lossless": {"ok": True if win else rng.random() > 0.3, "top1": 0.9991, "kl": 0.0004,
                         "max_regret": 0.011},
            "reason": f"{FX} " + ("all pairs positive, gain above bar" if win else "gain below bar")}


def write(out: str, seed: int = 7, now: float | None = None) -> dict:
    rng = random.Random(seed)
    now = time.time() if now is None else now
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "FIXTURE"), "w") as f:
        f.write("fake data written by forge.dash.fixtures\n")

    for rid, model, metric, hw in RECIPES:
        _w(os.path.join(out, "forge", "recipes", f"{rid}.json"),
           {"id": rid, "model": model, "engine": "vllm", "metric": metric, "hardware": hw,
            "serve": {"source": f"{FX} placeholder"}})

    machines = []
    for name, gpu, rate in MACHINES:
        if rate is None:
            continue
        started = now - rng.uniform(1.2, 3.5) * 3600
        machines.append({"name": name, "gpu_class": gpu, "rental_id": f"r-{name[-4:]}", "usd_per_hr": rate,
                         "started_at": started, "hours": 6, "status": "active"})
    machines.append({"name": "vast-2990", "gpu_class": GPU_SMALL, "rental_id": "r-2990", "usd_per_hr": 0.44,
                     "started_at": now - 5 * 3600, "returned_at": now - 2.5 * 3600, "status": "returned"})
    _w(os.path.join(out, "fleet", "machines.json"), machines)

    # finished missions: spec.json + events + row.json, as the agents write them
    finished = [("bolt-74b-a9b-c1", GPU_BIG, "rig", 182.4, 211.9, True),
                ("nano-9b-v2-c1", GPU_SMALL, "vast-3207", 251.0, 288.6, True),
                ("nano-9b-v2-c1", GPU_BIG, "vast-3101", 402.3, None, False),
                ("nemotron3-nano-30b-a3b-c1", GPU_SMALL, "vast-3208", 233.7, 247.1, True),
                ("bolt-74b-a9b-tput", GPU_BIG, "vast-3102", 1910.0, None, False)]
    gate_lines: dict[str, list] = {}
    for i, (rid, gpu, mach, base, best, win) in enumerate(finished):
        mid = f"m{i + 1:03d}"
        ts = now - rng.uniform(1800, 5400)
        v = _verdict(rng, ts, f"v{101 + i}", mach, rid, win)
        gate_lines.setdefault(mach, []).append(v)
        delta = round((best / base - 1) * 100, 2) if win else 0
        row = {"mission_id": mid, "recipe_id": rid, "machine": mach, "gpu_class": gpu, "metric": "decode_tok_s",
               "baseline": base, "best": best if win else 0, "delta_pct": delta,
               "gate_verdicts": [{"gate_id": v["verdict_id"], "verdict": "win" if win else "loss",
                                  "delta_pct": delta if win else -0.4, "pairs": v["pairs"]}],
               "change_summary": f"{FX} placeholder change {chr(65 + i)} ({rng.choice(LANES)} lane)",
               "done": {"status": "win" if win else "no_win", "met": win},
               "cost_usd": round(rng.uniform(3, 9), 2), "ended": ts, "minutes": 75}
        team = _team(i)
        _w(os.path.join(out, "missions", mid, "spec.json"),
           {"mission_id": mid, "recipe": {"id": rid, "model": MODEL[rid]},
            "machine": {"name": mach, "gpu_class": gpu}, "time_box_min": 75, "budget_usd": 15.0})
        _wl(os.path.join(out, "missions", mid, "events.jsonl"),
            [{"t": ts - 75 * 60, "kind": "mission_start", "team": team}, _llm(rng, ts - 60, team[0]),
             {"t": ts, "kind": "mission_end", "row": row}])
        _w(os.path.join(out, "missions", mid, "row.json"), row)
    _w(os.path.join(out, "results", "queue.json"),
       [{"recipe": "lightning-30b-a3b-c1", "gpu_class": GPU_SMALL},
        {"recipe": "super-vl-120b-c1", "gpu_class": GPU_BIG}])

    # live missions
    live = [("bolt-74b-a9b-c1", GPU_BIG, "rig"), ("super-vl-120b-c1", GPU_BIG, "vast-3101"),
            ("bolt-74b-a9b-tput", GPU_BIG, "vast-3102"), ("nano-9b-v2-tput", GPU_SMALL, "vast-3207"),
            ("nemotron3-nano-30b-a3b-c1", GPU_BIG, "vast-3101")]
    for i, (rid, gpu, mach) in enumerate(live):
        mid = f"m{len(finished) + i + 1:03d}"
        team = _team(i)
        started = now - rng.uniform(8, 60) * 60
        _w(os.path.join(out, "missions", mid, "spec.json"),
           {"mission_id": mid, "recipe": {"id": rid, "model": MODEL[rid]},
            "machine": {"name": mach, "gpu_class": gpu}, "time_box_min": 75, "budget_usd": 15.0,
            "team": {"workers": len(team) - 2}, "fixture_live": True})
        evs = [{"t": started, "kind": "mission_start", "team": team}]
        evs += [{"t": started + 5, "kind": "worker_task", "by": "lead", "worker": a["name"],
                 "lane": rng.choice(LANES), "task": f"{FX} task for {a['name']}"}
                for a in team if a["role"] == "worker"]
        t = started + 5
        for n in range(rng.randint(20, 60)):
            t += rng.uniform(15, 60)
            if t > now:
                break
            ag = rng.choice(team)
            evs.append(_step(rng, t, ag))
            evs.append(_llm(rng, t + 1, ag))
            if n % 9 == 4:
                evs.append(_decision(rng, t + 2, f"plan after step {n + 1}: evidence reviewed, next action set"))
            if n % 13 == 7:
                evs.append(_detector(rng, t + 3, ag))
        if i in (0, 2):
            evs.append({"t": t, "kind": "gate_submitted", "gate_id": f"q{i}", "agent": "verifier",
                        "bundle_hash": f"{rng.getrandbits(64):016x}", "claim": f"{FX} candidate bundle from {mid}"})
        _wl(os.path.join(out, "missions", mid, "events.jsonl"), evs)

    for mach, vs in gate_lines.items():
        _wl(os.path.join(out, "results", "gate", mach, "verdicts.jsonl"), vs)
    _w(os.path.join(out, "results", "gate", "rig", "queue.json"),
       {"machine": "rig", "running": {"request_id": "q0", "mission": "m006", "started_at": now - 140,
                                      "phase": "pairs 3/6"},
        "queued": [], "quiet": True})
    _w(os.path.join(out, "results", "gate", "vast-3102", "queue.json"),
       {"machine": "vast-3102", "running": None,
        "queued": [{"request_id": "q2", "mission": "m008", "ts": now - 60}], "quiet": True})
    for mach in ("vast-3101", "vast-3207", "vast-3208"):
        os.makedirs(os.path.join(out, "results", "gate", mach), exist_ok=True)

    _wl(os.path.join(out, "alerts.jsonl"), [
        {"ts": now - 900, "type": "infra", "severity": "critical", "machine": "vast-2990", "fault": "infra",
         "detector": "box_not_up", "message": f"{FX} box not up after 10 min, replaced"},
        {"ts": now - 300, "type": "infra", "severity": "warning", "machine": "rig", "fault": "infra",
         "detector": "landlord_unreachable", "message": f"{FX} landlord unreachable for 40 s, lease kept"}])

    statuses = ["measured"] * 5 + ["open"] * 7 + ["refuted"] * 3 + ["blocked"] * 2
    facts = []
    for i in range(30):
        rid = rng.choice(RECIPES)[0]
        facts.append({"fact_id": f"f{i + 1}", "t": now - (30 - i) * 170, "recipe_id": rid, "lane": rng.choice(LANES),
                      "status": rng.choice(statuses), "mission_id": f"m{rng.randint(1, 10):03d}", "by": "w1",
                      "claim": f"{FX} placeholder fact #{i + 1} about {rid}", "evidence": [f"b{i}"]})
    _wl(os.path.join(out, "brain", "facts.jsonl"), facts)

    # Version N+1 in the mechanic's log format: proposed, then promote/reject with per-recipe paired scores
    h = {k: f"{k}{rng.getrandbits(48):012x}" for k in ("pb0", "pb1", "pb2", "pb3")}
    _wl(os.path.join(out, "n1", "log.jsonl"), [
        {"t": now - 2.6 * 3600, "event": "proposed", "base": h["pb0"], "new": h["pb1"], "missions": ["m001", "m002"],
         "rationale": f"{FX} playbook revision 1", "changed": ["protocol"]},
        {"t": now - 2.5 * 3600, "event": "promote", "verdict": "promote", "old": h["pb0"], "new": h["pb1"],
         "reason": f"{FX} mean paired delta +0.85 over 2 recipes", "pairs": 2, "mean_delta": 0.85,
         "per_recipe": {"nano-9b-v2-c1": {"old": 1.5, "new": 3.2, "delta": 1.7},
                        "nemotron3-nano-30b-a3b-c1": {"old": 2.1, "new": 2.1, "delta": 0.0}}},
        {"t": now - 1.3 * 3600, "event": "proposed", "base": h["pb1"], "new": h["pb2"], "missions": ["m003"],
         "rationale": f"{FX} playbook revision 2", "changed": ["knobs"]},
        {"t": now - 1.2 * 3600, "event": "reject", "verdict": "reject", "old": h["pb1"], "new": h["pb2"],
         "reason": f"{FX} mean paired delta -0.40", "pairs": 2, "mean_delta": -0.4,
         "per_recipe": {"bolt-74b-a9b-tput": {"old": 1.0, "new": 0.5, "delta": -0.5},
                        "lightning-30b-a3b-c1": {"old": 2.3, "new": 2.0, "delta": -0.3}}},
        {"t": now - 600, "event": "proposed", "base": h["pb1"], "new": h["pb3"], "missions": ["m006", "m007"],
         "rationale": f"{FX} playbook revision 3", "changed": ["manual_notes"]}])
    return {"out": out, "missions": len(finished) + len(live)}


def live_loop(out: str, seed: int = 11, period: float = 1.5, max_bytes: int = 2_000_000) -> None:
    """Append plausible events to the live fixture missions so the SSE stream has something to show."""
    rng = random.Random(seed)
    live = []
    for d in sorted(os.listdir(os.path.join(out, "missions"))):
        try:
            with open(os.path.join(out, "missions", d, "spec.json")) as f:
                spec = json.load(f)
        except (OSError, ValueError):
            continue
        if spec.get("fixture_live"):
            live.append((d, _team(int(d[1:]) - 6)))
    n = 0
    while True:
        mid, team = rng.choice(live)
        path = os.path.join(out, "missions", mid, "events.jsonl")
        if os.path.getsize(path) > max_bytes:
            time.sleep(period)
            continue
        ag = rng.choice(team)
        now = time.time()
        n += 1
        evs = [_step(rng, now, ag), _llm(rng, now, ag)]
        r = rng.random()
        if r < 0.10:
            evs.append(_decision(rng, now, f"plan at event {n}"))
        elif r < 0.15:
            evs.append(_detector(rng, now, ag))
        elif r < 0.20:
            _a(os.path.join(out, "brain", "facts.jsonl"),
               [{"fact_id": f"f{n + 1000}", "t": now, "recipe_id": None, "mission_id": mid, "by": ag["name"],
                 "lane": rng.choice(LANES), "status": rng.choice(["open", "measured", "refuted", "blocked"]),
                 "claim": f"{FX} placeholder fact #{n + 1000} from {mid}"}])
        elif r < 0.23:
            evs.append(_gate_result(rng, now, f"g{n + 1000}", rng.random() < 0.4))
        _a(path, evs)
        time.sleep(period)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.expanduser("~/forge/dash_fixture"))
    ap.add_argument("--live", action="store_true", help="after writing, keep appending events")
    ap.add_argument("--period", type=float, default=1.5)
    args = ap.parse_args()
    out = os.path.abspath(os.path.expanduser(args.out))
    if os.path.exists(os.path.join(out, "missions")) and not os.path.exists(os.path.join(out, "FIXTURE")):
        raise SystemExit(f"{out} has missions but no FIXTURE marker: refusing to write fixtures into a real root")
    print(write(out))  # rewrites every fixture file, so timestamps are fresh
    if args.live:
        live_loop(out, period=args.period)


if __name__ == "__main__":
    main()
