"""Operator-only views. Never feed these displays back into agent context."""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path


def mapping(value):
    return value if isinstance(value, dict) else {}


def number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) and not isinstance(value, bool) else None
    except (TypeError, ValueError, OverflowError):
        return None


def rows(path):
    """Ignore incomplete final records, malformed rows and non-object JSON."""
    out = []
    try:
        with Path(path).open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if not line.endswith("\n"):
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    out.append(row)
    except OSError:
        pass
    return out


def state_root(root):
    root = Path(root).expanduser().resolve()
    return root / "state" if (root / "state" / "missions").is_dir() else root


def mission_dir(root, mid):
    if not mid or Path(mid).name != mid or mid in (".", "..") or "\\" in mid:
        return None
    path = state_root(root) / "missions" / mid
    return path if path.is_dir() and path.resolve().parent == path.parent.resolve() else None


def missions(root):
    try:
        return sorted(p.name for p in (state_root(root) / "missions").iterdir() if mission_dir(root, p.name))
    except OSError:
        return []


def metadata(root, mid):
    path = mission_dir(root, mid)
    meta = {}
    if path:
        for name in ("spec.json", "mission.json"):
            try:
                meta.update(mapping(json.loads((path / name).read_text(encoding="utf-8"))))
            except (OSError, ValueError, UnicodeError):
                pass
    recipe = meta.get("recipe_id") or meta.get("recipe")
    gpu = meta.get("gpu_class") or mapping(meta.get("machine")).get("gpu_class")
    return {"recipe": mapping(recipe).get("id") if isinstance(recipe, dict) else recipe,
            "gpu": gpu or mapping(recipe).get("gpu_class")}


def gate_data(root):
    """Fold requests, mission results and shared ledger updates into gate states."""
    records = []
    for mid in missions(root):
        path = mission_dir(root, mid)
        for row in rows(path / "gate_requests.jsonl"):
            records.append({"mission_id": mid, "status": "requested", **row,
                            "request_id": row.get("request_id") or row.get("id")})
        records.extend({"mission_id": mid, **r} for r in rows(path / "events.jsonl")
                       if r.get("kind") in ("gate_request", "gate_submitted", "gate_review", "gate_result"))
    records.extend(rows(state_root(root).parent / "gate" / "ledger.jsonl"))
    # Match the request ID to its eventual gate ID before folding any rows.
    aliases = {(str(r.get("mission_id")), str(r["request_id"])): str(r["gate_id"])
               for r in records if r.get("request_id") and r.get("gate_id")}
    folded = {}
    for i, row in enumerate(records):
        key = str(row.get("gate_id") or aliases.get((str(row.get("mission_id")), str(row.get("request_id")))))
        if key == "None":
            key = (str(row.get("mission_id")), str(row.get("request_id") or f"row-{i}"))
        folded[key] = {**folded.get(key, {}), **{k: v for k, v in row.items() if v is not None}}
    return list(folded.values())


def compare_data(root, mid=None):
    gates = gate_data(root)
    out = []
    for mission in ([mid] if mid else missions(root)):
        path = mission_dir(root, mission)
        if not path:
            continue
        best = None
        for row in rows(path / "benches.jsonl"):
            result = mapping(row.get("result"))
            ref = number(mapping(result.get("reference")).get("decode_tok_s"))
            cand = number(mapping(result.get("candidate")).get("decode_tok_s"))
            if ref is not None and ref > 0 and cand is not None and cand >= 0:
                if best is None or cand > best["candidate"]:
                    best = {"reference": ref, "candidate": cand, "delta": (cand / ref - 1) * 100}
        results = [r for r in gates if r.get("mission_id") == mission and r.get("verdict")]
        measured = [r for r in results if verified(r)]
        gate = max(measured, key=lambda r: number(r.get("delta_pct")),
                   default=results[-1] if results else None)
        out.append({"mission": mission, **metadata(root, mission), "quick": best, "gate": gate})
    def rank(row):
        delta = number(mapping(row["gate"]).get("delta_pct")) if verified(row["gate"]) else None
        quick = mapping(row["quick"]).get("delta")
        return (delta if delta is not None else -math.inf,
                quick if quick is not None else -math.inf)
    return sorted(out, key=rank, reverse=True)


def facts_data(root, mid=None):
    facts = rows(state_root(root) / "brain" / "facts.jsonl")
    recipe = metadata(root, mid)["recipe"] if mid else None
    # Like Brain.view: prioritize this recipe, while keeping the shared store visible.
    return sorted(facts, key=lambda r: (0 if recipe and r.get("recipe_id") == recipe else 1,
                                        -(number(r.get("t")) or 0)))


def n1_data(root, limit=12):
    """Root-scoped equivalent of winlib.reuse_stats; never change agent configuration."""
    path = state_root(root) / "winlib"
    entries = [r for r in rows(path / "entries.jsonl") if r.get("entry_id")]
    reuse = [r for r in rows(path / "reuse.jsonl") if r.get("entry_id") and number(r.get("delta_pct")) is not None]
    tried = {r["entry_id"] for r in reuse}
    stats = {"entries": len(entries), "missions": len({r["mission_id"] for r in entries + reuse if r.get("mission_id")}),
             "reuse_rate": sum(r["entry_id"] in tried for r in entries) / len(entries) if entries else 0,
             "help_rate": sum(number(r["delta_pct"]) > 0 for r in reuse) / len(reuse) if reuse else 0}
    def rank(row):
        outcome = mapping(row.get("outcome"))
        delta = number(outcome.get("delta_pct"))
        return (outcome.get("kind") == "gated_win", delta if delta is not None else -math.inf)
    return {"stats": stats, "entries": sorted(entries, key=rank, reverse=True)[:max(0, limit)]}


def control_data(root, mid, action, reason="", *, confirmed=False):
    if action not in ("pause", "resume", "stop"):
        raise ValueError("unknown control action")
    if action == "stop" and not confirmed:
        raise ValueError("stop requires confirmation")
    path = mission_dir(root, mid)
    if path is None:
        raise OSError("no focused mission directory")
    row = {"t": time.time(), "by": "operator", "action": action, "reason": reason}
    with (path / "control.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
    return row


REMOTE_NOTE_LAG = "Remote delivery status may lag ~60 s while events are pulled."


def steering_target(value):
    return value in ("lead", "verifier", "all") or bool(re.fullmatch(r"worker-[0-9]+", value))


def parse_steer(line):
    """Return (target, text, fleet) for free text or explicit steering syntax."""
    line = line.strip()
    target, fleet = "lead", False
    parts = line.split(None, 1)
    if parts and parts[0] == "/steer":
        line = parts[1].strip() if len(parts) == 2 else ""
        parts = line.split(None, 1)
        if parts and parts[0] == "--fleet":
            fleet = True
            line = parts[1].strip() if len(parts) == 2 else ""
        elif parts and steering_target(parts[0]):
            target = parts[0]
            line = parts[1].strip() if len(parts) == 2 else ""
        elif parts and parts[0].startswith(("--", "worker-", "@")):
            raise ValueError("invalid steering target or option")
    elif line.startswith("@"):
        target = parts[0][1:]
        if not steering_target(target):
            raise ValueError("invalid steering target")
        line = parts[1].strip() if len(parts) == 2 else ""
    elif line.startswith("/"):
        raise ValueError("not a steering command")
    if not line:
        raise ValueError("usage: /steer [lead|worker-N|verifier|all] <text> or /steer --fleet <text>")
    return target, line, fleet


def steer_data(root, mid, line):
    """Append only operator-authored text; never include dashboard data in notes."""
    target, message, fleet = parse_steer(line)
    selected = [m for m in missions(root)
                if not any(r.get("kind") == "mission_end"
                           for r in rows(mission_dir(root, m) / "events.jsonl"))] if fleet else [mid]
    sent = []
    for mission in selected:
        path = mission_dir(root, mission)
        if path is None:
            raise OSError("no focused mission directory")
        row = {"t": time.time(), "by": "operator", "to": target, "text": message}
        with (path / "operator_notes.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        sent.append((mission, row))
    return sent


def note_deliveries(notes, events):
    """FIFO matching: each acknowledgement consumes at most one earlier note.

    The runner only logs the requested target, not individual recipients for all.
    Text is deliberately never needed for matching.
    """
    statuses = ["queued"] * len(notes)
    for event in events:
        stamp = number(event.get("t"))
        if event.get("kind") != "operator_note" or stamp is None:
            continue
        for i, note in enumerate(notes):
            queued = number(note.get("t"))
            if (statuses[i] == "queued" and queued is not None and stamp > queued
                    and event.get("to") == note.get("to")):
                statuses[i] = "delivered to " + str(event["to"])
                break
    return statuses


def notes_data(root, mid, limit=20):
    path = mission_dir(root, mid)
    if path is None:
        return []
    notes = rows(path / "operator_notes.jsonl")
    statuses = note_deliveries(notes, rows(path / "events.jsonl"))
    return [dict(note, status=status) for note, status in zip(notes, statuses)][-limit:] if limit > 0 else []


def notes_lines(root, mid):
    return [REMOTE_NOTE_LAG] + [
        f"{note.get('t')} to {text(note.get('to'))}: {text(note.get('text'))} [{note['status']}]"
        for note in notes_data(root, mid)]


def text(value):
    return " ".join(str(value if value is not None else "-").split())


def metric(value):
    value = number(value)
    return "-" if value is None else f"{value:.2f}"


def verified(row):
    row = mapping(row)
    return row.get("verdict") in ("win", "loss", "gated_win", "gated_loss") and number(row.get("delta_pct")) is not None


def gate_line(row):
    ci = next((row[k] for k in ("ci", "ci95", "ci_pct", "confidence_interval") if row.get(k) is not None), None)
    if ci is None and ("ci_low" in row or "ci_high" in row):
        ci = [row.get("ci_low"), row.get("ci_high")]
    return (f"{text(row.get('mission_id'))}  {text(row.get('gate_id') or row.get('request_id'))}  "
            f"{text(row.get('verdict') or row.get('status') or 'requested')}  "
            f"delta {metric(row.get('delta_pct'))}%  CI {text(ci)}")


def compare_lines(data, fleet=False):
    out = []
    for row in data:
        quick, gate = mapping(row["quick"]), mapping(row["gate"])
        out.append(f"{row['mission']}  recipe {text(row['recipe'])}  GPU {text(row['gpu'])}")
        if fleet:
            label = "GATED (verified)" if verified(gate) else "GATED (no verified measurement)"
            out[-1] += (f" | {label} {metric(gate.get('delta_pct'))}%"
                        f" | quick bench (unverified) {metric(quick.get('delta'))}%")
            continue
        out.append("quick bench (unverified)  delta " + metric(quick.get("delta")) + "%")
        out.append(f"{'REFERENCE decode tok/s':<40} | BEST CANDIDATE decode tok/s")
        out.append(f"{metric(quick.get('reference')):<40} | {metric(quick.get('candidate'))}")
        scale = max(quick.get("reference", 0), quick.get("candidate", 0), 1)
        bars = ["#" * round(36 * quick.get(k, 0) / scale) for k in ("reference", "candidate")]
        out.extend([f"{bars[0]:<40} | {bars[1]}"] * 2)
        label = "GATED (verified)" if verified(gate) else "GATED result"
        out.append(label + "  " + gate_line(gate) if gate else "No gated measurement yet")
        out.append("")
    return out or ["no missions yet"]


def facts_lines(data):
    return [f"{text(r.get('fact_id'))} [{text(r.get('status'))}/{text(r.get('lane'))}] "
            f"{text(r.get('claim'))} | evidence {text(r.get('evidence'))} | "
            f"{text(r.get('recipe_id'))} / {text(r.get('gpu_class'))}" for r in data] or ["no shared facts yet"]


def n1_lines(data):
    stats = data["stats"]
    out = [f"entries {stats['entries']}  missions {stats['missions']}  "
           f"reuse rate {stats['reuse_rate']:.1%}  help rate {stats['help_rate']:.1%}"]
    for row in data["entries"]:
        outcome = mapping(row.get("outcome"))
        traits = mapping(row.get("traits"))
        where = {k: traits[k] for k in ("architecture", "layer_mix", "quantization", "drafter", "concurrency")
                 if traits.get(k) is not None}
        out.append(f"{text(outcome.get('kind'))} {metric(outcome.get('delta_pct'))}% | {text(row.get('lane'))} | "
                   f"{text(mapping(row.get('change')).get('summary'))} | where "
                   f"{text(row.get('mission_id'))} / {text(row.get('recipe_id'))} / {text(row.get('gpu_class'))} "
                   f"{text(where) if where else ''}")
    return out
