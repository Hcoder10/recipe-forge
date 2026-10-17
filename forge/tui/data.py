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


def safe_text(value):
    value = str(value if value is not None else "-")
    value = re.sub(r"(?i)(bearer\s+)[^\s,}]+", r"\1[REDACTED]", value)
    value = re.sub(r"(?i)([\"']?(?:api[_-]?key|access[_-]?token|token|password|secret|authorization)[\"']?\s*[:=]\s*)[\"']?[^\s,}\"']+", r"\1[REDACTED]", value)
    value = re.sub(r"(?:sk-|hf_)[A-Za-z0-9_-]{16,}", "[REDACTED]", value)
    return ''.join(c if c.isprintable() else ' ' for c in value)


def text(value):
    return " ".join(safe_text(value).split())


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
        out.append("UNVERIFIED / QUICK BENCH")
        out.append("+" + "-" * 54 + "+")
        out.append("|  DELTA  " + metric(quick.get("delta")) + "%" + " " * 8 + "quick bench (unverified)")
        out.append("+" + "-" * 54 + "+")
        out.append(f"{'REFERENCE decode tok/s':<40} | BEST CANDIDATE decode tok/s")
        out.append(f"{metric(quick.get('reference')):<40} | {metric(quick.get('candidate'))}")
        scale = max(quick.get("reference", 0), quick.get("candidate", 0), 1)
        bars = ["#" * round(36 * quick.get(k, 0) / scale) for k in ("reference", "candidate")]
        out.extend([f"{bars[0]:<40} | {bars[1]}"] * 2)
        label = "GATED (verified)" if verified(gate) else "GATED result"
        if gate.get('verdict') in ('win', 'gated_win') and verified(gate):
            out.append("VERIFIED / GATE WIN  " + metric(gate.get('delta_pct')) + "%")
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


def _short(value, limit=120):
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, default=str)
    return text(value)[:limit]


QUIET_KINDS = {"llm_call", "compaction", "done_locked"}


def _role(ev: dict) -> str:
    r = str(ev.get("role") or "").lower()
    if r:
        return r
    who = str(ev.get("agent") or ev.get("by") or ev.get("worker") or "").lower()
    for k in ("lead", "verifier", "worker"):
        if who.startswith(k):
            return k
    return "worker" if who[:1] == "w" and who[1:].isdigit() else "system"


def fmt_event(ev: dict) -> tuple[str, str, str] | None:
    """(role, tag, text) for one event, or None to hide it."""
    k = str(ev.get("kind") or ev.get("type") or "")
    if k in QUIET_KINDS:
        return None
    who = ev.get("agent") or ev.get("by") or ev.get("worker") or "system"
    role = _role(ev)
    if k == "tool_call":
        args = _short(ev.get("args"), 70)
        res = ev.get("error") or ev.get("result")
        ok = "" if ev.get("ok", True) else " FAILED"
        text = f"{ev.get('tool')} {args}{ok} -> {_short(res, 90)}" if res else f"{ev.get('tool')} {args}{ok}"
        if ev.get('tool') in ('bench', 'bench_pair') and isinstance(res, dict):
            delta = number(res.get('delta_pct'))
            text = f"{ev.get('tool')}{ok} -> delta {metric(delta)}% | {globals()['text'](res.get('verdict_hint'))} | quick / unverified"
    elif k == "decision":
        text = f"DECISION {str(ev.get('decision', '?')).upper()}: {_short(ev.get('plan') or ev.get('rationale'), 140)}"
    elif k in ("detector", "trigger", "tool_bug", "error"):
        infra = " [INFRA]" if ev.get("infra") or k in ("tool_bug", "error") else ""
        text = (f"ALERT{infra} {ev.get('signal') or ev.get('trigger_kind') or k}: "
                f"{_short(ev.get('detail') or ev.get('summary') or ev.get('error'), 120)}")
    elif k == "worker_task":
        who = ev.get("worker") or who
        role = "worker"
        text = f"task [{ev.get('lane')}] {_short(ev.get('task'), 130)}"
    elif k.startswith("gate"):
        role = "gate" if k == "gate_result" else role
        bits = [ev.get("gate_id"), ev.get("verdict"), ev.get("delta_pct") is not None and f"{number(ev['delta_pct']) or 0:+.2f}%"]
        text = f"{k.upper()} " + " ".join(str(b) for b in bits if b) + \
            (f" {_short(ev.get('claim') or ev.get('reason'), 90)}" if ev.get("claim") or ev.get("reason") else "")
    elif k == "ask_human":
        role = "human"
        text = f"ASK HUMAN [{ev.get('question_id')}] {_short(ev.get('question'), 140)}  (/answer {ev.get('question_id')} ...)"
    elif k == "human_answer":
        role = "operator"
        text = f"answer [{ev.get('question_id')}] {_short(ev.get('answer'), 140)}"
    elif k == "mission_start":
        text = "MISSION START " + _short(ev.get("playbook"), 20)
    elif k == "mission_ending":
        text = f"ending: {ev.get('reason')}"
    elif k == "mission_end":
        row = ev.get("row") or {}
        d = (row.get("done") or {}).get("status")
        text = f"MISSION END {d}: baseline {row.get('baseline')} best {row.get('best')} delta {row.get('delta_pct')}%"
    elif k == "worker_report":
        text = f"report: {_short(ev.get('report') or ev.get('summary') or ev.get('text'), 140)}"
    else:
        rest = {kk: v for kk, v in ev.items() if kk not in ("t", "kind", "mission_id", "agent", "role", "by")}
        text = f"{k} {_short(rest, 130)}"
    return role, str(who), globals()["text"](text)



COMMANDS = ('home', 'missions', 'mission', 'compare', 'gate', 'facts', 'n1', 'pause',
            'resume', 'stop', 'steer', 'notes', 'inbox', 'answer', 'leaderboard', 'help', 'quit')


def completions(prefix):
    return ['/' + command for command in COMMANDS if ('/' + command).startswith(prefix)] if prefix.startswith('/') else []


def clock_text(stamp):
    try:
        return time.strftime('%H:%M:%S', time.localtime(number(stamp) or 0))
    except (ValueError, OSError, OverflowError):
        return '--:--:--'


def agent_lines(mission):
    return [f"{text(a.get('name'))} | lane {text(a.get('lane'))} | {text(a.get('task') or a.get('tool') or 'waiting')}"
            for a in mission.get('team', []) if isinstance(a, dict)] or ['Lead | waiting for agent activity']


def fleet_data(root, snapshot, now=None):
    """Operator-only fleet projections. Never persist these into agent files."""
    now = time.time() if now is None else now
    gates = gate_data(root)
    out = []
    for mission in snapshot.get('missions', []):
        mid = mission['id']
        path = mission_dir(root, mid)
        events = rows(path / 'events.jsonl') if path else []
        benches = rows(path / 'benches.jsonl') if path else []
        best = None
        for bench in benches:
            result = mapping(bench.get('result'))
            ref = number(mapping(result.get('reference')).get('decode_tok_s'))
            cand = number(mapping(result.get('candidate')).get('decode_tok_s'))
            if ref is not None and ref > 0 and cand is not None and cand >= 0:
                delta = (cand / ref - 1) * 100
                if best is None or delta > best['delta']:
                    best = {'delta': delta, 'hint': result.get('verdict_hint') or bench.get('verdict_hint') or 'quick'}
        ended = next((e for e in reversed(events) if e.get('kind') == 'mission_end'), None)
        running = ended is None and mission.get('status') in ('running', 'active', 'ending', 'paused')
        start = number(mission.get('started_at'))
        finish = number(mapping(ended).get('t')) or number(mission.get('last_ts')) or now
        elapsed = max(0, (now if running else finish) - start) if start is not None else None
        deadline = number(mission.get('deadline'))
        box = max(0, deadline - start) if deadline is not None and start is not None else None
        recent = [e for e in events if (number(e.get('t')) or 0) >= now - 900]
        blocked = sum(e.get('kind') == 'env_blocked' or e.get('signal') == 'env_blocked' for e in recent)
        errors = sum(e.get('kind') in ('tool_bug', 'error', 'tool_error') or
                     (e.get('kind') == 'tool_call' and (e.get('ok') is False or bool(e.get('error')))) for e in recent)
        last_bench = max([number(b.get('t')) or 0 for b in benches] + [start or now])
        alerts = []
        if running:
            if blocked >= 2:
                alerts.append('env_blocked x' + str(blocked))
            if errors >= 2:
                alerts.append('tool errors x' + str(errors))
            if now - last_bench >= 900:
                alerts.append('no bench 15m')
        related = [g for g in gates if g.get('mission_id') == mid]
        gate = related[-1] if related else {}
        status = gate.get('verdict') or gate.get('status') or ('queued' if mission.get('pending_gate') else 'none')
        status = {'requested': 'queued', 'submitted': 'queued', 'gated_win': 'win', 'gated_loss': 'loss'}.get(status, status)
        out.append({**mission, 'state': 'running' if running else 'ended', 'elapsed': elapsed,
                    'box': box, 'benches': len(benches), 'quick': best, 'gate': status, 'alerts': alerts})
    return out


def duration(seconds):
    return '-' if seconds is None else f'{int(max(0, seconds) // 60)}m'


def fleet_columns(width):
    return (8, 6, 8, 5, 1, 9, 2, 10, 6, 5) if width < 100 else (13, 10, 16, 10, 7, 11, 4, 17, 7, 7)


def cells(values, width):
    def cell(value, size):
        value = text(value)
        return (value if len(value) <= size else value[:size - 1] + '~').ljust(size)
    return ' '.join(cell(v, n) for v, n in zip(values, fleet_columns(width)))


def fleet_header(width=120):
    return '  ' + cells(('MISSION', 'BOX', 'RECIPE', 'GPU', 'S', 'TIME/BOX', 'B#', 'QUICK/HINT', 'GATE', '$'), width)


def fleet_line(row, width=120):
    quick = mapping(row.get('quick'))
    delta = '-' if not quick else f"{quick['delta']:+.1f}%/{text(quick.get('hint'))}"
    state = row['state'] if width >= 100 else ('R' if row['state'] == 'running' else 'E')
    values = [row.get(k) for k in ('id', 'machine', 'recipe', 'gpu_class')]
    values += [state, f"{duration(row['elapsed'])}/{duration(row['box'])}", row['benches'], delta, row['gate'], metric(row.get('usd'))]
    return cells(values, width) + (' ! ' + ', '.join(row['alerts']) if row['alerts'] else '')


def home_lines(fleet, snapshot):
    vast = mapping(mapping(snapshot.get('spend')).get('vast'))
    return [f"FORGE FLEET | {len(fleet)} missions | Vast ${metric(vast.get('burn_usd_per_hr'))}/h",
            fleet_header()] + [fleet_line(r, 10000) for r in fleet] if fleet else ['FORGE FLEET', 'No missions yet.']


def fleet_snapshot(reader):
    snapshot = reader.snapshot()
    snapshot['missions'] = sorted((m.view(time.time()) for m in reader.missions.values()),
                                 key=lambda m: (m['status'] not in ('running', 'active', 'ending'), -(m.get('last_ts') or 0)))
    return snapshot
