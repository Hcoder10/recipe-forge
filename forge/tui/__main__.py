"""python -m forge.tui [--root ~/forge]

Live transcript of the focused mission's events.jsonl, a leaderboard strip, and a prompt:
  /missions  /mission <id>  /inbox  /answer <question_id> <text>  /leaderboard  /help  /quit
  /compare [mission|all]  /gate  /facts [mission]  /n1  /pause  /resume  /stop [reason]
  free text = an operator note to the focused mission's Lead (missions/<id>/operator_notes.jsonl)
Only stdlib (curses), so it runs with any python on the rig.
"""
from __future__ import annotations

import argparse
try:
    import curses
except ImportError:  # Data helpers and command tests also run on Windows.
    curses = None
import json
import os
import time

from forge.dash.reader import JsonlTail, Reader, _short, _ts
from forge.tui import data

ROLE_COLOR = {"lead": 1, "worker": 2, "verifier": 3, "gate": 4, "system": 5, "operator": 6, "human": 6}
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
        bits = [ev.get("gate_id"), ev.get("verdict"), ev.get("delta_pct") is not None and f"{ev['delta_pct']:+.2f}%"]
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
    return role, str(who), text


class App:
    def __init__(self, root: str):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.reader = Reader(self.root)
        self.snap: dict = {}
        self.focus: str | None = None      # mission id; None = follow the newest
        self.tail: JsonlTail | None = None
        self.lines: list[tuple[float, str, str, str]] = []
        self.buf = ""
        self.msg = "type /help for commands"
        self.view = "transcript"           # or an info panel name
        self.panel: list[str] = []
        self.scroll = 0
        self.last_poll = 0.0
        self.pending_stop: tuple[str, str] | None = None
        self.history: list[str] = []
        self.history_pos = 0
        self.history_draft = ""
        self.sent_notes = []

    # ---------------------------------------------------------------- data
    def mission_dir(self, mid: str) -> str | None:
        path = data.mission_dir(self.root, mid)
        return str(path) if path else None

    def current(self) -> str | None:
        if self.focus:
            return self.focus
        ms = self.snap.get("missions") or []
        return ms[0]["id"] if ms else None

    def poll(self) -> None:
        self.snap = self.reader.snapshot()
        mid = self.current()
        d = self.mission_dir(mid) if mid else None
        path = os.path.join(d, "events.jsonl") if d else None
        if path and (self.tail is None or self.tail.path != path):
            self.tail, self.lines = JsonlTail(path), []
        if self.tail:
            new, reset = self.tail.poll()
            if reset:
                self.lines = []
            for ev in new:
                f = fmt_event(ev)
                if f:
                    self.lines.append((_ts(ev.get("t") or ev.get("ts")) or 0, *f))
            self.lines = self.lines[-3000:]
        self.refresh_notes()

    def refresh_notes(self):
        if self.view == "notes":
            self.panel = data.notes_lines(self.root, self.current())
        if self.sent_notes:
            statuses = []
            for mission, note in self.sent_notes:
                records = data.notes_data(self.root, mission, limit=1000000)
                status = next((r["status"] for r in records if r.get("t") == note["t"]
                               and r.get("to") == note["to"]), "queued")
                statuses.append(f"{mission}: {status}")
            self.msg = "; ".join(statuses) + " | " + data.REMOTE_NOTE_LAG

    def recall(self, direction):
        if self.history_pos == len(self.history):
            self.history_draft = self.buf
        self.history_pos = max(0, min(len(self.history), self.history_pos + direction))
        self.buf = (self.history[self.history_pos] if self.history_pos < len(self.history)
                    else self.history_draft)

    def prompt(self):
        target, fleet = "lead", False
        try:
            target, _, fleet = data.parse_steer(self.buf)
        except ValueError:
            # Show the chosen recipient even before the message has been typed.
            try:
                target, _, fleet = data.parse_steer(self.buf + " message")
            except ValueError:
                pass
        return f"steer[{self.current() or 'no mission'}{' / fleet' if fleet else ''} -> {target}]> "

    def inbox(self, mid: str) -> list[dict]:
        d = self.mission_dir(mid)
        if not d:
            return []
        qs, answered = [], set()
        for name, out in (("inbox.jsonl", qs), ("answers.jsonl", None)):
            try:
                with open(os.path.join(d, name), encoding="utf-8") as f:
                    for line in f:
                        try:
                            r = json.loads(line)
                        except ValueError:
                            continue
                        if out is not None:
                            out.append(r)
                        else:
                            answered.add(str(r.get("question_id")))
            except OSError:
                pass
        return [dict(q, answered=str(q.get("question_id")) in answered) for q in qs]

    def append(self, mid: str, name: str, rec: dict) -> None:
        d = self.mission_dir(mid)
        if not d:
            raise OSError(f"no mission dir for {mid}")
        with open(os.path.join(d, name), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    # ---------------------------------------------------------------- commands
    def run_command(self, line: str) -> bool:
        line = line.strip()
        if not line:
            return True
        self.history.append(line)
        self.history_pos = len(self.history)
        self.history_draft = ""
        self.sent_notes = []
        mid = self.current()
        if self.pending_stop:
            target, reason = self.pending_stop
            if line.lower() not in ("y", "n"):
                self.msg = f"stop {target}? type y or n"
                return True
            self.pending_stop = None
            if line.lower() == "y":
                data.control_data(self.root, target, "stop", reason, confirmed=True)
                self.msg = f"stop requested for {target}"
            else:
                self.msg = "stop cancelled"
            return True
        if not line.startswith("/") or line.split(None, 1)[0] == "/steer":
            self.sent_notes = data.steer_data(self.root, mid, line)
            self.msg = "no running missions" if not self.sent_notes else "queued"
            self.refresh_notes()
            return True
        cmd, _, arg = line[1:].partition(" ")
        arg = arg.strip()
        if cmd in ("quit", "q", "exit"):
            return False
        if cmd == "help":
            self.show("help", ["/missions  /mission <id>  /inbox  /answer <question_id> <text>",
                               "/leaderboard  /compare [mission|all]  /gate  /facts [mission]  /n1",
                               "/pause [reason]  /resume [reason]  /stop [reason] (confirm y/n)",
                               "/steer [lead|worker-N|verifier|all] <text>; default: lead",
                               "@worker-2 <text>; /steer --fleet <text> sends to every RUNNING mission's lead",
                               "/notes: last 20 notes and delivery status; free text steers Lead",
                               data.REMOTE_NOTE_LAG,
                               "/help  /quit; keys: Up/Down input history, PgUp/PgDn scroll, Esc transcript"])
        elif cmd == "notes":
            self.show("notes", data.notes_lines(self.root, mid))
        elif cmd == "compare":
            fleet = arg in ("all", "fleet") or (not arg and not mid)
            target = None if fleet else arg or mid
            self.show("compare", data.compare_lines(data.compare_data(self.root, target), fleet))
        elif cmd == "gate":
            self.show("gate", [data.gate_line(r) for r in data.gate_data(self.root)] or ["no gate requests or results"])
        elif cmd == "facts":
            self.show("facts", data.facts_lines(data.facts_data(self.root, arg or mid)))
        elif cmd == "n1":
            self.show("n1", data.n1_lines(data.n1_data(self.root)))
        elif cmd in ("pause", "resume", "stop"):
            if not mid or not self.mission_dir(mid):
                self.msg = "no focused mission"
            elif cmd == "stop":
                self.pending_stop = (mid, arg)
                self.msg = f"stop {mid}? type y or n"
            else:
                data.control_data(self.root, mid, cmd, arg)
                self.msg = f"{cmd} requested for {mid}"
        elif cmd == "missions":
            out = []
            for m in self.snap.get("missions") or []:
                left = m.get("time_left_s")
                out.append(f"{m['id']:<14} {m['status']:<9} {str(m.get('recipe')):<28} {str(m.get('machine')):<16} "
                           f"${m['usd']:.2f}  {'' if left is None else f'{int(left // 60)}m left'}")
            self.show("missions", out or ["no missions yet"])
        elif cmd == "mission":
            if not arg:
                self.focus = None
                self.msg = "following the newest mission"
            elif self.mission_dir(arg):
                self.focus, self.tail = arg, None
                self.msg = f"focused on {arg}"
            else:
                self.msg = f"no mission {arg}"
            self.view = "transcript"
        elif cmd == "leaderboard":
            self.show("leaderboard", [self.lb_line(r) for r in self.snap.get("leaderboard") or []])
        elif cmd == "inbox":
            out = []
            for m in self.snap.get("missions") or []:
                for q in self.inbox(m["id"]):
                    if not q["answered"]:
                        out.append(f"{m['id']}  [{q.get('question_id')}] {q.get('by')}: {_short(q.get('question'), 150)}")
            self.show("inbox", out or ["no open questions"])
        elif cmd == "answer":
            qid, _, text = arg.partition(" ")
            if not qid or not text:
                self.msg = "usage: /answer <question_id> <text>"
                return True
            target = next((m["id"] for m in self.snap.get("missions") or []
                           if any(str(q.get("question_id")) == qid for q in self.inbox(m["id"]))), mid)
            self.append(target, "answers.jsonl", {"question_id": qid, "answer": text.strip(), "t": time.time()})
            self.msg = f"answered {qid} in {target}"
        else:
            self.msg = f"unknown command /{cmd} (try /help)"
        return True

    def show(self, name: str, lines: list[str]) -> None:
        self.view, self.panel, self.scroll = name, lines, 0

    @staticmethod
    def lb_line(r: dict) -> str:
        base = "-" if r.get("baseline_tok_s") is None else f"{r['baseline_tok_s']:.1f}"
        best = "-" if r.get("best_tok_s") is None else f"{r['best_tok_s']:.1f}"
        d = "" if r.get("delta_pct") is None else f"{r['delta_pct']:+.1f}%"
        return f"{r['recipe']:<34} {r['gpu_class']:<15} {base:>8} -> {best:>8} {d:>7}  {r['status']}"

    # ---------------------------------------------------------------- drawing
    def draw(self, scr) -> None:
        scr.erase()
        h, w = scr.getmaxyx()

        def put(y, x, s, attr=0):
            if 0 <= y < h and x < w:
                try:
                    scr.addnstr(y, x, s, max(0, w - x - 1), attr)
                except curses.error:
                    pass

        s = self.snap
        live = [m for m in s.get("missions") or [] if m["status"] in ("running", "active", "ending")]
        sp = s.get("spend") or {}
        crusoe = (sp.get("crusoe") or {}).get("total") or 0
        vast = sp.get("vast") or {}
        put(0, 0, f" recipe-forge  {time.strftime('%H:%M:%S')}  missions {len(live)} live  "
                  f"Crusoe ${crusoe:.2f}  Vast ${vast.get('burn_usd_per_hr', 0):.2f}/h (${vast.get('total_usd', 0):.2f})"
                  + ("  [FIXTURE]" if s.get("fixture") else ""), curses.A_REVERSE | curses.A_BOLD)
        lb = [r for r in s.get("leaderboard") or [] if r["status"] != "queued"][:5]
        nq = sum(1 for r in s.get("leaderboard") or [] if r["status"] == "queued")
        y = 1
        put(y, 0, f" LEADERBOARD  {'recipe':<32} {'gpu':<15} {'baseline':>8}    {'best':>8} {'delta':>7}  status",
            curses.A_DIM)
        y += 1
        for r in lb:
            col = curses.color_pair(2) if r["status"] == "verified" else curses.color_pair(1) if r["status"] == "running" else 0
            put(y, 1, "             " + self.lb_line(r), col)
            y += 1
        put(y, 1, f"             {nq} recipe x GPU queued", curses.A_DIM)
        y += 1
        mid = self.current()
        title = f" {self.view.upper()} " if self.view != "transcript" else \
            f" TRANSCRIPT {mid or '(no missions)'}{'' if self.focus else ' (newest)'} "
        put(y, 0, title.ljust(w - 1, "-"), curses.A_BOLD)
        top, bottom = y + 1, h - 3
        rows = bottom - top
        if self.view == "transcript":
            lines = self.lines
            end = len(lines) - self.scroll
            for i, (t, role, who, text) in enumerate(lines[max(0, end - rows):end]):
                yy = top + i
                stamp = time.strftime("%H:%M:%S", time.localtime(t)) if t else "--:--:--"
                tag = f"[{who}]"
                put(yy, 0, stamp, curses.A_DIM)
                put(yy, 9, tag, curses.color_pair(ROLE_COLOR.get(role, 5)) | curses.A_BOLD)
                put(yy, 10 + len(tag), text)
        else:
            for i, line in enumerate(self.panel[self.scroll:self.scroll + rows]):
                put(top + i, 1, line)
        put(h - 3, 0, "-" * (w - 1), curses.A_DIM)
        put(h - 2, 0, f" {self.msg}", curses.A_DIM)
        prompt = self.prompt()
        available = max(0, w - len(prompt) - 2)
        visible = self.buf[-available:] if available else ""
        put(h - 1, 0, prompt + visible, curses.A_BOLD)
        try:
            scr.move(h - 1, max(0, min(w - 2, len(prompt) + len(visible))))
        except curses.error:
            pass
        scr.refresh()

    def loop(self, scr) -> None:
        curses.curs_set(1)
        curses.start_color()
        curses.use_default_colors()
        for i, c in enumerate([curses.COLOR_CYAN, curses.COLOR_GREEN, curses.COLOR_MAGENTA, curses.COLOR_YELLOW,
                               curses.COLOR_WHITE, curses.COLOR_RED], start=1):
            curses.init_pair(i, c, -1)
        scr.timeout(250)
        while True:
            if time.time() - self.last_poll > 1.0:
                try:
                    self.poll()
                except Exception as e:  # keep the UI alive; show the problem
                    self.msg = f"read error: {type(e).__name__}: {e}"[:200]
                self.last_poll = time.time()
            self.draw(scr)
            try:
                ch = scr.get_wch() if hasattr(scr, "get_wch") else scr.getch()
            except curses.error:  # get_wch raises on the refresh timeout.
                continue
            if ch in (-1, curses.ERR):
                continue
            if ch in ("\n", "\r", curses.KEY_ENTER, 10, 13):
                line, self.buf = self.buf, ""
                try:
                    if not self.run_command(line):
                        return
                except Exception as e:
                    self.msg = f"error: {type(e).__name__}: {e}"[:200]
            elif ch in (curses.KEY_BACKSPACE, "\x7f", "\b", 127, 8):
                self.buf = self.buf[:-1]
            elif ch == "\x1b":
                if self.pending_stop:
                    self.pending_stop = None
                    self.msg = "stop cancelled"
                self.view, self.scroll = "transcript", 0
            elif ch == curses.KEY_PPAGE:
                self.scroll += 10
            elif ch == curses.KEY_NPAGE:
                self.scroll = max(0, self.scroll - 10)
            elif ch == curses.KEY_UP:
                self.recall(-1)
            elif ch == curses.KEY_DOWN:
                self.recall(1)
            elif ch == "?" and not self.buf:
                self.run_command("/help")
            elif isinstance(ch, str) and ch.isprintable():
                self.buf += ch


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m forge.tui")
    ap.add_argument("--root", default=os.environ.get("FORGE_ROOT", os.path.expanduser("~/forge")))
    ap.add_argument("--fixture", action="store_true", help="read ~/forge/dash_fixture (fake demo data)")
    ap.add_argument("--print", action="store_true", help="print one plain-text frame and exit (no curses)")
    args = ap.parse_args()
    root = os.path.join(args.root, "dash_fixture") if args.fixture else args.root
    if args.print:
        app = App(root)
        app.poll()
        print("LEADERBOARD")
        for r in [r for r in app.snap.get("leaderboard") or [] if r["status"] != "queued"][:8]:
            print("  " + app.lb_line(r))
        print(f"TRANSCRIPT {app.current()}")
        for t, role, who, text in app.lines[-25:]:
            print(f"{time.strftime('%H:%M:%S', time.localtime(t))} [{who}] {text}")
        return
    os.environ.setdefault("ESCDELAY", "25")
    if curses is None:
        ap.error("interactive TUI requires curses (run on Linux)")
    curses.wrapper(App(root).loop)


if __name__ == "__main__":
    main()
