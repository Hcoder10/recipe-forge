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
fmt_event = data.fmt_event


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
        self.view = "home"
        self.selected = 0
        self.fleet = []
        self.colors = False
        self.compare_target = None
        self.compare_fleet = False
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
        selected_id = self.fleet[self.selected]["id"] if self.fleet and self.selected < len(self.fleet) else None
        self.snap = data.fleet_snapshot(self.reader)
        self.fleet = data.fleet_data(self.root, self.snap)
        self.selected = next((i for i, row in enumerate(self.fleet) if row["id"] == selected_id),
                             min(self.selected, max(0, len(self.fleet) - 1)))
        if self.view == "compare":
            self.panel = data.compare_lines(data.compare_data(self.root, self.compare_target), self.compare_fleet)
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
                    if self.scroll and self.view == "transcript":
                        self.scroll += 1
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
        if cmd == "home":
            self.view, self.scroll = "home", 0
        elif cmd == "help":
            self.show("help", ["/missions  /mission <id>  /inbox  /answer <question_id> <text>",
                               "/leaderboard  /compare [mission|all]  /gate  /facts [mission]  /n1",
                               "/pause [reason]  /resume [reason]  /stop [reason] (confirm y/n)",
                               "/steer [lead|worker-N|verifier|all] <text>; default: lead",
                               "@worker-2 <text>; /steer --fleet <text> sends to every RUNNING mission's lead",
                               "/notes: last 20 notes and delivery status; free text steers Lead",
                               data.REMOTE_NOTE_LAG,
                               "/help  /quit; keys: j/k or arrows select; Enter focus; Tab complete; PgUp/PgDn scroll; Esc home; q quit"])
        elif cmd == "notes":
            self.show("notes", data.notes_lines(self.root, mid))
        elif cmd == "compare":
            fleet = arg in ("all", "fleet") or (not arg and (not mid or self.view == "home"))
            target = None if fleet else arg or mid
            self.compare_target, self.compare_fleet = target, fleet
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
    def color(self, pair):
        return curses.color_pair(pair) if self.colors else 0

    def draw(self, scr) -> None:
        scr.erase()
        h, w = scr.getmaxyx()

        def put(y, x, value, attr=0):
            if 0 <= y < h and 0 <= x < w - 1:
                try:
                    scr.addnstr(y, x, data.safe_text(value), max(0, w - x - 1), attr)
                except curses.error:
                    pass

        live = sum(r['state'] == 'running' for r in self.fleet)
        vast = data.mapping(data.mapping(self.snap.get('spend')).get('vast'))
        put(0, 0, f" FORGE / {self.view.upper()}   {live} live / {len(self.fleet)} missions   "
            f"Vast ${data.metric(vast.get('burn_usd_per_hr'))}/h", curses.A_REVERSE | curses.A_BOLD)
        top, bottom = 2, max(2, h - 4)
        if self.view == 'home':
            put(1, 0, data.fleet_header(w), curses.A_DIM)
            capacity = max(1, bottom - top)
            start = max(0, self.selected - capacity + 1)
            for i, row in enumerate(self.fleet[start:start + capacity], start):
                attr = (curses.A_REVERSE if i == self.selected else 0)
                if row['alerts']:
                    attr |= self.color(6) | curses.A_BOLD
                put(top + i - start, 0, ('>' if i == self.selected else ' ') + ('!' if row['alerts'] else ' ') + data.fleet_line(row, w), attr)
            if not self.fleet:
                put(top, 1, 'No missions yet. Waiting for state updates...')
        elif self.view == 'transcript':
            put(1, 0, f" MISSION {self.current() or '-'}   {'SCROLLBACK' if self.scroll else 'LIVE'}", curses.A_BOLD)
            mission = next((r for r in self.snap.get('missions', []) if r['id'] == self.current()), {})
            agents = data.agent_lines(mission)
            for line in agents[:max(0, min(len(agents), (bottom - top) // 3))]:
                put(top, 1, line, self.color(1))
                top += 1
            top += 1
            count = max(0, bottom - top)
            self.scroll = min(self.scroll, max(0, len(self.lines) - count))
            end = len(self.lines) - self.scroll
            for i, (stamp, role, who, message) in enumerate(self.lines[max(0, end - count):end] if count else []):
                put(top + i, 0, f"{data.clock_text(stamp)} [{who}] {message}", self.color(ROLE_COLOR.get(role, 5)))
        else:
            self.scroll = min(self.scroll, max(0, len(self.panel) - max(1, bottom - top)))
            for i, line in enumerate(self.panel[self.scroll:self.scroll + max(0, bottom - top)]):
                attr = curses.A_BOLD if ('VERIFIED' in line or '#' in line or 'DELTA' in line) else 0
                put(top + i, 0, line, attr | (self.color(2) if 'VERIFIED / GATE WIN' in line else 0))
        if self.buf.startswith('/'):
            put(h - 4, 0, '  '.join(data.completions(self.buf)) or 'Enter to run command', self.color(1))
        put(h - 3, 0, 'j/k select | Enter focus | / commands | Tab complete | PgUp/Dn scroll | Esc home | q quit', curses.A_DIM)
        detail = self.msg
        if self.view == 'home' and self.fleet and not self.buf:
            row = self.fleet[self.selected]
            detail = ' | '.join([data.text(row.get(k)) for k in ('id', 'machine', 'recipe', 'gpu_class')] + row['alerts'])
        put(h - 2, 0, detail, self.color(6) if self.view == 'home' and self.fleet and self.fleet[self.selected]['alerts'] else curses.A_DIM)
        prompt = self.prompt() if self.view != 'home' else 'fleet> '
        prompt = prompt[:max(0, w // 2)]
        available = max(0, w - len(prompt) - 2)
        visible = self.buf[-available:] if available else ''
        put(h - 1, 0, prompt + visible, curses.A_BOLD)
        try:
            scr.move(max(0, h - 1), max(0, min(w - 2, len(prompt) + len(visible))))
        except curses.error:
            pass
        scr.noutrefresh()
        curses.doupdate()

    def loop(self, scr) -> None:
        try:
            curses.curs_set(1)
        except curses.error:
            pass
        try:
            if curses.has_colors() and not os.environ.get('NO_COLOR'):
                curses.start_color()
                background = curses.COLOR_BLACK
                try:
                    curses.use_default_colors()
                    background = -1
                except curses.error:
                    pass
                for i, c in enumerate([curses.COLOR_CYAN, curses.COLOR_GREEN, curses.COLOR_MAGENTA,
                                       curses.COLOR_YELLOW, curses.COLOR_WHITE, curses.COLOR_RED], 1):
                    curses.init_pair(i, c, background)
                self.colors = True
        except curses.error:
            self.colors = False
        scr.keypad(True)
        scr.timeout(1000)
        while True:
            if time.time() - self.last_poll > 1.0:
                try:
                    self.poll()
                except Exception as e:  # keep the UI alive; show the problem
                    self.msg = f"read error: {type(e).__name__}"[:200]
                self.last_poll = time.time()
            self.draw(scr)
            try:
                ch = scr.get_wch() if hasattr(scr, "get_wch") else scr.getch()
            except curses.error:  # get_wch raises on the refresh timeout.
                continue
            if ch in (-1, curses.ERR):
                continue
            if ch in ("\n", "\r", curses.KEY_ENTER, 10, 13):
                if self.view == "home" and not self.buf and self.fleet:
                    self.run_command("/mission " + self.fleet[self.selected]["id"])
                    self.poll()
                    continue
                line, self.buf = self.buf, ""
                try:
                    if not self.run_command(line):
                        return
                except Exception as e:
                    self.msg = f"error: {type(e).__name__}"[:200]
            elif ch in (curses.KEY_BACKSPACE, "\x7f", "\b", 127, 8):
                self.buf = self.buf[:-1]
            elif ch == "\x15":
                self.buf = ""
            elif ch == "\x1b":
                if self.pending_stop:
                    self.pending_stop = None
                    self.msg = "stop cancelled"
                self.view, self.scroll = "home", 0
            elif ch == curses.KEY_PPAGE:
                self.scroll = max(0, self.scroll + (10 if self.view == "transcript" else -10))
            elif ch == curses.KEY_NPAGE:
                self.scroll = max(0, self.scroll + (-10 if self.view == "transcript" else 10))
            elif ch == 'q' and not self.buf and not self.pending_stop:
                return
            elif ch == '\t':
                choices = data.completions(self.buf)
                if choices:
                    self.buf = os.path.commonprefix(choices)
                    if len(choices) == 1:
                        self.buf += ' '
            elif ch in (curses.KEY_UP, curses.KEY_DOWN) or (self.view == 'home' and not self.buf and ch in ('j', 'k')):
                direction = -1 if ch in (curses.KEY_UP, 'k') else 1
                if self.view == 'home' and not self.buf:
                    self.selected = max(0, min(len(self.fleet) - 1, self.selected + direction))
                else:
                    self.recall(direction)
            elif ch == "?" and not self.buf:
                self.run_command("/help")
            elif isinstance(ch, str) and ch.isprintable():
                self.buf += ch


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m forge.tui")
    ap.add_argument("--root", default=os.environ.get("FORGE_ROOT", os.path.expanduser("~/forge")))
    ap.add_argument("--fixture", action="store_true", help="read ~/forge/dash_fixture (fake demo data)")
    ap.add_argument("--once", "--print", dest="once", action="store_true", help="print one plain-text frame and exit (no curses)")
    args = ap.parse_args()
    root = os.path.join(args.root, "dash_fixture") if args.fixture else args.root
    if args.once:
        app = App(root)
        app.poll()
        print("\n".join(data.home_lines(app.fleet, app.snap)))
        return
    os.environ.setdefault("ESCDELAY", "25")
    if curses is None:
        ap.error("interactive TUI requires curses (run on Linux)")
    curses.wrapper(App(root).loop)


if __name__ == "__main__":
    main()
