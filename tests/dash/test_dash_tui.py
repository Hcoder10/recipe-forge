"""TUI logic without curses: event one-liners, notes and answers land in the agreed files."""
import json

import tempfile
import unittest
from pathlib import Path

from forge.tui import data

from forge.dash import fixtures  # noqa: E402
from forge.tui.__main__ import App, fmt_event  # noqa: E402


def test_fmt_event_one_liners():
    role, who, text = fmt_event({"kind": "tool_call", "agent": "w2", "role": "worker", "tool": "bench",
                                 "args": {"config_delta": {"x": 1}}, "result": "+2.1% vs same-GPU ref"})
    assert (role, who) == ("worker", "w2") and text.startswith("bench") and "+2.1%" in text
    assert fmt_event({"kind": "llm_call"}) is None
    assert "REPLAN" in fmt_event({"kind": "decision", "by": "lead", "decision": "replan", "plan": "p"})[2]
    assert fmt_event({"kind": "ask_human", "question_id": "q1", "question": "?"})[0] == "human"


def test_notes_answers_and_views(tmp_path):
    root = str(tmp_path / "fx")
    fixtures.write(root)
    app = App(root)
    app.poll()
    assert app.current() and app.lines
    mid = app.current()
    app.run_command("please check the reference first")
    (note,) = data.rows(Path(root) / "missions" / mid / "operator_notes.jsonl")
    assert note["text"] == "please check the reference first" and note["to"] == "lead"
    with open(f"{root}/missions/{mid}/inbox.jsonl", "w") as f:
        f.write(json.dumps({"question_id": "q7", "by": "lead", "question": "use GPU 2?"}) + "\n")
    app.run_command("/inbox")
    assert any("q7" in line for line in app.panel)
    app.run_command("/answer q7 yes, work GPU only")
    (ans,) = data.rows(Path(root) / "missions" / mid / "answers.jsonl")
    assert ans == {"question_id": "q7", "answer": "yes, work GPU only", "t": ans["t"]}
    app.run_command("/inbox")
    assert app.panel == ["no open questions"]
    app.run_command("/missions")
    assert len(app.panel) == 10
    app.run_command("/mission m001")
    app.poll()
    assert app.current() == "m001" and any("MISSION END" in x[3] for x in app.lines)
    assert app.run_command("/quit") is False


class ExistingTuiTests(unittest.TestCase):
    def test_event_formatting(self):
        test_fmt_event_one_liners()

    def test_existing_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            test_notes_answers_and_views(Path(tmp))


class CommandDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "state"
        self.mission = self.root / "missions" / "a"
        self.mission.mkdir(parents=True)
        (self.mission / "spec.json").write_text(json.dumps({
            "recipe_id": "synthetic-recipe", "gpu_class": "synthetic-gpu"}), encoding="utf-8")

    def write_rows(self, path, records):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    def test_steering_target_parsing(self):
        for line, expected in [
            ("check logs", ("lead", "check logs", False)),
            ("/steer check logs", ("lead", "check logs", False)),
            ("/steer lead check logs", ("lead", "check logs", False)),
            ("/steer worker-2 check logs", ("worker-2", "check logs", False)),
            ("@worker-2 check logs", ("worker-2", "check logs", False)),
            ("/steer verifier check logs", ("verifier", "check logs", False)),
            ("/steer all check logs", ("all", "check logs", False)),
            ("/steer --fleet check logs", ("lead", "check logs", True)),
            (" /steer\tworker-2\tcheck logs ", ("worker-2", "check logs", False)),
        ]:
            with self.subTest(line=line):
                self.assertEqual(data.parse_steer(line), expected)
        for line in ("", "/steer", "/steer all", "@worker-2", "@unknown hello",
                     "/steer worker-x hello", "/steer --bad hello", "/steer --fleet"):
            with self.subTest(line=line), self.assertRaises(ValueError):
                data.parse_steer(line)

    def test_steering_fleet_fanout_and_append_contract(self):
        for mid in ("b", "ended"):
            (self.root / "missions" / mid).mkdir()
        self.write_rows(self.root / "missions" / "ended" / "events.jsonl",
                        [{"kind": "mission_end"}, {"kind": "other"}])
        self.write_rows(self.mission / "events.jsonl", [{"kind": "mission_start"}])
        first = data.steer_data(self.root, "a", "@worker-2 check logs")
        sent = data.steer_data(self.root.parent, None, "/steer --fleet review logs")
        self.assertEqual([mid for mid, _ in sent], ["a", "b"])
        self.assertEqual(data.rows(self.mission / "operator_notes.jsonl"), [first[0][1], sent[0][1]])
        self.assertFalse((self.root / "missions" / "ended" / "operator_notes.jsonl").exists())
        for _, row in sent:
            self.assertEqual(set(row), {"t", "by", "to", "text"})
            self.assertEqual((row["by"], row["to"], row["text"]), ("operator", "lead", "review logs"))
            self.assertIsInstance(row["t"], float)
        with self.assertRaises(OSError):
            data.steer_data(self.root, "missing", "check logs")

    def test_delivery_matching_consumes_events_once(self):
        notes = [{"t": 10, "to": "lead"}, {"t": 11, "to": "lead"},
                 {"t": 12, "to": "worker-2"}, {"t": 12, "to": "all"}]
        events = [{"kind": "operator_note", "t": 9, "to": "lead"},
                  {"kind": "operator_note", "t": 10, "to": "lead"},
                  {"kind": "other", "t": 13, "to": "lead"},
                  {"kind": "operator_note", "t": "nan", "to": "lead"},
                  {"kind": "operator_note", "t": 13, "to": "verifier"},
                  {"kind": "operator_note", "t": 13, "to": "lead"},
                  {"kind": "operator_note", "t": 14, "to": "worker-2"}]
        self.assertEqual(data.note_deliveries(notes, events),
                         ["delivered to lead", "queued", "delivered to worker-2", "queued"])
        events.extend([{"kind": "operator_note", "t": 15, "to": target} for target in ("lead", "all")])
        self.assertEqual(data.note_deliveries(notes, events),
                         ["delivered to lead", "delivered to lead", "delivered to worker-2", "delivered to all"])

    def test_notes_match_before_last_twenty_limit(self):
        notes = [{"t": i, "to": "lead", "text": "review logs"} for i in range(25)]
        self.write_rows(self.mission / "operator_notes.jsonl", notes)
        self.write_rows(self.mission / "events.jsonl", [{"kind": "operator_note", "t": 30, "to": "lead"}])
        recent = data.notes_data(self.root, "a")
        self.assertEqual(len(recent), 20)
        self.assertEqual(recent[0]["t"], 5)
        self.assertTrue(all(r["status"] == "queued" for r in recent))

    def test_steering_ui_history_prompt_and_live_delivery(self):
        app = App(str(self.root))
        app.focus = "a"
        app.buf = "@worker-2 check logs"
        self.assertIn("a -> worker-2", app.prompt())
        app.run_command(app.buf)
        self.assertIn("queued", app.msg)
        self.assertIn("~60 s", app.msg)
        note = data.rows(self.mission / "operator_notes.jsonl")[0]
        self.write_rows(self.mission / "events.jsonl", [
            {"kind": "operator_note", "t": note["t"] + 1, "to": "worker-2"}])
        app.refresh_notes()
        self.assertIn("delivered to worker-2", app.msg)
        app.buf = "draft"
        app.recall(-1)
        self.assertEqual(app.buf, "@worker-2 check logs")
        app.recall(-1)
        self.assertEqual(app.buf, "@worker-2 check logs")
        app.recall(1)
        self.assertEqual(app.buf, "draft")
        app.run_command("/notes")
        self.assertTrue(any("delivered to worker-2" in line for line in app.panel))
        app.run_command("/help")
        self.assertTrue(any("/steer" in line for line in app.panel))

    def bench(self, ref, cand):
        return {"result": {"reference": {"decode_tok_s": ref},
                           "candidate": {"decode_tok_s": cand}}}

    def test_compare_best_paired_reference_and_bars(self):
        self.write_rows(self.mission / "benches.jsonl", [self.bench(100, 120), self.bench(200, 220),
                        self.bench(0, 999), self.bench(100, "nan"), {"result": []}])
        result = data.compare_data(self.root, "a")
        self.assertTrue(result[0]["quick"]["reference"] == 200)
        self.assertTrue(result[0]["quick"]["candidate"] == 220)
        self.assertTrue(abs(result[0]["quick"]["delta"] - 10) < 1e-9)
        self.assertTrue(result[0]["recipe"] == "synthetic-recipe")
        lines = data.compare_lines(result)
        self.assertTrue(any("quick bench (unverified)" in line for line in lines))
        self.assertTrue(any("#" in line and "|" in line for line in lines))
        self.assertFalse(any("GATED (verified)" in line for line in lines))

    def test_compare_fleet_verified_then_quick(self):
        for mid in ("b", "c"):
            (self.root / "missions" / mid).mkdir()
        for mid, cand in (("a", 150), ("b", 120), ("c", 130)):
            self.write_rows(self.root / "missions" / mid / "benches.jsonl", [self.bench(100, cand)])
        self.write_rows(self.mission / "events.jsonl", [{"kind": "gate_result", "gate_id": "g1",
                        "verdict": "loss", "delta_pct": -1}])
        self.write_rows(self.root.parent / "gate" / "ledger.jsonl", [
            {"gate_id": "g2", "mission_id": "b", "status": "queued"},
            {"gate_id": "g2", "verdict": "win", "delta_pct": 2}])
        result = data.compare_data(self.root)
        self.assertTrue([r["mission"] for r in result] == ["b", "a", "c"])
        self.assertTrue(len(data.compare_lines(result, fleet=True)) == 3)
        self.assertTrue(any("GATED (verified)" in line for line in data.compare_lines(result[:1])))

    def test_gate_requests_results_ci_and_partial_updates(self):
        self.write_rows(self.mission / "gate_requests.jsonl", [{"id": "r1"}, {"id": "r2"}])
        self.write_rows(self.mission / "events.jsonl", [{"kind": "gate_result", "request_id": "r1",
                        "gate_id": "g1", "verdict": "win", "delta_pct": 2}])
        self.write_rows(self.root.parent / "gate" / "ledger.jsonl", [
            {"gate_id": "g1", "ci": [1, 3]}, {"gate_id": "g1", "delta_pct": None}])
        result = data.gate_data(self.root)
        self.assertTrue(len(result) == 2)
        done = next(r for r in result if r.get("gate_id") == "g1")
        self.assertTrue(done["delta_pct"] == 2 and done["mission_id"] == "a")
        self.assertTrue("CI [1, 3]" in data.gate_line(done))
        self.assertTrue(any(r.get("status") == "requested" for r in result))

    def test_facts_shared_recipe_priority(self):
        self.write_rows(self.root / "brain" / "facts.jsonl", [
            {"fact_id": "f1", "recipe_id": "elsewhere", "t": 9, "claim": "synthetic"},
            {"fact_id": "f2", "recipe_id": "synthetic-recipe", "t": 1, "claim": "synthetic"}])
        self.assertTrue(data.facts_data(self.root, "a")[0]["fact_id"] == "f2")
        self.assertTrue(data.facts_data(self.root)[0]["fact_id"] == "f1")
        self.assertTrue(len(data.facts_lines(data.facts_data(self.root))) == 2)

    def test_invalid_gate_is_visible_but_not_verified(self):
        self.write_rows(self.mission / "events.jsonl", [
            {"kind": "gate_result", "gate_id": "bad", "verdict": "invalid"}])
        result = data.compare_data(self.root, "a")
        lines = data.compare_lines(result)
        self.assertTrue(any("invalid" in line for line in lines))
        self.assertFalse(any("GATED (verified)" in line for line in lines))

    def test_pending_gate_links_submission_to_request(self):
        self.write_rows(self.mission / "gate_requests.jsonl", [{"id": "r1"}])
        self.write_rows(self.mission / "events.jsonl", [
            {"kind": "gate_submitted", "gate_id": "g1", "request_id": "r1"}])
        self.write_rows(self.root.parent / "gate" / "ledger.jsonl", [
            {"gate_id": "g1", "status": "running"}])
        result = data.gate_data(self.root)
        self.assertTrue(len(result) == 1 and result[0]["status"] == "running")

    def test_n1_stats_and_top_entries(self):
        self.write_rows(self.root / "winlib" / "entries.jsonl", [
            {"entry_id": "e1", "mission_id": "a", "outcome": {"kind": "quick_gain", "delta_pct": 3}},
            {"entry_id": "e2", "mission_id": "b", "outcome": {"kind": "gated_win", "delta_pct": 2}}])
        self.write_rows(self.root / "winlib" / "reuse.jsonl", [
            {"entry_id": "e1", "mission_id": "c", "delta_pct": 1},
            {"entry_id": "e1", "mission_id": "c", "delta_pct": -1}])
        result = data.n1_data(self.root)
        self.assertTrue(result["stats"] == {"entries": 2, "missions": 3, "reuse_rate": .5, "help_rate": .5})
        self.assertTrue(result["entries"][0]["entry_id"] == "e2")
        self.assertTrue("reuse rate" in data.n1_lines(result)[0])

    def test_missing_malformed_and_partial_files(self):
        path = self.mission / "benches.jsonl"
        path.write_bytes(b'null\n[]\ninvalid\n{}\n{"partial":')
        self.assertTrue(data.rows(path) == [{}])
        self.assertTrue(data.compare_data(self.root, "a")[0]["quick"] is None)
        self.assertTrue(data.compare_data(self.root, "missing") == [])
        self.assertTrue(data.gate_data(self.root) == [])
        self.assertTrue(data.facts_data(self.root) == [])
        self.assertTrue(data.n1_data(self.root)["stats"]["entries"] == 0)
        self.assertTrue(data.mission_dir(self.root, "../a") is None)
        self.assertTrue(data.state_root(self.root.parent) == self.root)

    def test_pause_resume_stop_contract(self):
        for action in ("pause", "resume"):
            data.control_data(self.root, "a", action, "operator reason")
        with self.assertRaises(ValueError):
            data.control_data(self.root, "a", "stop")
        data.control_data(self.root, "a", "stop", confirmed=True)
        records = data.rows(self.mission / "control.jsonl")
        self.assertTrue([r["action"] for r in records] == ["pause", "resume", "stop"])
        self.assertTrue(all(set(r) == {"t", "by", "action", "reason"} and r["by"] == "operator"
                            and isinstance(r["t"], float) for r in records))
        self.assertFalse((self.mission / "operator_notes.jsonl").exists())
        with self.assertRaises(OSError):
            data.control_data(self.root, "missing", "pause")

    def test_command_routing_and_stop_confirmation(self):
        app = App(str(self.root))
        app.focus = "a"
        for command in ("compare", "compare all", "facts", "gate", "n1"):
            self.assertTrue(app.run_command("/" + command))
            self.assertTrue(bool(app.panel))
        app.run_command("/pause reason")
        app.run_command("/resume")
        app.run_command("/stop reason")
        app.run_command("not an answer")
        self.assertTrue(len(data.rows(self.mission / "control.jsonl")) == 2)
        app.run_command("n")
        app.run_command("/stop reason")
        app.focus = "missing"
        app.run_command("y")
        self.assertTrue(data.rows(self.mission / "control.jsonl")[-1]["action"] == "stop")
        self.assertFalse((self.mission / "operator_notes.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
