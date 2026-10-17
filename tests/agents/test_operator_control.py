"""Operator controls and offline decision labels, without LLM or GPU work."""
import importlib
import inspect
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from forge.agents import events
from forge.agents.decisions import Decisions, labels_for
from forge.agents.mission import GateQueue, MissionRunner


class OperatorControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.r = MissionRunner.__new__(MissionRunner)
        self.r.dir = Path(self.tmp.name)
        self.r.finished = threading.Event()
        self.r.paused = threading.Event()
        self.r._operator_offsets = {}
        self.r.agents = {n: Mock() for n in ("lead", "worker-1", "worker-2", "verifier")}
        self.r.wakers = {n: threading.Event() for n in self.r.agents}
        self.r.log = events.EventLog(self.r.dir / "events.jsonl", "test")
        self.r.end_reason = ""

    def control(self, action):
        events.append(self.r.dir / "control.jsonl", {"by": "operator", "action": action, "reason": "test"})
        self.r._poll_operator()

    def test_pause_resume_stop(self):
        self.control("pause")
        completed = threading.Event()
        waiter = threading.Thread(target=lambda: (self.r._wait_unpaused(), completed.set()), daemon=True)
        waiter.start()
        self.assertFalse(completed.wait(0.1))
        self.control("resume")
        self.assertTrue(completed.wait(1))
        waiter.join(1)
        self.assertTrue(all(e.is_set() for e in self.r.wakers.values()))
        self.control("pause")
        self.control("stop")
        self.assertFalse(self.r._wait_unpaused())
        self.assertEqual(self.r.end_reason, "operator stop: test")
        kinds = [r["kind"] for r in events.read(self.r.dir / "events.jsonl")]
        self.assertEqual(kinds, ["operator_pause", "operator_resume", "operator_pause",
                                 "operator_stop", "mission_ending"])

    def test_notes_are_addressed_and_consumed_once(self):
        for target in ("lead", "worker-1", "all"):
            events.append(self.r.dir / "operator_notes.jsonl",
                          {"by": "operator", "to": target, "text": "Please review the task."})
        self.r._poll_operator()
        self.r._poll_operator()
        for name, agent in self.r.agents.items():
            self.assertEqual(agent.add_event_message.call_count, 2 if name in ("lead", "worker-1") else 1)
            agent.add_event_message.assert_called_with("OPERATOR NOTE: Please review the task.")
            self.assertTrue(self.r.wakers[name].is_set())
        self.assertEqual(len(events.read(self.r.dir / "events.jsonl")), 3)

    def test_partial_and_invalid_rows(self):
        path = self.r.dir / "control.jsonl"
        path.write_bytes(b'not json\n[]\n{"by":"operator","action":"pause"')
        self.r._poll_operator()
        self.assertFalse(self.r.paused.is_set())
        with path.open("ab") as out:
            out.write(b'}\n')
        self.r._poll_operator()
        self.assertTrue(self.r.paused.is_set())

    def test_worker_and_lead_do_not_step_while_paused(self):
        self.r.team_cfg = SimpleNamespace(lead_every_s=0.01, worker_step_gap_s=0)
        self.r.services = SimpleNamespace(team=SimpleNamespace(task_of=lambda name: {"task": "review"}))
        self.r.decisions = Decisions(self.r.dir / "decisions.jsonl", self.r.log)
        self.r._status = lambda agent: "status"
        self.r.agents["lead"].step.side_effect = lambda status: (self.r.end("test") or {"called": []})
        self.r.agents["worker-1"].step.side_effect = lambda status: (self.r.end("test") or {"called": []})
        for loop in (self.r._lead_loop, lambda: self.r._worker_loop("worker-1")):
            self.r.finished.clear()
            self.control("pause")
            for agent in self.r.agents.values():
                agent.reset_mock()
            thread = threading.Thread(target=loop, daemon=True)
            thread.start()
            self.assertFalse(self.r.finished.wait(0.1))
            self.assertFalse(any(a.step.called for a in self.r.agents.values()))
            self.control("resume")
            thread.join(1)
            self.assertFalse(thread.is_alive())

    def test_pause_allows_inflight_step_to_finish(self):
        entered, release, returned = threading.Event(), threading.Event(), threading.Event()
        self.r.team_cfg = SimpleNamespace(worker_step_gap_s=0)
        self.r.services = SimpleNamespace(team=SimpleNamespace(task_of=lambda name: {"task": "review"}))
        self.r._status = lambda agent: "status"

        def step(status):
            entered.set()
            release.wait(2)
            returned.set()
            return {"called": ["read"]}

        self.r.agents["worker-1"].step.side_effect = step
        thread = threading.Thread(target=lambda: self.r._worker_loop("worker-1"), daemon=True)
        thread.start()
        try:
            self.assertTrue(entered.wait(1))
            self.control("pause")
            release.set()
            self.assertTrue(returned.wait(1))
            self.assertFalse(self.r.finished.wait(0.1))
            self.assertEqual(self.r.agents["worker-1"].step.call_count, 1)
        finally:
            release.set()
            self.control("stop")
            thread.join(1)
        self.assertFalse(thread.is_alive())


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.decisions = Decisions(self.directory / "decisions.jsonl", Mock())

    def test_decision_context_options_and_citations(self):
        tid = self.decisions.trigger("time", "Review remaining work")
        self.decisions.set_context("lead", "s" * 7000, minutes_left=3, pending_candidates=[])
        self.decisions.decide("lead", "stop", [tid], "Reviewed B12, T1, f-0123456789 and B12")
        row = self.decisions.rows[0]
        self.assertEqual(len(row["context"]), 6000)
        self.assertEqual(row["options"], [{"id": tid, "kind": "time"}])
        self.assertEqual(row["action"], "stop")
        self.assertEqual(row["evidence_ids"], ["B12", "T1", "f-0123456789"])
        self.assertEqual(labels_for(self.directory)[0]["label"], "good")

    def test_verifier_records_review(self):
        runner = SimpleNamespace(dir=self.directory, decisions=self.decisions, log=Mock(), agents={})
        queue = GateQueue(runner)
        queue.reqs["R1"] = {"status": "pending_review", "bundle": "bundle-1", "hash": "hash-1", "by": "worker-1"}
        self.decisions.set_context("verifier", "Requests pending review: R1.")
        queue.review(SimpleNamespace(agent="verifier"), "R1", False, "Review B1")
        row = self.decisions.rows[0]
        self.assertEqual(row["action"], "reject")
        self.assertEqual(row["options"], ["R1"])
        self.assertEqual(row["evidence_ids"], ["B1"])
        self.assertEqual(row["context"], "Requests pending review: R1.")

    def test_labels_use_later_linked_facts(self):
        rows = [
            {"action": "approve", "request_id": "R1"},
            {"action": "approve", "request_id": "R2"},
            {"action": "approve", "request_id": "R3"},
            {"action": "reject", "request_id": "R4", "bundle_hash": "same"},
            {"action": "approve", "request_id": "R6"},
            {"action": "stop", "minutes_left": 1, "pending_candidates": ["R7"]},
            {"action": "stop", "minutes_left": 0, "pending_candidates": []},
            {"action": "stop", "minutes_left": 1},
        ]
        for row in rows:
            events.append(self.directory / "decisions.jsonl", {"t": 10, **row})
        events.append(self.directory / "gate_requests.jsonl",
                      {"event": "submit", "id": "R5", "hash": "same"})
        for rid, verdict, timestamp in [("R1", "win", 11), ("R2", "loss", 11), ("R3", "lossy", 11),
                                         ("R5", "win", 11), ("R6", "win", 9), ("R7", "win", 11)]:
            events.append(self.directory / "events.jsonl",
                          {"t": timestamp, "kind": "gate_result", "request_id": rid, "verdict": verdict})
        labeled = labels_for(self.directory)
        self.assertEqual([r["label"] for r in labeled],
                         ["good", "bad", "bad", "bad", "unknown", "unknown", "unknown", "unknown"])
        self.assertEqual(labels_for(self.directory), labeled)
        self.assertEqual(events.read(self.directory / "decisions_labeled.jsonl"), labeled)


def load_tests(loader, suite, pattern):
    """Include the older function tests in unittest discovery, without requiring pytest.

    Those two modules use only tmp_path and pytest.raises; supply their unittest equivalents.
    Keep the shim local to importing them so it cannot affect other engineers' tests.
    """
    shim = SimpleNamespace(raises=unittest.TestCase().assertRaises)
    for module_name in ("test_units", "test_mission_mock"):
        with patch.dict(sys.modules, {"pytest": shim}):
            module = importlib.import_module(module_name)
        # patch.dict restores the module table, including newly imported modules.
        # Keep the loaded legacy module so later discovery does not import pytest again.
        sys.modules[module_name] = module
        for name, fn in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_"):
                continue

            def run(fn=fn):
                with tempfile.TemporaryDirectory() as tmp:
                    kwargs = {"tmp_path": Path(tmp)} if "tmp_path" in inspect.signature(fn).parameters else {}
                    fn(**kwargs)

            suite.addTest(unittest.FunctionTestCase(run, description=f"{module_name}.{name}"))
    return suite


if __name__ == "__main__":
    unittest.main()
