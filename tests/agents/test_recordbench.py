"""CPU-only fixtures for record mode and gate priority; no measured findings."""
import json
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from forge.agents.mission import GateQueue
from forge.tools import benchjob, recordbench
from forge.tools.bench import RemoteBenchRunner
from forge.tools.builtin import build_registry
from forge.tools.registry import EnvBlocked, ToolError


def recipe(published=True):
    r = {"model": "fixture", "serve": {"args": []},
         "workload": {"concurrency": 1, "max_tokens": 7, "metric": "decode_tok_s"}}
    if published:
        r["published"] = {"concurrency": 2, "isl": 12, "osl": 6, "metric": "output_tok_s",
                          "unit": "tokens/s", "value": 1, "source_url": "https://example.invalid/fixture"}
    return r


class RecordTests(unittest.TestCase):
    def test_public_workload(self):
        r = recipe()
        wl, prompts = recordbench.workload(r)
        self.assertEqual((wl["concurrency"], wl["max_tokens"]), (2, 6))
        self.assertTrue(wl["ignore_eos"])
        self.assertEqual(len(prompts), 2)
        self.assertTrue(all(len(p["prompt"].split()) == 12 for p in prompts))
        self.assertEqual(prompts, recordbench.workload(r)[1])
        self.assertEqual(r["workload"]["concurrency"], 1)

    def test_fallback_workload(self):
        wl, prompts = recordbench.workload(recipe(False))
        self.assertEqual((wl["concurrency"], wl["max_tokens"]), (1, 7))
        self.assertIsNone(prompts)

    def test_incomplete_published_metadata_rejected(self):
        r = recipe()
        r["published"]["isl"] = None
        with self.assertRaises(ValueError):
            recordbench.workload(r)

    def test_abab_shape_with_mocked_server(self):
        gpu = {"name": "fixture GPU", "power_limit": "fixture"}
        result = {"n": 2, "fixed_length": True, "output_tok_s": 2,
                  "decode_tok_s": 1, "ttft_ms_mean": 3}
        with tempfile.TemporaryDirectory() as td, patch.object(benchjob, "serve_and_measure",
                                                              return_value=result) as serve:
            report = recordbench.run_record({"record_recipe": recipe(), "reps": 3,
                          "candidate": {"bundle_dir": "/candidate"}, "sandbox": False}, Path(td), gpu)
            self.assertEqual([r["side"] for r in report["runs"]], ["reference", "candidate"] * 3)
            self.assertEqual([c.args[1] for c in serve.call_args_list], [None, "/candidate"] * 3)
            self.assertTrue(all(c.args[0]["sandbox"] for c in serve.call_args_list))
            self.assertTrue(all(r["gpu"] == gpu for r in report["runs"]))
            self.assertEqual(report["means"]["candidate"]["output_tok_s_per_user"], 1)
            self.assertEqual(report["published"], recipe()["published"])
            self.assertIn("not a substitute", report["note"])
            self.assertEqual(report["workload_source"], "published")

    def test_fallback_report_and_incomplete_run(self):
        result = {"n": 1, "fixed_length": True, "output_tok_s": 1,
                  "decode_tok_s": 1, "ttft_ms_mean": 1}
        with tempfile.TemporaryDirectory() as td, patch.object(benchjob, "serve_and_measure",
                                                              return_value=result):
            spec = {"record_recipe": recipe(False), "reps": 1, "prompts": "/public"}
            report = recordbench.run_record(spec, Path(td), {})
            self.assertIsNone(report["published"])
            self.assertIn("no published block", report["workload_source"])
            result["fixed_length"] = False
            with self.assertRaises(ValueError):
                recordbench.run_record(spec, Path(td), {})

    def test_roles_receipt_and_scheduler(self):
        tool = build_registry().tools["record_bench"]
        self.assertEqual(set(tool.roles), {"lead", "verifier"})
        ex = Mock()
        report = {"ok": True, "mode": "record", "means": {"fixture": 1}}
        ex.run.return_value = SimpleNamespace(infra_error=None, rc=0, stdout="BENCH_RESULT " + json.dumps(report))
        with tempfile.TemporaryDirectory() as td:
            m = SimpleNamespace(id="m", recipe_id="r", recipe=recipe(), python="/python",
                                bench_prompts="/public", bench_log=Path(td) / "missions/m/benches.jsonl")
            ctx = SimpleNamespace(mission=m, role="lead", agent="lead", log_event=Mock(),
                                  services=SimpleNamespace(bench=RemoteBenchRunner(ex, "/forge", "/python")))
            receipt = recordbench.record_bench(ctx)
            self.assertEqual(set(receipt), {"record_id", "status", "note"})
            stored = json.loads((Path(td) / "records.jsonl").read_text())
            self.assertEqual(stored["means"], report["means"])
            self.assertEqual(ex.run.call_args.kwargs["kind"], "bench")
            self.assertEqual(ex.run.call_args.kwargs["gpus"], 1)
            spec = json.loads(ex.write_text.call_args.args[1])
            self.assertTrue(spec["sandbox"])
            self.assertEqual(spec["mode"], "record")
            ctx.log_event.assert_called_once_with("record_bench", record_id=receipt["record_id"], by="lead")
            ex.run.return_value = SimpleNamespace(infra_error="private diagnostic", rc=1, stdout="")
            with self.assertRaisesRegex(EnvBlocked, "^record measurement unavailable; no report was accepted$"):
                recordbench.record_bench(ctx)
            ctx.role = "worker"
            with self.assertRaises(ToolError):
                recordbench.record_bench(ctx)

    def test_strict_sampling_does_not_retry_without_ignore_eos(self):
        from urllib.error import HTTPError
        with patch.object(benchjob.urllib.request, "urlopen", side_effect=HTTPError("url", 400, "fixture", {}, None)) as post:
            with self.assertRaises(HTTPError):
                benchjob.stream_one(1, "fixture", {"prompt": "tree", "strict_fixed_length": True}, 4, None)
            self.assertEqual(post.call_count, 1)
            body = json.loads(post.call_args.args[0].data)
            self.assertTrue(body["ignore_eos"])
            self.assertEqual(body["min_tokens"], 4)


class FasttrackTests(unittest.TestCase):
    def test_selection(self):
        valid = {"confirmed": True, "delta_pct": 8, "verdict_hint": "gain"}
        with patch.dict(os.environ, {"FORGE_FASTTRACK_PCT": "8"}):
            self.assertTrue(GateQueue.fasttrack(valid))
            for update in ({"confirmed": False}, {"confirmed": "true"}, {"delta_pct": 7.9},
                           {"delta_pct": None}, {"delta_pct": float("nan")},
                           {"delta_pct": float("inf")}, {"verdict_hint": "within_noise"}):
                self.assertFalse(GateQueue.fasttrack({**valid, **update}))
        with patch.dict(os.environ, {"FORGE_FASTTRACK_PCT": "9"}):
            self.assertFalse(GateQueue.fasttrack(valid))

    def test_auto_approval_and_review_path(self):
        for confirmed in (True, False):
            with self.subTest(confirmed=confirmed), tempfile.TemporaryDirectory() as td:
                gate = SimpleNamespace(ledger=Mock(return_value=[]), request=Mock(return_value={"gate_id": "g"}))
                m = SimpleNamespace(recipe_id="r", id="m", bundle_dir=lambda b: "/bundle",
                    bundles={"b": {"hash": "h", "manifest": {}}},
                    benches={"B1": {"bundle": "b", "result": {"confirmed": confirmed,
                                                               "delta_pct": 8, "verdict_hint": "gain"}}})
                runner = SimpleNamespace(dir=Path(td), mission=m, services=SimpleNamespace(gate=gate),
                    log=Mock(), wake=Mock(), decisions=Mock(), agents={}, playbook={"knobs": {}})
                q = GateQueue(runner)
                with patch.object(q, "_watch"), patch.dict(os.environ, {"FORGE_FASTTRACK_PCT": "8"}):
                    result = q.submit(SimpleNamespace(agent="worker-1"), "b", "fixture", "B1")
                if confirmed:
                    self.assertEqual(result["status"], "gated")
                    self.assertEqual(gate.request.call_args.kwargs["priority"], "high")
                    runner.wake.assert_not_called()
                    runner.log.assert_any_call("gate_fasttrack", request_id="R1", bench_id="B1")
                else:
                    self.assertEqual(result["status"], "pending_review")
                    gate.request.assert_not_called()
                    runner.wake.assert_called_once_with("verifier")


@unittest.skipIf(sys.platform == "win32", "gate ledger requires Linux fcntl")
class QueueTests(unittest.TestCase):
    def setUp(self):
        from forge.gate import records
        self.l = records
        self.requests = {}
        self.rows = {}
        self.nulls = {}
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(records, "latest", side_effect=lambda: self.rows))
        stack.enter_context(patch.object(records, "get_request", side_effect=lambda gid: self.requests[gid]))
        stack.enter_context(patch.object(records, "_save_request", side_effect=lambda r: self.requests.update({r["gate_id"]: r})))
        stack.enter_context(patch.object(records, "_n_null", side_effect=lambda hk: self.nulls.get(hk, 0)))
        stack.enter_context(patch.object(records, "submit", side_effect=self.submit))

    def add(self, gid, kind, hk="r", priority="normal", status="queued", reps=4):
        self.requests[gid] = {"gate_id": gid, "kind": kind, "history_key": hk, "recipe_id": hk,
            "recipe": {}, "python": "/python", "priority": priority, "submitted": len(self.requests), "reps": reps}
        self.rows[gid] = {"gate_id": gid, "kind": kind, "status": status}
        return self.requests[gid]

    def submit(self, kind, recipe_id, **kw):
        return self.add("generated-" + str(len(self.requests)), kind, recipe_id, reps=kw.get("reps", 4))

    def test_high_before_unrelated_reference(self):
        self.add("ref", "reference", "other")
        self.add("candidate", "candidate", priority="high")
        self.nulls["r"] = 8
        self.assertEqual(self.l.next_request()["gate_id"], "candidate")

    def test_exact_missing_controls_then_candidate(self):
        self.add("unrelated", "reference", "other")
        self.add("ref", "reference", reps=6)
        self.add("candidate", "candidate", priority="high")
        self.nulls["r"] = 5
        control = self.l.next_request()
        self.assertEqual((control["gate_id"], control["reps"]), ("ref", 3))
        self.rows["ref"]["status"] = "done"
        self.nulls["r"] = 8
        self.assertEqual(self.l.next_request()["gate_id"], "candidate")

    def test_generated_controls_exact_total(self):
        self.add("candidate", "candidate", priority="high")
        self.nulls["r"] = 1
        total = 0
        while self.nulls["r"] < 8:
            control = self.l.next_request()
            self.assertEqual(control["kind"], "reference")
            total += control["reps"]
            self.nulls["r"] += control["reps"]
            self.rows[control["gate_id"]]["status"] = "done"
        self.assertEqual(total, 7)
        self.assertEqual(self.l.next_request()["gate_id"], "candidate")

    def test_no_preemption(self):
        self.add("running", "reference", "other", status="running")
        self.add("candidate", "candidate", priority="high")
        self.assertIsNone(self.l.next_request())

    def test_normal_order_preserved(self):
        self.add("candidate", "candidate")
        self.add("ref", "reference")
        self.nulls["r"] = 8
        self.assertEqual(self.l.next_request()["gate_id"], "ref")


if __name__ == "__main__":
    unittest.main()
