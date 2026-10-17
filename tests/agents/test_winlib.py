import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from forge.agents import events, winlib


class WinlibTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch = patch("forge.agents.config.state_root", return_value=self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.mission = self.root / "missions" / "m1"
        self.recipe = {"id": "r1", "workload": {"concurrency": 8}}

    def bench(self, bid, delta, **kw):
        events.append(self.mission / "benches.jsonl", {"bench_id": bid, "by": "worker-1",
                      "config_delta": {"max_num_seqs": 8}, "claim": "reduce batch size",
                      "result": {"delta_pct": delta}, **kw})

    def distill(self):
        return winlib.distill(self.mission, self.recipe, None, "H100")

    def test_hybrid_traits(self):
        cfg = {"architectures": ["HybridForCausalLM"], "model_type": "nemotron_h",
               "hybrid_override_pattern": "M*EM*E", "hidden_size": 4096, "num_hidden_layers": 6,
               "num_key_value_heads": 8, "n_routed_experts": 64, "num_experts_per_tok": 4,
               "quantization_config": {"quant_method": "fp8"}}
        recipe = {**self.recipe, "drafter": {"model": "draft"}, "serve": {"args": [
            "--speculative-config", '{"method":"dflash","model":"draft"}']}}
        traits = winlib.arch_traits(recipe, cfg)
        self.assertEqual(traits["layer_mix"], {"ssm": 2, "attention": 2, "moe": 2})
        self.assertEqual(traits["architecture"], "HybridForCausalLM")
        self.assertEqual(traits["model_type"], "nemotron_h")
        self.assertEqual(traits["num_experts"], 64)
        self.assertEqual(traits["num_experts_per_tok"], 4)
        self.assertEqual(traits["quantization"], "fp8")
        self.assertTrue(traits["drafter"])
        self.assertEqual(traits["drafter_method"], "dflash")
        self.assertEqual(traits["concurrency"], 8)

    def test_attention_and_missing_traits(self):
        cfg = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 2048,
               "num_hidden_layers": 24, "num_key_value_heads": 4}
        traits = winlib.arch_traits({"serve": {"args": ["--quantization", "awq"]}}, cfg)
        self.assertEqual(traits["layer_mix"], {"ssm": 0, "attention": 24, "moe": 0})
        self.assertEqual((traits["hidden_size"], traits["num_layers"], traits["kv_heads"]), (2048, 24, 4))
        self.assertEqual(traits["quantization"], "awq")
        self.assertFalse(traits["drafter"])
        self.assertIsNone(winlib.arch_traits({}, None)["architecture"])
        for key in ("layers_block_type", "layer_types"):
            self.assertEqual(winlib.arch_traits({}, {key: ["mamba", "attention", "moe"]})["layer_mix"],
                             {"ssm": 1, "attention": 1, "moe": 1})

    def test_distill_gates_and_idempotency(self):
        events.append(self.mission / "events.jsonl", {"kind": "worker_task", "mission_id": "m1",
                      "worker": "worker-1", "lane": "kernel"})
        self.bench("B1", 3, bundle="abc")
        self.bench("B2", 4, bundle="abc", config_delta={"--max-num-seqs": "8"})
        events.append(self.mission / "gate_requests.jsonl", {"id": "R1", "bundle": "abc", "hash": "abcdef",
                      "manifest": {"overlays": ["kernel.py"]}})
        events.append(self.mission / "events.jsonl", {"kind": "gate_result", "gate_id": "G1",
                      "request_id": "R1", "verdict": "lossy", "delta_pct": 5})
        self.bench("B3", -2, config_delta={"x": 1})
        self.bench("B4", None, config_delta={"y": 1})
        self.bench("B5", 2, config_delta={"z": 1}, lane="config")
        rows = self.distill()
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["outcome"], {"kind": "lossy", "delta_pct": 5.0,
                                             "evidence": ["B1", "G1", "B2"]})
        self.assertEqual(rows[0]["lane"], "kernel")
        self.assertEqual(rows[0]["change"]["files"], ["kernel.py"])
        self.assertEqual(rows[0]["change"]["bundle_hash"], "abcdef")
        self.assertEqual(rows[1]["outcome"]["kind"], "quick_loss")
        self.assertEqual(rows[2]["outcome"]["kind"], "quick_gain")
        self.assertIn("t", rows[0])
        self.assertEqual(self.distill(), [])
        self.assertEqual(len(events.read(self.root / "winlib" / "entries.jsonl")), 3)

    def test_gate_verdicts(self):
        for i, verdict in enumerate(("win", "loss", "error")):
            self.bench(f"B{i}", 2, config_delta={"x": i})
            events.append(self.mission / "events.jsonl", {"kind": "gate_result", "bench_id": f"B{i}",
                          "gate_id": f"G{i}", "verdict": verdict, "delta_pct": -1})
        self.assertEqual([r["outcome"]["kind"] for r in self.distill()],
                         ["gated_win", "gated_loss", "quick_gain"])

    def test_relevant_render_and_matching(self):
        self.bench("B1", 2)
        base = self.distill()[0]
        for eid, kind, traits, gpu in (("win", "gated_win", base["traits"], "H100"),
                                      ("loss", "gated_loss", base["traits"], "H100"),
                                      ("other", "gated_win", {"architecture": "Other"}, "T4")):
            events.append(self.root / "winlib" / "entries.jsonl", {**base, "entry_id": eid,
                          "traits": traits, "gpu_class": gpu, "outcome": {**base["outcome"], "kind": kind}})
        rows = winlib.relevant(base["traits"], "H100")
        self.assertEqual([r["entry_id"] for r in rows], ["win", base["entry_id"], "loss", "other"])
        self.assertEqual(winlib.relevant({}, "H100", k=0), [])
        self.assertEqual(winlib.match_entry({"--max-num-seqs": "8"}, {}, rows), "win")
        self.assertIsNone(winlib.match_entry({"--max-num-seqs": 9}, {}, rows))
        self.assertIsNone(winlib.match_entry({}, {}, rows))
        bundled = {**base, "change": {**base["change"], "bundle_hash": "abc"}}
        self.assertEqual(winlib.match_entry({"max_num_seqs": 8.0}, None, [bundled]), base["entry_id"])
        bundled["change"]["files"] = ["kernel.py"]
        self.assertIsNone(winlib.match_entry({"max_num_seqs": 8}, None, [bundled]))
        long = {**base, "change": {"summary": "huge\n" * 10000}}
        rendered = winlib.render(rows + [long] * 100)
        self.assertTrue(rendered.startswith("WIN LIBRARY (measured by earlier missions; re-verify on this model)"))
        self.assertLessEqual(len(rendered.encode()), 1500)
        self.assertIn("evidence B1", rendered)
        self.assertEqual(winlib.render([]), "")

    def test_reuse_stats(self):
        self.assertEqual(winlib.reuse_stats()["help_rate"], 0)
        self.bench("B1", 2)
        eid = self.distill()[0]["entry_id"]
        winlib.record_reuse("m2", eid, "B1", 4)
        winlib.record_reuse("m2", eid, "B1", 4)
        winlib.record_reuse("m2", eid, "B2", -1)
        winlib.record_reuse("m3", eid, "B1", None)
        stats = winlib.reuse_stats()
        self.assertEqual(stats["per_entry"][eid], {"tried": 2, "helped": 1})
        self.assertEqual((stats["missions"], stats["entries"]), (2, 1))
        self.assertEqual(stats["reuse_rate"], 1)
        self.assertEqual(stats["help_rate"], .5)


if __name__ == "__main__":
    unittest.main()
