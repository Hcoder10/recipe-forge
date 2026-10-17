"""Synthetic arithmetic and procedure-only prompt checks; no GPU or timed runs."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from forge.agents import playbook, prompts
from forge.tools.builtin import LANES, build_registry, t_arch_profile
from forge.tools.registry import ToolError
from forge.tools.sol import GPU_SPECS, sol_ledger, t_sol_ledger


def hybrid():
    return dict(hidden_size=16, num_layers=4,
                layer_mix=dict(attention=1, ssm=1, moe=1, mlp=1),
                attention_heads=4, head_dim=4, kv_heads=2, context_length=8,
                intermediate_size=32, moe_intermediate_size=8,
                num_experts=4, num_experts_per_tok=2,
                ssm_inner_size=8, ssm_state_size=4, ssm_weight_params_per_layer=64,
                vocab_size=32, weight_bytes=2, kv_bytes=2, ssm_state_bytes=4)


class SolTests(unittest.TestCase):
    def rows(self, structure, **kw):
        return {r["category"]: r for r in sol_ledger(structure, gpu="RTX 5090", **kw)["rows"]}

    def test_hybrid_arithmetic(self):
        rows = self.rows(hybrid(), batch=2)
        # Q/K/V/O weights plus both KV context reads and append writes.
        self.assertEqual(rows["attention"]["bytes_per_step"], 768 * 2 + 2 * 2 * 9 * 2 * 4 * 2)
        self.assertEqual(rows["attention"]["flops_per_step"], 2 * 2 * 768 + 4 * 2 * 4 * 4 * 8)
        self.assertEqual(rows["moe"]["bytes_per_step"], (4 * 3 * 16 * 8 + 16 * 4) * 2)
        self.assertEqual(rows["moe"]["flops_per_step"], 2 * 2 * (2 * 3 * 16 * 8 + 16 * 4))
        self.assertEqual(rows["ssm"]["bytes_per_step"], 2 * 2 * 8 * 4 * 4 + 64 * 2)
        self.assertEqual(rows["ssm"]["flops_per_step"], 2 * (5 * 8 * 4 + 2 * 64))
        self.assertEqual(rows["gemm/quant"]["bytes_per_step"], 3 * 16 * 32 * 2)
        for row in rows.values():
            expected = max(row["bytes_per_step"] / 1792e9, row["flops_per_step"] / 209.5e12) * 1000
            self.assertAlmostEqual(row["sol_ms_per_step"], expected)
            self.assertIsNone(row["gap_ms"])

    def test_routing_reuse_and_storage(self):
        s = hybrid()
        s.update(active_experts_per_step=2, num_shared_experts=1, weight_bytes=.5)
        self.assertEqual(self.rows(s, batch=2)["moe"]["bytes_per_step"], (3 * 3 * 16 * 8 + 16 * 4) * .5)

    def test_absolute_measurements_aliases_and_sort(self):
        result = sol_ledger({"structure": hybrid(), "time_by_category_ms_per_step":
                            {"ssm/mamba": 2, "dense gemm": 1, "quant/dequant": 3}}, gpu="H100")
        self.assertEqual(result["rows"][0]["category"], "gemm/quant")
        row = result["rows"][0]
        self.assertEqual(row["measured_ms_per_step"], 4)
        self.assertAlmostEqual(row["gap_ms"], 4 - row["sol_ms_per_step"])
        self.assertAlmostEqual(row["gap_pct"], row["gap_ms"] / 4 * 100)

    def test_percentages_need_correct_denominator(self):
        envelope = {"structure": hybrid(), "time_by_category_pct": {"attention": 50}, "step_ms": 10}
        self.assertNotIn("measured_ms_per_step", self.rows(envelope)["attention"])
        envelope["category_total_ms_per_step"] = 8
        self.assertEqual(self.rows(envelope)["attention"]["measured_ms_per_step"], 4)

    def test_missing_inputs_are_unknown(self):
        s = hybrid()
        del s["context_length"]
        self.assertIsNone(self.rows(s)["attention"]["sol_ms_per_step"])
        self.assertTrue(all(r["gap_ms"] is None for r in self.rows(s).values()))

    def test_explicit_compute_bound_work(self):
        row = self.rows({"category_work": {"moe": {"bytes_per_step": 0, "flops_per_step": 209.5e12}}})["moe"]
        self.assertEqual(row["sol_ms_per_step"], 1000)

    def test_validation(self):
        for batch in (0, -1, 1.5, True, float("nan")):
            with self.assertRaises(ToolError):
                self.rows(hybrid(), batch=batch)
        for gpu in ("unknown", "H100 PCIe", "H100 NVL"):
            with self.assertRaises(ToolError):
                sol_ledger(hybrid(), gpu=gpu)
        for value in (-1, float("inf"), "bad"):
            s = hybrid()
            s["hidden_size"] = value
            with self.assertRaises(ToolError):
                self.rows(s)

    def test_gpu_table_and_input_immutable(self):
        s = hybrid()
        original = copy.deepcopy(s)
        for gpu in GPU_SPECS:
            self.assertEqual(len(sol_ledger(s, gpu=gpu)["rows"]), 7)
        self.assertEqual(s, original)

    def test_mission_cache_and_registration(self):
        out = {"structure": hybrid(), "gpu": {"name": "RTX 5090"}}
        mission = SimpleNamespace(recipe_id="synthetic", gpu_class="RTX 5090", id="synthetic")
        ctx = SimpleNamespace(mission=mission, agent="worker", services=SimpleNamespace(
            bench=SimpleNamespace(arch_profile=Mock(return_value=out)), brain=SimpleNamespace(add=Mock(return_value="fact"))))
        t_arch_profile(ctx)
        self.assertEqual(t_sol_ledger(ctx), sol_ledger(out))
        ctx.services.bench.arch_profile.assert_called_once()
        tool = build_registry().tools["sol_ledger"]
        self.assertTrue(tool.light)
        self.assertIn("lead", tool.roles)
        for lane in ("acceptance", "norm", "sampler"):
            self.assertIn(lane, LANES)
        other = SimpleNamespace(mission=SimpleNamespace(gpu_class="RTX 5090"))
        with self.assertRaises(ToolError):
            t_sol_ledger(other)

    def test_playbook_hash_and_method_delivery(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pb = playbook.load(None, root)
            digest = playbook.save(pb, root)
            self.assertEqual(playbook.load(digest, root)["hash"], digest)
            changed = copy.deepcopy(pb)
            changed["method"] += " Inspect calibration."
            self.assertNotEqual(playbook.content_hash(changed), digest)
            self.assertIn(pb["method"], pb["protocol"])
            self.assertEqual(pb["knobs"], {"verifier_bar_pct": 3.0, "facts_refresh_turns": 8, "keep_turns": 12})

    def test_static_prompts_have_no_numeric_advice(self):
        from forge.agents.mechanic import leak_check
        self.assertEqual(leak_check(playbook.DEFAULT), [])
        for key in ("lead", "worker", "protocol", "method", "techniques", "verifier"):
            self.assertNotRegex(playbook.DEFAULT[key], r"\d")
            self.assertNotRegex(playbook.DEFAULT[key], r"--[a-z][a-z-]*[= ]")
        self.assertNotRegex(prompts.SUMMARIZE, r"\d")
        self.assertNotIn("private-marker", prompts.context_block("private-marker", "private-marker", "private-marker"))
        m = SimpleNamespace(recipe={"serve": {"args": "private-marker", "env": {"private-marker": "private-marker"}},
                                    "workload": {"private-marker": "private-marker"}},
                            id="synthetic", recipe_id="synthetic", metric="decode_tok_s", machine={},
                            gpu_class="synthetic", time_box_min=1, budget_usd=1, n_workers=1,
                            worker_names=["worker"], ws="workspace")
        brief = prompts.mission_brief(m)
        self.assertNotIn("private-marker", brief)
        self.assertIn("ledger rows", brief)
        self.assertIn("acceptance", brief)


if __name__ == "__main__":
    unittest.main()
