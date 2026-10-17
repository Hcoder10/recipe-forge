import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from forge.agents import events, winlib
from forge.tools import benchjob, builtin
from forge.tools.bench import RemoteBenchRunner


class TraceTests(unittest.TestCase):
    def test_discovery_clock_conversion_and_owned_cleanup(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old = root / "existing.json"
            old.write_text("do not remove")
            (root / "frontend.json").write_text(json.dumps({"traceEvents": []}))
            trace = {"baseTimeNanoseconds": 1000000, "traceEvents": [
                {"ph": "X", "cat": "kernel", "name": "prefill", "ts": 0, "dur": 10},
                {"ph": "X", "cat": "kernel", "name": "gemm", "ts": 20, "dur": 10}]}
            with gzip.open(root / "worker.pt.trace.json.gz", "wt") as f:
                json.dump(trace, f)
            out = benchjob.parse_profile_traces(root, {old}, 1, [(1020, 1040)])
            self.assertEqual([k["name"] for k in out["top_kernels"]], ["gemm"])
            self.assertTrue(out["trace_removed"])
            self.assertIsNone(out["trace_path"])
            self.assertEqual(list(root.iterdir()), [old])

    def test_bad_trace_is_removed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "bad.json").write_text("invalid")
            with self.assertRaises(ValueError):
                benchjob.parse_profile_traces(root, set(), 1, [(0, 1)])
            self.assertEqual(list(root.iterdir()), [])

    def test_categories_and_overlap(self):
        names = ["flash_fwd", "selective_scan", "fused_moe", "cublas_gemm",
                 "rmsnorm", "nvfp4_quant", "softmax", "ncclAllReduce", "memcpy", "unknown"]
        trace = {"traceEvents": [
            {"ph": "X", "cat": "kernel" if name != "memcpy" else "gpu_memcpy",
             "name": name, "ts": i * 10, "dur": 10} for i, name in enumerate(names)] + [
            {"ph": "X", "cat": "cpu_op", "name": "gemm", "ts": 0, "dur": 1000}]}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "trace.json.gz"
            with gzip.open(path, "wt") as f:
                json.dump(trace, f)
            result = benchjob.parse_trace(path, 10, [(0, 200)])
        self.assertEqual(set(result["time_by_category_pct"].values()), {10.0})
        self.assertEqual(result["idle_pct"], 50)
        self.assertEqual(result["launches_per_step"], .9)
        self.assertEqual(len(result["top_kernels"]), 9)

    def test_clipping_union_and_counts(self):
        trace = {"traceEvents": [
            {"ph": "X", "cat": "kernel", "name": "gemm", "ts": 0, "dur": 80},
            {"ph": "X", "cat": "kernel", "name": "gemm", "ts": 20, "dur": 80}]}
        result = benchjob.parse_trace(trace, 2, [(10, 90), (120, 140)])
        self.assertEqual(result["idle_pct"], 20)
        self.assertEqual(result["top_kernels"][0]["count"], 2)
        self.assertEqual(result["top_kernels"][0]["time_us"], 140)
        with self.assertRaises(ValueError):
            benchjob.parse_trace(trace, 2, [(200, 300)])
        with self.assertRaises(ValueError):
            benchjob.parse_trace(trace, 2, [])

    def test_specialized_kernels_before_generic_gemm(self):
        for name, expected in [("grouped_gemm_fp8", "moe"), ("cutlass_moe", "moe"),
                               ("sm100_xmma", "dense gemm"), ("scaled_gemm", "quant/dequant"),
                               ("topk", "sampler/topk/softmax")]:
            result = benchjob.parse_trace({"traceEvents": [
                {"ph": "X", "cat": "kernel", "name": name, "ts": 0, "dur": 1}]}, 1)
            self.assertEqual(result["time_by_category_pct"][expected], 100)


class StructureTests(unittest.TestCase):
    def test_dotted_speculative_config(self):
        result = benchjob.extract_structure({}, {"serve_args": [
            "--speculative_config.model", "draft", "--speculative_config.method", "dflash",
            "--speculative_config.num_speculative_tokens", "7"]})
        self.assertEqual(result["drafter_config"], {
            "model": "draft", "method": "dflash", "num_speculative_tokens": 7})

    def test_missing_profiler_preserves_structure(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "config.json").write_text('{"architectures":["Hybrid"]}')
            with patch.object(benchjob.subprocess, "run", side_effect=OSError):
                structure, files, available = benchjob.inspect_profile_target(
                    {"python": "missing", "recipe": {"model": td}})
            self.assertEqual(structure["architecture"], "Hybrid")
            self.assertEqual(files, [])
            self.assertFalse(available)

    def test_hybrid(self):
        result = benchjob.extract_structure({"architectures": ["HybridForCausalLM"],
            "text_config": {"hybrid_override_pattern": "M*EM*-", "n_routed_experts": 64,
                            "moe_top_k": 4, "hidden_size": 4096, "num_key_value_heads": 8,
                            "quantization_config": {"quant_method": "fp8"}}},
            {"serve_args": ["--speculative-config", '{"model":"draft","method":"mtp"}']})
        self.assertEqual(result["layer_mix"], {"ssm": 2, "attention": 2, "moe": 1, "mlp": 1})
        self.assertEqual(result["num_experts"], 64)
        self.assertEqual(result["num_experts_per_tok"], 4)
        self.assertEqual(result["kv_heads"], 8)
        self.assertEqual(result["quantization"], "fp8")
        self.assertEqual(result["drafter_config"], {"model": "draft", "method": "mtp"})

    def test_block_types(self):
        self.assertEqual(benchjob.extract_structure({"layers_block_type": ["mamba", "attention", "moe", "mlp"]})[
            "layer_mix"], {"ssm": 1, "attention": 1, "moe": 1, "mlp": 1})

    def test_installed_registry_inspection(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text(json.dumps({"architectures": ["HybridForCausalLM"]}))
            package = root / "vllm"
            models = package / "model_executor/models"
            models.mkdir(parents=True)
            (models / "registry.py").write_text('_MODELS = {"HybridForCausalLM": ("hybrid", "HybridForCausalLM")}')
            (models / "hybrid.py").write_text("# model implementation")
            (package / "profiler.py").write_text('# start_profile stop_profile "VLLM_TORCH_PROFILER_DIR"')
            spec = {"python": "python", "recipe": {"model": str(model)}}
            with patch.object(benchjob.subprocess, "run", return_value=SimpleNamespace(stdout=json.dumps(str(package)))):
                structure, files, available = benchjob.inspect_profile_target(spec)
            self.assertTrue(available)
            self.assertEqual(files, [str(models / "hybrid.py")])
            self.assertEqual(structure["architectures"], ["HybridForCausalLM"])

    def test_modern_profiler_options_are_supported_fields_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config.json").write_text('{}')
            package = root / "vllm"
            (package / "config").mkdir(parents=True)
            (package / "config/profiler.py").write_text(
                'class ProfilerConfig:\n    torch_profiler_dir: str = ""\n'
                '    ignore_frontend: bool = False\n    torch_profiler_with_stack: bool = True\n'
                '# start_profile stop_profile\n')
            spec = {"python": "python", "recipe": {"model": str(root)}}
            with patch.object(benchjob.subprocess, "run", return_value=SimpleNamespace(
                    stdout=json.dumps(str(package)))):
                _, _, available = benchjob.inspect_profile_target(spec)
            self.assertTrue(available)
            self.assertTrue(spec["profiler_config"])
            self.assertEqual(spec["profiler_options"], {
                "ignore_frontend": True, "torch_profiler_with_stack": False})

    def test_fallback_and_cache(self):
        with tempfile.TemporaryDirectory() as td:
            spec = {"recipe": {}, "ref_cache_dir": td}
            gpu = {"uuid": "GPU-test"}
            with patch.object(benchjob, "inspect_profile_target", return_value=({"architecture": "Hybrid"}, [], False)):
                result = benchjob.run_profile(spec, Path(td), gpu)
            self.assertIsNone(result["time_by_category_pct"])
            self.assertEqual(result["structure"]["architecture"], "Hybrid")
            self.assertNotIn("\n", result["reason"])
            with patch.object(benchjob, "inspect_profile_target", return_value=({}, [], True)), patch.object(
                    benchjob, "serve_and_measure", return_value={"time_by_category_pct": {"other": 100}}) as serve:
                benchjob.run_profile(spec, Path(td), gpu)
                self.assertTrue(benchjob.run_profile(spec, Path(td), gpu)["cached"])
                self.assertEqual(serve.call_count, 1)


class CodeReuseTests(unittest.TestCase):
    def test_code_match_requires_file_and_function(self):
        targets = winlib.code_targets({"vllm/model_executor/models/hybrid.py": "overlays/hybrid.py"},
            "--- a/vllm/model_executor/models/hybrid.py\n+++ b/vllm/model_executor/models/hybrid.py\n"
            "@@ -4,1 +4,1 @@ def forward(self):\n-    old()\n+    new()")
        entry = {"entry_id": "entry", "targets": targets}
        self.assertEqual(winlib.match_code([{"file": "/venv/site-packages/vllm/model_executor/models/hybrid.py",
                                            "functions": ["forward"]}], [entry]), "entry")
        self.assertIsNone(winlib.match_code([{"file": targets[0]["file"], "functions": ["other"]}], [entry]))
        self.assertIsNone(winlib.match_code([targets[0]["file"]], [entry]))
        self.assertIsNone(winlib.match_code([{"file": "another.py", "functions": ["forward"]}], [entry]))

    def test_distill_targets_category_and_safe_render(self):
        with tempfile.TemporaryDirectory() as td, patch("forge.agents.config.state_root", return_value=Path(td)):
            mission = Path(td) / "missions/m1"
            events.append(mission / "benches.jsonl", {"bench_id": "b1", "bundle": "bundle",
                "result": {"delta_pct": 1, "arch_profile_category": "ssm/mamba"},
                "manifest": {"overlays": {"vllm/model_executor/models/hybrid.py": {
                    "diff": "@@ -1 +1 @@ def forward(self):\n-old\n+new"}}},
                "claim": "DO_NOT_RENDER_FINDINGS"})
            rows = winlib.distill(mission, {}, {"architectures": ["Hybrid"]}, "gpu")
            self.assertEqual(rows[0]["targets"][0]["functions"], ["forward"])
            self.assertEqual(rows[0]["targets"][0]["category"], "ssm/mamba")
            text = winlib.render(rows)
            self.assertIn("forward", text)
            self.assertNotIn("DO_NOT_RENDER_FINDINGS", text)
            self.assertNotIn("1.0%", text)

    def test_relevance_boost(self):
        with tempfile.TemporaryDirectory() as td, patch("forge.agents.config.state_root", return_value=Path(td)):
            base = {"traits": {}, "outcome": {"kind": "quick_gain", "delta_pct": 1}}
            for eid, targets in [("generic", []), ("category", [{"file": "vllm/x.py", "category": "ssm/mamba"}]),
                                  ("model", [{"file": "vllm/model_executor/models/hybrid.py"}])]:
                events.append(Path(td) / "winlib/entries.jsonl", {**base, "entry_id": eid, "targets": targets})
            rows = winlib.relevant({"model_files": ["vllm/model_executor/models/hybrid.py"], "layer_mix": {"ssm": 2}}, "gpu")
            self.assertEqual([r["entry_id"] for r in rows], ["model", "category", "generic"])


class ToolTests(unittest.TestCase):
    def test_profile_uses_served_name_and_cleans_failed_request(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trace = root / "trace"
            trace.mkdir()
            old = trace / "old.json"
            old.write_text("preserve")
            def request(*args, **kwargs):
                if kwargs.get("token_clock"):
                    (trace / "new.json").write_text("partial capture")
                    raise ValueError("request failed")
                return {}
            with patch.object(benchjob, "stream_one", side_effect=request) as stream, patch.object(
                    benchjob.urllib.request, "urlopen", return_value=response):
                with self.assertRaises(ValueError):
                    benchjob.profile_workload(1234, {"model": "repo", "serve_args": [
                        "--served-model-name", "alias"]}, {"workload": {}}, root)
            self.assertEqual(stream.call_args.args[1], "alias")
            self.assertEqual(list(trace.iterdir()), [old])

    def test_profile_stops_after_request_failure(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        with patch.object(benchjob, "stream_one", side_effect=[{}, ValueError("request failed")]), patch.object(
                benchjob.urllib.request, "urlopen", return_value=response) as post:
            with self.assertRaises(ValueError):
                benchjob.profile_workload(1234, {"model": "model"}, {"workload": {}}, Path("unused"))
        self.assertEqual([c.args[0].full_url for c in post.call_args_list],
                         ["http://127.0.0.1:1234/start_profile", "http://127.0.0.1:1234/stop_profile"])

    def test_fact_uses_arch_lane(self):
        brain = SimpleNamespace(add=Mock(return_value="fact"))
        runner = SimpleNamespace(arch_profile=Mock(return_value={"structure": {}, "model_files": [],
                                                                "time_by_category_pct": None}))
        ctx = SimpleNamespace(agent="worker-1", mission=SimpleNamespace(id="m", recipe_id="r", gpu_class="g"),
                              services=SimpleNamespace(brain=brain, bench=runner))
        self.assertEqual(builtin.t_arch_profile(ctx)["fact_id"], "fact")
        self.assertEqual(brain.add.call_args.args[0]["lane"], "arch")
        with self.assertRaises(builtin.ToolError):
            builtin.t_arch_profile(ctx, steps=0)

    def test_remote_profile_uses_gpu_executor_and_sandbox(self):
        executor = Mock()
        executor.workspace.return_value = "/workspace"
        executor.run.return_value = SimpleNamespace(infra_error=None, stdout='BENCH_RESULT {"ok":true}', rc=0)
        runner = RemoteBenchRunner(executor, "/forge", "/python", sandbox=False)
        mission = SimpleNamespace(id="m", python="/recipe/python", bench_prompts="/prompts",
            recipe={"model": "/model", "serve": {"args": []}, "workload": {}})
        runner.arch_profile(mission)
        spec = json.loads(executor.write_text.call_args.args[1])
        self.assertEqual(spec["mode"], "profile")
        self.assertTrue(spec["sandbox"])
        self.assertIsNone(spec["candidate"])
        self.assertEqual(executor.run.call_args.kwargs["kind"], "bench")
        self.assertEqual(executor.run.call_args.kwargs["gpus"], 1)


if __name__ == "__main__":
    unittest.main()
