"""CPU-only catalog checks: python -m unittest forge.recipes.test_record_targets -v.

Runs the existing pure catalog test functions through unittest without importing
pytest or the rest of the fleet test module. Never starts a server or a job.
"""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

from forge import recipes
from forge.agents.mission import normalize_recipe
from forge.agents.prompts import mission_brief
from forge.tools.bundle import serve_command

ROOT = Path(__file__).resolve().parents[2]
NEW = (
    "bolt-74b-lilicorr-5090",
    "lightning-30b-a3b-nvfp4-dspark-h100",
    "super3-120b-a12b-nvfp4",
    "gpt-oss-20b-mxfp4",
)


class RecordTargetCatalog(unittest.TestCase):
    def test_all_catalog_metadata(self):
        keys = {"value", "metric", "unit", "hardware", "concurrency", "isl", "osl", "source_url", "date"}
        for base in recipes.bases():
            if base["id"] == "cosmos3-nano-video":
                continue
            with self.subTest(recipe=base["id"]):
                self.assertIn("published", base)
                p = base["published"]
                if p is not None:
                    self.assertEqual(set(p), keys)
                    self.assertIsInstance(p["value"], (int, float))
                    self.assertTrue(p["source_url"].startswith("https://"))
                    for k in ("concurrency", "isl", "osl"):
                        self.assertTrue(p[k] is None or isinstance(p[k], int))
                self.assertTrue(base.get("published_note"))

    def test_new_recipes_normalize(self):
        for rid in NEW:
            base = recipes.load_base(str(Path(recipes.HERE) / (rid + ".yaml")))
            for gpu in base["hardware"]["fits"]:
                for expanded in recipes.expand(base, gpu):
                    with self.subTest(recipe=expanded["id"], gpu=gpu):
                        r = normalize_recipe(expanded)
                        self.assertTrue(r["workload"]["concurrency"] > 0)
                        self.assertTrue(all(isinstance(a, str) for a in r["serve"]["args"]))
                        self.assertTrue(r["serve"]["source"].startswith("https://"))
                        self.assertTrue(r["serve"]["quoted"].strip())
                        self.assertTrue(r["vllm"]["run_image"])
                        argv, _ = serve_command("python", r["model"], r["serve"]["args"], None, 18000)
                        self.assertEqual(argv[:5], ["python", "-m", "vllm.entrypoints.cli.main", "serve", r["model"]])

    def test_bolt_capacity_adaptation(self):
        original = recipes.get("bolt-74b-lilicorr:c1")
        r = recipes.get("bolt-74b-lilicorr-5090:c1", "rtx5090_32g")
        self.assertEqual(r["serve"]["published_args"], original["serve"]["args"])
        expected = recipes.apply_flags(original["serve"]["args"], {
            "--max-model-len": "4096", "--gpu-memory-utilization": "0.95"})
        self.assertEqual(r["serve"]["args"], expected)
        self.assertEqual(normalize_recipe(r)["serve"]["args"], expected)
        self.assertEqual(r["workload"]["concurrency"], 1)
        self.assertTrue(r["serve"]["adapted"]["why"])
        argv, _ = serve_command("python", r["model"], r["serve"]["args"], None, 18000)
        self.assertEqual(argv[argv.index("--max-model-len") + 1], "4096")
        self.assertEqual(argv[argv.index("--gpu-memory-utilization") + 1], "0.95")

    def test_published_metadata_not_in_mission_brief(self):
        for r in recipes.load_all():
            r = copy.deepcopy(r)
            r["published"] = {"sentinel": "OPERATOR_METADATA_SENTINEL"}
            r["published_note"] = "OPERATOR_METADATA_SENTINEL"
            m = SimpleNamespace(recipe=r, id="static-check", recipe_id=r["id"],
                metric=r["workload"]["metric"], machine={"name": "test"}, gpu_class="test",
                time_box_min=1, budget_usd=1, n_workers=1, worker_names=["worker-1"],
                ws="/workspace", done_hash="0" * 64,
                criteria={"success": "gate", "termination": "budget"})
            text = mission_brief(m)
            self.assertNotIn("OPERATOR_METADATA_SENTINEL", text)
            self.assertNotIn("record_targets.md", text)


def load_tests(loader, standard_tests, pattern):
    names = {
        "test_catalog_loads_and_cites_sources",
        "test_variants_respect_published_max_num_seqs",
        "test_ids_and_lookup", "test_adapt_only_for_its_gpu_class",
        "test_apply_flags", "test_public_prompts_deterministic",
    }
    source = ROOT / "tests" / "fleet" / "test_fleet.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {f.name for f in funcs} != names:
        raise AssertionError("Existing catalog test names changed; review the wrapper")
    namespace = {name: getattr(recipes, name) for name in
                 ("apply_flags", "bases", "get", "load_all", "load_prompts")}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(source), "exec"), namespace)
    for name in sorted(names):
        standard_tests.addTest(unittest.FunctionTestCase(namespace[name], description=name))
    source = ROOT / "tests" / "agents" / "test_units.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "test_normalize_catalog_recipe"]
    if len(funcs) != 1:
        raise AssertionError("Existing normalization test missing")
    namespace = {"normalize_recipe": normalize_recipe}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(source), "exec"), namespace)
    standard_tests.addTest(unittest.FunctionTestCase(namespace[funcs[0].name]))
    return standard_tests
