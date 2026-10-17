"""CPU-only nullpool tests; run with python -m unittest."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from forge.fleet import nullpool as n


class NullPoolTests(unittest.TestCase):
    def setUp(self):
        self.hw = {"name": "GPU", "gpu_class": "class-a", "power": 300}
        self.verdict = {"gate_id": "g1", "kind": "reference", "pairs": [{"log_gain": 0.0}]}
        self.h = {"null_gains": [0.0], "controls": ["g1"], "null_probe": [], "null_decode": []}

    def pairs(self):
        return n.extract_pairs(self.h, "one", self.hw, lambda _: self.verdict)

    def test_compatibility(self):
        self.assertTrue(n.compatible(self.hw, self.hw))
        self.assertFalse(n.compatible(self.hw, {**self.hw, "gpu_class": "class-b"}))
        self.assertTrue(n.compatible(self.hw, {**self.hw, "name": "alias", "power": 330}))
        self.assertFalse(n.compatible(self.hw, {**self.hw, "name": "alias", "power": 600}))
        self.assertFalse(n.compatible(self.hw, {**self.hw, "name": "alias", "power": None}))

    def test_legacy_and_repeat(self):
        ps = self.pairs()
        self.assertEqual(ps[0]["pooled_from"], "one")
        self.assertEqual(n.merge_pairs(ps, ps, self.hw), ps)
        self.h["nullpool_pairs"] = ps
        self.assertEqual(self.pairs(), ps)

    def test_new_local_control_after_pooling(self):
        self.h["nullpool_pairs"] = self.pairs()
        self.h["null_gains"].append(0.0)
        self.h["controls"].append("g2")
        ps = n.extract_pairs(self.h, "one", self.hw,
                             lambda gid: {**self.verdict, "gate_id": gid})
        self.assertEqual([p["gate_id"] for p in ps], ["g1", "g2"])

    def test_unmapped_fails_closed(self):
        self.h["null_gains"].append(0.0)
        with self.assertRaises(ValueError):
            self.pairs()

    def test_conflict_and_cross_class(self):
        ps = self.pairs()
        with self.assertRaises(ValueError):
            n.merge_pairs(ps, [{**ps[0], "pooled_from": "other"}], self.hw)
        other = {**ps[0], "hardware": {**self.hw, "gpu_class": "class-b"}}
        self.assertEqual(n.merge_pairs([], [other], self.hw), [])

    def test_atomic_preserves_unknown_fields(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "history.json"
            n.atomic_json(p, {**self.h, "future": True})
            self.assertTrue(json.loads(p.read_text())["future"])
            self.assertEqual(len(list(Path(td).iterdir())), 1)

    def test_shrink_only_queued_targets(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "gate/requests").mkdir(parents=True)
            rows = {}
            for gid, status, recipe in [("g1", "queued", "super3-120b-a12b-nvfp4:c1"),
                                        ("g2", "running", "super3-120b-a12b-nvfp4:c1"),
                                        ("g3", "queued", "unrelated:c1")]:
                req = {"gate_id": gid, "kind": "reference", "recipe_key": recipe, "reps": 6,
                       "history_key": recipe, "lock": "original"}
                n.atomic_json(root / "gate/requests" / (gid + ".json"), req)
                rows[gid] = {**req, "status": status}
            changed = n.shrink_locked(root, rows)
            self.assertEqual(len(changed), 1)
            self.assertEqual(changed[0]["old"], "g1")
            new = n.read_json(root / "gate/requests" / (changed[0]["new"] + ".json"))
            self.assertEqual(new["reps"], 4)
            self.assertEqual(new["lock"], "original")
            self.assertEqual(n.current_rows(root)["g1"]["status"], "superseded")

    @unittest.skipIf(os.name == "nt", "gate locking is Linux-only")
    def test_remote_merge_is_idempotent_and_preserves_local_bands(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "gate/history").mkdir(parents=True)
            h = {"null_gains": [], "null_probe": [{"local": True}], "null_decode": [], "controls": []}
            p = root / "gate/history/recipe:c1.json"
            n.atomic_json(p, h)
            payload = {"machine": {"name": "two", "forge_root": td}, "op": "merge",
                       "histories": {"recipe:c1": self.pairs()}}
            with patch.object(n, "hardware_for", return_value=self.hw):
                first = n.remote(payload)
                second = n.remote(payload)
            self.assertEqual(first["counts"]["recipe:c1"]["after"], 1)
            self.assertEqual(second["counts"]["recipe:c1"]["before"], 1)
            saved = n.read_json(p)
            self.assertEqual(saved["null_probe"], h["null_probe"])
            self.assertEqual(saved["controls"], [])
            self.assertEqual(saved["nullpool_pairs"][0]["pooled_from"], "one")

    @unittest.skipIf(os.name == "nt", "gate locking is Linux-only")
    def test_busy_service_cannot_write(self):
        import fcntl
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "gate").mkdir()
            with (root / "gate/service.lock").open("a") as f:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(n, "hardware_for", return_value=self.hw):
                    result = n.remote({"machine": {"name": "test", "forge_root": td}, "op": "shrink"})
                self.assertEqual(result["status"], "service_lock_busy")
            self.assertFalse((root / "gate/ledger.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
