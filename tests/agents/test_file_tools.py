"""File tool contracts against a temporary, simulated remote filesystem."""
import fnmatch
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from forge.tools.builtin import (build_registry, t_apply_patch, t_edit, t_glob,
                                 t_ls, t_read, t_search)
from forge.tools.registry import EnvBlocked, ToolError


class FakeExecutor:
    def __init__(self, root):
        self.root = Path(root)
        self.writes = []
        self.commands = []

    def local(self, path):
        return self.root / path.lstrip("/")

    def exists(self, path):
        return self.local(path).exists()

    def read_text(self, path):
        return self.local(path).read_text(encoding="utf-8")

    def write_text(self, path, text):
        p = self.local(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        self.writes.append(path)

    def remove(self, path):
        self.writes.append(path)
        self.local(path).unlink(missing_ok=True)

    def list_files(self, root):
        return sorted(p.relative_to(self.local(root)).as_posix()
                      for p in self.local(root).rglob("*") if p.is_file())

    def listdir(self, path):
        return sorted(p.name for p in self.local(path).iterdir())

    def run(self, cmd, **kwargs):
        self.commands.append(cmd)
        result = SimpleNamespace(rc=0, stdout="", stderr="", infra_error=None)
        if cmd[0] == "test":
            result.rc = 0 if self.local(cmd[-1]).is_dir() else 1
        elif cmd[0] == "find":
            root = self.local(cmd[1])
            if not root.exists():
                result.rc, result.stderr = 1, "missing directory"
                return result
            for p in sorted(root.rglob("*")):
                name = p.relative_to(root).as_posix()
                if "-maxdepth" in cmd:
                    if len(name.split("/")) <= int(cmd[cmd.index("-maxdepth") + 1]):
                        result.stdout += f"{name}\0{'d' if p.is_dir() else 'f'}\0{p.stat().st_size}\0"
                elif p.is_file():
                    patterns = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-path"]
                    if any(fnmatch.fnmatchcase(cmd[1].rstrip("/") + "/" + name, pat) for pat in patterns):
                        result.stdout += name + "\0"
        elif cmd[0] == "grep":
            pattern, root = cmd[-2:]
            regex = re.compile(pattern, re.I if "-i" in cmd else 0)
            includes = [arg.split("=", 1)[1] for arg in cmd if arg.startswith("--include=")]
            context = int(cmd[cmd.index("-C") + 1])
            for p in sorted(self.local(root).rglob("*")):
                if not p.is_file() or not any(fnmatch.fnmatchcase(p.name, pat) for pat in includes):
                    continue
                lines = p.read_text().splitlines()
                matches = {i for i, line in enumerate(lines) if regex.search(line)}
                selected = {j for i in matches for j in range(max(0, i-context), min(len(lines), i+context+1))}
                for i in sorted(selected):
                    sep = ":" if i in matches else "-"
                    path = root.rstrip("/") + "/" + p.relative_to(self.local(root)).as_posix()
                    result.stdout += f"{path}{sep}{i+1}{sep}{lines[i]}\n"
            result.rc = 0 if result.stdout else 1
        else:
            raise AssertionError(cmd)
        return result


class FileToolsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ex = FakeExecutor(self.tmp.name)
        self.ex.local("/ws").mkdir()
        mission = SimpleNamespace(ws="/ws", site_packages="/site", readable_roots=["/model"],
                                  id="test", packages=lambda: ["vllm"],
                                  overlay_root=lambda agent: "/copies/" + agent)
        self.ctx = SimpleNamespace(executor=self.ex, mission=mission, agent="a")

    def patch(self, body):
        return t_apply_patch(self.ctx, "*** Begin Patch\n" + body + "*** End Patch\n")

    def test_patch_add_update_delete_and_move(self):
        result = self.patch("*** Add File: a.txt\n+one\n+two\n")
        self.assertEqual(result["files"], ["a.txt"])
        self.assertIn("+one", result["diff"])
        self.patch("*** Update File: a.txt\n@@\n one\n-two\n+three\n")
        self.assertEqual(self.ex.read_text("/ws/a.txt"), "one\nthree\n")
        self.patch("*** Update File: a.txt\n*** Move to: sub/b.txt\n@@ one\n-three\n+four\n*** End of File\n")
        self.assertFalse(self.ex.exists("/ws/a.txt"))
        self.assertEqual(self.ex.read_text("/ws/sub/b.txt"), "one\nfour\n")
        self.patch("*** Delete File: sub/b.txt\n")
        self.assertFalse(self.ex.exists("/ws/sub/b.txt"))

    def test_atomic_failure(self):
        self.ex.write_text("/ws/a", "one\ntwo\n")
        self.ex.writes.clear()
        with self.assertRaisesRegex(ToolError, "a: hunk did not match.*missing"):
            self.patch("*** Add File: new\n+created\n*** Update File: a\n@@\n-one\n+ONE\n@@\n-missing\n+bad\n")
        self.assertEqual(self.ex.writes, [])
        self.assertFalse(self.ex.exists("/ws/new"))
        self.assertEqual(self.ex.read_text("/ws/a"), "one\ntwo\n")

    def test_overlay_copy_on_write_and_revert(self):
        self.ex.write_text("/site/vllm/a.py", "original\n")
        t_edit(self.ctx, "vllm/a.py", content="edited\n")
        self.patch("*** Update File: vllm/a.py\n@@\n-edited\n+patched\n")
        self.assertEqual(self.ex.read_text("/site/vllm/a.py"), "original\n")
        self.assertIn("patched", t_read(self.ctx, "vllm/a.py")["text"])
        result = self.patch("*** Delete File: vllm/a.py\n")
        self.assertIn("+original", result["diff"])
        self.assertFalse(self.ex.exists("/copies/a/vllm/a.py"))
        self.assertIn("original", t_read(self.ctx, "vllm/a.py")["text"])

    def test_whitespace_edit(self):
        for actual, old in [("  hello  \n  world\n", "  hello\n  world\n"),
                            ("    hello\n    world\n", "hello\nworld\n")]:
            self.ex.write_text("/ws/a", actual)
            result = t_edit(self.ctx, "a", old=old, new="done\n")
            self.assertEqual(self.ex.read_text("/ws/a"), "done\n")
            self.assertIn("+done", result["diff"])
        self.ex.write_text("/ws/a", " hi \n hi \n")
        with self.assertRaises(ToolError):
            t_edit(self.ctx, "a", old="hi\n", new="x", replace_all=True)
        self.ex.write_text("/ws/a", "prefix hello  \nhello  \n")
        t_edit(self.ctx, "a", old="hello\n", new="done\n")
        self.assertEqual(self.ex.read_text("/ws/a"), "prefix hello  \ndone\n")

    def test_replace_all_and_diff_limit(self):
        self.ex.write_text("/ws/a", "x x x")
        with self.assertRaises(ToolError):
            t_edit(self.ctx, "a", old="x", new="y")
        self.assertEqual(t_edit(self.ctx, "a", old="x", new="y", replace_all=True)["replaced"], 3)
        self.assertEqual(self.ex.read_text("/ws/a"), "y y y")
        self.assertLessEqual(len(t_edit(self.ctx, "a", content="z" * 5000)["diff"]), 2000)

    def test_ls_glob_and_directory_read(self):
        self.ex.write_text("/site/vllm/mamba.py", "abc")
        self.ex.write_text("/site/vllm/sub/mamba2.py", "abcd")
        t_edit(self.ctx, "vllm/mamba.py", content="edited")
        t_edit(self.ctx, "vllm/added.py", content="new")
        rows = {r["path"]: r for r in t_ls(self.ctx, "vllm")["entries"]}
        self.assertEqual(rows["mamba.py"]["size"], 6)
        self.assertTrue(rows["mamba.py"]["edited"])
        self.assertIn("added.py", rows)
        self.assertNotIn("sub/mamba2.py", rows)
        self.assertEqual(t_read(self.ctx, "vllm"), t_ls(self.ctx, "vllm"))
        self.assertEqual(len(t_ls(self.ctx, "vllm", depth=2)["entries"]), 4)
        self.assertEqual(t_glob(self.ctx, "**/*mamba*.py")["files"],
                         ["vllm/mamba.py", "vllm/sub/mamba2.py"])
        self.assertEqual(t_ls(self.ctx)["entries"], [])
        self.assertEqual(t_read(self.ctx, "/ws")["entries"], [])

    def test_search_context_and_filters(self):
        self.ex.write_text("/site/vllm/a.cpp", "before\nNEEDLE\nafter\n")
        self.ex.write_text("/site/vllm/b.py", "needle\n")
        result = t_search(self.ctx, "needle", include="*.cpp", ignore_case=True, context=1)
        self.assertEqual(result["hits"], ["vllm/a.cpp-1-before", "vllm/a.cpp:2:NEEDLE", "vllm/a.cpp-3-after"])
        self.assertEqual(t_search(self.ctx, "needle", ignore_case=True)["total"], 2)
        self.assertTrue(t_search(self.ctx, "needle", ignore_case=True, max_hits=1)["truncated"])

    def test_limits_and_readonly(self):
        for i in range(305):
            self.ex.write_text(f"/ws/{i}.py", "x")
        self.assertEqual(len(t_ls(self.ctx)["entries"]), 300)
        result = t_glob(self.ctx, "*.py", path=".")
        self.assertEqual(len(result["files"]), 200)
        self.assertTrue(result["truncated"])
        with self.assertRaises(ToolError):
            self.patch("*** Add File: /model/file\n+x\n")
        self.assertFalse(self.ex.exists("/model/file"))

    def test_registry_and_infrastructure(self):
        reg = build_registry()
        for name in ("apply_patch", "ls", "glob"):
            self.assertIn("worker", reg.tools[name].roles)
        self.ex.run = lambda *a, **kw: SimpleNamespace(infra_error="offline")
        with self.assertRaises(EnvBlocked):
            t_glob(self.ctx, "*", ".")


if __name__ == "__main__":
    unittest.main()
