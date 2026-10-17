"""ABI compatibility and achieved-level reporting; no GPU or pytest required."""
import errno
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from forge import sandbox
from forge.sandbox import enter, landlock


class AbiTest(unittest.TestCase):
    def setUp(self):
        policy = mock.patch.dict(os.environ, {
            "FORGE_SANDBOX_MIN_ABI": "1", "FORGE_SANDBOX_ALLOW_NONE": "0"})
        policy.start()
        self.addCleanup(policy.stop)

    def test_rule_sets(self):
        for abi in (0, 1, 2, 3, 4, 5, 6):
            with self.subTest(abi=abi):
                attrs, paths, ports = [], [], []

                def syscall(number, *args):
                    if number == landlock.SYS_CREATE_RULESET:
                        attr = args[0]._obj
                        attrs.append((attr.handled_access_fs, attr.handled_access_net,
                                      attr.scoped, args[1].value))
                        return 100
                    if number == landlock.SYS_ADD_RULE:
                        rule = args[2]._obj
                        if args[1].value == landlock.RULE_PATH_BENEATH:
                            paths.append(rule.allowed_access)
                        else:
                            ports.append((rule.allowed_access, rule.port))
                    return 0

                libc = mock.Mock()
                libc.syscall.side_effect = syscall
                libc.prctl.return_value = 0
                with mock.patch.object(landlock, "abi_version", return_value=abi), \
                     mock.patch.object(landlock, "_libc", return_value=libc), \
                     mock.patch.object(os, "O_PATH", 0, create=True), \
                     mock.patch.object(os, "O_CLOEXEC", 0, create=True), \
                     mock.patch.object(os, "open", return_value=101), \
                     mock.patch.object(os, "close"), \
                     mock.patch.object(os.path, "isdir", return_value=True):
                    result = landlock.apply([("/allowed", (1 << 16) - 1)], [123], [456],
                                            allow_none=True)
                self.assertEqual(result, abi)
                if abi == 0:
                    self.assertEqual(attrs, [])
                    libc.prctl.assert_not_called()
                    continue
                expected = (1 << 13) - 1
                if abi >= 2:
                    expected |= landlock.REFER
                if abi >= 3:
                    expected |= landlock.TRUNCATE
                if abi >= 5:
                    expected |= landlock.IOCTL_DEV
                self.assertEqual(attrs, [(expected, 3 if abi >= 4 else 0,
                                         3 if abi >= 6 else 0,
                                         24 if abi >= 6 else 16 if abi >= 4 else 8)])
                self.assertEqual(paths, [expected])
                self.assertEqual(ports, [(1, 123), (2, 456)] if abi >= 4 else [])
                self.assertEqual(libc.syscall.call_args.args[0], landlock.SYS_RESTRICT_SELF)

    def test_none_requires_explicit_opt_in(self):
        for failure in (0, OSError(errno.ENOSYS, "unavailable"),
                        OSError(errno.EOPNOTSUPP, "disabled")):
            with self.subTest(failure=failure):
                probe = dict(side_effect=failure) if isinstance(failure, OSError) else dict(return_value=0)
                with mock.patch.object(landlock, "abi_version", **probe), \
                     mock.patch.object(landlock, "_libc") as libc:
                    with self.assertRaises(landlock.LandlockUnavailable):
                        landlock.apply([], [], [])
                    with mock.patch.dict(os.environ, {"FORGE_SANDBOX_ALLOW_NONE": "1"}):
                        self.assertEqual(landlock.apply([], [], []), 0)
                    libc.assert_not_called()

    def test_minimum_and_launch_rejection(self):
        with mock.patch.object(landlock, "abi_version", return_value=3):
            self.assertEqual(landlock.checked_abi(), 3)
            with mock.patch.dict(os.environ, {"FORGE_SANDBOX_MIN_ABI": "4",
                                              "FORGE_SANDBOX_ALLOW_NONE": "1"}):
                with self.assertRaises(landlock.LandlockUnavailable):
                    landlock.apply([], [], [])
                with self.assertRaises(sandbox.SandboxError):
                    sandbox.launch(sandbox.Spec(cmd=["true"], job_dir="unused"))

    def test_unexpected_probe_error_fails_closed(self):
        with mock.patch.object(landlock, "abi_version", side_effect=OSError(errno.EPERM, "denied")):
            with self.assertRaises(OSError):
                landlock.checked_abi(allow_none=True)

    def test_unrestricted_network_is_not_reported_as_enforced(self):
        with tempfile.TemporaryDirectory() as td:
            spec_path = Path(td) / "spec.json"
            spec_path.write_text(json.dumps({"env": {}, "paths": [], "cwd": td,
                                            "cmd": ["/bin/true"]}))
            with mock.patch.object(landlock, "apply", return_value=6), \
                 mock.patch.object(os, "chdir"), mock.patch.object(os, "umask"), \
                 mock.patch.object(enter.shutil, "which", return_value="/bin/true"), \
                 mock.patch.object(os, "execve"):
                enter.main(str(spec_path))
            self.assertFalse(json.loads((Path(td) / "sandbox_level.json").read_text())["net_rules"])

    def test_entry_records_applied_level(self):
        for abi in (0, 1, 3, 4, 6):
            for containment in ("systemd", "pgroup"):
                with self.subTest(abi=abi, containment=containment), tempfile.TemporaryDirectory() as td:
                    spec_path = Path(td) / "spec.json"
                    spec_path.write_text(json.dumps({
                        "cmd": ["/bin/true"], "env": {}, "cwd": td, "paths": [],
                        "bind_ports": [], "connect_ports": [], "min_abi": 1, "allow_none": True}))
                    with mock.patch.object(landlock, "apply", return_value=abi) as apply, \
                         mock.patch.object(os, "chdir"), mock.patch.object(os, "umask"), \
                         mock.patch.object(enter.shutil, "which", return_value="/bin/true"), \
                         mock.patch.object(os, "execve") as execute:
                        enter.main(str(spec_path), containment)
                    self.assertEqual(json.loads((Path(td) / "sandbox_level.json").read_text()), {
                        "landlock_abi": abi, "fs_rules": abi > 0, "net_rules": abi >= 4,
                        "containment": containment})
                    self.assertTrue(apply.call_args.kwargs["allow_none"])
                    execute.assert_called_once()

    def test_failed_apply_does_not_report_success_or_exec(self):
        with tempfile.TemporaryDirectory() as td:
            spec_path = Path(td) / "spec.json"
            spec_path.write_text(json.dumps({"env": {}, "paths": []}))
            with mock.patch.object(landlock, "apply", side_effect=OSError("failed")), \
                 mock.patch.object(os, "umask"), mock.patch.object(os, "execve") as execute:
                with self.assertRaises(OSError):
                    enter.main(str(spec_path))
            self.assertEqual((Path(td) / "sandbox_level.json").read_text(), "")
            execute.assert_not_called()


@unittest.skipUnless(sys.platform == "linux", "Linux process containment required")
class LevelIntegrationTest(unittest.TestCase):
    def test_launch_records_level(self):
        # This also exercises ABI 0 hosts with an explicit operator opt-in.
        with tempfile.TemporaryDirectory(dir=os.path.expanduser("~")) as td, \
             mock.patch.dict(os.environ, {"FORGE_SANDBOX_MIN_ABI": "1",
                                          "FORGE_SANDBOX_ALLOW_NONE": "1",
                                          "FORGE_SANDBOX_NO_SYSTEMD": "1"}):
            job = sandbox.launch(sandbox.Spec(cmd=["/bin/true"], job_dir=td))
            try:
                self.assertEqual(job.wait(timeout=10), 0)
            finally:
                job.stop(grace=0)
            self.assertEqual(job.sandbox_level, {
                "landlock_abi": landlock.abi_version(),
                "fs_rules": landlock.abi_version() > 0,
                "net_rules": landlock.abi_version() >= 4, "containment": "pgroup"})
            self.assertEqual(job.sandbox_level,
                             json.loads((Path(td) / "sandbox_level.json").read_text()))


if __name__ == "__main__":
    unittest.main()
