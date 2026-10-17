"""Fallback containment integration tests; run on a Linux rig with Landlock ABI 6."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from forge import sandbox


class DetectionTest(unittest.TestCase):
    def test_probe_result_cached(self):
        for result in (0, 1):
            with self.subTest(result=result), \
                 mock.patch.dict(os.environ, {"FORGE_SANDBOX_NO_SYSTEMD": "0"}), \
                 mock.patch.object(sandbox, "_systemd_available", None), \
                 mock.patch.object(sandbox, "_user_bus_env", return_value={}), \
                 mock.patch.object(sandbox.subprocess, "run", return_value=subprocess.CompletedProcess([], result)) as run:
                self.assertEqual(sandbox._use_systemd(), result == 0)
                self.assertEqual(sandbox._use_systemd(), result == 0)
                run.assert_called_once()

    def test_missing_or_timed_out_systemd_cached(self):
        for error in (FileNotFoundError(), subprocess.TimeoutExpired('systemd-run', 10)):
            with self.subTest(error=error), \
                 mock.patch.dict(os.environ, {"FORGE_SANDBOX_NO_SYSTEMD": "0"}), \
                 mock.patch.object(sandbox, "_systemd_available", None), \
                 mock.patch.object(sandbox, "_user_bus_env", return_value={}), \
                 mock.patch.object(sandbox.subprocess, "run", side_effect=error) as run:
                self.assertFalse(sandbox._use_systemd())
                self.assertFalse(sandbox._use_systemd())
                run.assert_called_once()

    def test_force_overrides_cached_success(self):
        with mock.patch.dict(os.environ, {"FORGE_SANDBOX_NO_SYSTEMD": "1"}), \
             mock.patch.object(sandbox, "_systemd_available", True), \
             mock.patch.object(sandbox.subprocess, "run") as run:
            self.assertFalse(sandbox._use_systemd())
            run.assert_not_called()


@unittest.skipUnless(sys.platform == 'linux' and sandbox.landlock.abi_version() >= 6,
                     'needs Linux with Landlock ABI >= 6; run on the rig')
class FallbackTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='.forge_fallback_', dir=os.path.expanduser('~'))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.force = mock.patch.dict(os.environ, {'FORGE_SANDBOX_NO_SYSTEMD': '1'})
        self.force.start()
        self.addCleanup(self.force.stop)

    def launch(self, code, **kw):
        job_dir = tempfile.mkdtemp(prefix='job-', dir=self.root)
        job = sandbox.launch(sandbox.Spec(
            cmd=[sys.executable, '-c', code], job_dir=job_dir,
            ro=list({sys.prefix, sys.base_prefix}), **kw))
        self.addCleanup(lambda: job.stop(grace=0))
        self.assertEqual(job.containment, 'pgroup-fallback')
        return job

    def wait_for_pid(self, job):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            for line in job.tail().splitlines():
                if line.startswith('GRANDCHILD '):
                    return int(line.split()[1])
            if job.poll() is not None:
                self.fail(job.tail())
            time.sleep(0.02)
        self.fail('grandchild did not start: ' + job.tail())

    def assert_gone(self, pid):
        deadline = time.monotonic() + 5
        while Path(f'/proc/{pid}').exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertFalse(Path(f'/proc/{pid}').exists(), f'process {pid} survived')

    def test_setsid_grandchild_killed(self):
        # The intermediate process exits immediately; the grandchild ignores TERM.
        # Test both inherited and stripped environments (subreaper ancestry path).
        for stripped in (False, True):
            with self.subTest(stripped=stripped):
                middle = (
                    "import subprocess; "
                    "p = subprocess.Popen(['/bin/sh', '-c', 'trap \\\"\\\" TERM; exec sleep 300'], "
                    f"start_new_session=True, env={'{}' if stripped else 'None'}); "
                    "print('GRANDCHILD', p.pid, flush=True)"
                )
                job = self.launch(
                    f"import subprocess, sys, time; subprocess.run([sys.executable, '-c', {middle!r}]); "
                    "time.sleep(300)")
                pid = self.wait_for_pid(job)
                self.assertEqual(job.kill(grace=0.1), [])
                self.assert_gone(pid)
                self.assertIsNotNone(job.poll())
                self.assertEqual(job.wait(), job.returncode)
                with open(job._report_path) as f:
                    self.assertIn(pid, json.load(f)['pids'])

    def test_normal_exit_reaps_detached_child(self):
        job = self.launch("import subprocess; p = subprocess.Popen(['sleep', '300'], start_new_session=True); "
                          "print('GRANDCHILD', p.pid, flush=True)")
        self.assertEqual(job.wait(timeout=10), 0, job.tail())
        pid = int(job.tail().split('GRANDCHILD ')[1].split()[0])
        self.assert_gone(pid)
        self.assertEqual(job.survivors, [])

    def test_timeout_reaps(self):
        job = self.launch("import time; time.sleep(300)")
        self.assertIsNotNone(job.wait(timeout=0.1))
        self.assertEqual(job.survivors, [])

    def test_landlock_read_denied_and_clean_env(self):
        secret = self.root / 'secret'
        secret.write_text('hidden')
        with mock.patch.dict(os.environ, {'FORGE_PARENT_SECRET': 'must-not-leak'}):
            job = self.launch(
                f"import os\ntry:\n open({str(secret)!r}).read()\n"
                "except PermissionError:\n print('READ_DENIED')\n"
                "else:\n raise AssertionError('secret readable')\n"
                "open('allowed', 'w').write('ok')\n"
                "assert 'FORGE_PARENT_SECRET' not in os.environ\n"
                "assert 'FORGE_JOB_TOKEN' in os.environ\n")
        self.assertEqual(job.wait(timeout=10), 0, job.tail())
        self.assertIn('READ_DENIED', job.tail())
        self.assertEqual((Path(job.job_dir) / 'ws/allowed').read_text(), 'ok')

    def test_scope_start_failure_falls_back_and_is_cached(self):
        real_popen = subprocess.Popen
        attempts = []

        def popen(argv, **kwargs):
            if argv[0] == 'systemd-run':
                attempts.append(argv)
                return real_popen(['/bin/false'], **kwargs)
            return real_popen(argv, **kwargs)

        with mock.patch.dict(os.environ, {'FORGE_SANDBOX_NO_SYSTEMD': '0'}), \
             mock.patch.object(sandbox, '_systemd_available', True), \
             mock.patch.object(sandbox.subprocess, 'Popen', side_effect=popen):
            for _ in range(2):
                job = self.launch("print('started once')")
                self.assertEqual(job.wait(timeout=10), 0, job.tail())
                self.assertEqual(job.tail().count('started once'), 1)
            self.assertEqual(len(attempts), 1)


if __name__ == '__main__':
    unittest.main()
