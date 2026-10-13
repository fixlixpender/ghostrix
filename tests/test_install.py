"""install.sh tests, using a fake ollama binary. No models are pulled for real."""

import os
import stat
import subprocess
import tempfile
import textwrap
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTALL = os.path.join(REPO, "install.sh")

FAKE_OLLAMA = textwrap.dedent("""\
    #!/usr/bin/env bash
    # Fake ollama: `list` prints FAKE_MODELS; `pull` logs to FAKE_LOG.
    case "$1" in
      list)
        echo "NAME                ID      SIZE    MODIFIED"
        for m in $FAKE_MODELS; do echo "$m  abc  1GB  now"; done
        ;;
      pull)
        echo "$2" >> "$FAKE_LOG"
        if [ "$2" = "$FAKE_FAIL_MODEL" ]; then exit 1; fi
        ;;
      *) exit 0 ;;
    esac
""")


class InstallScriptTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.fake = os.path.join(self.dir, "ollama")
        self.log = os.path.join(self.dir, "pulls.log")
        with open(self.fake, "w") as f:
            f.write(FAKE_OLLAMA)
        os.chmod(self.fake, os.stat(self.fake).st_mode | stat.S_IEXEC)

    def tearDown(self):
        self._tmp.cleanup()

    def run_install(self, models="", args=(), fail_model=""):
        env = {
            **os.environ,
            "OLLAMA_BIN": self.fake,
            "FAKE_MODELS": models,
            "FAKE_LOG": self.log,
            "FAKE_FAIL_MODEL": fail_model,
        }
        return subprocess.run(
            ["bash", INSTALL, *args], env=env, capture_output=True, text=True
        )

    def pulls(self):
        if not os.path.exists(self.log):
            return []
        with open(self.log) as f:
            return [line.strip() for line in f if line.strip()]

    def test_pulls_both_models_when_none_present(self):
        r = self.run_install(models="")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.pulls(), ["qwen3:latest", "nomic-embed-text"])

    def test_skips_models_already_present(self):
        r = self.run_install(models="qwen3:latest")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.pulls(), ["nomic-embed-text"])
        self.assertIn("ok: qwen3:latest already present", r.stdout)

    def test_latest_tag_counts_as_present(self):
        r = self.run_install(models="nomic-embed-text:latest qwen3:latest")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.pulls(), [])

    def test_dry_run_changes_nothing(self):
        r = self.run_install(models="", args=("--dry-run",))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.pulls(), [])
        self.assertIn("would pull: qwen3:latest", r.stdout)

    def test_missing_ollama_exits_1_with_hint(self):
        env = {**os.environ, "OLLAMA_BIN": os.path.join(self.dir, "no-such-binary")}
        r = subprocess.run(["bash", INSTALL], env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn("Ollama not found", r.stderr)

    def test_failed_pull_exits_2(self):
        r = self.run_install(models="", fail_model="qwen3:latest")
        self.assertEqual(r.returncode, 2)
        self.assertIn("failed to pull qwen3:latest", r.stderr)

    def test_unknown_argument_rejected(self):
        r = self.run_install(args=("--yolo",))
        self.assertEqual(r.returncode, 1)


if __name__ == "__main__":
    unittest.main()
