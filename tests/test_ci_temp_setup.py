"""CI environment setup must preserve the directory and all existing env entries."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import configure_ci_temp as setup  # noqa: E402


class CiTempSetupTests(unittest.TestCase):
    def test_all_temp_aliases_use_the_same_resolved_existing_directory(self):
        with tempfile.TemporaryDirectory() as name:
            expected = str(Path(name).resolve(strict=True))
            with patch.object(setup.tempfile, "gettempdir", return_value=name):
                self.assertEqual(setup.canonical_temp_environment(),
                                 dict(TMPDIR=expected, TEMP=expected, TMP=expected))

    def test_short_path_spelling_is_resolved_not_reused(self):
        with patch.object(setup.tempfile, "gettempdir", return_value="RUNNER~1/temp"), \
                patch.object(setup.Path, "resolve", return_value=Path("runneradmin/temp")) as resolve:
            result = setup.canonical_temp_environment()
        resolve.assert_called_once_with(strict=True)
        self.assertEqual(set(result.values()), {str(Path("runneradmin/temp"))})

    def test_appends_without_overwriting_existing_github_environment(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "github-env"
            path.write_text("EXISTING=preserved\n", encoding="utf-8")
            with patch.dict(os.environ, GITHUB_ENV=str(path)), \
                    patch.object(setup.tempfile, "gettempdir", return_value=name):
                self.assertEqual(setup.main(), 0)
            expected = str(Path(name).resolve(strict=True))
            self.assertEqual(path.read_text(encoding="utf-8"),
                             "EXISTING=preserved\n" + "".join(
                                 f"{key}={expected}\n" for key in ("TMPDIR", "TEMP", "TMP")))

    def test_rejects_multiline_environment_values(self):
        with patch.object(setup.Path, "resolve", return_value=Path("bad\nTEMP=other")), \
                self.assertRaisesRegex(ValueError, "single line"):
            setup.canonical_temp_environment()


if __name__ == "__main__":
    unittest.main()
