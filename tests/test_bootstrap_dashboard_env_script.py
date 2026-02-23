from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_bootstrap_dashboard_env_rejects_python_397_with_clear_message(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "bootstrap_dashboard_env.sh"
    fake_python = tmp_path / "python397"

    _write_executable(
        fake_python,
        """#!/usr/bin/env bash
if [[ "$1" == "-c" ]]; then
  echo "3.9.7"
  exit 0
fi
echo "unexpected invocation: $*" >&2
exit 2
""",
    )

    result = subprocess.run(
        ["bash", str(script_path), str(tmp_path / "venv-dashboard"), str(fake_python)],
        cwd=str(repo_root),
        env=dict(os.environ),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    combined = (result.stdout or "") + (result.stderr or "")
    assert "Python 3.9.7" in combined
    assert "nao e compativel com streamlit==1.39.0" in combined
    assert "Python 3.11 (recomendado)" in combined
    assert "PYTHON_BIN=python3.11" in combined

