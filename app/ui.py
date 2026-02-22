from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    from streamlit.web import cli as stcli

    repo_root = Path(__file__).resolve().parents[1]
    app_path = repo_root / "src" / "talent_scouting_intel" / "ui" / "streamlit_app.py"

    args = sys.argv[1:]
    sys.argv = ["streamlit", "run", str(app_path), *args]
    raise SystemExit(stcli.main())


if __name__ == "__main__":
    main()
