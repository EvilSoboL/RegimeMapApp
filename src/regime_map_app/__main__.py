from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    project_src = Path(__file__).resolve().parents[1]
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))
    from regime_map_app.app import main
else:
    from .app import main


if __name__ == "__main__":
    raise SystemExit(main())
