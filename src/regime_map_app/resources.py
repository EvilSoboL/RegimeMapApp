from __future__ import annotations

import sys
from pathlib import Path


def resolve_app_icon_path() -> Path | None:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        candidate = Path(sys._MEIPASS) / "regime_map_app" / "assets" / "app_icon.ico"
    else:
        candidate = Path(__file__).resolve().parent / "assets" / "app_icon.ico"
    return candidate if candidate.exists() else None
