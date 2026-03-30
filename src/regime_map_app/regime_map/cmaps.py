from __future__ import annotations

from matplotlib import colormaps

DEFAULT_CMAP_NAME = "viridis"
CMAP_REFERENCE_URL = "https://matplotlib.org/stable/gallery/color/colormap_reference.html"
AVAILABLE_CMAP_NAMES = tuple(sorted(colormaps))
_CMAP_NAME_LOOKUP = {name.casefold(): name for name in AVAILABLE_CMAP_NAMES}


def resolve_cmap_name(name: str) -> str | None:
    normalized_name = name.strip()
    if not normalized_name:
        return DEFAULT_CMAP_NAME
    return _CMAP_NAME_LOOKUP.get(normalized_name.casefold())
