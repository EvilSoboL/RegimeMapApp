from __future__ import annotations

from pathlib import Path


project_root = Path.cwd()
icon_path = project_root / "src" / "regime_map_app" / "assets" / "app_icon.ico"


a = Analysis(
    ["src/regime_map_app/__main__.py"],
    pathex=[str(project_root), str(project_root / "src")],
    binaries=[],
    datas=[
        (str(icon_path), "regime_map_app/assets"),
    ],
    hiddenimports=[
        "matplotlib.backends.backend_qtagg",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="RegimeMapApp",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(icon_path),
)
