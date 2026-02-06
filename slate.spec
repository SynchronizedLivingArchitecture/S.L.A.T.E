# -*- mode: python ; coding: utf-8 -*-
# Modified: 2026-02-06T22:30:00Z | Author: COPILOT | Change: SLATE PyInstaller spec (replaces Aurora.spec)
# ═══════════════════════════════════════════════════════════════════════════════
# S.L.A.T.E. PyInstaller Build Specification
# Builds the SLATEPI standalone executable for Windows
# ═══════════════════════════════════════════════════════════════════════════════

import os
import sys

block_cipher = None

a = Analysis(
    ['slate/slate_sdk.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('slate/*.py', 'slate'),
        ('agents/*.py', 'agents'),
        ('slate_web/*.html', 'slate_web'),
        ('slate_web/*.css', 'slate_web'),
        ('slate_web/*.js', 'slate_web'),
        ('docs/assets/*.svg', 'docs/assets'),
        ('requirements.txt', '.'),
        ('pyproject.toml', '.'),
    ],
    hiddenimports=[
        'slate.slate_status',
        'slate.slate_runtime',
        'slate.slate_runner_manager',
        'slate.slate_mcp_server',
        'agents.slate_dashboard_server',
        'agents.install_api',
        'agents.runner_api',
        'psutil',
        'aiohttp',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'scipy'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SLATEPI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SLATEPI',
)
