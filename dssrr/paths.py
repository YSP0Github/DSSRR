# -*- coding: utf-8 -*-
"""Canonical filesystem locations for DSSRR user data.

DSSRR keeps a very small amount of state in the user's home directory rather
than in the repository or the package directory:

============================  ==========================================
File                          Purpose
============================  ==========================================
``~/.dssrr_settings.json``    GUI / batch detection + repair parameters
``~/.dssrr_moonquake_catalog.json``
                              Optional user override of the bundled
                              Apollo moonquake event catalogue
============================  ==========================================

Both files are **optional**: if they are absent, sensible defaults are used.

Legacy names
------------
Earlier releases (and the ``seisy`` desktop application this package was
extracted from) used the ``seisy`` prefix::

    ~/.seisy_repair_settings.json
    ~/.seisy_moonquake_catalog.json

Reading still honours those names so an existing installation keeps working,
but everything is *written* under the ``dssrr`` name.  Call
:func:`settings_read_path` / :func:`catalog_override_read_path` rather than
hard-coding a filename, and the migration happens transparently.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Application identity (also used for QSettings / QStandardPaths organisation)
# ---------------------------------------------------------------------------
APP_NAME = "DSSRR"
APP_ORG = "DSSRR"
LEGACY_APP_NAME = "SeisY"

# ---------------------------------------------------------------------------
# File names
# ---------------------------------------------------------------------------
SETTINGS_FILENAME = ".dssrr_settings.json"
LEGACY_SETTINGS_FILENAME = ".seisy_repair_settings.json"

CATALOG_OVERRIDE_FILENAME = ".dssrr_moonquake_catalog.json"
LEGACY_CATALOG_OVERRIDE_FILENAME = ".seisy_moonquake_catalog.json"


def _home(*parts: str) -> str:
    """Join ``parts`` onto the current user's home directory."""
    return os.path.join(os.path.expanduser("~"), *parts)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
def user_settings_path() -> str:
    """Absolute path DSSRR **writes** its settings to."""
    return _home(SETTINGS_FILENAME)


def legacy_user_settings_path() -> str:
    """Absolute path of the pre-rename settings file (read-only fallback)."""
    return _home(LEGACY_SETTINGS_FILENAME)


def settings_read_path() -> str:
    """Absolute path DSSRR should **read** settings from.

    Prefers the current name; falls back to the legacy file when only that one
    exists; otherwise returns the current name (so a "file not found" simply
    means "use defaults").
    """
    current = user_settings_path()
    if os.path.exists(current):
        return current
    legacy = legacy_user_settings_path()
    if os.path.exists(legacy):
        return legacy
    return current


# ---------------------------------------------------------------------------
# Moonquake catalogue override
# ---------------------------------------------------------------------------
def catalog_override_path() -> str:
    """Absolute path of the user's moonquake-catalogue override file."""
    return _home(CATALOG_OVERRIDE_FILENAME)


def legacy_catalog_override_path() -> str:
    """Absolute path of the pre-rename catalogue override (read-only fallback)."""
    return _home(LEGACY_CATALOG_OVERRIDE_FILENAME)


def catalog_override_read_path() -> str | None:
    """Existing catalogue override path, or ``None`` when the user has none."""
    for path in (catalog_override_path(), legacy_catalog_override_path()):
        if os.path.exists(path):
            return path
    return None


__all__ = [
    "APP_NAME",
    "APP_ORG",
    "LEGACY_APP_NAME",
    "SETTINGS_FILENAME",
    "LEGACY_SETTINGS_FILENAME",
    "CATALOG_OVERRIDE_FILENAME",
    "LEGACY_CATALOG_OVERRIDE_FILENAME",
    "user_settings_path",
    "legacy_user_settings_path",
    "settings_read_path",
    "catalog_override_path",
    "legacy_catalog_override_path",
    "catalog_override_read_path",
]
