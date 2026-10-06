# -*- coding: utf-8 -*-
"""Backward-compatibility alias for :mod:`dssrr.config`.

The DSSRR code base used to carry **two** configuration dictionaries: the
trimmed public one in ``dssrr/config.py`` and a larger private one here that
additionally declared the ``detect``, ``reference`` and ``layered_synthesis``
sections used by the GUI / batch pipelines.

Both have been merged into the single canonical :data:`dssrr.config.DEFAULT_CONFIG`
— it is a superset, so the lightweight core simply ignores the sections it does
not read, and the whole project now shares one configuration object.

There is now exactly one place to change defaults: :mod:`dssrr.config`.

See also
--------
:func:`dssrr.config.get_config`
    Returns a deep copy of the defaults with optional per-section overrides.
:data:`dssrr.config.DEFAULT_CONFIG`
    The merged default dictionary.
"""
from __future__ import annotations

from ...config import DEFAULT_CONFIG, get_config

__all__ = ["DEFAULT_CONFIG", "get_config"]
