"""Active language selector — resolves strings_tr or strings_en per config.language.

Modules import this (`from . import strings as S`); the concrete language is chosen
once in config.yaml (language: "en" | "tr"). strings_tr.py is kept for reference.
"""
from . import config as _cfg

if str(_cfg.CFG.get("language", "tr")).lower() == "en":
    from .strings_en import *  # noqa: F401,F403
else:
    from .strings_tr import *  # noqa: F401,F403
