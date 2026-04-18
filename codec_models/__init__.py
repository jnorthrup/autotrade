"""
Codec implementations for the 24 SOTA models
=============================================

Registry / discovery module.  Import from the autotrade root:

    from codec_models import load_all_codecs, list_available_codecs

All 24 codecs work without MLX (numpy-only fallback).  If mlx is
available the codecs transparently upgrade to Apple-Silicon-native ops.
"""

import os
import glob
import importlib
from typing import List, Dict, Any, Type

from .base_codec import BaseExpert


# ── Loading ────────────────────────────────────────────────────────────

def load_all_codecs() -> List[Type[BaseExpert]]:
    """
    Dynamically loads all 24 codec experts from the codec_models directory.
    Guarantees canonical ordering according to GOALS.md.
    """
    codec_dir = os.path.dirname(__file__)
    codec_files = sorted(glob.glob(os.path.join(codec_dir, "codec_[0-2][0-9]_*.py")))

    codecs: List[Type[BaseExpert]] = []

    for file_path in codec_files:
        module_name = os.path.basename(file_path)[:-3]
        module = importlib.import_module(f".{module_name}", package="codec_models")

        # Find the class that inherits from BaseExpert
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, BaseExpert) and attr is not BaseExpert:
                codecs.append(attr)
                break

    return codecs


# ── Registry / discovery ───────────────────────────────────────────────

def list_available_codecs() -> List[Dict[str, Any]]:
    """
    Return a list of dicts describing every available codec.

    Each entry contains:
        id          – integer codec ID (1-24)
        class_name  – e.g. 'Codec01'
        name        – human-readable strategy name (e.g. 'volatility_breakout')
        module      – source module filename
        class_obj   – the actual class (for instantiation)

    Example:
        >>> from codec_models import list_available_codecs
        >>> for info in list_available_codecs():
        ...     print(f"  {info['id']:>2}  {info['name']:<30}  {info['class_name']}")
    """
    codecs = load_all_codecs()
    registry: List[Dict[str, Any]] = []

    for cls in codecs:
        # Derive numeric ID from class name (Codec01 -> 1, Codec24 -> 24)
        raw = cls.__name__.replace("Codec", "")
        try:
            codec_id = int(raw)
        except ValueError:
            codec_id = 0

        # Human-readable name comes from the default instance config
        try:
            instance = cls()
            name = instance.name
        except Exception:
            name = cls.__name__

        module_file = os.path.basename(cls.__module__.replace("codec_models.", ""))
        if module_file == "__init__":
            module_file = cls.__module__

        registry.append({
            "id": codec_id,
            "class_name": cls.__name__,
            "name": name,
            "module": module_file,
            "class_obj": cls,
        })

    return registry


__all__ = ["BaseExpert", "load_all_codecs", "list_available_codecs"]
