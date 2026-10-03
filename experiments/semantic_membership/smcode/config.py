"""Загрузка конфигурации стенда и разрешение путей (DESIGN.md §11)."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]  # experiments/semantic_membership/
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Читает YAML-конфиг; `overrides` — вложенный словарь, объединяемый поверх."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(path)
    if overrides:
        _deep_update(cfg, overrides)
    return cfg


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    """Путь из cfg['paths'][key], относительно корня стенда (ROOT), с созданием каталога."""
    p = Path(cfg["paths"][key])
    if not p.is_absolute():
        p = ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_rng(cfg: dict[str, Any], salt: str = "") -> random.Random:
    """Детерминированный ГПСЧ: seed из конфига + соль (имя этапа)."""
    seed = int(cfg.get("seed", 0))
    return random.Random(f"{seed}:{salt}")


def hmac_key(cfg: dict[str, Any]) -> bytes | None:
    """Ключ HMAC отпечатков из переменной окружения (DESIGN.md §6, M1). None — без ключа (только тесты)."""
    env = cfg.get("fingerprint", {}).get("hmac_key_env", "SMCODE_INDEX_KEY")
    val = os.environ.get(env)
    return val.encode("utf-8") if val else None
