#!/usr/bin/env python
"""Шаг 02: дедупликация корпуса функций (DESIGN.md §2.3): точные дубликаты внутри сплитов,
near-dup фильтр hard_neg/public_* против protected по winnowing-отпечаткам, stats.json.

    python scripts/02_dedup_split.py --config configs/default.yaml [--workers 4] [--force]
Идемпотентно: при наличии data/functions/stats.json шаг пропускается без --force.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smcode.config import load_config  # noqa: E402
from smcode.data.dedup_split import run_dedup  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Дедупликация data/functions/*.jsonl")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="выполнить заново даже при наличии stats.json")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    stats = run_dedup(cfg, workers=args.workers, force=args.force)
    summary = {s: {k: v[k] for k in ("n_raw", "n_final", "removed_exact", "removed_near_dup")} for s, v in stats.get("splits", {}).items()}
    logging.getLogger("02_dedup_split").info("итог: %s", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
