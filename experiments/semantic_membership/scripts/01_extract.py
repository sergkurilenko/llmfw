#!/usr/bin/env python
"""Шаг 01: извлечение функций из клонов (DESIGN.md §2.2) -> data/functions/<split>.jsonl
и data/functions/protected_windows.jsonl.

    python scripts/01_extract.py --config configs/default.yaml [--splits protected public_train]
                                 [--workers 4] [--no-windows] [--force]
Идемпотентно: готовые файлы сплитов пропускаются без --force.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smcode.config import load_config  # noqa: E402
from smcode.data.extract import run_extract  # noqa: E402
from smcode.types import SPLITS  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Извлечение функций tree-sitter из data/raw")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--splits", nargs="*", default=None, choices=list(SPLITS), help="подмножество сплитов")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-windows", action="store_true", help="не строить файловые окна protected")
    ap.add_argument("--force", action="store_true", help="пересчитать даже при наличии выходных файлов")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    outputs = run_extract(cfg, splits=args.splits, workers=args.workers, force=args.force, with_windows=not args.no_windows)
    for name, path in outputs.items():
        logging.getLogger("01_extract").info("%-17s -> %s", name, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
