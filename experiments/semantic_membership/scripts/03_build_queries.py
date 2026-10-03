#!/usr/bin/env python
"""Шаг 03: построение наборов запросов (DESIGN.md §5) из data/functions/*.jsonl → data/queries/*.jsonl.

Примеры:
  python scripts/03_build_queries.py --config configs/default.yaml
  python scripts/03_build_queries.py --config configs/default.yaml --sets protected public_calib --force
  python scripts/03_build_queries.py --config configs/default.yaml --llm        # GPU: paraphrase/translate (vLLM)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smcode.config import load_config  # noqa: E402
from smcode.eval.build_queries import ALL_SETS, build_queries  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Построение наборов запросов (программные и LLM-преобразования).")
    ap.add_argument("--config", default="configs/default.yaml", help="путь к YAML-конфигу")
    ap.add_argument("--sets", nargs="*", default=None, help=f"наборы (по умолчанию: {' '.join(ALL_SETS)})")
    ap.add_argument("--max-per-split", type=int, default=None, help="максимум функций на набор (cfg: transforms.max_per_split, 5000)")
    ap.add_argument("--force", action="store_true", help="перестроить, даже если файлы уже есть")
    ap.add_argument("--llm", action="store_true", help="добавить LLM-преобразования paraphrase/translate (GPU, vLLM)")
    ap.add_argument("--llm-sample", type=int, default=None, help="функций на набор для LLM (cfg: transforms.llm_sample_per_split)")
    ap.add_argument("--llm-only", action="store_true", help="пропустить программные преобразования, только LLM")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    if not args.llm_only:
        res = build_queries(cfg, splits=args.sets, force=args.force, max_per_split=args.max_per_split)
        for k, v in res.items():
            logging.info("%s: %s", k, "skipped (exists)" if v < 0 else f"{v} queries")
    if args.llm or args.llm_only:
        from smcode.transforms.llm_paraphrase import build_llm_queries

        res = build_llm_queries(cfg, splits=args.sets, sample=args.llm_sample, force=args.force)
        for k, v in res.items():
            logging.info("llm %s: %s", k, "skipped (exists)" if v < 0 else f"{v} queries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
