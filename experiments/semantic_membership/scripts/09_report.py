#!/usr/bin/env python
"""Шаг 09: метрики → results/summary.json (+ summary_schema.md) → results/tables/T1..T7.md → paper/figures/F1..F7.png.

    python scripts/09_report.py --config configs/default.yaml [--methods exact,winnowing] [--bootstrap 200]
                                [--no-tables] [--no-figures] [--force]
Идемпотентно: готовый results/summary.json переиспользуется, если он актуален (тот же набор каталогов
results/scores/*, нет файлов скоров новее summary.json, --methods/--bootstrap не заданы); иначе — пересчёт
(--force — всегда). Таблицы и рисунки строятся из summary всегда. В конце печатается краткая сводка на русском.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smcode.config import load_config  # noqa: E402
from smcode.eval.report import format_summary, run_report  # noqa: E402

log = logging.getLogger("09_report")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Метрики, таблицы и рисунки по results/scores/")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--methods", default=None, help="через запятую (по умолчанию все каталоги results/scores/*)")
    ap.add_argument("--bootstrap", type=int, default=None, help="число повторов бутстрэпа (cfg.eval.bootstrap)")
    ap.add_argument("--no-tables", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--force", action="store_true", help="пересчитать summary.json даже если он актуален")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()] if args.methods else None
    res = run_report(cfg, force=args.force, bootstrap=args.bootstrap, methods=methods, tables=not args.no_tables, figures=not args.no_figures)
    log.info("summary.json %s", f"пересчитан ({res['reason']})" if res["recomputed"] else "переиспользован")
    print(format_summary(res["summary"], res["tables"], res["figures"]))
    return 0 if res["summary"].get("methods") else 1


if __name__ == "__main__":
    raise SystemExit(main())
