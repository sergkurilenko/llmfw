#!/usr/bin/env python
"""Шаг 07: прогон методов по наборам запросов (DESIGN.md §10) → results/scores/<method>/<set>.jsonl.

    python scripts/07_eval.py --config configs/default.yaml [--methods exact,winnowing,minhash,semantic,hybrid]
                              [--sets protected hard_neg public_calib public_test] [--batch 256] [--limit N]
                              [--bench 20] [--force]
Индекс берётся из data/indexes/<method>/. Для semantic/hybrid при наличии data/embeddings/queries/<set>.npz
используется numpy-путь (без torch); иначе query_batch (нужен энкодер). Идемпотентно: готовые файлы скоров
пропускаются без --force. --bench N — дополнительный замер латентности батча 1 на N запросах
(cfg.eval.latency_repeats повторов) → <set>.meta.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smcode.config import load_config  # noqa: E402
from smcode.eval.run_eval import DEFAULT_BATCH, available_methods, available_sets, run_eval  # noqa: E402
from smcode.fingerprint.index import REGISTRY  # noqa: E402

log = logging.getLogger("07_eval")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Оценка методов на наборах запросов → results/scores/")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--methods", default=None, help=f"через запятую (по умолчанию все с готовым индексом); известные: {','.join(REGISTRY)}")
    ap.add_argument("--sets", nargs="*", default=None, help="наборы запросов (по умолчанию все существующие)")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="размер батча numpy-пути semantic/hybrid")
    ap.add_argument("--limit", type=int, default=None, help="только первые N запросов каждого набора (отладка)")
    ap.add_argument("--bench", type=int, default=0, help="замер латентности батча 1 на N запросах (0 — выкл.)")
    ap.add_argument("--force", action="store_true", help="пересчитать даже при наличии файлов скоров")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()] if args.methods else available_methods(cfg)
    unknown = [m for m in methods if m not in REGISTRY]
    if unknown:
        ap.error(f"unknown methods {unknown}; known: {sorted(REGISTRY)}")
    if not methods:
        log.error("нет готовых индексов в data/indexes/ (запустите scripts/04_build_indexes.py)")
        return 1
    sets = args.sets or available_sets(cfg)
    if not sets:
        log.error("нет наборов запросов в data/queries/ (запустите scripts/03_build_queries.py)")
        return 1
    log.info("methods: %s; sets: %s", ",".join(methods), ",".join(sets))
    res = run_eval(cfg, methods=methods, sets=sets, force=args.force, batch=args.batch, limit=args.limit, bench=args.bench)
    brief = {m: ({"error": r["error"]} if "error" in r else {s: (v.get("n_scored", "skipped") if isinstance(v, dict) else v) for s, v in r.items()})
             for m, r in res.items()}
    print(json.dumps(brief, ensure_ascii=False, indent=1))
    return 1 if any("error" in r for r in res.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
