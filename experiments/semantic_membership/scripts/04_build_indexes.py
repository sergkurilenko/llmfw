#!/usr/bin/env python
"""Шаг 04: построение индексов членства (DESIGN.md §6) из data/functions/protected.jsonl
(+ protected_windows.jsonl для winnowing при --with-windows) -> data/indexes/<method>/;
память и время сборки -> results/index_stats.json.

    python scripts/04_build_indexes.py --config configs/default.yaml [--methods exact,winnowing,minhash]
                                       [--with-windows] [--workers 4] [--force]
Идемпотентно: индекс с готовым meta.json пропускается без --force. Методы semantic/hybrid
делегируются make_index(...).build (нужен torch); при ImportError шаг пропускается с сообщением.
Фильтр общности winnowing подгоняется по data/functions/public_train.jsonl, если файл есть.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smcode.config import load_config  # noqa: E402
from smcode.fingerprint.index import FINGERPRINT_METHODS, GPU_METHODS, REGISTRY, build_from_corpus, write_index_stats  # noqa: E402

log = logging.getLogger("04_build_indexes")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Построение индексов членства из data/functions/protected.jsonl")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--methods", default=",".join(FINGERPRINT_METHODS), help=f"через запятую; известные: {','.join(REGISTRY)}")
    ap.add_argument("--with-windows", action="store_true", help="добавить protected_windows.jsonl в индекс winnowing")
    ap.add_argument("--workers", type=int, default=1, help="процессов для отпечатков (cfg.fingerprint.workers)")
    ap.add_argument("--force", action="store_true", help="пересобрать даже при наличии индекса")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    cfg.setdefault("fingerprint", {})["workers"] = max(1, args.workers)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in REGISTRY]
    if unknown:
        ap.error(f"unknown methods {unknown}; known: {sorted(REGISTRY)}")

    stats: dict[str, dict] = {}
    failed: list[str] = []
    for m in methods:
        try:
            stats[m] = build_from_corpus(m, cfg, with_windows=args.with_windows, force=args.force)
        except ImportError as exc:
            if m in GPU_METHODS:
                log.error("%s: пропуск — нужны torch/transformers (GPU-машина): %s", m, exc)
            else:
                log.error("%s: ImportError: %s", m, exc)
            failed.append(m)
            continue
        except FileNotFoundError as exc:
            log.error("%s: %s", m, exc)
            failed.append(m)
            continue
        except Exception as exc:  # noqa: BLE001 — GPU-методы могут падать по множеству причин
            if m in GPU_METHODS:
                log.exception("%s: сборка не удалась: %s", m, exc)
                failed.append(m)
                continue
            raise
        s = stats[m]
        log.info(
            "%-10s records=%s build=%.1fs memory=%.1f MB disk=%.1f MB%s",
            m, s.get("n_records"), float(s.get("build_seconds") or 0.0), (s.get("memory_bytes") or 0) / 1e6,
            (s.get("disk_bytes") or 0) / 1e6, " (skipped, exists)" if s.get("skipped") else "",
        )
    if stats:
        path = write_index_stats(cfg, stats)
        log.info("stats -> %s", path)
        print(json.dumps({m: {k: v for k, v in s.items() if k != "meta"} for m, s in stats.items()}, ensure_ascii=False, indent=1))
    if failed:
        log.warning("не построены: %s", ", ".join(failed))
    return 1 if any(m in FINGERPRINT_METHODS for m in failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
