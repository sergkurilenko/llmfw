#!/usr/bin/env python
"""Шаг 04: построение индексов членства (DESIGN.md §6) из data/functions/protected.jsonl
(+ protected_windows.jsonl для всех методов при --with-windows) -> data/indexes/<method>/;
память и время сборки -> results/index_stats.json.

    python scripts/04_build_indexes.py --config configs/default.yaml [--methods exact,winnowing,minhash]
                                       [--with-windows] [--workers 4] [--set key=value ...] [--force]
    # zero-shot вариант семантического индекса в отдельный каталог (абляция, DESIGN §7):
    python scripts/04_build_indexes.py --methods semantic --set semantic.checkpoint=null --set paths.indexes=data/indexes_zeroshot
--set key=value (повторяемый, значение — YAML) переопределяет конфиг только для этого запуска — те же правила,
что у scripts/07_eval.py. Идемпотентно: индекс с готовым meta.json пропускается без --force (индекс без окон пересобирается,
если запрошены окна; индекс старше входных файлов data/functions/*.jsonl, data/embeddings/*.npz пересобирается).
Методы semantic/hybrid делегируются make_index(...).build (готовые эмбеддинги из data/embeddings, иначе torch).
Код возврата: 1, если хотя бы один запрошенный метод не построен (run_all.sh останавливается); --allow-missing-gpu
делает ImportError GPU-методов (нет torch на CPU-машине) нефатальным. Фильтр общности winnowing подгоняется по
data/functions/public_train.jsonl, если файл есть. Без $SMCODE_INDEX_KEY индекс строится без ключа
(предупреждение): для реального эксперимента ключ обязателен (DESIGN §1(d), §6), и тот же ключ
должен быть задан при оценке (07): несовпадение ключа при загрузке индекса — ошибка.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smcode.config import load_config  # noqa: E402
from smcode.eval.run_eval import apply_overrides, parse_overrides  # noqa: E402
from smcode.fingerprint.index import (  # noqa: E402
    FINGERPRINT_METHODS,
    GPU_METHODS,  # noqa: F401 — реестр методов, требующих torch
    REGISTRY,
    build_from_corpus,
    warn_if_unkeyed,
    write_index_stats,
)

log = logging.getLogger("04_build_indexes")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Построение индексов членства из data/functions/protected.jsonl")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--methods", default=",".join(FINGERPRINT_METHODS), help=f"через запятую; известные: {','.join(REGISTRY)}")
    ap.add_argument("--with-windows", action="store_true", help="добавить protected_windows.jsonl в индексы (все методы)")
    ap.add_argument("--workers", type=int, default=None, help="процессов для отпечатков (по умолчанию cfg.fingerprint.workers)")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                    help="переопределение конфига, например semantic.checkpoint=null или paths.indexes=data/indexes_zeroshot (повторяемый)")
    ap.add_argument("--force", action="store_true", help="пересобрать даже при наличии индекса")
    ap.add_argument("--allow-missing-gpu", action="store_true",
                    help="код 0, если semantic/hybrid не построены только из-за отсутствия torch/transformers (CPU-машина)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    try:
        overrides = parse_overrides(args.overrides)
    except ValueError as exc:
        ap.error(str(exc))
    if overrides:
        cfg = apply_overrides(cfg, overrides)
        log.info("config overrides: %s", overrides)
    if args.workers is not None:
        cfg.setdefault("fingerprint", {})["workers"] = max(1, args.workers)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in REGISTRY]
    if unknown:
        ap.error(f"unknown methods {unknown}; known: {sorted(REGISTRY)}")
    warn_if_unkeyed(cfg, "indexes")

    # вариантный каталог индексов (--set paths.indexes=...) не должен затирать статистику базовых индексов в index_stats.json
    variant_tag = Path(str(cfg["paths"]["indexes"])).name if "indexes" in (overrides.get("paths") or {}) else None
    stats: dict[str, dict] = {}
    failed: list[str] = []
    missing_gpu: list[str] = []  # GPU-методы, не построенные только из-за ImportError (нет torch)
    for m in methods:
        key = f"{m}@{variant_tag}" if variant_tag else m
        try:
            stats[key] = build_from_corpus(m, cfg, with_windows=args.with_windows, force=args.force)
        except ImportError as exc:
            if m in GPU_METHODS:
                log.error("%s: пропуск — нужны torch/transformers (GPU-машина): %s", m, exc)
                missing_gpu.append(m)
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
        s = stats[key]
        log.info(
            "%-10s records=%s build=%.1fs memory=%.1f MB disk=%.1f MB windows=%s%s",
            key, s.get("n_records"), float(s.get("build_seconds") or 0.0), (s.get("memory_bytes") or 0) / 1e6,
            (s.get("disk_bytes") or 0) / 1e6, s.get("with_windows"), " (skipped, exists)" if s.get("skipped") else "",
        )
    if stats:
        path = write_index_stats(cfg, stats)
        log.info("stats -> %s", path)
        print(json.dumps({m: {k: v for k, v in s.items() if k != "meta"} for m, s in stats.items()}, ensure_ascii=False, indent=1))
    if failed:
        tolerated = args.allow_missing_gpu and all(m in missing_gpu for m in failed)
        log.log(logging.WARNING if tolerated else logging.ERROR, "не построены: %s%s", ", ".join(failed),
                " (допустимо: --allow-missing-gpu)" if tolerated else "")
        return 0 if tolerated else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
