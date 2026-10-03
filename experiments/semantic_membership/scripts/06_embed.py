#!/usr/bin/env python
"""Шаг 06 (GPU/CPU): эмбеддинги функций и запросов выбранным чекпойнтом + латентность энкодера.

    python scripts/06_embed.py --config configs/default.yaml [--checkpoint runs/unixcoder_ft/best]
                               [--splits protected public_train protected_windows] [--sets protected hard_neg ...]
                               [--batch-size 64] [--no-latency] [--own-model] [--device cuda] [--force]

Выход: data/embeddings/<split>.npz (ids, emb float16), data/embeddings/queries/<set>.npz (qids, emb),
results/latency_encoder.json (CPU с cfg.eval.latency_threads потоками — по умолчанию 1 и 4, GPU; batch 1 и 32).
По умолчанию кодируются только сплиты, которые читают 04 и 08 (protected, public_train, protected_windows), и все
наборы запросов (QUERY_SETS + окна из cfg.windows.splits); другие сплиты — через --splits.
Идемпотентно по модели и содержимому: файлы, полученные этим же чекпойнтом (meta.model), пропускаются, если содержат
все id входного файла (иначе дозаписываются недостающие строки, например LLM-запросы после `03 --llm-only`); файлы
другой модели заменяются, а их копия остаётся в data/embeddings/by_model/<model_tag>/ (zero-shot и дообученные
эмбеддинги сосуществуют; повторный запуск с прежним чекпойнтом восстанавливает их из кэша без кодирования).
Без --checkpoint берётся cfg.semantic.checkpoint, иначе cfg.semantic.base_model (zero-shot).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smcode.config import load_config  # noqa: E402
from smcode.semantic.embed import DEFAULT_QUERY_SETS, DEFAULT_SPLITS, run_embed  # noqa: E402

log = logging.getLogger("06_embed")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Эмбеддинги функций/запросов и латентность энкодера.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", default=None, help="каталог чекпойнта (runs/<name>/best) или имя HF-модели")
    ap.add_argument("--splits", nargs="*", default=None, help=f"сплиты функций (по умолчанию {' '.join(DEFAULT_SPLITS)})")
    ap.add_argument("--sets", nargs="*", default=None,
                    help=f"наборы запросов (по умолчанию {' '.join(DEFAULT_QUERY_SETS)} + окна негативов из cfg.windows.splits)")
    ap.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=None)
    ap.add_argument("--no-latency", action="store_true", help="не замерять латентность")
    ap.add_argument("--latency-threads", "--latency_threads", dest="latency_threads", type=int, nargs="*", default=None,
                    help="число CPU-потоков для замера (по умолчанию cfg.eval.latency_threads = 1 4)")
    ap.add_argument("--own-model", "--own_model", dest="own_model", action="store_true", help="чекпойнт своей модели")
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    try:
        res = run_embed(cfg, checkpoint=args.checkpoint, splits=args.splits, sets=args.sets, batch_size=args.batch_size,
                        force=args.force, latency=not args.no_latency, own_model=args.own_model, device=args.device,
                        latency_threads=args.latency_threads or None)
    except ImportError as exc:
        log.error("нужен torch/transformers (GPU-машина, requirements-gpu.txt): %s", exc)
        return 2
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 2
    print(json.dumps({k: v for k, v in res.items() if k != "latency_report"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
