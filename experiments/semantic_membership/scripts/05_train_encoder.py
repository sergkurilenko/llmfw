#!/usr/bin/env python
"""Шаг 05 (GPU): контрастивное дообучение семантического энкодера (DESIGN.md §7) → runs/<name>/best/.

    python scripts/05_train_encoder.py --config configs/default.yaml
    python scripts/05_train_encoder.py --config configs/default.yaml --max_steps 20 --batch_size 8   # smoke
    python scripts/05_train_encoder.py --config configs/default.yaml --own_model                     # своя модель с нуля
    python scripts/05_train_encoder.py --config configs/default.yaml --adapt_on_protected            # абляция

Идемпотентно: готовый runs/<name>/best/encoder_meta.json пропускается без --force. Результат (путь best/)
указывается далее через --checkpoint в scripts/06_embed.py или cfg.semantic.checkpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smcode.config import load_config  # noqa: E402

log = logging.getLogger("05_train_encoder")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Контрастивное дообучение энкодера кода (GPU).")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--name", default="", help="имя запуска (runs/<name>); по умолчанию из модели")
    ap.add_argument("--base_model", "--base-model", dest="base_model", default=None, help="HF-модель (cfg.semantic.base_model)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max_steps", "--max-steps", dest="max_steps", type=int, default=None, help="ограничение шагов (smoke)")
    ap.add_argument("--adapt_on_protected", "--adapt-on-protected", dest="adapt_on_protected", action="store_true",
                    help="абляция: добавить protected в якоря (заказчик-специфичная адаптация)")
    ap.add_argument("--own_model", "--own-model", dest="own_model", action="store_true", help="своя модель с нуля (own_small_model)")
    ap.add_argument("--hard_negatives", "--hard-negatives", dest="hard_negatives", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max_length", "--max-length", dest="max_length", type=int, default=None)
    ap.add_argument("--grad_checkpointing", "--grad-checkpointing", dest="grad_checkpointing", action="store_true")
    ap.add_argument("--amp", choices=["auto", "bf16", "fp16", "none"], default="auto")
    ap.add_argument("--num_workers", "--num-workers", dest="num_workers", type=int, default=4)
    ap.add_argument("--val_pairs", "--val-pairs", dest="val_pairs", type=int, default=2000)
    ap.add_argument("--max_anchors", "--max-anchors", dest="max_anchors", type=int, default=None, help="подвыборка якорей (smoke)")
    ap.add_argument("--p_llm", "--p-llm", dest="p_llm", type=float, default=0.5, help="доля LLM-позитивов при их наличии")
    ap.add_argument("--log_every", "--log-every", dest="log_every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None, help="cuda|cpu (по умолчанию cuda, если доступна)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    try:
        from smcode.semantic.train import TrainConfig, train
    except ImportError as exc:
        log.error("нужен torch/transformers (GPU-машина, requirements-gpu.txt): %s", exc)
        return 2
    tc = TrainConfig(name=args.name, base_model=args.base_model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                     max_steps=args.max_steps, adapt_on_protected=args.adapt_on_protected, own_model=args.own_model,
                     hard_negatives=args.hard_negatives, temperature=args.temperature, max_length=args.max_length,
                     grad_checkpointing=args.grad_checkpointing, amp=args.amp, num_workers=args.num_workers,
                     val_pairs=args.val_pairs, max_anchors=args.max_anchors, p_llm=args.p_llm, log_every=args.log_every,
                     seed=args.seed, device=args.device, force=args.force)
    try:
        summary = train(cfg, tc)
    except ImportError as exc:
        log.error("нужен torch/transformers (GPU-машина, requirements-gpu.txt): %s", exc)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
