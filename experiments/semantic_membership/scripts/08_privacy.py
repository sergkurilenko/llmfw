#!/usr/bin/env python
"""Шаг 08: приватность индекса эмбеддингов (DESIGN.md §9) → results/privacy.json (таблица T6, рисунок F6).

    python scripts/08_privacy.py --config configs/default.yaml [--defences none,proj,proj+int8,proj+noise0.1]
                                 [--leaked-pairs 0 100 1000 10000] [--a1-backend auto|torch|ridge] [--a3]
                                 [--limit-train 50000] [--limit-eval 10000] [--force]

Входы: data/embeddings/{protected,public_train}.npz, data/embeddings/queries/<set>.npz (scripts/06_embed.py),
data/functions/{protected,public_train}.jsonl. Без torch A1 обучается гребневой регрессией, A3 пропускается.
Идемпотентно: при наличии results/privacy.json шаг пропускается без --force.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smcode.config import load_config  # noqa: E402
from smcode.privacy.run import default_defences, run_privacy  # noqa: E402

log = logging.getLogger("08_privacy")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Приватность индекса эмбеддингов: защиты × (полезность, атаки A1/A2/A3)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--defences", "--defenses", dest="defences", default=None,
                    help="через запятую: none, proj, proj+int8, proj+int4, proj+bin, proj+noise0.1, ... (по умолчанию из конфига)")
    ap.add_argument("--leaked-pairs", "--leaked_pairs", dest="leaked_pairs", nargs="*", type=int, default=None,
                    help="n утёкших пар для A2 (по умолчанию cfg.privacy.attack_leaked_pairs)")
    ap.add_argument("--a1-backend", "--a1_backend", dest="a1_backend", choices=["auto", "torch", "ridge"], default=None)
    ap.add_argument("--a3", action="store_true", help="запустить генеративную инверсию A3 (нужен torch, GPU)")
    ap.add_argument("--limit-train", "--limit_train", dest="limit_train", type=int, default=None, help="макс. функций public_train для атак")
    ap.add_argument("--limit-eval", "--limit_eval", dest="limit_eval", type=int, default=None, help="макс. функций protected для оценки атак")
    ap.add_argument("--list-defences", action="store_true", help="показать защиты по умолчанию и выйти")
    ap.add_argument("--force", action="store_true", help="пересчитать даже при наличии results/privacy.json")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    if args.list_defences:
        print("\n".join(d.name for d in default_defences(cfg)))
        return 0
    defences = [d.strip() for d in args.defences.split(",") if d.strip()] if args.defences else None
    try:
        res = run_privacy(cfg, force=args.force, defences=defences, leaked_pairs=args.leaked_pairs, a1_backend=args.a1_backend,
                          with_a3=args.a3, limit_train=args.limit_train, limit_eval=args.limit_eval)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1
    brief = {"skipped": res.get("skipped"), "n_index": res.get("n_index"), "dim": res.get("dim"), "a1_backend": (res.get("a1") or {}).get("backend"),
             "defenses": [{"name": d["name"], "memory_ratio": d.get("memory_ratio"),
                           "tpr": ((d.get("tpr") or {}).get("semantic") or {}).get("value"),
                           "A1_f1": ((d.get("attacks") or {}).get("A1") or {}).get("f1"),
                           "A2_f1": {n: v.get("f1") for n, v in ((d.get("attacks") or {}).get("A2") or {}).items()}}
                          for d in res.get("defenses", [])],
             "errors": res.get("errors")}
    print(json.dumps(brief, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
