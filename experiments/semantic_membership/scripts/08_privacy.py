#!/usr/bin/env python
"""Шаг 08: приватность индекса эмбеддингов (DESIGN.md §9) → results/privacy.json (таблица T6, рисунок F6).

    python scripts/08_privacy.py --config configs/default.yaml [--defences none,proj,proj+int8,proj+noise0.5]
                                 [--leaked-pairs 0 100 1000 10000] [--a1-backend auto|torch|ridge] [--a3]
                                 [--limit-train 50000] [--limit-eval 10000] [--no-full-df] [--force]

Входы: data/embeddings/{protected,protected_windows,public_train}.npz, data/embeddings/queries/<set>.npz (scripts/06_embed.py),
data/functions/{protected,protected_windows,public_train}.jsonl, data/queries/<set>.jsonl (коды запросов для полезности M4).
Состав индекса — как у основного индекса M3 (data/indexes/semantic/meta.json: with_windows; --with-windows/--no-windows —
явно), все npz должны быть одной модели (meta.model), иначе ошибка. Полезность M4 — с комбинатором основного индекса
(--no-hybrid — отключить). Без torch A1 обучается гребневой регрессией, A3 пропускается.
Защиты по умолчанию (см. --list-defences): если σ из cfg.privacy.noise_sigma дают ожидаемый cos > 0.97, сетка шума
дополняется σ ∈ {0.25, 0.5, 1, 2} (relative). Частоты df для словаря A1 считаются по всему public_train
(--no-full-df — по подвыборке атакующего). Идемпотентно: при наличии results/privacy.json шаг пропускается без --force.
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
    ap.add_argument("--no-full-df", dest="full_df", action="store_false", help="df словаря A1 по подвыборке атакующего, а не по всему public_train")
    ap.add_argument("--no-windows", dest="with_windows", action="store_const", const=False, default=None,
                    help="индекс только из функций protected (по умолчанию состав — как у data/indexes/semantic: with_windows из meta.json)")
    ap.add_argument("--with-windows", dest="with_windows", action="store_const", const=True, help="включить окна protected в индекс")
    ap.add_argument("--no-hybrid", dest="hybrid", action="store_const", const=False, default=None,
                    help="не считать полезность M4 под защитами (по умолчанию cfg.privacy.utility_hybrid)")
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
                          with_a3=args.a3, limit_train=args.limit_train, limit_eval=args.limit_eval, full_df=args.full_df,
                          with_windows=args.with_windows, hybrid=args.hybrid)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1
    brief = {"skipped": res.get("skipped"), "n_index": res.get("n_index"), "index_composition": res.get("index_composition"),
             "model": res.get("model"), "dim": res.get("dim"), "a1_backend": (res.get("a1") or {}).get("backend"),
             "hybrid_utility": (res.get("hybrid_utility") or {}).get("available"),
             "defenses": [{"name": d["name"], "memory_ratio": d.get("memory_ratio"), "expected_cos": d.get("expected_cos"),
                           "tpr": ((d.get("tpr") or {}).get("semantic") or {}).get("value"),
                           "tpr_hybrid": ((d.get("tpr") or {}).get("hybrid") or {}).get("value"),
                           "A1_f1": ((d.get("attacks") or {}).get("A1") or {}).get("f1"),
                           "A1_rare_recall": ((d.get("attacks") or {}).get("A1") or {}).get("rare_id_recall"),
                           "A1_rare_recall_chance": ((d.get("attacks") or {}).get("A1") or {}).get("rare_id_recall_chance"),
                           "A2_f1": {n: v.get("f1") for n, v in ((d.get("attacks") or {}).get("A2") or {}).items()}}
                          for d in res.get("defenses", [])],
             "errors": res.get("errors")}
    print(json.dumps(brief, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
