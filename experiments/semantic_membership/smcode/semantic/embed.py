"""Батчевый инференс энкодера и хранение эмбеддингов (DESIGN.md §7, embed.py).

Контракты на диске:
  data/embeddings/<split>.npz           — ids (unicode), emb (float16, (N, d)), meta (JSON-строка)
  data/embeddings/queries/<set>.npz     — qids (unicode), emb (float16, (M, d)), meta
  results/latency_encoder.json          — латентность энкодера: CPU 1 поток / все потоки / GPU, batch 1 и 32

Функции чтения (load_embeddings, embeddings_for_records при наличии файлов) работают без torch;
torch импортируется лениво внутри run_embed / measure_latency.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from smcode.config import get_rng, resolve_path
from smcode.types import SPLITS, read_jsonl

log = logging.getLogger(__name__)

EMBEDDINGS_DIR_DEFAULT = "data/embeddings"
QUERIES_SUBDIR = "queries"
LATENCY_FILE = "latency_encoder.json"
WINDOWS_SPLIT = "protected_windows"
DEFAULT_QUERY_SETS: tuple[str, ...] = ("protected", "hard_neg", "public_calib", "public_test", "protected_windows")
LATENCY_BATCH_SIZES: tuple[int, ...] = (1, 32)
LATENCY_SAMPLE = 64

Encoder = Any  # CodeEncoder | OwnEncoder (smcode.semantic.model / own_model)


# ----------------------------------------------------------------------------- пути


def embeddings_dir(cfg: dict[str, Any]) -> Path:
    """paths.embeddings (по умолчанию data/embeddings), каталог создаётся."""
    cfg.setdefault("paths", {}).setdefault("embeddings", EMBEDDINGS_DIR_DEFAULT)
    return resolve_path(cfg, "embeddings")


def split_embeddings_path(cfg: dict[str, Any], split: str) -> Path:
    return embeddings_dir(cfg) / f"{split}.npz"


def query_embeddings_path(cfg: dict[str, Any], set_name: str) -> Path:
    d = embeddings_dir(cfg) / QUERIES_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{set_name}.npz"


# ----------------------------------------------------------------------------- чтение/запись


def save_embeddings(path: str | Path, ids: Sequence[str], emb: np.ndarray, key: str = "ids",
                    meta: dict[str, Any] | None = None) -> Path:
    """Сохраняет npz: <key> (unicode), emb (float16), meta (JSON-строка)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    emb = np.asarray(emb)
    if emb.ndim != 2 or emb.shape[0] != len(ids):
        raise ValueError(f"emb shape {emb.shape} does not match {len(ids)} ids")
    arrays = {key: np.asarray(list(ids), dtype=str), "emb": emb.astype(np.float16),
              "meta": np.asarray(json.dumps(meta or {}, ensure_ascii=False))}
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    return path


def load_embeddings(path: str | Path) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    """Читает npz с ids|qids + emb → (ids, emb float32 (N, d), meta)."""
    with np.load(Path(path), allow_pickle=False) as z:
        key = "ids" if "ids" in z.files else "qids"
        ids = [str(x) for x in z[key].tolist()]
        emb = np.asarray(z["emb"], dtype=np.float32)
        meta: dict[str, Any] = {}
        if "meta" in z.files:
            try:
                meta = json.loads(str(z["meta"]))
            except (json.JSONDecodeError, TypeError):
                meta = {}
    return ids, emb, meta


def _records_of(cfg: dict[str, Any], split: str) -> list[dict[str, Any]]:
    path = resolve_path(cfg, "functions") / f"{split}.jsonl"
    if not path.exists():
        return []
    return list(read_jsonl(path))


def _queries_of(cfg: dict[str, Any], set_name: str) -> list[dict[str, Any]]:
    path = resolve_path(cfg, "queries") / f"{set_name}.jsonl"
    if not path.exists():
        return []
    return list(read_jsonl(path))


# ----------------------------------------------------------------------------- эмбеддинги записей


def embeddings_for_records(
    cfg: dict[str, Any],
    encoder: Encoder | Callable[[], Encoder],
    ids: Sequence[str],
    codes: Sequence[str],
    langs: Sequence[str],
    split: str | None = None,
    batch_size: int | None = None,
    reuse: bool | None = None,
) -> np.ndarray:
    """Эмбеддинги (N, d) float32 для записей: берутся из data/embeddings/<split>.npz, если файл есть
    и покрывает id (cfg.semantic.reuse_embeddings, по умолчанию True); остальное кодируется энкодером.
    ``encoder`` — объект с encode(codes, langs, batch_size) либо фабрика без аргументов (ленивая загрузка)."""
    scfg = cfg.get("semantic", {}) or {}
    reuse = bool(scfg.get("reuse_embeddings", True)) if reuse is None else reuse
    bs = int(batch_size or scfg.get("batch_size", 32))
    n = len(ids)
    out: np.ndarray | None = None
    missing = np.ones(n, dtype=bool)
    if reuse and split:
        path = split_embeddings_path(cfg, split)
        if path.exists():
            f_ids, f_emb, meta = load_embeddings(path)
            pos = {rid: i for i, rid in enumerate(f_ids)}
            rows = np.array([pos.get(rid, -1) for rid in ids], dtype=np.int64)
            found = rows >= 0
            if found.any():
                out = np.zeros((n, f_emb.shape[1]), dtype=np.float32)
                out[found] = f_emb[rows[found]]
                missing = ~found
                log.info("embeddings for %s: %d/%d reused from %s (model=%s)", split, int(found.sum()), n, path,
                         meta.get("model"))
    if missing.any():
        enc = encoder() if callable(encoder) and not hasattr(encoder, "encode") else encoder
        idx = np.flatnonzero(missing)
        emb = enc.encode([codes[i] for i in idx], [langs[i] for i in idx], batch_size=bs)
        emb = np.asarray(emb, dtype=np.float32)
        if out is None:
            out = np.zeros((n, emb.shape[1]), dtype=np.float32)
        elif out.shape[1] != emb.shape[1]:
            raise ValueError(f"embedding dim mismatch: file {out.shape[1]} vs encoder {emb.shape[1]}")
        out[idx] = emb
    if out is None:  # n == 0
        out = np.zeros((0, 0), dtype=np.float32)
    return out


def embed_split(cfg: dict[str, Any], split: str, encoder: Encoder, batch_size: int | None = None,
                force: bool = False) -> Path | None:
    """data/functions/<split>.jsonl → data/embeddings/<split>.npz (ids, emb). None, если файла функций нет."""
    out = split_embeddings_path(cfg, split)
    if out.exists() and not force:
        log.info("embed %s: exists, skipping (%s)", split, out)
        return out
    rows = _records_of(cfg, split)
    if not rows:
        log.warning("embed %s: no functions file, skipping", split)
        return None
    bs = int(batch_size or cfg.get("semantic", {}).get("batch_size", 32))
    t0 = time.perf_counter()
    emb = encoder.encode([r["code"] for r in rows], [r.get("lang", "") for r in rows], batch_size=bs, show_progress=True)
    dt = time.perf_counter() - t0
    meta = {"split": split, "model": getattr(encoder, "model_name", None), "dim": int(emb.shape[1]),
            "n": len(rows), "seconds": round(dt, 2), "device": str(getattr(encoder, "device", "")),
            "max_length": getattr(encoder, "max_length", None)}
    save_embeddings(out, [r["id"] for r in rows], emb, key="ids", meta=meta)
    log.info("embed %s: %d functions in %.1fs → %s", split, len(rows), dt, out)
    return out


def embed_query_set(cfg: dict[str, Any], set_name: str, encoder: Encoder, batch_size: int | None = None,
                    force: bool = False) -> Path | None:
    """data/queries/<set>.jsonl → data/embeddings/queries/<set>.npz (qids, emb)."""
    out = query_embeddings_path(cfg, set_name)
    if out.exists() and not force:
        log.info("embed queries %s: exists, skipping (%s)", set_name, out)
        return out
    rows = _queries_of(cfg, set_name)
    if not rows:
        log.warning("embed queries %s: no query file, skipping", set_name)
        return None
    bs = int(batch_size or cfg.get("semantic", {}).get("batch_size", 32))
    t0 = time.perf_counter()
    emb = encoder.encode([r["code"] for r in rows], [r.get("lang", "") for r in rows], batch_size=bs, show_progress=True)
    dt = time.perf_counter() - t0
    meta = {"set": set_name, "model": getattr(encoder, "model_name", None), "dim": int(emb.shape[1]),
            "n": len(rows), "seconds": round(dt, 2), "device": str(getattr(encoder, "device", ""))}
    save_embeddings(out, [r["qid"] for r in rows], emb, key="qids", meta=meta)
    log.info("embed queries %s: %d queries in %.1fs → %s", set_name, len(rows), dt, out)
    return out


# ----------------------------------------------------------------------------- латентность


_SYNTH_CODES = [
    ("def add(a, b):\n    total = a + b\n    if total > 10:\n        return total * 2\n    return total\n", "python"),
    ("int sum_arr(const int *a, int n) {\n    int s = 0;\n    for (int i = 0; i < n; i++) s += a[i];\n    return s;\n}\n", "c"),
    ("func Max(a, b int) int {\n\tif a > b {\n\t\treturn a\n\t}\n\treturn b\n}\n", "go"),
    ("public static int clamp(int v, int lo, int hi) {\n    if (v < lo) return lo;\n    if (v > hi) return hi;\n    return v;\n}\n", "java"),
    ("function norm(v) {\n  let s = 0;\n  for (const x of v) s += x * x;\n  return Math.sqrt(s);\n}\n", "javascript"),
]


def latency_sample(cfg: dict[str, Any], n: int = LATENCY_SAMPLE) -> tuple[list[str], list[str]]:
    """Выборка кодов для замера латентности: protected → public_train → синтетика (детерминированно)."""
    rng = get_rng(cfg, "latency_sample")
    for split in ("protected", "public_train", "public_calib"):
        rows = _records_of(cfg, split)
        if rows:
            rows = sorted(rows, key=lambda r: r["id"])
            rng.shuffle(rows)
            rows = rows[:n]
            return [r["code"] for r in rows], [r.get("lang", "") for r in rows]
    codes, langs = zip(*_SYNTH_CODES)
    return list(codes) * max(1, n // len(codes)), list(langs) * max(1, n // len(langs))


def _percentiles(times_ms: Sequence[float]) -> dict[str, float]:
    t = np.asarray(times_ms, dtype=np.float64)
    return {"mean_ms": float(t.mean()), "p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95)),
            "min_ms": float(t.min()), "n": int(t.size)}


def measure_latency(
    cfg: dict[str, Any],
    make_encoder: Callable[[str], Encoder],
    codes: Sequence[str],
    langs: Sequence[str],
    repeats: int | None = None,
    batch_sizes: Iterable[int] = LATENCY_BATCH_SIZES,
    out_path: str | Path | None = None,
    devices: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Латентность encode(): CPU (1 поток и все потоки) и GPU, batch 1 и 32 → results/latency_encoder.json.
    ``make_encoder(device)`` создаёт энкодер на устройстве. Требует torch (GPU-часть)."""
    import torch  # ленивый импорт (GPU-часть)

    reps = int(repeats or cfg.get("eval", {}).get("latency_repeats", 200))
    n_threads_all = max(1, os.cpu_count() or 1)
    configs: list[tuple[str, int | None]] = [("cpu", 1), ("cpu", n_threads_all)]
    if devices is None:
        if torch.cuda.is_available():
            configs.append(("cuda", None))
    else:
        configs = [(d, (1 if d == "cpu" else None)) for d in devices]
    results: list[dict[str, Any]] = []
    model_name = None
    dim = None
    for device, threads in configs:
        if device == "cpu" and threads:
            torch.set_num_threads(int(threads))
        enc = make_encoder(device)
        model_name = getattr(enc, "model_name", model_name)
        for bs in batch_sizes:
            bs = int(bs)
            times: list[float] = []
            n_rep = reps if bs == 1 else max(10, reps // 10)
            for i in range(3 + n_rep):  # 3 прогрева
                start = (i * bs) % max(1, len(codes))
                batch_codes = [codes[(start + j) % len(codes)] for j in range(bs)]
                batch_langs = [langs[(start + j) % len(langs)] for j in range(bs)]
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                emb = enc.encode(batch_codes, batch_langs, batch_size=bs)
                if device == "cuda":
                    torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) * 1000.0
                if i >= 3:
                    times.append(dt)
                dim = int(emb.shape[1])
            stats = _percentiles(times)
            row = {"device": device, "threads": threads if device == "cpu" else None, "batch": bs,
                   "per_item_ms": stats["p50_ms"] / bs, **stats}
            results.append(row)
            log.info("latency %s threads=%s batch=%d: p50=%.1f ms p95=%.1f ms (%.2f ms/item)", device, threads, bs,
                     stats["p50_ms"], stats["p95_ms"], row["per_item_ms"])
        del enc
        if device == "cuda":
            torch.cuda.empty_cache()
    report = {"model": model_name, "dim": dim, "max_length": cfg.get("semantic", {}).get("max_length"),
              "n_sample_codes": len(codes), "repeats": reps, "cpu_threads_all": n_threads_all,
              "torch": torch.__version__, "cuda": torch.cuda.is_available(),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "configs": results}
    out = Path(out_path) if out_path else resolve_path(cfg, "results") / LATENCY_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    log.info("latency report → %s", out)
    return report


# ----------------------------------------------------------------------------- шаг 06


def run_embed(
    cfg: dict[str, Any],
    checkpoint: str | None = None,
    splits: Iterable[str] | None = None,
    sets: Iterable[str] | None = None,
    batch_size: int | None = None,
    force: bool = False,
    latency: bool = True,
    own_model: bool = False,
    device: str | None = None,
) -> dict[str, Any]:
    """Шаг 06: эмбеддинги всех сплитов функций и наборов запросов выбранным чекпойнтом + латентность.
    Идемпотентно: готовые npz пропускаются без force. Требует torch."""
    from smcode.semantic.model import load_encoder  # ленивый импорт torch

    enc = load_encoder(cfg, checkpoint, device=device, own_model=own_model)
    split_list = list(splits) if splits else [*SPLITS, WINDOWS_SPLIT]
    set_list = list(sets) if sets else list(DEFAULT_QUERY_SETS)
    done: dict[str, Any] = {"model": enc.model_name, "device": str(enc.device), "splits": {}, "queries": {}}
    for split in split_list:
        p = embed_split(cfg, split, enc, batch_size=batch_size, force=force)
        done["splits"][split] = str(p) if p else None
    for s in set_list:
        p = embed_query_set(cfg, s, enc, batch_size=batch_size, force=force)
        done["queries"][s] = str(p) if p else None
    if latency:
        out = resolve_path(cfg, "results") / LATENCY_FILE
        if out.exists() and not force:
            log.info("latency report exists, skipping (%s)", out)
            done["latency"] = str(out)
        else:
            codes, langs = latency_sample(cfg)
            del enc
            report = measure_latency(cfg, lambda dev: load_encoder(cfg, checkpoint, device=dev, own_model=own_model),
                                     codes, langs, out_path=out)
            done["latency"] = str(out)
            done["latency_report"] = report
    return done
