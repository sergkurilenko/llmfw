"""Батчевый инференс энкодера и хранение эмбеддингов (DESIGN.md §7, embed.py).

Контракты на диске («активные» эмбеддинги, которые читают индексы, eval и privacy):
  data/embeddings/<split>.npz           — ids (unicode), emb (float16, (N, d)), meta (JSON-строка: model, max_length, …)
  data/embeddings/queries/<set>.npz     — qids (unicode), emb (float16, (M, d)), meta
  data/embeddings/active_model.json     — какой моделью получены активные эмбеддинги (справочно)
  results/latency_encoder.json          — латентность энкодера: CPU (cfg.eval.latency_threads, по умолчанию 1 и 4 потока)
                                          и GPU, batch 1 и 32
Кэш по моделям (zero-shot и дообученные эмбеддинги сосуществуют):
  data/embeddings/by_model/<model_tag>/<split>.npz, .../by_model/<model_tag>/queries/<set>.npz
Активный файл — жёсткая ссылка (или копия) на файл кэша. Переиспользование всегда проверяется по
meta.model (model_key энкодера) и meta.max_length: файл другой модели никогда не выдаётся за текущую,
а при смене чекпойнта активные файлы заменяются (старые остаются в кэше своей модели).

Функции чтения (load_embeddings, embeddings_for_records при наличии файлов) работают без torch;
torch импортируется лениво внутри run_embed / measure_latency.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from smcode.config import get_rng, resolve_path
from smcode.semantic.model import encoder_model_key, expected_model_key, model_key, model_tag
from smcode.types import read_jsonl

log = logging.getLogger(__name__)

EMBEDDINGS_DIR_DEFAULT = "data/embeddings"
QUERIES_SUBDIR = "queries"
BY_MODEL_SUBDIR = "by_model"
ACTIVE_MODEL_FILE = "active_model.json"
LATENCY_FILE = "latency_encoder.json"
WINDOWS_SPLIT = "protected_windows"
WINDOW_SUFFIX = "_windows"
# сплиты функций, которые читают 04 (semantic/hybrid: protected + окна) и 08 (protected, public_train); остальные — по --splits
DEFAULT_SPLITS: tuple[str, ...] = ("protected", "public_train", WINDOWS_SPLIT)
DEFAULT_QUERY_SETS: tuple[str, ...] = ("protected", "hard_neg", "public_calib", "public_test", "protected_windows")


def windows_split(split: str) -> str:
    """Имя «сплита» окон для файла эмбеддингов: <split>_windows (protected → protected_windows)."""
    return split if split.endswith(WINDOW_SUFFIX) else f"{split}{WINDOW_SUFFIX}"


def default_query_sets(cfg: dict[str, Any]) -> list[str]:
    """DEFAULT_QUERY_SETS + наборы окон негативов из cfg.windows.splits (public_calib_windows, ...)."""
    out = list(DEFAULT_QUERY_SETS)
    for s in (cfg.get("windows") or {}).get("splits") or []:
        name = windows_split(str(s))
        if name not in out:
            out.append(name)
    return out
LATENCY_BATCH_SIZES: tuple[int, ...] = (1, 32)
LATENCY_THREADS_DEFAULT: tuple[int, ...] = (1, 4)  # DESIGN §7: CPU 1 и 4 потока
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


def model_embeddings_dir(cfg: dict[str, Any], model: str | None) -> Path:
    """Каталог кэша эмбеддингов модели: data/embeddings/by_model/<model_tag> (создаётся только при записи)."""
    return embeddings_dir(cfg) / BY_MODEL_SUBDIR / model_tag(model)


def cached_split_path(cfg: dict[str, Any], split: str, model: str | None) -> Path:
    return model_embeddings_dir(cfg, model) / f"{split}.npz"


def cached_query_path(cfg: dict[str, Any], set_name: str, model: str | None) -> Path:
    return model_embeddings_dir(cfg, model) / QUERIES_SUBDIR / f"{set_name}.npz"


# ----------------------------------------------------------------------------- чтение/запись


def save_embeddings(path: str | Path, ids: Sequence[str], emb: np.ndarray, key: str = "ids",
                    meta: dict[str, Any] | None = None) -> Path:
    """Сохраняет npz: <key> (unicode), emb (float16), meta (JSON-строка). Запись атомарная (tmp + replace)."""
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
        meta = _parse_meta(z)
    return ids, emb, meta


def _parse_meta(z: Any) -> dict[str, Any]:
    if "meta" not in z.files:
        return {}
    try:
        meta = json.loads(str(z["meta"]))
    except (json.JSONDecodeError, TypeError):
        return {}
    return meta if isinstance(meta, dict) else {}


def read_embeddings_meta(path: str | Path) -> dict[str, Any]:
    """Только meta из npz (дёшево: массив emb не читается); {} если файла/meta нет."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with np.load(path, allow_pickle=False) as z:
            return _parse_meta(z)
    except (OSError, ValueError) as exc:
        log.warning("cannot read embeddings meta from %s: %s", path, exc)
        return {}


def embeddings_match(meta: dict[str, Any], model: str | None, max_length: int | None = None) -> tuple[bool, str]:
    """Получены ли эмбеддинги (по meta) моделью ``model`` (model_key) с той же max_length. (ok, причина)."""
    want = model_key(model)
    have = meta.get("model")
    if not have:
        return False, "embeddings file has no model in meta (unverifiable)"
    if model_key(have) != want:
        return False, f"embeddings produced by model {have!r}, current model {want!r}"
    if max_length is not None and meta.get("max_length") is not None and int(meta["max_length"]) != int(max_length):
        return False, f"embeddings produced with max_length={meta['max_length']}, current {max_length}"
    return True, "ok"


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


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.exists() and b.exists() and os.path.samefile(a, b)
    except OSError:
        return False


def _activate(src: Path, dst: Path) -> Path:
    """Делает dst жёсткой ссылкой на src (или копией, если ссылки недоступны); атомарно заменяет dst."""
    if _same_file(src, dst):
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.npz")
    if tmp.exists():
        tmp.unlink()
    try:
        os.link(src, tmp)
    except OSError:
        shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    return dst


def _archive_foreign(cfg: dict[str, Any], active: Path, meta: dict[str, Any], kind: str, name: str) -> None:
    """Активный файл другой модели переносится в кэш своей модели (если его там ещё нет) и удаляется."""
    other = meta.get("model")
    if other:
        target = cached_query_path(cfg, name, other) if kind == "query" else cached_split_path(cfg, name, other)
        if _same_file(active, target):
            active.unlink()
            return
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(active, target)
            log.info("embeddings %s: file of model %r moved to cache %s", name, other, target)
            return
    active.unlink()


def _encoder_props(encoder: Any) -> tuple[str, int | None]:
    """(model_key, max_length) энкодера."""
    ml = getattr(encoder, "max_length", None)
    return encoder_model_key(encoder), (int(ml) if ml is not None else None)


def write_active_model(cfg: dict[str, Any], model: str, extra: dict[str, Any] | None = None) -> Path:
    """data/embeddings/active_model.json — справка о модели активных эмбеддингов."""
    path = embeddings_dir(cfg) / ACTIVE_MODEL_FILE
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"model": model_key(model), "tag": model_tag(model), "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   **(extra or {})}, f, ensure_ascii=False, indent=1)
    return path


# ----------------------------------------------------------------------------- эмбеддинги записей


def embeddings_for_records_ex(
    cfg: dict[str, Any],
    encoder: Encoder | Callable[[], Encoder],
    ids: Sequence[str],
    codes: Sequence[str],
    langs: Sequence[str],
    split: str | None = None,
    batch_size: int | None = None,
    reuse: bool | None = None,
    model: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Эмбеддинги (N, d) float32 записей + сведения {model, n_reused, n_encoded, source}.

    Переиспользуются только векторы той же модели (meta.model == model_key ожидаемой модели): сначала активный
    data/embeddings/<split>.npz, затем кэш by_model/<tag>/<split>.npz; остальное кодируется энкодером.
    Ожидаемая модель: ``model`` → model_name объекта-энкодера → cfg.semantic.checkpoint/base_model (как load_encoder).
    ``encoder`` — объект с encode(codes, langs, batch_size) либо фабрика без аргументов (ленивая загрузка torch)."""
    scfg = cfg.get("semantic", {}) or {}
    reuse = bool(scfg.get("reuse_embeddings", True)) if reuse is None else reuse
    bs = int(batch_size or scfg.get("batch_size", 32))
    is_obj = hasattr(encoder, "encode")
    want = model_key(model) if model else (encoder_model_key(encoder) if is_obj else expected_model_key(cfg))
    n = len(ids)
    out: np.ndarray | None = None
    missing = np.ones(n, dtype=bool)
    info: dict[str, Any] = {"model": want, "n_reused": 0, "n_encoded": 0, "source": None}
    if reuse and split and n:
        for path in (split_embeddings_path(cfg, split), cached_split_path(cfg, split, want)):
            if not path.exists():
                continue
            meta = read_embeddings_meta(path)
            ok, why = embeddings_match(meta, want)
            if not ok:
                log.warning("embeddings for %s: not reusing %s: %s", split, path, why)
                continue
            f_ids, f_emb, _ = load_embeddings(path)
            pos = {rid: i for i, rid in enumerate(f_ids)}
            rows = np.array([pos.get(rid, -1) for rid in ids], dtype=np.int64)
            found = rows >= 0
            if found.any():
                out = np.zeros((n, f_emb.shape[1]), dtype=np.float32)
                out[found] = f_emb[rows[found]]
                missing = ~found
                info.update({"n_reused": int(found.sum()), "source": str(path)})
                log.info("embeddings for %s: %d/%d reused from %s (model=%s)", split, int(found.sum()), n, path, want)
                break
    if missing.any():
        enc = encoder if is_obj else encoder()
        have = encoder_model_key(enc)
        if have and have != want:
            raise ValueError(f"encoder model {have!r} differs from the expected model {want!r} for {split or 'records'}")
        idx = np.flatnonzero(missing)
        emb = np.asarray(enc.encode([codes[i] for i in idx], [langs[i] for i in idx], batch_size=bs), dtype=np.float32)
        if out is None:
            out = np.zeros((n, emb.shape[1]), dtype=np.float32)
        elif out.shape[1] != emb.shape[1]:
            raise ValueError(f"embedding dim mismatch: file {out.shape[1]} vs encoder {emb.shape[1]}")
        out[idx] = emb
        info["n_encoded"] = int(idx.size)
    if out is None:  # n == 0
        out = np.zeros((0, 0), dtype=np.float32)
    return out, info


def embeddings_for_records(cfg: dict[str, Any], encoder: Encoder | Callable[[], Encoder], ids: Sequence[str],
                           codes: Sequence[str], langs: Sequence[str], split: str | None = None,
                           batch_size: int | None = None, reuse: bool | None = None, model: str | None = None) -> np.ndarray:
    """Как embeddings_for_records_ex, но возвращает только матрицу (N, d) float32."""
    return embeddings_for_records_ex(cfg, encoder, ids, codes, langs, split=split, batch_size=batch_size, reuse=reuse, model=model)[0]


# ----------------------------------------------------------------------------- шаг 06: сплиты и наборы


def _missing_rows(rows: list[dict[str, Any]], id_field: str, have_ids: Sequence[str]) -> list[dict[str, Any]]:
    """Строки входного файла, которых нет в готовом npz (например, LLM-строки, добавленные 03 --llm-only после 06)."""
    pos = set(have_ids)
    return [r for r in rows if r[id_field] not in pos]


def _embed_file(cfg: dict[str, Any], kind: str, name: str, encoder: Encoder, rows: list[dict[str, Any]], id_field: str,
                key: str, batch_size: int | None, force: bool) -> Path | None:
    """Общая логика embed_split / embed_query_set: проверка активного файла по модели, кэш by_model, кодирование.

    Готовый файл той же модели пропускается, только если содержит все id входного файла; иначе кодируются лишь
    недостающие строки и файл переписывается (порядок строк — как во входном файле)."""
    active = query_embeddings_path(cfg, name) if kind == "query" else split_embeddings_path(cfg, name)
    want, max_len = _encoder_props(encoder)
    cached = cached_query_path(cfg, name, want) if kind == "query" else cached_split_path(cfg, name, want)
    label = f"embed queries {name}" if kind == "query" else f"embed {name}"
    have: tuple[list[str], np.ndarray] | None = None  # готовые векторы той же модели (для дозаписи)
    if active.exists():
        meta = read_embeddings_meta(active)
        ok, why = embeddings_match(meta, want, max_len)
        if ok and not force:
            f_ids, f_emb, _ = load_embeddings(active)
            missing = _missing_rows(rows, id_field, f_ids)
            if not missing:
                log.info("%s: exists for model %s, skipping (%s)", label, want, active)
                if not cached.exists():
                    _activate(active, cached)
                return active
            log.warning("%s: active file %s lacks %d of %d items (input file changed after embedding) → encoding the missing rows",
                         label, active, len(missing), len(rows))
            have = (f_ids, f_emb)
        elif not ok:
            log.warning("%s: active file %s is stale: %s → re-encoding", label, active, why)
            _archive_foreign(cfg, active, meta, kind, name)
    if have is None and cached.exists() and not force:
        ok, why = embeddings_match(read_embeddings_meta(cached), want, max_len)
        if ok:
            f_ids, f_emb, _ = load_embeddings(cached)
            missing = _missing_rows(rows, id_field, f_ids)
            if not missing:
                _activate(cached, active)
                log.info("%s: restored from model cache %s", label, cached)
                return active
            log.info("%s: cache %s lacks %d of %d items → encoding the missing rows", label, cached, len(missing), len(rows))
            have = (f_ids, f_emb)
        else:
            log.warning("%s: cache file %s unusable: %s", label, cached, why)
    if not rows:
        log.warning("%s: no input file, skipping", label)
        return None
    bs = int(batch_size or cfg.get("semantic", {}).get("batch_size", 32))
    todo = rows if have is None else _missing_rows(rows, id_field, have[0])
    t0 = time.perf_counter()
    emb_new = np.asarray(encoder.encode([r["code"] for r in todo], [r.get("lang", "") for r in todo], batch_size=bs, show_progress=True),
                         dtype=np.float32) if todo else np.zeros((0, 0), dtype=np.float32)
    dt = time.perf_counter() - t0
    if have is None:
        emb = emb_new
    else:
        pos = {rid: i for i, rid in enumerate(have[0])}
        d = int(emb_new.shape[1]) if emb_new.size else int(have[1].shape[1])
        if emb_new.size and have[1].shape[1] != d:
            raise ValueError(f"{label}: embedding dim mismatch: file {have[1].shape[1]} vs encoder {d}")
        emb = np.zeros((len(rows), d), dtype=np.float32)
        j = 0
        for i, r in enumerate(rows):
            k = pos.get(r[id_field])
            if k is not None:
                emb[i] = have[1][k]
            else:
                emb[i] = emb_new[j]
                j += 1
    meta = {("set" if kind == "query" else "split"): name, "model": want, "model_name": str(getattr(encoder, "model_name", "")),
            "kind": str(getattr(encoder, "kind", "")), "dim": int(emb.shape[1]), "n": len(rows), "n_encoded": len(todo),
            "n_reused": len(rows) - len(todo), "seconds": round(dt, 2),
            "device": str(getattr(encoder, "device", "")), "max_length": max_len, "pooling": getattr(encoder, "pooling", None),
            "input_prefix": getattr(encoder, "input_prefix", None)}
    save_embeddings(cached, [r[id_field] for r in rows], emb, key=key, meta=meta)
    _activate(cached, active)
    log.info("%s: %d items (%d encoded) in %.1fs → %s (cache %s)", label, len(rows), len(todo), dt, active, cached)
    return active


def embed_split(cfg: dict[str, Any], split: str, encoder: Encoder, batch_size: int | None = None,
                force: bool = False) -> Path | None:
    """data/functions/<split>.jsonl → data/embeddings/<split>.npz (ids, emb) той же моделью, что энкодер.
    Готовый файл пропускается, только если получен этой моделью (meta.model/max_length); иначе — перекодирование
    (или восстановление из кэша by_model/<tag>). None, если файла функций нет."""
    return _embed_file(cfg, "split", split, encoder, _records_of(cfg, split), "id", "ids", batch_size, force)


def embed_query_set(cfg: dict[str, Any], set_name: str, encoder: Encoder, batch_size: int | None = None,
                    force: bool = False) -> Path | None:
    """data/queries/<set>.jsonl → data/embeddings/queries/<set>.npz (qids, emb); проверка модели как в embed_split."""
    return _embed_file(cfg, "query", set_name, encoder, _queries_of(cfg, set_name), "qid", "qids", batch_size, force)


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


def latency_thread_counts(cfg: dict[str, Any]) -> list[int]:
    """cfg.eval.latency_threads (по умолчанию [1, 4], DESIGN §7); уникальные положительные значения по возрастанию."""
    raw = (cfg.get("eval", {}) or {}).get("latency_threads", list(LATENCY_THREADS_DEFAULT))
    if isinstance(raw, int):
        raw = [raw]
    vals = sorted({int(t) for t in raw if int(t) > 0})
    return vals or list(LATENCY_THREADS_DEFAULT)


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
    threads: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Латентность encode(): CPU с cfg.eval.latency_threads потоками (по умолчанию 1 и 4) и GPU, batch 1 и 32
    → results/latency_encoder.json. ``make_encoder(device)`` создаёт энкодер на устройстве. Требует torch (GPU-часть).
    В отчёте: cpu_count (ядер на машине) и cpu_threads_all (наибольшее из измеренных число потоков)."""
    import torch  # ленивый импорт (GPU-часть)

    reps = int(repeats or cfg.get("eval", {}).get("latency_repeats", 200))
    cpu_count = max(1, os.cpu_count() or 1)
    thread_list = sorted({int(t) for t in threads}) if threads is not None else latency_thread_counts(cfg)
    configs: list[tuple[str, int | None]] = [("cpu", t) for t in thread_list]
    if devices is None:
        if torch.cuda.is_available():
            configs.append(("cuda", None))
    else:
        configs = [(d, t) for d in devices for t in (thread_list if d == "cpu" else [None])]
    results: list[dict[str, Any]] = []
    model_name = None
    dim = None
    for device, n_threads in configs:
        if device == "cpu" and n_threads:
            torch.set_num_threads(int(n_threads))
        enc = make_encoder(device)
        model_name = encoder_model_key(enc) or model_name
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
            row = {"device": device, "threads": n_threads if device == "cpu" else None, "batch": bs,
                   "per_item_ms": stats["p50_ms"] / bs, **stats}
            results.append(row)
            log.info("latency %s threads=%s batch=%d: p50=%.1f ms p95=%.1f ms (%.2f ms/item)", device, n_threads, bs,
                     stats["p50_ms"], stats["p95_ms"], row["per_item_ms"])
        del enc
        if device == "cuda":
            torch.cuda.empty_cache()
    report = {"model": model_name, "dim": dim, "max_length": cfg.get("semantic", {}).get("max_length"),
              "n_sample_codes": len(codes), "repeats": reps, "cpu_threads": thread_list,
              "cpu_threads_all": max(thread_list) if thread_list else None, "cpu_count": cpu_count,
              "torch": torch.__version__, "cuda": torch.cuda.is_available(),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "configs": results}
    out = Path(out_path) if out_path else resolve_path(cfg, "results") / LATENCY_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    log.info("latency report → %s", out)
    return report


def _latency_report_matches(path: Path, model: str) -> bool:
    if not path.exists():
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            rep = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return model_key(rep.get("model")) == model_key(model)


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
    latency_threads: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Шаг 06: эмбеддинги сплитов функций (по умолчанию DEFAULT_SPLITS — те, что читают 04 и 08) и всех наборов
    запросов (default_query_sets) выбранным чекпойнтом + латентность. Идемпотентно по модели и по содержимому:
    готовые npz той же модели пропускаются, недостающие строки дозаписываются, файлы другой модели заменяются
    (старые остаются в by_model/<tag>). Требует torch."""
    from smcode.semantic.model import load_encoder  # ленивый импорт torch

    enc = load_encoder(cfg, checkpoint, device=device, own_model=own_model)
    model = encoder_model_key(enc)
    split_list = list(splits) if splits else list(DEFAULT_SPLITS)
    set_list = list(sets) if sets else default_query_sets(cfg)
    done: dict[str, Any] = {"model": model, "model_name": enc.model_name, "device": str(enc.device), "splits": {}, "queries": {}}
    for split in split_list:
        p = embed_split(cfg, split, enc, batch_size=batch_size, force=force)
        done["splits"][split] = str(p) if p else None
    for s in set_list:
        p = embed_query_set(cfg, s, enc, batch_size=batch_size, force=force)
        done["queries"][s] = str(p) if p else None
    done["active_model_file"] = str(write_active_model(cfg, model, {"checkpoint": checkpoint, "kind": getattr(enc, "kind", None)}))
    if latency:
        out = resolve_path(cfg, "results") / LATENCY_FILE
        if _latency_report_matches(out, model) and not force:
            log.info("latency report for model %s exists, skipping (%s)", model, out)
            done["latency"] = str(out)
        else:
            codes, langs = latency_sample(cfg)
            del enc
            report = measure_latency(cfg, lambda dev: load_encoder(cfg, checkpoint, device=dev, own_model=own_model),
                                     codes, langs, out_path=out, threads=latency_threads)
            done["latency"] = str(out)
            done["latency_report"] = report
    return done
