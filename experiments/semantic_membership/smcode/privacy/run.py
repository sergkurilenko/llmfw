"""Шаг 08 — приватность индекса эмбеддингов (DESIGN.md §9): защиты × (полезность, атаки) → results/privacy.json.

Защиты (cfg.privacy): none; proj (ключевая ортогональная проекция); proj+int8/int4/bin (quant_bits);
proj+noise<σ> (noise_sigma, σ > 0). Имя защиты — компоненты через «+»: none | proj | int8 | int4 | bin | noise<σ>.
Порядок применения к векторам индекса: проекция → шум (с нормировкой) → квантование (деквантованный float32).
Запросы на шлюзе проецируются тем же ключом (без шума и квантования).

Полезность: TPR при FPR = α (cfg.calibration.alphas[0], по умолчанию 0.01) метода M3 на защищённом индексе
(SemanticIndex.build_from_embeddings/score_embeddings), порог — split conformal по защищённым скорам
public_calib (smcode.calibration.conformal_threshold); FPR на public_test и hard_neg с ДИ Клоппера–Пирсона.
Атаки: A1 (модель обучена на plain-эмбеддингах public_train, применена напрямую к защищённому индексу),
A2 (n утёкших пар → Q̂ → инверсия → A1), A3 (опционально, torch: генеративная инверсия после A2 при max n).

Схема results/privacy.json:
{
 "schema": "smcode.privacy/1", "created": ISO, "alpha": 0.01, "leaked_pairs": [0, 100, ...],
 "n_index": N, "dim": d, "noise_mode": "relative", "align_method": "procrustes",
 "a1": {"backend": "ridge"|"torch", "vocab_size": V, "n_rare_vocab": R, "n_train": ..., "n_eval": ..., "val_f1": ...,
        "prior_baseline": {"f1", "rare_id_recall", ...}},
 "queries": {"protected": M, "public_calib": M, ...},
 "defenses": [{"name": "proj+int8", "projection": true, "bits": 8, "sigma": 0.0, "memory_ratio": 0.5, "memory_bytes": ...,
               "expected_cos": 0.97 | null,
               "threshold": τ, "tpr": {"semantic": RATE, "hybrid": null}, "tpr_by_alpha": {"0.01": {"semantic": RATE}},
               "fpr": {"public_test": RATE, "hard_neg": RATE},
               "attacks": {"A1": {"f1", "precision", "recall", "rare_id_recall", ...},
                           "A2": {"0": {...}, "100": {"f1", ..., "q_rel_error", "recovered_cos"}, ...},
                           "A3": {"bleu", "id_acc", "id_exact"} | null}}],
 "errors": [...]
}
RATE = {"value", "ci": [lo, hi], "n", "k"} (ДИ Клоппера–Пирсона 95 %). Таблица T6 / рисунок F6 читают этот файл.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from smcode.calibration import conformal_threshold, rate_with_ci
from smcode.config import ROOT, get_rng, resolve_path
from smcode.privacy import attack_align, attack_bow, projection, quantize
from smcode.types import read_jsonl

log = logging.getLogger(__name__)

SCHEMA = "smcode.privacy/1"
PRIVACY_FILE = "privacy.json"
EMBEDDINGS_DIR_DEFAULT = "data/embeddings"
QUERY_SETS: tuple[str, ...] = ("protected", "public_calib", "public_test", "hard_neg")
DEFAULT_ALPHA = 0.01
DEFAULT_A1_TRAIN_MAX = 50000
DEFAULT_A1_EVAL_MAX = 10000
DEFAULT_LEAK_POOL_MAX = 20000


# ----------------------------------------------------------------------------- защиты


@dataclass(frozen=True)
class DefenceSpec:
    """Конфигурация защиты: проекция, квантование (8/4/1 бит или None), шум σ (0 — нет)."""

    name: str
    projection: bool = False
    bits: int | None = None
    sigma: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_BITS_NAMES = {8: "int8", 4: "int4", 1: "bin"}
_NAMES_BITS = {v: k for k, v in _BITS_NAMES.items()}


def defence_name(projection: bool, bits: int | None, sigma: float) -> str:
    parts: list[str] = []
    if projection:
        parts.append("proj")
    if sigma > 0:
        parts.append(f"noise{sigma:g}")
    if bits is not None:
        parts.append(_BITS_NAMES[bits])
    return "+".join(parts) if parts else "none"


def make_defence(projection: bool = False, bits: int | None = None, sigma: float = 0.0) -> DefenceSpec:
    if bits is not None and bits not in quantize.BITS:
        raise ValueError(f"bits must be one of {quantize.BITS} or None, got {bits}")
    return DefenceSpec(defence_name(projection, bits, float(sigma)), projection, bits, float(sigma))


def parse_defence(name: str) -> DefenceSpec:
    """'none' | 'proj' | 'proj+int8' | 'proj+noise0.1' | 'int4' | ... → DefenceSpec."""
    projection, bits, sigma = False, None, 0.0
    for part in [p.strip() for p in name.split("+") if p.strip()]:
        if part == "none":
            continue
        if part == "proj":
            projection = True
        elif part in _NAMES_BITS:
            bits = _NAMES_BITS[part]
        elif part.startswith("noise"):
            sigma = float(part[len("noise"):])
        else:
            raise ValueError(f"unknown defence component {part!r} in {name!r}")
    return make_defence(projection, bits, sigma)


def default_defences(cfg: dict[str, Any]) -> list[DefenceSpec]:
    """none; proj; proj+<bits> для quant_bits; proj+noise<σ> для σ > 0 из noise_sigma."""
    p = cfg.get("privacy", {}) or {}
    out = [make_defence(), make_defence(projection=True)]
    for b in p.get("quant_bits", [8, 4, 1]) or []:
        out.append(make_defence(projection=True, bits=int(b)))
    for s in p.get("noise_sigma", [0.05, 0.1, 0.2]) or []:
        if float(s) > 0:
            out.append(make_defence(projection=True, sigma=float(s)))
    return out


def apply_defence(emb: np.ndarray, spec: DefenceSpec, Q: np.ndarray | None, rng: Any, noise_mode: str = "relative") -> np.ndarray:
    """Защищённые векторы индекса (float32): проекция → шум → квантование (деквантованные значения)."""
    x = np.asarray(emb, dtype=np.float32)
    if spec.projection:
        if Q is None:
            raise ValueError("projection requested but Q is None")
        x = projection.apply(x, Q)
    if spec.sigma > 0:
        x = quantize.add_noise(x, spec.sigma, rng, mode=noise_mode)
    if spec.bits is not None:
        x = quantize.quantize_full(x, spec.bits).dequantize()
    return np.ascontiguousarray(x, dtype=np.float32)


def defend_queries(q_emb: np.ndarray, spec: DefenceSpec, Q: np.ndarray | None) -> np.ndarray:
    """Запросы шлюза: только проекция (ключ известен шлюзу)."""
    x = np.asarray(q_emb, dtype=np.float32)
    return projection.apply(x, Q) if spec.projection and Q is not None else x


def defence_memory(spec: DefenceSpec, n: int, d: int) -> tuple[float, int]:
    """(память относительно float16, байт на индекс)."""
    ratio = quantize.memory_ratio(d, spec.bits)
    nbytes = n * (quantize.bytes_per_vector(d, spec.bits) if spec.bits is not None else 2 * d)
    return ratio, int(nbytes)


# ----------------------------------------------------------------------------- входные данные


@dataclass
class PrivacyInputs:
    """Входы шага 08 в памяти (см. load_inputs): индекс protected, обучающий корпус public_train, запросы по наборам."""

    index_ids: list[str]
    index_emb: np.ndarray  # plain (N, d)
    index_codes: list[str]
    index_langs: list[str]
    train_ids: list[str]
    train_emb: np.ndarray  # plain (T, d)
    train_codes: list[str]
    train_langs: list[str]
    queries: dict[str, tuple[list[str], np.ndarray]]  # set → (qids, emb)
    meta: dict[str, Any]

    @property
    def dim(self) -> int:
        return int(self.index_emb.shape[1]) if self.index_emb.ndim == 2 else 0


def embeddings_dir(cfg: dict[str, Any]) -> Path:
    cfg.setdefault("paths", {}).setdefault("embeddings", EMBEDDINGS_DIR_DEFAULT)
    return resolve_path(cfg, "embeddings")


def _load_npz(path: Path) -> tuple[list[str], np.ndarray]:
    """(ids|qids, emb float32) из npz по контракту data/embeddings."""
    with np.load(path, allow_pickle=False) as z:
        key = "ids" if "ids" in z.files else "qids"
        ids = [str(x) for x in z[key].tolist()]
        emb = np.asarray(z["emb"], dtype=np.float32)
    if emb.ndim != 2 or emb.shape[0] != len(ids):
        raise ValueError(f"bad embeddings file {path}: emb {emb.shape}, ids {len(ids)}")
    return ids, emb


def _join_codes(ids: Sequence[str], functions_path: Path) -> tuple[list[str], list[str], np.ndarray]:
    """(codes, langs, маска найденных) для ids по файлу функций."""
    pos = {rid: i for i, rid in enumerate(ids)}
    codes: list[str | None] = [None] * len(ids)
    langs: list[str] = [""] * len(ids)
    for row in read_jsonl(functions_path):
        i = pos.get(row.get("id"))
        if i is not None:
            codes[i] = row.get("code", "")
            langs[i] = row.get("lang", "")
    found = np.array([c is not None for c in codes], dtype=bool)
    return [c or "" for c in codes], langs, found


def _subsample(n: int, max_n: int | None, rng: random.Random) -> np.ndarray:
    if max_n is None or n <= max_n:
        return np.arange(n)
    return np.sort(quantize.as_generator(rng).choice(n, size=int(max_n), replace=False))


def load_inputs(cfg: dict[str, Any], limit_train: int | None = None, limit_eval: int | None = None, sets: Iterable[str] = QUERY_SETS,
                rng: random.Random | None = None) -> PrivacyInputs:
    """Читает data/embeddings/{protected,public_train}.npz, data/functions/*.jsonl и data/embeddings/queries/<set>.npz.

    limit_train/limit_eval — подвыборки (с seed) для атак; индекс и запросы берутся целиком.
    """
    rng = rng or get_rng(cfg, "privacy_inputs")
    edir = embeddings_dir(cfg)
    fdir = resolve_path(cfg, "functions")
    p_idx = edir / "protected.npz"
    if not p_idx.exists():
        raise FileNotFoundError(f"protected embeddings not found: {p_idx} (run scripts/06_embed.py)")
    index_ids, index_emb = _load_npz(p_idx)
    index_codes, index_langs, found = _join_codes(index_ids, fdir / "protected.jsonl")
    if not found.all():
        log.warning("privacy: %d/%d protected embeddings without code (dropped)", int((~found).sum()), len(index_ids))
        keep = np.flatnonzero(found)
        index_ids = [index_ids[i] for i in keep]
        index_emb = index_emb[keep]
        index_codes = [index_codes[i] for i in keep]
        index_langs = [index_langs[i] for i in keep]
    train_ids: list[str] = []
    train_emb = np.zeros((0, index_emb.shape[1]), dtype=np.float32)
    train_codes: list[str] = []
    train_langs: list[str] = []
    p_tr = edir / "public_train.npz"
    if p_tr.exists():
        train_ids, train_emb = _load_npz(p_tr)
        train_codes, train_langs, found = _join_codes(train_ids, fdir / "public_train.jsonl")
        keep = np.flatnonzero(found)
        keep = keep[_subsample(len(keep), limit_train, rng)]
        train_ids = [train_ids[i] for i in keep]
        train_emb = train_emb[keep]
        train_codes = [train_codes[i] for i in keep]
        train_langs = [train_langs[i] for i in keep]
    else:
        log.warning("privacy: %s not found — атаки A1/A2 недоступны", p_tr)
    queries: dict[str, tuple[list[str], np.ndarray]] = {}
    for s in sets:
        p = edir / "queries" / f"{s}.npz"
        if p.exists():
            queries[s] = _load_npz(p)
        else:
            log.warning("privacy: query embeddings for %s not found (%s)", s, p)
    meta = {"index_file": str(p_idx), "train_file": str(p_tr) if p_tr.exists() else None, "limit_train": limit_train, "limit_eval": limit_eval}
    return PrivacyInputs(index_ids, index_emb, index_codes, index_langs, train_ids, train_emb, train_codes, train_langs, queries, meta)


# ----------------------------------------------------------------------------- полезность


def utility(spec: DefenceSpec, inputs: PrivacyInputs, Q: np.ndarray | None, defended_index: np.ndarray, alphas: Sequence[float],
            top_k: int = 1) -> dict[str, Any]:
    """TPR/FPR метода M3 на защищённом индексе с конформным порогом по защищённым скорам public_calib."""
    from smcode.semantic.semantic_index import SemanticIndex

    out: dict[str, Any] = {"threshold": None, "tpr": {"semantic": None, "hybrid": None}, "tpr_by_alpha": {}, "fpr": {}, "scores_mean": {}}
    if "protected" not in inputs.queries or "public_calib" not in inputs.queries:
        out["error"] = "query embeddings for protected/public_calib are missing"
        return out
    idx = SemanticIndex({"semantic": {"use_faiss": False, "ann_top_k": top_k}})
    idx.build_from_embeddings(inputs.index_ids, defended_index)
    scores: dict[str, np.ndarray] = {}
    for s, (_, q) in inputs.queries.items():
        sc, _, _, _ = idx.score_embeddings(defend_queries(q, spec, Q), top_k=top_k)
        scores[s] = np.asarray(sc, dtype=np.float64)
        out["scores_mean"][s] = float(sc.mean()) if sc.size else None
    for a in alphas:
        tau = conformal_threshold(scores["public_calib"], float(a))
        key = f"{float(a):g}"
        block = {"threshold": tau if np.isfinite(tau) else "inf", "semantic": rate_with_ci(scores["protected"], tau),
                 "fpr": {s: rate_with_ci(scores[s], tau) for s in ("public_test", "hard_neg") if s in scores}}
        out["tpr_by_alpha"][key] = block
    first = f"{float(alphas[0]):g}"
    out["threshold"] = out["tpr_by_alpha"][first]["threshold"]
    out["tpr"]["semantic"] = out["tpr_by_alpha"][first]["semantic"]
    out["fpr"] = out["tpr_by_alpha"][first]["fpr"]
    out["alpha"] = float(alphas[0])
    return out


# ----------------------------------------------------------------------------- прогон


def _a3_available() -> bool:
    return attack_bow.torch_available()


def run_privacy_inputs(
    cfg: dict[str, Any],
    inputs: PrivacyInputs,
    defences: Sequence[DefenceSpec] | None = None,
    leaked_pairs: Sequence[int] | None = None,
    a1_backend: str | None = None,
    with_a3: bool = False,
    key: bytes | None = None,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Полный прогон §9 на данных в памяти → словарь по схеме privacy.json (без записи на диск)."""
    t0 = time.perf_counter()
    rng = rng or get_rng(cfg, "privacy")
    p = cfg.get("privacy", {}) or {}
    defences = list(defences) if defences else default_defences(cfg)
    ns = [int(n) for n in (leaked_pairs if leaked_pairs is not None else p.get("attack_leaked_pairs", [0, 100, 1000, 10000]))]
    alphas = [float(a) for a in (cfg.get("calibration", {}) or {}).get("alphas", [DEFAULT_ALPHA])] or [DEFAULT_ALPHA]
    noise_mode = str(p.get("noise_mode", "relative"))
    align_method = str(p.get("align_method", "procrustes"))
    d = inputs.dim
    if str(p.get("projection", "keyed_orthogonal")) != "keyed_orthogonal":
        raise ValueError(f"unsupported projection {p.get('projection')!r}")
    Q = projection.keyed_orthogonal(d, key if key is not None else projection.projection_key(cfg)) if d else None
    errors: list[str] = []
    result: dict[str, Any] = {
        "schema": SCHEMA, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "alpha": alphas[0], "alphas": alphas, "leaked_pairs": ns,
        "n_index": len(inputs.index_ids), "dim": d, "noise_mode": noise_mode, "align_method": align_method,
        "queries": {s: len(q[0]) for s, q in inputs.queries.items()}, "a1": None, "defenses": [], "errors": errors, **inputs.meta,
    }

    # --- A1: модель атакующего (plain public_train) и цели на protected (подвыборка)
    model = vocab = None
    Y_eval = None
    eval_idx = np.arange(len(inputs.index_ids))
    attacks_enabled = inputs.train_emb.shape[0] >= 4 and len(inputs.index_ids) > 0
    if attacks_enabled:
        eval_max = inputs.meta.get("limit_eval") or int(p.get("a1_eval_max", DEFAULT_A1_EVAL_MAX))
        eval_idx = _subsample(len(inputs.index_ids), eval_max, random.Random(rng.random()))
        train_bags = attack_bow.identifier_bags(inputs.train_codes, inputs.train_langs, show_progress=True)
        eval_bags = attack_bow.identifier_bags([inputs.index_codes[i] for i in eval_idx], [inputs.index_langs[i] for i in eval_idx], show_progress=True)
        try:
            model, vocab, Y_train = attack_bow.train_bow_attack(inputs.train_emb, train_bags, cfg, rng=random.Random(rng.random()), backend=a1_backend)
        except ImportError as exc:
            log.warning("A1 torch backend unavailable (%s), falling back to ridge", exc)
            model, vocab, Y_train = attack_bow.train_bow_attack(inputs.train_emb, train_bags, cfg, rng=random.Random(rng.random()), backend="ridge")
        Y_eval = attack_bow.targets(eval_bags, vocab)
        n_val = min(Y_train.shape[0] // 10 + 1, attack_bow.DEFAULT_VAL_MAX)
        result["a1"] = {"backend": model.backend, "vocab_size": len(vocab), "n_rare_vocab": vocab.n_rare, "rare_df": vocab.rare_df,
                        "n_train": model.n_train, "n_eval": int(eval_idx.size), "val_f1": model.val_f1, "threshold": model.threshold,
                        "prior_baseline": attack_bow.prior_baseline(Y_train[n_val:], Y_train[:n_val], Y_eval, vocab)}
    else:
        errors.append("attacks skipped: public_train embeddings/codes or protected index missing")
    pool_max = int(p.get("leak_pool_max", DEFAULT_LEAK_POOL_MAX))
    pool_idx = _subsample(inputs.train_emb.shape[0], max(pool_max, max(ns) if ns else 0), random.Random(rng.random()))
    pool_plain = inputs.train_emb[pool_idx]
    eval_plain = inputs.index_emb[eval_idx]
    a3_model = a3_vocab = None
    use_a3 = bool(with_a3) and attacks_enabled
    if with_a3 and not _a3_available():
        errors.append("A3 skipped: torch not available")
        use_a3 = False

    # --- защиты
    for spec in defences:
        ts = time.perf_counter()
        drng = random.Random(f"{rng.random()}:{spec.name}")
        defended = apply_defence(inputs.index_emb, spec, Q, drng, noise_mode)
        ratio, nbytes = defence_memory(spec, len(inputs.index_ids), d)
        row: dict[str, Any] = {**spec.to_dict(), "memory_ratio": ratio, "memory_bytes": nbytes,
                               "expected_cos": quantize.expected_cosine(spec.sigma, noise_mode, d) if spec.sigma > 0 else None,
                               "attacks": {"A1": None, "A2": None, "A3": None}}
        try:
            row.update(utility(spec, inputs, Q, defended, alphas))
        except Exception as exc:  # noqa: BLE001 — полезность не должна ронять атаки
            log.exception("utility failed for %s: %s", spec.name, exc)
            errors.append(f"utility {spec.name}: {exc}")
        if attacks_enabled and model is not None and Y_eval is not None:
            target = defended[eval_idx]
            row["attacks"]["A1"] = model.evaluate(target, Y_eval)
            pool_def = apply_defence(pool_plain, spec, Q, random.Random(f"{drng.random()}:pool"), noise_mode)
            row["attacks"]["A2"] = attack_align.run_attack_align(model, pool_plain, pool_def, target, Y_eval, ns, random.Random(drng.random()),
                                                                 method=align_method, Q_true=Q if spec.projection else np.eye(d, dtype=np.float32),
                                                                 target_plain=eval_plain)
            if use_a3:
                try:
                    from smcode.privacy import attack_gen

                    n_max = max(ns) if ns else 0
                    pp, dd = attack_align.sample_leaked_pairs(pool_plain, pool_def, n_max, random.Random(drng.random()))
                    Q_hat = attack_align.estimate_projection(pp, dd, align_method) if pp.shape[0] else np.eye(d, dtype=np.float32)
                    rec = attack_align.recover(target, Q_hat)
                    codes = [inputs.index_codes[i] for i in eval_idx]
                    langs = [inputs.index_langs[i] for i in eval_idx]
                    a3, a3_model, a3_vocab = attack_gen.run_attack_gen(inputs.train_emb, inputs.train_codes, inputs.train_langs, rec, codes, langs,
                                                                       cfg, rng=random.Random(drng.random()), model=a3_model, vocab=a3_vocab)
                    a3["n_leaked_pairs"] = int(pp.shape[0])
                    row["attacks"]["A3"] = a3
                except ImportError as exc:
                    errors.append(f"A3 skipped: {exc}")
                    use_a3 = False
        row["seconds"] = round(time.perf_counter() - ts, 2)
        tpr = (row.get("tpr") or {}).get("semantic") or {}
        a1 = row["attacks"]["A1"] or {}
        log.info("defence %-14s mem=%.3f tpr=%s A1 f1=%s (%.1fs)", spec.name, ratio, tpr.get("value"), a1.get("f1"), row["seconds"])
        result["defenses"].append(row)
    result["seconds"] = round(time.perf_counter() - t0, 2)
    return result


def privacy_path(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg, "results") / PRIVACY_FILE


def write_privacy(cfg: dict[str, Any], result: dict[str, Any]) -> Path:
    path = privacy_path(cfg)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1, default=_json_default)
    log.info("privacy results → %s", path)
    return path


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def run_privacy(
    cfg: dict[str, Any],
    force: bool = False,
    defences: Sequence[str] | None = None,
    leaked_pairs: Sequence[int] | None = None,
    a1_backend: str | None = None,
    with_a3: bool = False,
    limit_train: int | None = None,
    limit_eval: int | None = None,
) -> dict[str, Any]:
    """Шаг 08 с файлами: data/embeddings → results/privacy.json. Идемпотентно (файл есть → пропуск без force)."""
    path = privacy_path(cfg)
    if path.exists() and not force:
        log.info("privacy.json exists, skipping (%s); use --force to recompute", path)
        with open(path, "r", encoding="utf-8") as f:
            res = json.load(f)
        res["skipped"] = True
        return res
    p = cfg.get("privacy", {}) or {}
    lt = limit_train if limit_train is not None else p.get("a1_train_max", DEFAULT_A1_TRAIN_MAX)
    lt = max(int(lt), max(int(n) for n in (leaked_pairs or p.get("attack_leaked_pairs", [0])))) if lt else None
    inputs = load_inputs(cfg, limit_train=lt, limit_eval=limit_eval)
    specs = [parse_defence(n) for n in defences] if defences else None
    result = run_privacy_inputs(cfg, inputs, defences=specs, leaked_pairs=leaked_pairs, a1_backend=a1_backend, with_a3=with_a3)
    result["config"] = {"privacy": p, "calibration": cfg.get("calibration"), "seed": cfg.get("seed"), "config_path": cfg.get("_config_path"),
                        "root": str(ROOT)}
    write_privacy(cfg, result)
    result["skipped"] = False
    return result
