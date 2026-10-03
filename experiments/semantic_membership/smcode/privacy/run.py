"""Шаг 08 — приватность индекса эмбеддингов (DESIGN.md §9): защиты × (полезность, атаки) → results/privacy.json.

Защиты (cfg.privacy): none; proj (ключевая ортогональная проекция); proj+int8/int4/bin (quant_bits);
proj+noise<σ> (noise_sigma, σ > 0). Имя защиты — компоненты через «+»: none | proj | int8 | int4 | bin | noise<σ>.
Порядок применения к векторам индекса: проекция → шум (с нормировкой) → квантование (деквантованный float32).
Запросы на шлюзе проецируются тем же ключом (без шума и квантования).

Шум: σ трактуется по cfg.privacy.noise_mode («relative», по умолчанию: σ = отношение норм шума и вектора,
cos ≈ 1/√(1+σ²); «absolute»: σ на координату, cos ≈ 1/√(1+σ²d)). Если все σ из конфига дают ожидаемый cos > 0.97
(как [0.05, 0.1, 0.2] в relative-режиме), ветвь шума кривой F6 неинформативна — default_defences дописывает сетку
DEFAULT_NOISE_SIGMA (relative 0.25/0.5/1/2 → cos ≈ 0.97/0.89/0.71/0.45) с предупреждением; явный список --defences
не расширяется.

Полезность: TPR при FPR = α (cfg.calibration.alphas[0], по умолчанию 0.01) методов M3 (SemanticIndex.build_from_embeddings/
score_embeddings) и M4 (HybridIndex.build_from_embeddings с теми же кодами и комбинатором основного индекса; cfg.privacy.
utility_hybrid) на защищённом индексе, порог — split conformal по защищённым скорам public_calib
(smcode.calibration.conformal_threshold); FPR на public_test и hard_neg. Состав индекса — как у основного индекса M3
(data/indexes/semantic/meta.json: with_windows → protected.npz + protected_windows.npz), поэтому строка защиты «none»
воспроизводит TPR M3 основной оценки (T2/T4). Все npz должны быть получены одной моделью (meta.model), иначе ValueError.
ДИ: Клоппера–Пирсона по запросам («ci», без учёта кластеров) и персентильный кластерный бутстрэп по исходной функции
запроса (qid до «#», «ci_cluster», как в основной оценке) — запросы одной функции (~13 преобразований) не независимы.
Атаки: A1 (модель обучена на plain-эмбеддингах public_train, применена напрямую к защищённому индексу),
A2 (n утёкших пар → Q̂ → инверсия → A1; выборки пар вложены по n — одна перестановка пула на защиту),
A3 (опционально, torch: генеративная инверсия после A2 при max n на одной и той же подвыборке protected для всех защит).

Схема results/privacy.json:
{
 "schema": "smcode.privacy/1", "created": ISO, "alpha": 0.01, "alphas": [...], "leaked_pairs": [0, 100, ...],
 "n_index": N, "dim": d, "noise_mode": "relative", "align_method": "procrustes", "bootstrap": B,
 "index_composition": "protected (functions) + protected_windows" | "protected (functions)", "with_windows": bool,
 "n_index_functions": N_f, "n_index_windows": N_w, "model": model_key энкодера (по meta npz) | null,
 "hybrid_utility": {"available": bool, "rule": "logistic"|"two_threshold", "combiner": ..., "reason": str|null},
 "a1": {"backend": "ridge"|"torch", "vocab_size": V, "n_rare_vocab": R, "rare_df": 3, "rare_df_range": [2, 3],
        "df_corpus_size": число функций public_train, по которым считался df, "n_train": ..., "n_eval": ..., "val_f1": ...,
        "thr_min_precision": 0.1, "thr_min_f1": 0.1, "prior_baseline": {"f1", "rare_id_recall", ...},
        "protected_coverage": {"occurrence_coverage": доля вхождений идентификаторов protected, покрытых словарём A1,
                               "n_protected_ids": ..., "n_protected_only_ids": идентификаторы с df = 0 в public_train,
                               "protected_only_occurrence_fraction": ...}},   # A1/A2 — нижняя оценка утечки (только
                               # идентификаторы, известные по public_train); protected-only покрывает лишь A3
 "queries": {"protected": M, "public_calib": M, ...},
 "a3": {"n_eval": ..., "eval_seed": ...} | null,
 "defenses": [{"name": "proj+int8", "projection": true, "bits": 8, "sigma": 0.0, "noise_mode": null|"relative",
               "memory_ratio": 0.5, "memory_bytes": ..., "expected_cos": 0.97 | null,
               "threshold": τ, "tpr": {"semantic": RATE, "hybrid": RATE | null},
               "tpr_by_alpha": {"0.01": {"threshold", "semantic": RATE, "fpr": {...}, "hybrid": RATE|null, "threshold_hybrid", "fpr_hybrid"}},
               "fpr": {"public_test": RATE, "hard_neg": RATE}, "fpr_hybrid": {...} | null,
               "attacks": {"A1": {"f1", "precision", "recall", "rare_id_recall", "rare_id_precision", "rare_id_recall_chance", ...},
                           "A2": {"0": {...}, "100": {"f1", ..., "n", "n_effective", "q_rel_error", "recovered_cos"}, ...},
                           "A3": {"bleu", "id_acc", "id_exact", "n_eval"} | null}}],
 "errors": [...]
}
RATE = {"value", "ci": [lo, hi] (Клоппер–Пирсон 95 %, без кластеров), "n", "k", "ci_cluster": [lo, hi] | null
(кластерный бутстрэп по исходной функции), "n_clusters"}. Таблица T6 / рисунок F6 читают этот файл.
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

from smcode.calibration import conformal_threshold, decide, rate_with_ci
from smcode.config import ROOT, get_rng, resolve_path
from smcode.privacy import attack_align, attack_bow, attack_gen, projection, quantize
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
DEFAULT_BOOTSTRAP = 1000
DEFAULT_NOISE_SIGMA: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)  # relative: cos ≈ 0.97, 0.89, 0.71, 0.45
INFORMATIVE_COS = 0.97  # если все σ конфига дают cos > …, сетка шума дополняется DEFAULT_NOISE_SIGMA
NOISE_MODES: tuple[str, ...] = ("relative", "absolute")


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


def noise_mode(cfg: dict[str, Any]) -> str:
    """cfg.privacy.noise_mode: relative (по умолчанию) | absolute."""
    mode = str((cfg.get("privacy", {}) or {}).get("noise_mode", "relative"))
    if mode not in NOISE_MODES:
        raise ValueError(f"unknown noise mode {mode!r}; known: {NOISE_MODES}")
    return mode


def noise_grid(mode: str, d: int | None = None) -> list[float] | None:
    """Сетка σ с cos ≈ 0.97…0.45: DEFAULT_NOISE_SIGMA в relative-режиме; в absolute — σ/√d (нужна размерность d;
    без неё None)."""
    if mode == "relative":
        return [float(s) for s in DEFAULT_NOISE_SIGMA]
    if d is None or d <= 0:
        return None
    return [float(s) / float(np.sqrt(d)) for s in DEFAULT_NOISE_SIGMA]


def default_defences(cfg: dict[str, Any], d: int | None = None, extend_noise: bool = True) -> list[DefenceSpec]:
    """none; proj; proj+<bits> для quant_bits; proj+noise<σ> для σ > 0 из noise_sigma (по умолчанию DEFAULT_NOISE_SIGMA).

    Если все σ конфига дают ожидаемый cos > INFORMATIVE_COS (ветвь шума неотличима от «proj»), сетка дополняется
    noise_grid(mode, d) с предупреждением (extend_noise=False — не дополнять). Ключ noise_sigma, заданный явно
    без положительных σ, означает «без шума».
    """
    p = cfg.get("privacy", {}) or {}
    mode = noise_mode(cfg)
    out = [make_defence(), make_defence(projection=True)]
    for b in p.get("quant_bits", [8, 4, 1]) or []:
        out.append(make_defence(projection=True, bits=int(b)))
    configured = p.get("noise_sigma", list(DEFAULT_NOISE_SIGMA))
    sigmas = sorted({float(s) for s in (configured or []) if float(s) > 0})
    if extend_noise and sigmas:
        extra = noise_grid(mode, d)
        cos = [quantize.expected_cosine(s, mode, d) for s in sigmas] if (mode == "relative" or d) else []
        if extra is not None and cos and all(c > INFORMATIVE_COS for c in cos):
            log.warning("privacy: noise_sigma=%s (%s) дают ожидаемый cos ≥ %.3f — ветвь шума неинформативна; добавлена сетка σ=%s",
                        sigmas, mode, min(cos), [round(s, 4) for s in extra])
            sigmas = sorted(set(sigmas) | set(extra))
        elif extra is None and extend_noise:
            log.warning("privacy: noise_mode=absolute без размерности d — проверка информативности σ пропущена")
    for s in sigmas:
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
    """Входы шага 08 в памяти (см. load_inputs): индекс protected, обучающий корпус public_train (подвыборка атакующего),
    запросы по наборам; train_df/df_corpus_size — документные частоты идентификаторов по полному public_train
    (None — считать по обучающей подвыборке)."""

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
    train_df: dict[str, int] | None = None
    df_corpus_size: int | None = None
    query_codes: dict[str, tuple[list[str], list[str]]] | None = None  # set → (codes, langs), выровнены с qids (для M4)
    model: str | None = None  # model_key энкодера по meta npz (None — файлы без meta)

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


class _ModelCheck:
    """Собирает meta.model всех прочитанных npz: разные модели — ValueError (иначе скоры между векторами разных энкодеров
    были бы молча мусором; шаг 07 делает ту же проверку)."""

    def __init__(self) -> None:
        self.models: dict[str, str] = {}
        self.unverified: list[str] = []

    def add(self, path: Path) -> None:
        from smcode.semantic.embed import read_embeddings_meta
        from smcode.semantic.model import model_key

        m = read_embeddings_meta(path).get("model")
        if not m:
            self.unverified.append(path.name)
            log.warning("privacy: %s has no meta.model — encoder consistency cannot be verified", path)
            return
        self.models[str(path)] = model_key(m)
        if len(set(self.models.values())) > 1:
            raise ValueError(f"privacy: embeddings produced by different encoders: {self.models}; re-run scripts/06_embed.py for all "
                             "splits/sets with one checkpoint (or point paths.embeddings at one by_model/<tag>/ cache)")

    @property
    def model(self) -> str | None:
        vals = set(self.models.values())
        return next(iter(vals)) if vals else None


def index_with_windows(cfg: dict[str, Any]) -> bool | None:
    """with_windows основного индекса M3 (data/indexes/semantic/meta.json); None, если индекса нет."""
    from smcode.fingerprint.index import META_FILE, index_dir

    path = index_dir(cfg, "semantic") / META_FILE
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return bool(json.load(f).get("with_windows"))
    except (OSError, json.JSONDecodeError):
        return None


def _join_codes(ids: Sequence[str], *functions_paths: Path) -> tuple[list[str], list[str], np.ndarray]:
    """(codes, langs, маска найденных) для ids по файлам функций (функции и окна)."""
    pos = {rid: i for i, rid in enumerate(ids)}
    codes: list[str | None] = [None] * len(ids)
    langs: list[str] = [""] * len(ids)
    for functions_path in functions_paths:
        if not functions_path.exists():
            continue
        for row in read_jsonl(functions_path):
            i = pos.get(row.get("id"))
            if i is not None:
                codes[i] = row.get("code", "")
                langs[i] = row.get("lang", "")
    found = np.array([c is not None for c in codes], dtype=bool)
    return [c or "" for c in codes], langs, found


def _query_codes(cfg: dict[str, Any], set_name: str, qids: Sequence[str]) -> tuple[list[str], list[str]] | None:
    """(codes, langs) запросов набора, выровненные с qids (data/queries/<set>.jsonl); None, если файла нет."""
    path = resolve_path(cfg, "queries") / f"{set_name}.jsonl"
    if not path.exists():
        log.warning("privacy: queries file %s not found — hybrid utility unavailable for %s", path, set_name)
        return None
    pos = {q: i for i, q in enumerate(qids)}
    codes: list[str] = [""] * len(qids)
    langs: list[str] = [""] * len(qids)
    n = 0
    for row in read_jsonl(path):
        i = pos.get(row.get("qid"))
        if i is not None:
            codes[i], langs[i] = str(row.get("code", "")), str(row.get("lang", ""))
            n += 1
    if n < len(qids):
        log.warning("privacy: %s: %d/%d query embeddings without code in %s", set_name, len(qids) - n, len(qids), path.name)
    return codes, langs


def _subsample(n: int, max_n: int | None, rng: random.Random) -> np.ndarray:
    if max_n is None or n <= max_n:
        return np.arange(n)
    return np.sort(quantize.as_generator(rng).choice(n, size=int(max_n), replace=False))


def load_inputs(cfg: dict[str, Any], limit_train: int | None = None, limit_eval: int | None = None, sets: Iterable[str] = QUERY_SETS,
                rng: random.Random | None = None, full_df: bool = True, with_windows: bool | None = None,
                hybrid: bool | None = None) -> PrivacyInputs:
    """Читает data/embeddings/{protected[,protected_windows],public_train}.npz, data/functions/*.jsonl и
    data/embeddings/queries/<set>.npz.

    with_windows — включить окна protected в индекс (как основной индекс M3; None — по data/indexes/semantic/meta.json,
    без индекса — False). hybrid — читать коды запросов data/queries/<set>.jsonl для полезности M4 (None —
    cfg.privacy.utility_hybrid, по умолчанию true). Все npz проверяются на одну модель (meta.model) — ValueError иначе.
    limit_train/limit_eval — подвыборки (с seed) для атак; индекс и запросы берутся целиком. full_df=True: если
    public_train подвыбран, документные частоты идентификаторов (train_df, для словаря A1 и определения «редкий»)
    считаются по всему public_train (~1.5 мс на функцию), иначе — по подвыборке.
    """
    rng = rng or get_rng(cfg, "privacy_inputs")
    edir = embeddings_dir(cfg)
    fdir = resolve_path(cfg, "functions")
    p_idx = edir / "protected.npz"
    if not p_idx.exists():
        raise FileNotFoundError(f"protected embeddings not found: {p_idx} (run scripts/06_embed.py)")
    check = _ModelCheck()
    check.add(p_idx)
    index_ids, index_emb = _load_npz(p_idx)
    n_functions = len(index_ids)
    if with_windows is None:
        iw = index_with_windows(cfg)
        with_windows = bool(iw)
        if iw is None:
            log.info("privacy: no data/indexes/semantic/meta.json — index composition defaults to functions only")
    p_win = edir / "protected_windows.npz"
    n_windows = 0
    if with_windows:
        if p_win.exists():
            check.add(p_win)
            w_ids, w_emb = _load_npz(p_win)
            if w_emb.shape[1] != index_emb.shape[1]:
                raise ValueError(f"privacy: dim mismatch between {p_idx} ({index_emb.shape[1]}) and {p_win} ({w_emb.shape[1]})")
            index_ids, index_emb = index_ids + w_ids, np.concatenate([index_emb, w_emb], axis=0)
            n_windows = len(w_ids)
            log.info("privacy: index = %d protected functions + %d windows (as the main semantic index)", n_functions, n_windows)
        else:
            log.warning("privacy: main index contains windows but %s is missing — utility measured on functions only "
                        "(run scripts/06_embed.py --splits protected_windows)", p_win)
            with_windows = False
    index_codes, index_langs, found = _join_codes(index_ids, fdir / "protected.jsonl", fdir / "protected_windows.jsonl")
    if not found.all():
        log.warning("privacy: %d/%d protected embeddings without code (dropped)", int((~found).sum()), len(index_ids))
        keep = np.flatnonzero(found)
        index_ids = [index_ids[i] for i in keep]
        index_emb = index_emb[keep]
        index_codes = [index_codes[i] for i in keep]
        index_langs = [index_langs[i] for i in keep]
        n_functions = sum(1 for i in index_ids if ":w" not in i.rsplit(":", 1)[-1])
        n_windows = len(index_ids) - n_functions
    train_ids: list[str] = []
    train_emb = np.zeros((0, index_emb.shape[1]), dtype=np.float32)
    train_codes: list[str] = []
    train_langs: list[str] = []
    train_df: dict[str, int] | None = None
    df_corpus_size: int | None = None
    p_tr = edir / "public_train.npz"
    if p_tr.exists():
        check.add(p_tr)
        train_ids, train_emb = _load_npz(p_tr)
        train_codes, train_langs, found = _join_codes(train_ids, fdir / "public_train.jsonl")
        all_keep = np.flatnonzero(found)
        keep = all_keep[_subsample(len(all_keep), limit_train, rng)]
        if full_df and keep.size < all_keep.size:
            log.info("privacy: document frequencies over the full public_train corpus (%d functions; attack subsample %d)",
                     all_keep.size, keep.size)
            train_df = dict(attack_bow.identifier_df([train_codes[i] for i in all_keep], [train_langs[i] for i in all_keep], show_progress=True))
            df_corpus_size = int(all_keep.size)
        train_ids = [train_ids[i] for i in keep]
        train_emb = train_emb[keep]
        train_codes = [train_codes[i] for i in keep]
        train_langs = [train_langs[i] for i in keep]
    else:
        log.warning("privacy: %s not found — атаки A1/A2 недоступны", p_tr)
    queries: dict[str, tuple[list[str], np.ndarray]] = {}
    want_hybrid = bool((cfg.get("privacy", {}) or {}).get("utility_hybrid", True)) if hybrid is None else bool(hybrid)
    query_codes: dict[str, tuple[list[str], list[str]]] | None = {} if want_hybrid else None
    for s in sets:
        p = edir / "queries" / f"{s}.npz"
        if p.exists():
            check.add(p)
            queries[s] = _load_npz(p)
            if query_codes is not None:
                qc = _query_codes(cfg, s, queries[s][0])
                if qc is None:
                    query_codes = None  # без кодов хотя бы одного набора полезность M4 не считается
                else:
                    query_codes[s] = qc
        else:
            log.warning("privacy: query embeddings for %s not found (%s)", s, p)
    model = check.model
    if model is not None:
        from smcode.semantic.model import expected_model_key, model_key

        want = model_key(expected_model_key(cfg))
        if model != want:
            raise ValueError(f"privacy: embeddings in {edir} were produced by {model!r}, but the config expects {want!r} "
                             "(semantic.checkpoint/base_model): run scripts/06_embed.py with the config's checkpoint or "
                             "set paths.embeddings to the matching by_model/<tag>/ cache")
    meta = {"index_file": str(p_idx), "windows_file": str(p_win) if with_windows else None, "with_windows": bool(with_windows),
            "index_composition": "protected (functions) + protected_windows" if with_windows else "protected (functions)",
            "n_index_functions": int(n_functions), "n_index_windows": int(n_windows), "model": model,
            "embeddings_unverified": check.unverified or None,
            "train_file": str(p_tr) if p_tr.exists() else None, "limit_train": limit_train, "limit_eval": limit_eval}
    return PrivacyInputs(index_ids, index_emb, index_codes, index_langs, train_ids, train_emb, train_codes, train_langs, queries, meta,
                         train_df=train_df, df_corpus_size=df_corpus_size, query_codes=query_codes, model=model)


# ----------------------------------------------------------------------------- полезность


def query_cluster(qid: str) -> str:
    """Кластер запроса — исходная функция: qid до первого «#» (DESIGN §5: "<source_id>#<transform>#<k>")."""
    return qid.split("#", 1)[0]


class ClusterBoot:
    """Персентильный бутстрэп доли с ресемплированием кластеров (исходных функций): одна матрица счётчиков
    (B × C) на набор запросов, общая для всех порогов/защит → согласованные ДИ."""

    def __init__(self, clusters: Sequence[str], n_boot: int, rng: np.random.Generator) -> None:
        self.uniq, self.cidx = np.unique(np.asarray(list(clusters), dtype=str), return_inverse=True)
        self.C = int(self.uniq.size)
        self.B = int(n_boot)
        self.counts: np.ndarray | None = None
        if self.B > 0 and self.C > 0:
            self.counts = rng.multinomial(self.C, np.full(self.C, 1.0 / self.C), size=self.B).astype(np.float32)
        self._sizes = np.bincount(self.cidx, minlength=self.C).astype(np.float64) if self.C else np.zeros(0)

    def rate_ci(self, hits: np.ndarray, conf: float = 0.95) -> list[float] | None:
        """[lo, hi] доли hits (bool по запросам) по ресемплированным кластерам; None без бутстрэпа."""
        if self.counts is None or hits.size != self.cidx.size or hits.size == 0:
            return None
        h = np.bincount(self.cidx, weights=hits.astype(np.float64), minlength=self.C)
        num, den = self.counts @ h, self.counts @ self._sizes
        ok = den > 0
        if not ok.any():
            return None
        vals = num[ok] / den[ok]
        q = (1.0 - conf) / 2.0
        return [float(np.percentile(vals, 100 * q)), float(np.percentile(vals, 100 * (1 - q)))]


def make_cluster_boots(inputs: PrivacyInputs, n_boot: int, rng: random.Random) -> dict[str, ClusterBoot]:
    """ClusterBoot на каждый набор запросов (seed из rng; один и тот же бутстрэп для всех защит)."""
    g = quantize.as_generator(rng)
    return {s: ClusterBoot([query_cluster(q) for q in qids], n_boot, np.random.default_rng(int(g.integers(0, 2**31 - 1))))
            for s, (qids, _) in inputs.queries.items()}


def rate_block(scores: np.ndarray, tau: float, boot: ClusterBoot | None = None) -> dict[str, Any]:
    """RATE: доля score > τ с ДИ Клоппера–Пирсона (ci, по запросам) и кластерным бутстрэпом (ci_cluster, по функциям)."""
    out = rate_with_ci(scores, tau)
    out["ci_cluster"] = boot.rate_ci(decide(scores, tau)) if boot is not None else None
    out["n_clusters"] = boot.C if boot is not None else None
    return out


def prepare_hybrid(cfg: dict[str, Any], inputs: PrivacyInputs) -> tuple[Any, dict[str, Any]]:
    """HybridIndex для полезности M4 на plain-индексе (отпечатки считаются один раз; матрица подменяется на защищённую в
    utility) с комбинатором основного индекса: правило two_threshold — без обучения; logistic — кэш
    hybrid.combiner_path(cfg, model) той же сигнатуры. Возвращает (индекс | None, сведения {available, rule, combiner, reason})."""
    info: dict[str, Any] = {"available": False, "rule": None, "combiner": None, "reason": None}
    if inputs.query_codes is None:
        info["reason"] = "query codes unavailable (data/queries/<set>.jsonl missing or privacy.utility_hybrid=false)"
        return None, info
    missing = [s for s in inputs.queries if s not in inputs.query_codes]
    if missing:
        info["reason"] = f"no query codes for {missing}"
        return None, info
    from smcode.semantic.combiner import Combiner
    from smcode.semantic.hybrid import HybridIndex, combiner_matches, combiner_path

    hyb = HybridIndex(cfg)
    hyb.sem.use_faiss = False
    hyb.build_from_embeddings(inputs.index_ids, inputs.index_emb, inputs.index_codes, inputs.index_langs, meta={"model": inputs.model})
    info["rule"] = hyb.rule
    if hyb.rule == "logistic":
        path = combiner_path(cfg, hyb.sem.model_name)
        comb = Combiner.load(path) if path.exists() else None
        ok, why = combiner_matches(comb, hyb.combiner_signature())
        if not ok:
            info["reason"] = f"no fitted logistic combiner for this index ({path}: {why}); build the hybrid index (scripts/04_build_indexes.py) first"
            return None, info
        hyb.set_combiner(comb)
        info["combiner"] = comb.describe()
    info["available"] = True
    return hyb, info


def utility(spec: DefenceSpec, inputs: PrivacyInputs, Q: np.ndarray | None, defended_index: np.ndarray, alphas: Sequence[float],
            top_k: int = 1, boots: dict[str, ClusterBoot] | None = None, hybrid: Any = None) -> dict[str, Any]:
    """TPR/FPR методов M3 (и M4, если передан prepare_hybrid-индекс) на защищённом индексе с конформным порогом по
    защищённым скорам public_calib; boots — кластерный бутстрэп по наборам (make_cluster_boots) для ci_cluster."""
    from smcode.semantic.semantic_index import SemanticIndex

    out: dict[str, Any] = {"threshold": None, "tpr": {"semantic": None, "hybrid": None}, "tpr_by_alpha": {}, "fpr": {}, "fpr_hybrid": None,
                           "scores_mean": {}}
    if "protected" not in inputs.queries or "public_calib" not in inputs.queries:
        out["error"] = "query embeddings for protected/public_calib are missing"
        return out
    boots = boots or {}
    idx = SemanticIndex({"semantic": {"use_faiss": False, "ann_top_k": top_k}})
    idx.build_from_embeddings(inputs.index_ids, defended_index)
    scores: dict[str, np.ndarray] = {}
    scores_h: dict[str, np.ndarray] = {}
    if hybrid is not None:  # та же защищённая матрица, отпечатки и комбинатор — как в основном индексе M4
        hybrid.sem.build_from_embeddings(inputs.index_ids, defended_index, meta={"model": hybrid.sem.model_name})
    for s, (_, q) in inputs.queries.items():
        qd = defend_queries(q, spec, Q)
        sc, _, _, _ = idx.score_embeddings(qd, top_k=top_k)
        scores[s] = np.asarray(sc, dtype=np.float64)
        out["scores_mean"][s] = float(sc.mean()) if sc.size else None
        if hybrid is not None and inputs.query_codes is not None:
            codes, langs = inputs.query_codes[s]
            sh, _, _ = hybrid.score_embeddings_with_codes(qd, codes, langs, top_k=hybrid.top_k)
            scores_h[s] = np.asarray(sh, dtype=np.float64)
    for a in alphas:
        tau = conformal_threshold(scores["public_calib"], float(a))
        key = f"{float(a):g}"
        block = {"threshold": tau if np.isfinite(tau) else "inf", "semantic": rate_block(scores["protected"], tau, boots.get("protected")),
                 "fpr": {s: rate_block(scores[s], tau, boots.get(s)) for s in ("public_test", "hard_neg") if s in scores},
                 "hybrid": None, "threshold_hybrid": None, "fpr_hybrid": None}
        if scores_h:
            tau_h = conformal_threshold(scores_h["public_calib"], float(a))
            block.update({"threshold_hybrid": tau_h if np.isfinite(tau_h) else "inf",
                          "hybrid": rate_block(scores_h["protected"], tau_h, boots.get("protected")),
                          "fpr_hybrid": {s: rate_block(scores_h[s], tau_h, boots.get(s)) for s in ("public_test", "hard_neg") if s in scores_h}})
        out["tpr_by_alpha"][key] = block
    first = f"{float(alphas[0]):g}"
    out["threshold"] = out["tpr_by_alpha"][first]["threshold"]
    out["tpr"]["semantic"] = out["tpr_by_alpha"][first]["semantic"]
    out["tpr"]["hybrid"] = out["tpr_by_alpha"][first]["hybrid"]
    out["fpr"] = out["tpr_by_alpha"][first]["fpr"]
    out["fpr_hybrid"] = out["tpr_by_alpha"][first]["fpr_hybrid"]
    out["alpha"] = float(alphas[0])
    return out


# ----------------------------------------------------------------------------- прогон


def _a3_available() -> bool:
    return attack_bow.torch_available()


def protected_coverage(eval_bags: Sequence[set[str]], vocab: Any, train_df: dict[str, int] | Any) -> dict[str, Any]:
    """Покрытие словарём A1 идентификаторов protected: доля вхождений, покрытых словарём; идентификаторы с df = 0 в
    public_train (только protected: внутренние API, символы проекта) структурно не измеримы A1/A2 — их доля даётся отдельно."""
    total = sum(len(b) for b in eval_bags)
    ids = {t for b in eval_bags for t in b}
    in_vocab = sum(1 for b in eval_bags for t in b if t in vocab.index)
    only = {t for t in ids if int(train_df.get(t, 0)) == 0}
    only_occ = sum(1 for b in eval_bags for t in b if t in only)
    return {"occurrence_coverage": (in_vocab / total) if total else None, "n_protected_ids": len(ids),
            "n_protected_ids_in_vocab": sum(1 for t in ids if t in vocab.index), "n_protected_only_ids": len(only),
            "protected_only_occurrence_fraction": (only_occ / total) if total else None,
            "note": "A1/A2 измеряют утечку только идентификаторов, известных по public_train (нижняя оценка); protected-only покрывает A3"}


def run_privacy_inputs(
    cfg: dict[str, Any],
    inputs: PrivacyInputs,
    defences: Sequence[DefenceSpec] | None = None,
    leaked_pairs: Sequence[int] | None = None,
    a1_backend: str | None = None,
    with_a3: bool = False,
    key: bytes | None = None,
    rng: random.Random | None = None,
    hybrid: bool | None = None,
) -> dict[str, Any]:
    """Полный прогон §9 на данных в памяти → словарь по схеме privacy.json (без записи на диск).
    hybrid — считать полезность M4 (None — cfg.privacy.utility_hybrid; нужны inputs.query_codes)."""
    t0 = time.perf_counter()
    rng = rng or get_rng(cfg, "privacy")
    p = cfg.get("privacy", {}) or {}
    d = inputs.dim
    defences = list(defences) if defences else default_defences(cfg, d=d or None)
    ns = [int(n) for n in (leaked_pairs if leaked_pairs is not None else p.get("attack_leaked_pairs", [0, 100, 1000, 10000]))]
    alphas = [float(a) for a in (cfg.get("calibration", {}) or {}).get("alphas", [DEFAULT_ALPHA])] or [DEFAULT_ALPHA]
    mode = noise_mode(cfg)
    align_method = str(p.get("align_method", "procrustes"))
    n_boot = int(p.get("bootstrap", (cfg.get("eval", {}) or {}).get("bootstrap", DEFAULT_BOOTSTRAP)))
    if str(p.get("projection", "keyed_orthogonal")) != "keyed_orthogonal":
        raise ValueError(f"unsupported projection {p.get('projection')!r}")
    Q = projection.keyed_orthogonal(d, key if key is not None else projection.projection_key(cfg)) if d else None
    errors: list[str] = []
    result: dict[str, Any] = {
        "schema": SCHEMA, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "alpha": alphas[0], "alphas": alphas, "leaked_pairs": ns,
        "n_index": len(inputs.index_ids), "dim": d, "noise_mode": mode, "align_method": align_method, "bootstrap": n_boot,
        "queries": {s: len(q[0]) for s, q in inputs.queries.items()}, "a1": None, "a3": None, "defenses": [], "errors": errors, **inputs.meta,
    }
    boots = make_cluster_boots(inputs, n_boot, random.Random(rng.random()))
    want_hybrid = bool(p.get("utility_hybrid", True)) if hybrid is None else bool(hybrid)
    hyb = None
    hyb_info: dict[str, Any] = {"available": False, "rule": None, "combiner": None, "reason": "disabled"}
    if want_hybrid:
        try:
            hyb, hyb_info = prepare_hybrid(cfg, inputs)
        except Exception as exc:  # noqa: BLE001 — полезность M4 не должна ронять шаг
            log.exception("privacy: hybrid utility unavailable: %s", exc)
            hyb, hyb_info = None, {"available": False, "rule": None, "combiner": None, "reason": str(exc)}
        if hyb is None:
            log.warning("privacy: TPR M4 under defences not computed: %s", hyb_info.get("reason"))
    result["hybrid_utility"] = hyb_info

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
        df_corpus = inputs.df_corpus_size if inputs.train_df is not None else len(train_bags)
        kw = {"df": inputs.train_df, "n_docs": df_corpus}
        try:
            model, vocab, Y_train = attack_bow.train_bow_attack(inputs.train_emb, train_bags, cfg, rng=random.Random(rng.random()), backend=a1_backend, **kw)
        except ImportError as exc:
            log.warning("A1 torch backend unavailable (%s), falling back to ridge", exc)
            model, vocab, Y_train = attack_bow.train_bow_attack(inputs.train_emb, train_bags, cfg, rng=random.Random(rng.random()), backend="ridge", **kw)
        Y_eval = attack_bow.targets(eval_bags, vocab)
        n_val = min(Y_train.shape[0] // 10 + 1, attack_bow.DEFAULT_VAL_MAX)
        result["a1"] = {"backend": model.backend, "vocab_size": len(vocab), "n_rare_vocab": vocab.n_rare, "rare_df": vocab.rare_df,
                        "rare_df_range": [int(vocab.min_df), int(vocab.rare_df)], "df_corpus_size": int(df_corpus),
                        "df_corpus": "public_train (full)" if inputs.train_df is not None else "public_train (attack subsample)",
                        "n_train": model.n_train, "n_eval": int(eval_idx.size), "val_f1": model.val_f1, "val_f1_rules": model.val_f1_rules,
                        "decision": model.decision, "threshold": model.threshold, "thr_min_precision": model.min_precision,
                        "thr_min_f1": model.min_f1, "n_tokens_firing": int(np.isfinite(model.thresholds).sum()) if model.thresholds is not None else None,
                        "prior_baseline": attack_bow.prior_baseline(Y_train[n_val:], Y_train[:n_val], Y_eval, vocab),
                        "protected_coverage": protected_coverage(eval_bags, vocab, inputs.train_df if inputs.train_df is not None
                                                                 else attack_bow.document_frequency(train_bags))}
    else:
        errors.append("attacks skipped: public_train embeddings/codes or protected index missing")
    pool_max = int(p.get("leak_pool_max", DEFAULT_LEAK_POOL_MAX))
    pool_idx = _subsample(inputs.train_emb.shape[0], max(pool_max, max(ns) if ns else 0), random.Random(rng.random()))
    pool_plain = inputs.train_emb[pool_idx]
    eval_plain = inputs.index_emb[eval_idx]
    a3_model = a3_vocab = None
    a3_sel: np.ndarray | None = None
    a3_seed = int(rng.random() * 2**31)
    use_a3 = bool(with_a3) and attacks_enabled
    if with_a3 and not _a3_available():
        errors.append("A3 skipped: torch not available")
        use_a3 = False
    if use_a3:  # одна и та же подвыборка protected для всех защит → BLEU/id_acc сравнимы между строками T6
        a3_sel = attack_gen.eval_subset(int(eval_idx.size), cfg, random.Random(a3_seed))
        result["a3"] = {"n_eval": int(a3_sel.size), "eval_seed": a3_seed, "n_leaked_pairs": max(ns) if ns else 0}

    # --- защиты
    for spec in defences:
        ts = time.perf_counter()
        drng = random.Random(f"{rng.random()}:{spec.name}")
        defended = apply_defence(inputs.index_emb, spec, Q, drng, mode)
        ratio, nbytes = defence_memory(spec, len(inputs.index_ids), d)
        row: dict[str, Any] = {**spec.to_dict(), "noise_mode": mode if spec.sigma > 0 else None, "memory_ratio": ratio, "memory_bytes": nbytes,
                               "expected_cos": quantize.expected_cosine(spec.sigma, mode, d) if spec.sigma > 0 else None,
                               "attacks": {"A1": None, "A2": None, "A3": None}}
        try:
            row.update(utility(spec, inputs, Q, defended, alphas, boots=boots, hybrid=hyb))
        except Exception as exc:  # noqa: BLE001 — полезность не должна ронять атаки
            log.exception("utility failed for %s: %s", spec.name, exc)
            errors.append(f"utility {spec.name}: {exc}")
        if attacks_enabled and model is not None and Y_eval is not None:
            target = defended[eval_idx]
            row["attacks"]["A1"] = model.evaluate(target, Y_eval)
            pool_def = apply_defence(pool_plain, spec, Q, random.Random(f"{drng.random()}:pool"), mode)
            perm = attack_align.leak_permutation(pool_plain.shape[0], random.Random(drng.random()))  # вложенные выборки пар
            row["attacks"]["A2"] = attack_align.run_attack_align(model, pool_plain, pool_def, target, Y_eval, ns, None, method=align_method,
                                                                 Q_true=Q if spec.projection else np.eye(d, dtype=np.float32),
                                                                 target_plain=eval_plain, perm=perm)
            if use_a3 and a3_sel is not None:
                try:
                    n_max = max(ns) if ns else 0
                    pp, dd = attack_align.sample_leaked_pairs(pool_plain, pool_def, n_max, perm=perm)  # те же пары, что у A2 при max n
                    Q_hat = attack_align.estimate_projection(pp, dd, align_method) if pp.shape[0] else np.eye(d, dtype=np.float32)
                    rec = attack_align.recover(target, Q_hat)
                    codes = [inputs.index_codes[i] for i in eval_idx]
                    langs = [inputs.index_langs[i] for i in eval_idx]
                    a3, a3_model, a3_vocab = attack_gen.run_attack_gen(inputs.train_emb, inputs.train_codes, inputs.train_langs, rec, codes, langs,
                                                                       cfg, rng=random.Random(a3_seed), model=a3_model, vocab=a3_vocab, eval_idx=a3_sel)
                    a3["n_leaked_pairs"] = int(pp.shape[0])
                    a3["eval_seed"] = a3_seed
                    row["attacks"]["A3"] = a3
                except ImportError as exc:
                    errors.append(f"A3 skipped: {exc}")
                    use_a3 = False
        row["seconds"] = round(time.perf_counter() - ts, 2)
        tpr = (row.get("tpr") or {}).get("semantic") or {}
        tpr_h = (row.get("tpr") or {}).get("hybrid") or {}
        a1 = row["attacks"]["A1"] or {}
        log.info("defence %-14s mem=%.3f tpr M3=%s M4=%s A1 f1=%s rare=%s (chance %s) (%.1fs)", spec.name, ratio, tpr.get("value"),
                 tpr_h.get("value"), a1.get("f1"), a1.get("rare_id_recall"), a1.get("rare_id_recall_chance"), row["seconds"])
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
    full_df: bool = True,
    with_windows: bool | None = None,
    hybrid: bool | None = None,
) -> dict[str, Any]:
    """Шаг 08 с файлами: data/embeddings → results/privacy.json. Идемпотентно (файл есть → пропуск без force).
    full_df — документные частоты для словаря A1 по всему public_train; with_windows/hybrid — см. load_inputs."""
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
    inputs = load_inputs(cfg, limit_train=lt, limit_eval=limit_eval, full_df=full_df, with_windows=with_windows, hybrid=hybrid)
    specs = [parse_defence(n) for n in defences] if defences else None
    result = run_privacy_inputs(cfg, inputs, defences=specs, leaked_pairs=leaked_pairs, a1_backend=a1_backend, with_a3=with_a3, hybrid=hybrid)
    result["config"] = {"privacy": p, "calibration": cfg.get("calibration"), "seed": cfg.get("seed"), "config_path": cfg.get("_config_path"),
                        "root": str(ROOT)}
    write_privacy(cfg, result)
    result["skipped"] = False
    return result
