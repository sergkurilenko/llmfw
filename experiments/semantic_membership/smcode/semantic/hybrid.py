"""M4 hybrid: top-k кандидатов по косинусу (SemanticIndex) → признаки по winnowing-отпечаткам →
логистический комбинатор → score = max по кандидатам (DESIGN.md §6).

Признаки кандидата: [cos, overlap, longest_common_run / n_q_fps, log(1 + n_tokens_q)] (combiner.FEATURE_NAMES).
Отпечатки записей считаются при сборке (smcode.fingerprint.winnowing, k/w и HMAC-ключ из cfg.fingerprint)
и хранятся CSR-массивами (rec_offsets, rec_hashes). numpy-путь без torch:
build_from_embeddings(ids, matrix, codes, langs) + score_embeddings_with_codes(q_emb, q_codes, q_langs).
Комбинатор обучается на public_train (train_combiner_from_corpus: члены = преобразования функций
индексной половины public_train, не-члены = функции других репозиториев public_train) и хранится в
cfg.semantic.combiner_path (по умолчанию data/indexes/hybrid/combiner.pkl) и внутри state.pkl индекса.
Абляция: cfg.semantic.hybrid_rule = two_threshold (правило двух порогов без обучения).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from smcode.config import ROOT, get_rng, hmac_key, resolve_path
from smcode.fingerprint.winnowing import longest_common_run, winnow_fingerprints
from smcode.normalize import abstract_tokens, canon_lang, tokenize
from smcode.semantic.combiner import DEFAULT_C, DEFAULT_THRESHOLDS, FEATURE_NAMES, N_FEATURES, Combiner, candidate_labels
from smcode.semantic.semantic_index import BaseIndex, SemanticIndex, deep_sizeof, iter_records
from smcode.types import FunctionRecord, QueryResult, read_functions

try:
    from smcode.fingerprint.index import chunked, pool_context, pool_map
except ImportError:  # pragma: no cover
    from contextlib import contextmanager

    def chunked(it, n=2000):  # type: ignore[misc]
        buf = []
        for x in it:
            buf.append(x)
            if len(buf) >= n:
                yield buf
                buf = []
        if buf:
            yield buf

    @contextmanager
    def pool_context(workers):  # type: ignore[misc]
        yield None

    def pool_map(pool, fn, tasks):  # type: ignore[misc]
        return [fn(t) for t in tasks]

log = logging.getLogger(__name__)

DEFAULT_K = 25
DEFAULT_W = 25
DEFAULT_RULE = "logistic"
COMBINER_PATH_DEFAULT = "data/indexes/hybrid/combiner.pkl"
COMBINER_SPLIT = "public_train"
DEFAULT_COMBINER_N_INDEX = 20000
DEFAULT_COMBINER_N_ANCHORS = 2000
DEFAULT_COMBINER_N_NEG = 2000


def _fp_task(args: tuple[str, str, int, int, bytes | None]) -> tuple[np.ndarray, int]:
    """(последовательность отпечатков uint64, число abstract-токенов) записи; пустая, если токенов < k."""
    code, lang, k, w, key = args
    try:
        toks = abstract_tokens(tokenize(code, canon_lang(lang)), mode="full")
    except Exception as exc:  # noqa: BLE001 — неподдерживаемый язык/ошибка парсера
        log.debug("tokenize failed (%s): %s", lang, exc)
        return np.zeros(0, dtype=np.uint64), 0
    if len(toks) < k:
        return np.zeros(0, dtype=np.uint64), len(toks)
    fps = winnow_fingerprints(toks, k, w, key)
    return np.fromiter((h for h, _ in fps), dtype=np.uint64, count=len(fps)), len(toks)


def combiner_path(cfg: dict[str, Any]) -> Path:
    """cfg.semantic.combiner_path (относительно корня стенда)."""
    p = Path((cfg.get("semantic", {}) or {}).get("combiner_path", COMBINER_PATH_DEFAULT))
    return p if p.is_absolute() else ROOT / p


class HybridIndex(BaseIndex):
    """M4: семантический индекс + winnowing-отпечатки записей + комбинатор."""

    name = "hybrid"

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(cfg)
        self.sem = SemanticIndex(cfg)
        self.k = DEFAULT_K
        self.w = DEFAULT_W
        self.key: bytes | None = None
        self.workers = 1
        self.top_k = self.sem.top_k
        self.rule = DEFAULT_RULE
        self.thresholds: dict[str, float] = dict(DEFAULT_THRESHOLDS)
        self.C = DEFAULT_C
        self.combiner: Combiner | None = None
        self.rec_offsets = np.zeros(1, dtype=np.int64)
        self.rec_hashes = np.zeros(0, dtype=np.uint64)
        self._configure(self.cfg)

    def _configure(self, cfg: dict[str, Any] | None) -> None:
        if not cfg:
            return
        self.cfg = cfg
        self.sem._configure(cfg)
        fp = cfg.get("fingerprint", {}) or {}
        scfg = cfg.get("semantic", {}) or {}
        self.k = int(fp.get("k", DEFAULT_K))
        self.w = int(fp.get("w", DEFAULT_W))
        self.workers = int(fp.get("workers", 1) or 1)
        self.key = hmac_key(cfg)
        self.top_k = int(scfg.get("ann_top_k", self.sem.top_k))
        self.rule = str(scfg.get("hybrid_rule", DEFAULT_RULE))
        self.thresholds = {**DEFAULT_THRESHOLDS, **(scfg.get("hybrid_thresholds", {}) or {})}
        self.C = float(scfg.get("combiner_C", DEFAULT_C))
        if self.combiner is None and self.rule == "two_threshold":
            self.combiner = Combiner(rule="two_threshold", thresholds=self.thresholds)

    # --- отпечатки

    def fingerprint_seq(self, code: str, lang: str) -> tuple[np.ndarray, int]:
        """(последовательность отпечатков запроса, число токенов)."""
        return _fp_task((code, lang, self.k, self.w, self.key))

    def record_fps(self, idx: int) -> np.ndarray:
        return self.rec_hashes[self.rec_offsets[idx] : self.rec_offsets[idx + 1]]

    def _compute_fps(self, codes: Sequence[str], langs: Sequence[str]) -> None:
        parts: list[np.ndarray] = []
        counts: list[int] = []
        with pool_context(self.workers) as pool:
            for chunk in chunked(list(zip(codes, langs))):
                tasks = [(c, l, self.k, self.w, self.key) for c, l in chunk]
                for seq, _ in pool_map(pool, _fp_task, tasks):
                    counts.append(int(seq.size))
                    if seq.size:
                        parts.append(seq)
        self.rec_hashes = np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint64)
        self.rec_offsets = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(np.asarray(counts, dtype=np.int64), out=self.rec_offsets[1:])

    # --- построение

    def build(self, records: Iterable[FunctionRecord], cfg: dict[str, Any]) -> None:
        """Эмбеддинги (энкодер или готовые npz) + отпечатки записей + комбинатор (загрузка/обучение)."""
        self._configure(cfg)
        recs = list(iter_records(records))
        t0 = time.perf_counter()
        self.sem.build(recs, cfg)
        self._compute_fps([r.code for r in recs], [r.lang for r in recs])
        self.ids = list(self.sem.ids)
        self.ensure_combiner(cfg)
        self.meta = {**self.sem.meta, "k": self.k, "w": self.w, "keyed": self.key is not None, "rule": self.rule,
                     "n_fps_total": int(self.rec_hashes.size), "combiner": self.combiner.describe() if self.combiner else None,
                     "build_seconds": round(time.perf_counter() - t0, 3)}
        log.info("hybrid: %d records, %d fps, rule=%s, combiner fitted=%s", len(self.ids), self.rec_hashes.size, self.rule,
                 bool(self.combiner and self.combiner.fitted))

    def build_from_embeddings(self, ids: Sequence[str], matrix: np.ndarray, codes: Sequence[str], langs: Sequence[str],
                              meta: dict[str, Any] | None = None) -> None:
        """numpy-путь: готовые эмбеддинги + коды (для отпечатков). Комбинатор не обучается (set_combiner)."""
        if not (len(ids) == len(codes) == len(langs)):
            raise ValueError("ids, codes and langs must have the same length")
        self._reset()
        self.sem.build_from_embeddings(ids, matrix, meta=meta)
        self._compute_fps(codes, langs)
        self.ids = list(self.sem.ids)
        self.meta = {**self.sem.meta, "k": self.k, "w": self.w, "keyed": self.key is not None, "rule": self.rule,
                     "n_fps_total": int(self.rec_hashes.size)}

    # --- комбинатор

    def set_combiner(self, combiner: Combiner | None) -> None:
        self.combiner = combiner
        if combiner is not None:
            self.rule = combiner.rule
        self.meta["rule"] = self.rule
        self.meta["combiner"] = combiner.describe() if combiner else None

    def ensure_combiner(self, cfg: dict[str, Any], encoder: Any = None) -> Combiner:
        """two_threshold → правило; logistic → загрузка cfg.semantic.combiner_path либо обучение на public_train
        (нужен энкодер/torch); при невозможности — предупреждение и правило двух порогов."""
        if self.rule == "two_threshold":
            self.set_combiner(Combiner(rule="two_threshold", thresholds=self.thresholds))
            return self.combiner  # type: ignore[return-value]
        if self.combiner is not None and self.combiner.fitted:
            return self.combiner
        path = combiner_path(cfg)
        if path.exists():
            self.set_combiner(Combiner.load(path))
            log.info("hybrid: combiner loaded from %s (%s)", path, self.combiner.describe() if self.combiner else None)
            return self.combiner  # type: ignore[return-value]
        try:
            comb = train_combiner_from_corpus(cfg, encoder=encoder or (lambda: self.sem.encoder), out_path=path)
            self.set_combiner(comb)
        except FileNotFoundError as exc:
            log.warning("hybrid: cannot train combiner (%s); using two_threshold rule", exc)
            self.set_combiner(Combiner(rule="two_threshold", thresholds=self.thresholds))
        return self.combiner  # type: ignore[return-value]

    # --- признаки и оценка

    def candidate_features(self, q_seq: np.ndarray, n_tokens: int, cand_idx: np.ndarray, cand_sims: np.ndarray) -> np.ndarray:
        """Матрица признаков (k, N_FEATURES) для кандидатов одного запроса."""
        k = int(cand_idx.size)
        X = np.zeros((k, N_FEATURES), dtype=np.float32)
        if k == 0:
            return X
        X[:, 0] = cand_sims
        X[:, 3] = np.log1p(max(0, int(n_tokens)))
        if q_seq.size == 0:
            return X
        q_unique = np.unique(q_seq)
        q_pairs = [(int(h), i) for i, h in enumerate(q_seq)]
        for j, ci in enumerate(cand_idx):
            if ci < 0:
                continue
            c_seq = self.record_fps(int(ci))
            if c_seq.size == 0:
                continue
            X[j, 1] = float(np.isin(q_unique, c_seq).sum() / q_unique.size)
            if X[j, 1] > 0:
                c_pairs = [(int(h), i) for i, h in enumerate(c_seq)]
                X[j, 2] = longest_common_run(q_pairs, c_pairs) / max(1, q_seq.size)
        return X

    def features_for_queries(self, q_emb: np.ndarray, q_codes: Sequence[str], q_langs: Sequence[str],
                             top_k: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """(features (M, k, F), cand_idx (M, k), cand_sims (M, k), per-query info)."""
        q_emb = np.asarray(q_emb, dtype=np.float32)
        if q_emb.ndim == 1:
            q_emb = q_emb[None, :]
        M = q_emb.shape[0]
        if not (M == len(q_codes) == len(q_langs)):
            raise ValueError("q_emb, q_codes and q_langs must have the same length")
        _, _, idx, sims = self.sem.score_embeddings(q_emb, top_k=top_k or self.top_k)
        k = idx.shape[1]
        feats = np.zeros((M, k, N_FEATURES), dtype=np.float32)
        info: list[dict[str, Any]] = []
        for i in range(M):
            q_seq, n_tok = self.fingerprint_seq(q_codes[i], q_langs[i])
            feats[i] = self.candidate_features(q_seq, n_tok, idx[i], sims[i])
            info.append({"n_tokens": int(n_tok), "n_fps": int(q_seq.size)})
        return feats, idx, sims, info

    def score_candidates(self, feats: np.ndarray) -> np.ndarray:
        """Вероятности комбинатора для (M, k, F) → (M, k)."""
        M, k = feats.shape[0], feats.shape[1]
        if M == 0 or k == 0:
            return np.zeros((M, k), dtype=np.float32)
        comb = self.combiner
        if comb is None:
            log.warning("hybrid: no combiner set; using two_threshold rule")
            comb = self.combiner = Combiner(rule="two_threshold", thresholds=self.thresholds)
        return comb.predict_proba(feats.reshape(M * k, N_FEATURES)).reshape(M, k)

    def score_embeddings_with_codes(self, q_emb: np.ndarray, q_codes: Sequence[str], q_langs: Sequence[str],
                                    top_k: int | None = None) -> tuple[np.ndarray, list[str | None], list[dict[str, Any]]]:
        """numpy-путь: эмбеддинги + коды запросов → (scores (M,), best_ids, details по запросу)."""
        feats, idx, sims, info = self.features_for_queries(q_emb, q_codes, q_langs, top_k=top_k)
        M, k = idx.shape
        probs = self.score_candidates(feats)
        scores = np.zeros(M, dtype=np.float32)
        best_ids: list[str | None] = [None] * M
        details: list[dict[str, Any]] = []
        for i in range(M):
            d: dict[str, Any] = {**info[i], "rule": self.rule, "n_candidates": int((idx[i] >= 0).sum()) if k else 0}
            if k == 0 or d["n_candidates"] == 0:
                details.append(d)
                continue
            valid = idx[i] >= 0
            p = np.where(valid, probs[i], -1.0)
            j = int(np.argmax(p))
            scores[i] = float(np.clip(probs[i, j], 0.0, 1.0))
            best_ids[i] = self.ids[int(idx[i, j])]
            d.update({"best_rank": j, "sem_best_id": self.ids[int(idx[i, 0])], "sem_cos": float(sims[i, 0]),
                      **{name: float(feats[i, j, f]) for f, name in enumerate(FEATURE_NAMES)}})
            details.append(d)
        return scores, best_ids, details

    def query(self, code: str, lang: str) -> QueryResult:
        emb = self.sem.encode_queries([code], [lang])
        scores, best, details = self.score_embeddings_with_codes(emb, [code], [lang])
        return QueryResult(score=float(scores[0]), best_id=best[0], details=details[0])

    def query_batch(self, items: list[tuple[str, str]]) -> list[QueryResult]:
        """Батчевое кодирование (cfg.semantic.query_batch_size); latency_ms амортизирована на запрос."""
        bs = self.sem.query_batch_size
        if bs <= 1:
            return super().query_batch(items)
        out: list[QueryResult] = []
        for s in range(0, len(items), bs):
            chunk = items[s : s + bs]
            codes = [c for c, _ in chunk]
            langs = [l for _, l in chunk]
            t0 = time.perf_counter()
            emb = self.sem.encode_queries(codes, langs)
            scores, best, details = self.score_embeddings_with_codes(emb, codes, langs)
            lat = (time.perf_counter() - t0) * 1000.0 / max(1, len(chunk))
            for i in range(len(chunk)):
                details[i]["latency_mode"] = "amortized_batch"
                out.append(QueryResult(score=float(scores[i]), best_id=best[i], details=details[i], latency_ms=lat))
        return out

    # --- память и сериализация

    def memory_bytes(self) -> int:
        return int(self.sem.memory_bytes() + self.rec_offsets.nbytes + self.rec_hashes.nbytes + deep_sizeof(self.combiner))

    def _state(self) -> dict[str, Any]:
        st = self.sem._state()
        return {"matrix": st["matrix"], "rec_offsets": self.rec_offsets, "rec_hashes": self.rec_hashes,
                "combiner": self.combiner,
                "params": {**st["params"], "k": self.k, "w": self.w, "keyed": self.key is not None, "rule": self.rule,
                           "thresholds": dict(self.thresholds)}}

    def _restore(self, state: dict[str, Any]) -> None:
        ids = list(self.ids)
        meta = dict(self.meta)
        p = state.get("params", {}) or {}
        self.k = int(p.get("k", self.k))
        self.w = int(p.get("w", self.w))
        self.rule = str(p.get("rule", self.rule))
        self.thresholds = {**self.thresholds, **(p.get("thresholds", {}) or {})}
        self.sem.model_name = p.get("model", self.sem.model_name)
        self.sem.build_from_embeddings(ids, np.asarray(state["matrix"], dtype=np.float32), meta=meta)
        self.rec_offsets = np.asarray(state["rec_offsets"], dtype=np.int64)
        self.rec_hashes = np.asarray(state["rec_hashes"], dtype=np.uint64)
        self.ids = ids
        self.meta = meta
        comb = state.get("combiner")
        self.combiner = comb if isinstance(comb, Combiner) else None
        if p.get("keyed") and self.key is None:
            log.warning("hybrid index was built with an HMAC key, but none is set now: overlap features will be 0")


# ----------------------------------------------------------------------------- обучение комбинатора


def collect_features(index: HybridIndex, q_emb: np.ndarray, q_codes: Sequence[str], q_langs: Sequence[str],
                 q_labels: Sequence[int], q_source_ids: Sequence[str | None],
                 top_k: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Обучающая выборка комбинатора: (X (M·k, F), y (M·k,), group (M·k,) — номер запроса).
    y = 1 ⇔ запрос — член и кандидат — его исходная функция (DESIGN.md §6)."""
    feats, idx, _, _ = index.features_for_queries(q_emb, q_codes, q_langs, top_k=top_k)
    M, k = idx.shape
    if M == 0 or k == 0:
        return np.zeros((0, N_FEATURES), np.float32), np.zeros(0, np.int64), np.zeros(0, np.int64)
    cand_ids = [[index.ids[int(c)] if c >= 0 else None for c in row] for row in idx]
    y = candidate_labels(q_labels, q_source_ids, cand_ids).reshape(-1)
    valid = (idx >= 0).reshape(-1)
    X = feats.reshape(M * k, N_FEATURES)
    groups = np.repeat(np.arange(M, dtype=np.int64), k)
    return X[valid], y[valid], groups[valid]


def _split_by_repo(records: Sequence[FunctionRecord], rng, frac: float = 0.5) -> tuple[list[FunctionRecord], list[FunctionRecord]]:
    repos = sorted({r.repo for r in records})
    rng.shuffle(repos)
    n_a = max(1, int(round(len(repos) * frac))) if len(repos) > 1 else len(repos)
    a = set(repos[:n_a])
    return [r for r in records if r.repo in a], [r for r in records if r.repo not in a]


def _sample(items: list[Any], n: int | None, rng) -> list[Any]:
    items = list(items)
    if n is None or len(items) <= n:
        return items
    rng.shuffle(items)
    return items[:n]


def combiner_training_queries(cfg: dict[str, Any], records: Sequence[FunctionRecord] | None = None,
                              n_index: int | None = None, n_anchors: int | None = None, n_neg: int | None = None,
                              rng: Any = None) -> tuple[list[FunctionRecord], list[dict[str, Any]]]:
    """Готовит (записи индекса, запросы) для обучения комбинатора из public_train (без torch):
    репозитории делятся пополам; индекс и члены — из первой половины (все программные преобразования,
    smcode.eval.build_queries.queries_for_record), не-члены — функции второй половины с теми же преобразованиями."""
    from smcode.eval.build_queries import queries_for_record

    scfg = cfg.get("semantic", {}) or {}
    rng = rng or get_rng(cfg, "combiner")
    if records is None:
        path = resolve_path(cfg, "functions") / f"{COMBINER_SPLIT}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found (needed to train the hybrid combiner)")
        records = [r for r in read_functions(path) if (r.kind or "function") == "function"]
    records = sorted(records, key=lambda r: r.id)
    if not records:
        raise FileNotFoundError("no public_train records for the combiner")
    part_a, part_b = _split_by_repo(records, rng)
    if not part_b:  # один репозиторий: делим функции пополам
        half = len(part_a) // 2
        part_a, part_b = part_a[:half] or part_a, part_a[half:] or part_a
    index_recs = _sample(part_a, int(n_index or scfg.get("combiner_n_index", DEFAULT_COMBINER_N_INDEX)), rng)
    anchors = _sample(index_recs, int(n_anchors or scfg.get("combiner_n_anchors", DEFAULT_COMBINER_N_ANCHORS)), rng)
    negs = _sample(part_b, int(n_neg or scfg.get("combiner_n_neg", DEFAULT_COMBINER_N_NEG)), rng)
    queries: list[dict[str, Any]] = []
    for label, recs in ((1, anchors), (0, negs)):
        for rec in recs:
            for q in queries_for_record(rec, f"combiner{label}", cfg):
                queries.append({"qid": q.qid, "code": q.code, "lang": q.lang, "label": label, "source_id": rec.id,
                                "transform": q.transform})
    log.info("combiner data: index=%d (%d anchors) → %d member queries; %d negative functions → %d non-member queries",
             len(index_recs), len(anchors), sum(q["label"] for q in queries), len(negs), sum(1 - q["label"] for q in queries))
    return index_recs, queries


def train_combiner_from_corpus(cfg: dict[str, Any], encoder: Any = None, records: Sequence[FunctionRecord] | None = None,
                               n_index: int | None = None, n_anchors: int | None = None, n_neg: int | None = None,
                               out_path: str | Path | None = None, force: bool = False) -> Combiner:
    """Обучает логистический комбинатор на public_train (эмбеддинги — энкодером или из data/embeddings/public_train.npz)
    и сохраняет в cfg.semantic.combiner_path. ``encoder`` — объект с encode() или фабрика (torch нужен для кодирования)."""
    from smcode.semantic.embed import embeddings_for_records

    out = Path(out_path) if out_path else combiner_path(cfg)
    if out.exists() and not force:
        log.info("combiner exists at %s, loading", out)
        return Combiner.load(out)
    scfg = cfg.get("semantic", {}) or {}
    index_recs, queries = combiner_training_queries(cfg, records, n_index, n_anchors, n_neg)
    if encoder is None:
        from smcode.semantic.model import load_encoder

        encoder = lambda: load_encoder(cfg)  # noqa: E731
    enc_obj: Any = None

    def get_enc() -> Any:
        nonlocal enc_obj
        if enc_obj is None:
            enc_obj = encoder() if callable(encoder) and not hasattr(encoder, "encode") else encoder
        return enc_obj

    ids = [r.id for r in index_recs]
    emb = embeddings_for_records(cfg, get_enc, ids, [r.code for r in index_recs], [r.lang for r in index_recs], split=COMBINER_SPLIT)
    index = HybridIndex(cfg)
    index.rule = "logistic"
    index.build_from_embeddings(ids, emb, [r.code for r in index_recs], [r.lang for r in index_recs], meta={"purpose": "combiner"})
    q_emb = np.asarray(get_enc().encode([q["code"] for q in queries], [q["lang"] for q in queries],
                                        batch_size=int(scfg.get("batch_size", 32)), show_progress=True), dtype=np.float32)
    X, y, _ = collect_features(index, q_emb, [q["code"] for q in queries], [q["lang"] for q in queries],
                               [q["label"] for q in queries], [q["source_id"] for q in queries])
    comb = Combiner(rule="logistic", C=float(scfg.get("combiner_C", DEFAULT_C)), thresholds=index.thresholds).fit(X, y)
    comb.train_stats.update({"n_index": len(index_recs), "n_queries": len(queries), "model": getattr(get_enc(), "model_name", None)})
    comb.save(out)
    log.info("combiner saved → %s: %s", out, comb.describe())
    return comb
