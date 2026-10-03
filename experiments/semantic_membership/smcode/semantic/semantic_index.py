"""M3 semantic: индекс эмбеддингов энкодера, score = max cosine по top-k (DESIGN.md §6).

numpy-часть (build_from_embeddings, score_embeddings, save/load) работает без torch и используется
модулями eval и privacy; query/query_batch кодируют запрос энкодером (model.load_encoder, нужен torch).
build(records) использует готовые data/embeddings/<split>.npz, если они покрывают записи
(cfg.semantic.reuse_embeddings), иначе кодирует энкодером. Артефакты: data/indexes/semantic/
(arrays.npz: matrix float16/float32; state.pkl: ids, params; meta.json).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from smcode.semantic.ann import ANN, l2_normalize
from smcode.types import FunctionRecord, QueryResult

log = logging.getLogger(__name__)

try:  # базовый класс пишет модуль fingerprint; при его отсутствии — минимальная локальная замена
    from smcode.fingerprint.index import BaseIndex, deep_sizeof, iter_records
except ImportError:  # pragma: no cover
    import pickle
    import sys

    def deep_sizeof(obj: Any) -> int:  # type: ignore[misc]
        return sys.getsizeof(obj)

    def iter_records(records: Iterable[Any]) -> Iterable[FunctionRecord]:  # type: ignore[misc]
        for r in records:
            yield r if isinstance(r, FunctionRecord) else FunctionRecord.from_dict(r)

    class BaseIndex:  # type: ignore[no-redef]
        """Минимальный базовый класс (контракт DESIGN.md §6) на случай отсутствия smcode.fingerprint.index."""

        name = "base"

        def __init__(self, cfg: dict[str, Any] | None = None) -> None:
            self.cfg = cfg or {}
            self.ids: list[str] = []
            self.meta: dict[str, Any] = {}

        @property
        def n_records(self) -> int:
            return len(self.ids)

        def _reset(self) -> None:
            self.ids, self.meta = [], {}

        def query(self, code: str, lang: str) -> QueryResult:
            raise NotImplementedError

        def query_batch(self, items: list[tuple[str, str]]) -> list[QueryResult]:
            out = []
            for code, lang in items:
                t0 = time.perf_counter()
                r = self.query(code, lang)
                r.latency_ms = (time.perf_counter() - t0) * 1000.0
                out.append(r)
            return out

        def _state(self) -> dict[str, Any]:
            return {}

        def _restore(self, state: dict[str, Any]) -> None:
            pass

        def save(self, path: str | Path) -> None:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            state = dict(self._state())
            arrays = {k: v for k, v in state.items() if isinstance(v, np.ndarray)}
            rest = {k: v for k, v in state.items() if not isinstance(v, np.ndarray)}
            rest["ids"] = list(self.ids)
            np.savez(path / "arrays.npz", **arrays)
            with open(path / "state.pkl", "wb") as f:
                pickle.dump(rest, f)

        def load(self, path: str | Path) -> None:
            path = Path(path)
            with open(path / "state.pkl", "rb") as f:
                state = pickle.load(f)
            with np.load(path / "arrays.npz", allow_pickle=False) as z:
                state.update({k: z[k] for k in z.files})
            self._reset()
            self.ids = list(state.pop("ids", []))
            self._restore(state)

        def memory_bytes(self) -> int:
            return sum(v.nbytes for v in self._state().values() if isinstance(v, np.ndarray))


DEFAULT_TOP_K = 20
DEFAULT_QUERY_BATCH = 32


class SemanticIndex(BaseIndex):
    """M3: матрица L2-нормированных эмбеддингов + ANN; score = max cos по top-k (обрезан в [0, 1])."""

    name = "semantic"

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(cfg)
        self.matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.ann: ANN | None = None
        self._encoder: Any = None
        self.top_k = DEFAULT_TOP_K
        self.query_batch_size = DEFAULT_QUERY_BATCH
        self.use_faiss: bool | None = None
        self.index_fp16 = True
        self.checkpoint: str | None = None
        self.model_name: str | None = None
        self._configure(self.cfg)

    def _configure(self, cfg: dict[str, Any] | None) -> None:
        if not cfg:
            return
        self.cfg = cfg
        scfg = cfg.get("semantic", {}) or {}
        self.top_k = int(scfg.get("ann_top_k", DEFAULT_TOP_K))
        self.query_batch_size = int(scfg.get("query_batch_size", DEFAULT_QUERY_BATCH))
        self.use_faiss = scfg.get("use_faiss", None)
        self.index_fp16 = bool(scfg.get("index_fp16", True))
        self.checkpoint = scfg.get("checkpoint")
        self.model_name = str(scfg.get("checkpoint") or scfg.get("base_model") or "")

    # --- энкодер (torch, лениво)

    @property
    def encoder(self) -> Any:
        """Энкодер запросов (CodeEncoder/OwnEncoder); загружается при первом обращении (нужен torch)."""
        if self._encoder is None:
            from smcode.semantic.model import load_encoder

            own = bool((self.cfg.get("semantic", {}).get("own_small_model", {}) or {}).get("enabled", False))
            self._encoder = load_encoder(self.cfg, self.checkpoint, own_model=own and bool(self.checkpoint))
            self.model_name = getattr(self._encoder, "model_name", self.model_name)
        return self._encoder

    def set_encoder(self, encoder: Any) -> None:
        self._encoder = encoder
        self.model_name = getattr(encoder, "model_name", self.model_name)

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1]) if self.matrix.ndim == 2 and self.matrix.size else 0

    # --- построение

    def build(self, records: Iterable[FunctionRecord], cfg: dict[str, Any]) -> None:
        """Кодирует записи (или берёт готовые эмбеддинги сплита) и строит индекс."""
        from smcode.semantic.embed import embeddings_for_records

        self._configure(cfg)
        recs = list(iter_records(records))
        t0 = time.perf_counter()
        ids = [r.id for r in recs]
        codes = [r.code for r in recs]
        langs = [r.lang for r in recs]
        splits = {r.split for r in recs}
        emb = np.zeros((len(recs), 0), dtype=np.float32)
        if recs:
            if len(splits) == 1:
                emb = embeddings_for_records(cfg, lambda: self.encoder, ids, codes, langs, split=next(iter(splits)))
            else:  # смесь сплитов (например, protected + окна): по сплитам
                parts = []
                for s in sorted(splits):
                    sel = [i for i, r in enumerate(recs) if r.split == s]
                    e = embeddings_for_records(cfg, lambda: self.encoder, [ids[i] for i in sel], [codes[i] for i in sel],
                                               [langs[i] for i in sel], split=s)
                    parts.append((sel, e))
                d = parts[0][1].shape[1]
                emb = np.zeros((len(recs), d), dtype=np.float32)
                for sel, e in parts:
                    emb[sel] = e
        self.build_from_embeddings(ids, emb, meta={"model": self.model_name, "build_seconds": round(time.perf_counter() - t0, 3),
                                                   "splits": sorted(splits)})

    def build_from_embeddings(self, ids: Sequence[str], matrix: np.ndarray, meta: dict[str, Any] | None = None) -> None:
        """numpy-путь: ids + матрица (N, d) → нормировка → ANN. Используется eval/privacy и load()."""
        ids = list(ids)
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(ids):
            raise ValueError(f"matrix shape {matrix.shape} does not match {len(ids)} ids")
        self._reset()
        self.ids = ids
        self.matrix = l2_normalize(matrix) if matrix.size else np.zeros((0, matrix.shape[1] if matrix.ndim == 2 else 0), np.float32)
        self.ann = ANN(self.matrix, use_faiss=self.use_faiss)
        self.meta = {"dim": self.dim, "n_records": len(ids), "ann_backend": self.ann.backend, "top_k": self.top_k,
                     "model": self.model_name, **(meta or {})}
        log.info("semantic: %d vectors (d=%d), ann=%s", len(ids), self.dim, self.ann.backend)

    # --- оценка

    def score_embeddings(self, q: np.ndarray, top_k: int | None = None) -> tuple[np.ndarray, list[str | None], np.ndarray, np.ndarray]:
        """(scores (M,), best_ids, top_idx (M, k), top_sims (M, k)); score = max cos, обрезанный в [0, 1]."""
        q = l2_normalize(q)
        M = q.shape[0]
        k = int(top_k or self.top_k)
        if self.ann is None or self.n_records == 0 or M == 0:
            return np.zeros(M, dtype=np.float32), [None] * M, np.zeros((M, 0), dtype=np.int64), np.zeros((M, 0), dtype=np.float32)
        sims, idx = self.ann.search(q, k)
        scores = np.clip(sims[:, 0], 0.0, 1.0).astype(np.float32)
        best_ids: list[str | None] = [self.ids[int(i)] if i >= 0 else None for i in idx[:, 0]]
        return scores, best_ids, idx, sims

    def encode_queries(self, codes: Sequence[str], langs: Sequence[str]) -> np.ndarray:
        """Эмбеддинги запросов энкодером (нужен torch)."""
        return np.asarray(self.encoder.encode(list(codes), list(langs), batch_size=max(1, self.query_batch_size)), dtype=np.float32)

    def _result(self, score: float, best_id: str | None, idx_row: np.ndarray, sims_row: np.ndarray) -> QueryResult:
        details: dict[str, Any] = {"n_candidates": int((idx_row >= 0).sum()) if idx_row.size else 0}
        if sims_row.size:
            details["cos"] = float(sims_row[0])
            details["top"] = [(self.ids[int(i)], round(float(s), 4)) for i, s in zip(idx_row[:5], sims_row[:5]) if i >= 0]
        return QueryResult(score=float(score), best_id=best_id, details=details)

    def query(self, code: str, lang: str) -> QueryResult:
        emb = self.encode_queries([code], [lang])
        scores, best, idx, sims = self.score_embeddings(emb)
        return self._result(scores[0], best[0], idx[0], sims[0])

    def query_batch(self, items: list[tuple[str, str]]) -> list[QueryResult]:
        """Батчевое кодирование (cfg.semantic.query_batch_size); latency_ms — амортизированная на запрос.
        При query_batch_size ≤ 1 — поштучно с точной латентностью (BaseIndex.query_batch)."""
        if self.query_batch_size <= 1:
            return super().query_batch(items)
        out: list[QueryResult] = []
        bs = self.query_batch_size
        for s in range(0, len(items), bs):
            chunk = items[s : s + bs]
            t0 = time.perf_counter()
            emb = self.encode_queries([c for c, _ in chunk], [l for _, l in chunk])
            scores, best, idx, sims = self.score_embeddings(emb)
            lat = (time.perf_counter() - t0) * 1000.0 / max(1, len(chunk))
            for i in range(len(chunk)):
                r = self._result(scores[i], best[i], idx[i], sims[i])
                r.latency_ms = lat
                r.details["latency_mode"] = "amortized_batch"
                out.append(r)
        return out

    # --- память и сериализация

    def memory_bytes(self) -> int:
        ann_extra = (self.ann.memory_bytes() - int(self.matrix.nbytes)) if self.ann is not None else 0
        return int(self.matrix.nbytes) + max(0, ann_extra) + int(deep_sizeof(self.ids))

    def _state(self) -> dict[str, Any]:
        m = self.matrix.astype(np.float16) if self.index_fp16 else self.matrix
        return {"matrix": m, "params": {"top_k": self.top_k, "model": self.model_name, "dim": self.dim,
                                        "index_fp16": self.index_fp16, "n_records": self.n_records}}

    def _restore(self, state: dict[str, Any]) -> None:
        ids = list(self.ids)
        meta = dict(self.meta)
        p = state.get("params", {}) or {}
        self.model_name = p.get("model", self.model_name)
        m = np.asarray(state["matrix"], dtype=np.float32)
        if m.ndim != 2:
            m = m.reshape(len(ids), -1) if len(ids) else np.zeros((0, 0), np.float32)
        self.build_from_embeddings(ids, m, meta=meta)
