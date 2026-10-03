"""Атака A1 — инверсия «мешок токенов» (DESIGN.md §9).

По эмбеддингу функции предсказывается наличие каждого из V лексических идентификаторов (токены
normalize.tokenize с kind == "id"). Словарь строится по public_train: частые идентификаторы (df > rare_df)
по убыванию df плюс квота редких (min_df ≤ df ≤ rare_df), чтобы метрика «доля восстановленных редких
идентификаторов» была измерима. Модель: MLP на torch (если установлен) или гребневая регрессия
(замкнутая форма, numpy). Решающее правило: порог на каждый токен, максимизирующий F1 этого токена на
вневыборочных скорах обучающего корпуса (ridge — точные leave-one-out скоры через рычаги h_ii; MLP — 2-кратный
cross-fitting; decision="per_token" — иначе редкие токены никогда не предсказываются), либо один глобальный порог
по micro-F1 на отложенной части public_train (decision="global"); decision="auto" (по умолчанию) выбирает правило
с большим micro-F1 на валидации. В метриках: f1/precision/recall — по выбранному правилу (плюс f1_global и
f1_per_token), rare_id_recall/precision — всегда по правилу per_token (rare_id_rule).
Оценка на protected: micro precision/recall/F1 по токенам, recall редких идентификаторов, базовая линия
«априорное предсказание» (одинаковый набор токенов для всех функций).
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import sparse

from smcode.normalize import tokenize
from smcode.privacy.quantize import as_generator

log = logging.getLogger(__name__)

DEFAULT_VOCAB = 20000
DEFAULT_MIN_DF = 2
DEFAULT_RARE_DF = 3
DEFAULT_RARE_FRACTION = 0.25
DEFAULT_RIDGE_ALPHA = 1.0
DEFAULT_VAL_FRACTION = 0.1
DEFAULT_VAL_MAX = 2000
N_THRESHOLDS = 64
CHUNK = 1024
COL_BLOCK = 256
DEFAULT_THR_MEM = 4e9  # байт на матрицу скоров float16 для порогов MLP
DECISIONS: tuple[str, ...] = ("auto", "per_token", "global")


# ----------------------------------------------------------------------------- токены и словарь


def identifier_tokens(code: str, lang: str) -> list[str]:
    """Лексические идентификаторы функции (с повторами); пустой список при ошибке парсера/языке."""
    try:
        return [t.text for t in tokenize(code, lang) if t.kind == "id"]
    except ValueError:
        return []
    except Exception as exc:  # noqa: BLE001 — одна запись не должна ронять атаку
        log.debug("tokenize failed: %s", exc)
        return []


def identifier_bags(codes: Sequence[str], langs: Sequence[str], show_progress: bool = False) -> list[set[str]]:
    """Множества идентификаторов по функциям."""
    it: Iterable[tuple[str, str]] = zip(codes, langs)
    if show_progress:
        try:
            from tqdm import tqdm

            it = tqdm(it, total=len(codes), desc="identifiers", unit="fn", leave=False)
        except ImportError:  # pragma: no cover
            pass
    return [set(identifier_tokens(c, l)) for c, l in it]


def document_frequency(bags: Iterable[set[str]]) -> Counter:
    df: Counter = Counter()
    for b in bags:
        df.update(b)
    return df


@dataclass
class Vocab:
    """Словарь идентификаторов атаки: tokens, df (в public_train), граница редкости rare_df."""

    tokens: list[str]
    df: np.ndarray
    rare_df: int = DEFAULT_RARE_DF
    index: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.df = np.asarray(self.df, dtype=np.int64)
        if not self.index:
            self.index = {t: i for i, t in enumerate(self.tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def rare_mask(self) -> np.ndarray:
        return self.df <= self.rare_df

    @property
    def n_rare(self) -> int:
        return int(self.rare_mask.sum())

    def to_dict(self) -> dict[str, Any]:
        return {"tokens": list(self.tokens), "df": self.df.tolist(), "rare_df": int(self.rare_df)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Vocab":
        return cls(tokens=list(d["tokens"]), df=np.asarray(d["df"], dtype=np.int64), rare_df=int(d.get("rare_df", DEFAULT_RARE_DF)))


def build_vocab(
    bags: Sequence[set[str]],
    max_size: int = DEFAULT_VOCAB,
    min_df: int = DEFAULT_MIN_DF,
    rare_df: int = DEFAULT_RARE_DF,
    rare_fraction: float = DEFAULT_RARE_FRACTION,
    rng: random.Random | None = None,
) -> Vocab:
    """Словарь из top-частых идентификаторов (df > rare_df) и квоты редких (min_df ≤ df ≤ rare_df).

    Квота редких — rare_fraction·max_size (случайная выборка с rng; при rng=None — первые по (−df, токен)).
    Если частых меньше, чем их доля, остаток заполняется редкими, и наоборот.
    """
    df = document_frequency(bags)
    items = [(t, c) for t, c in df.items() if c >= min_df]
    frequent = sorted((x for x in items if x[1] > rare_df), key=lambda x: (-x[1], x[0]))
    rare = sorted((x for x in items if x[1] <= rare_df), key=lambda x: (-x[1], x[0]))
    if rng is not None:
        rng.shuffle(rare)
    n_rare = min(len(rare), int(round(max_size * rare_fraction)))
    n_freq = min(len(frequent), max_size - n_rare)
    n_rare = min(len(rare), max_size - n_freq)
    chosen = frequent[:n_freq] + rare[:n_rare]
    chosen.sort(key=lambda x: (-x[1], x[0]))
    tokens = [t for t, _ in chosen]
    counts = np.asarray([c for _, c in chosen], dtype=np.int64)
    log.info("bow vocab: %d tokens (%d frequent, %d rare df≤%d) из %d кандидатов с df≥%d", len(tokens), n_freq, n_rare,
             rare_df, len(items), min_df)
    return Vocab(tokens=tokens, df=counts, rare_df=rare_df)


def targets(bags: Sequence[set[str]], vocab: Vocab) -> sparse.csr_matrix:
    """Бинарная разреженная матрица (N, V): 1, если идентификатор есть в функции."""
    rows: list[int] = []
    cols: list[int] = []
    for i, b in enumerate(bags):
        for t in b:
            j = vocab.index.get(t)
            if j is not None:
                rows.append(i)
                cols.append(j)
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(bags), len(vocab)), dtype=np.float32)


# ----------------------------------------------------------------------------- метрики


def _dense_rows(Y: sparse.csr_matrix, lo: int, hi: int) -> np.ndarray:
    return np.asarray(Y[lo:hi].toarray() > 0)


def f1_from_counts(tp: float, n_pred: float, n_true: float) -> dict[str, float]:
    p = tp / n_pred if n_pred > 0 else 0.0
    r = tp / n_true if n_true > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}


def threshold_curve(scores_fn: Any, Y: sparse.csr_matrix, thresholds: np.ndarray, chunk: int = CHUNK) -> np.ndarray:
    """micro-F1 для каждого порога: scores_fn(lo, hi) → (hi−lo, V) скоры строк lo..hi."""
    n = Y.shape[0]
    tp = np.zeros(len(thresholds), dtype=np.float64)
    n_pred = np.zeros(len(thresholds), dtype=np.float64)
    n_true = float(Y.nnz)
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        s = scores_fn(lo, hi)
        y = _dense_rows(Y, lo, hi)
        for k, t in enumerate(thresholds):
            pred = s > t
            n_pred[k] += float(pred.sum())
            tp[k] += float((pred & y).sum())
    f1 = np.array([f1_from_counts(tp[k], n_pred[k], n_true)["f1"] for k in range(len(thresholds))])
    return f1


def best_threshold(scores_fn: Any, Y: sparse.csr_matrix, lo_hi: tuple[float, float], n_thresholds: int = N_THRESHOLDS) -> tuple[float, float]:
    """Порог с максимальным micro-F1 на сетке между lo_hi; возвращает (порог, F1)."""
    lo, hi = lo_hi
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    grid = np.linspace(lo, hi, n_thresholds + 2)[1:-1]
    f1 = threshold_curve(scores_fn, Y, grid)
    k = int(np.argmax(f1))
    return float(grid[k]), float(f1[k])


def per_token_thresholds(scores_cols: Any, Y: sparse.csr_matrix, block: int = COL_BLOCK) -> np.ndarray:
    """Порог на каждый токен, максимизирующий его F1 (правило score > t) по скорам обучающих строк.

    scores_cols(lo, hi) → (N, hi−lo) скоры столбцов lo..hi. Токены без положительных примеров (или с F1 = 0) → +inf.
    F1 при отсечении top-k = 2·tp_k / (k + n_pos), поэтому достаточно отсортировать скоры по убыванию.
    """
    n, v = Y.shape
    thr = np.full(v, np.inf, dtype=np.float32)
    Yc = Y.tocsc()
    k = np.arange(1, n + 1, dtype=np.float32)[:, None]
    for lo in range(0, v, block):
        hi = min(v, lo + block)
        S = np.asarray(scores_cols(lo, hi), dtype=np.float32)
        Yb = np.asarray(Yc[:, lo:hi].toarray() > 0)
        npos = Yb.sum(axis=0).astype(np.float32)
        order = np.argsort(-S, axis=0, kind="stable")
        Ss = np.take_along_axis(S, order, axis=0)
        tp = np.cumsum(np.take_along_axis(Yb, order, axis=0), axis=0, dtype=np.float32)
        f1 = 2.0 * tp / (k + np.maximum(npos, 1.0)[None, :])
        best = np.argmax(f1, axis=0)
        cols = np.arange(hi - lo)
        s_best = Ss[best, cols]
        s_next = np.where(best + 1 < n, Ss[np.minimum(best + 1, n - 1), cols], s_best - 1e-6)
        t = np.where(s_next < s_best, 0.5 * (s_best + s_next), np.nextafter(s_best, -np.inf))
        ok = (npos > 0) & (f1[best, cols] > 0)
        thr[lo:hi] = np.where(ok, t, np.inf)
    return thr


def bow_metrics(pred_fn: Any, Y: sparse.csr_matrix, vocab: Vocab, chunk: int = CHUNK) -> dict[str, Any]:
    """Метрики инверсии: micro P/R/F1 по (функция, токен), recall/precision/F1 по редким токенам (df ≤ rare_df),
    доля функций с полностью восстановленным набором, средний Жаккар. pred_fn(lo, hi) → bool (hi−lo, V)."""
    n = Y.shape[0]
    rare = vocab.rare_mask
    tp = fp = fn = 0.0
    tp_r = fp_r = fn_r = 0.0
    exact = 0
    jacc = 0.0
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        p = pred_fn(lo, hi)
        y = _dense_rows(Y, lo, hi)
        inter = p & y
        tp += float(inter.sum())
        fp += float((p & ~y).sum())
        fn += float((~p & y).sum())
        tp_r += float(inter[:, rare].sum())
        fp_r += float((p & ~y)[:, rare].sum())
        fn_r += float((~p & y)[:, rare].sum())
        union = (p | y).sum(axis=1)
        inter_n = inter.sum(axis=1)
        exact += int(((p == y).all(axis=1) & (y.any(axis=1))).sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            jacc += float(np.where(union > 0, inter_n / np.maximum(union, 1), 1.0).sum())
    out = f1_from_counts(tp, tp + fp, tp + fn)
    rare_m = f1_from_counts(tp_r, tp_r + fp_r, tp_r + fn_r)
    out.update({
        "rare_id_recall": rare_m["recall"] if (tp_r + fn_r) > 0 else None,
        "rare_id_precision": rare_m["precision"] if (tp_r + fp_r) > 0 else None,
        "rare_id_f1": rare_m["f1"] if (tp_r + fn_r) > 0 else None,
        "n_rare_true": int(tp_r + fn_r),
        "exact_set_rate": exact / n if n else 0.0,
        "mean_jaccard": jacc / n if n else 0.0,
        "n_eval": int(n), "n_true_tokens": int(tp + fn), "n_pred_tokens": int(tp + fp),
    })
    return out


def prior_baseline(Y_train: sparse.csr_matrix, Y_val: sparse.csr_matrix, Y_test: sparse.csr_matrix, vocab: Vocab) -> dict[str, Any]:
    """Базовая линия «случайного угадывания»: один и тот же набор самых частых токенов для всех функций
    (порог по априорной частоте подобран по micro-F1 на валидации)."""
    prior = np.asarray(Y_train.mean(axis=0)).ravel().astype(np.float32)
    scores_fn = lambda lo, hi: np.broadcast_to(prior, (hi - lo, prior.size))  # noqa: E731
    thr, _ = best_threshold(scores_fn, Y_val, (0.0, float(prior.max()) if prior.size else 1.0))
    pred = prior > thr
    m = bow_metrics(lambda lo, hi: np.broadcast_to(pred, (hi - lo, pred.size)), Y_test, vocab)
    m["threshold"] = thr
    m["n_pred_per_fn"] = int(pred.sum())
    return m


# ----------------------------------------------------------------------------- модели


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


class _RidgeHead:
    """Гребневая регрессия на бинарные цели в замкнутой форме: W = (XᵀX + λI)⁻¹ Xᵀ Y, с перехватом."""

    def __init__(self, alpha: float = DEFAULT_RIDGE_ALPHA) -> None:
        self.alpha = float(alpha)
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.mu: np.ndarray | None = None
        self.G_inv: np.ndarray | None = None
        self.n: int = 0

    def fit(self, X: np.ndarray, Y: sparse.csr_matrix) -> "_RidgeHead":
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        mu = X.mean(axis=0)
        Xc = X - mu
        ymean = np.asarray(Y.mean(axis=0)).ravel().astype(np.float64)
        G = Xc.T @ Xc + self.alpha * np.eye(d)
        XtY = np.asarray((Y.T @ Xc).T) - np.outer(Xc.sum(axis=0), ymean)  # Xcᵀ (Y − ȳ)
        W = np.linalg.solve(G, XtY)
        self.W = W.astype(np.float32)
        self.b = (ymean - mu @ W).astype(np.float32)
        self.mu, self.G_inv, self.n = mu, np.linalg.inv(G), n
        return self

    def leverage(self, X: np.ndarray) -> np.ndarray:
        """Диагональ hat-матрицы h_ii = 1/n + x̃ᵢᵀ(X̃ᵀX̃ + λI)⁻¹x̃ᵢ для обучающих строк (точный leave-one-out)."""
        assert self.mu is not None and self.G_inv is not None
        Xc = np.asarray(X, dtype=np.float64) - self.mu
        h = 1.0 / self.n + np.einsum("ij,jk,ik->i", Xc, self.G_inv, Xc)
        return np.clip(h, 0.0, 0.99).astype(np.float32)

    def loo_scores_cols(self, X: np.ndarray, Y: sparse.csc_matrix, h: np.ndarray, lo: int, hi: int) -> np.ndarray:
        """Leave-one-out скоры столбцов lo..hi: (f − h·y)/(1 − h) (тождество PRESS для линейного сглаживателя)."""
        S = self.scores_cols(X, lo, hi)
        Yb = np.asarray(Y[:, lo:hi].toarray(), dtype=np.float32)
        return ((S - h[:, None] * Yb) / (1.0 - h[:, None])).astype(np.float32)

    def scores(self, X: np.ndarray) -> np.ndarray:
        assert self.W is not None and self.b is not None
        return (np.asarray(X, dtype=np.float32) @ self.W + self.b).astype(np.float32)

    def scores_cols(self, X: np.ndarray, lo: int, hi: int) -> np.ndarray:
        assert self.W is not None and self.b is not None
        return (np.asarray(X, dtype=np.float32) @ self.W[:, lo:hi] + self.b[lo:hi]).astype(np.float32)

    def state(self) -> dict[str, Any]:
        return {"W": self.W, "b": self.b, "alpha": self.alpha}


class _TorchMLP:
    """MLP d → hidden → V с BCEWithLogits (GPU-часть, torch импортируется лениво; на CPU без torch не исполняется)."""

    def __init__(self, hidden: int = 1024, epochs: int = 10, lr: float = 1e-3, batch: int = 256, dropout: float = 0.1,
                 device: str | None = None, seed: int = 0) -> None:
        self.hidden, self.epochs, self.lr, self.batch, self.dropout, self.seed = int(hidden), int(epochs), float(lr), int(batch), float(dropout), int(seed)
        self.device = device
        self.model: Any = None
        self.v: int = 0

    def _build(self, d: int, v: int) -> Any:
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = nn.Sequential(nn.Linear(d, self.hidden), nn.GELU(), nn.Dropout(self.dropout), nn.Linear(self.hidden, v))
        return model.to(self.device)

    def fit(self, X: np.ndarray, Y: sparse.csr_matrix) -> "_TorchMLP":
        import torch
        from torch import nn

        n, d = X.shape
        v = Y.shape[1]
        self.v = int(v)
        self.model = self._build(d, v)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
        rng = np.random.default_rng(self.seed)
        self.model.train()
        for ep in range(self.epochs):
            order = rng.permutation(n)
            total = 0.0
            for lo in range(0, n, self.batch):
                idx = order[lo : lo + self.batch]
                xb = Xt[torch.as_tensor(idx, dtype=torch.long)].to(self.device)
                yb = torch.as_tensor(Y[idx].toarray(), dtype=torch.float32, device=self.device)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()
                total += float(loss.item()) * len(idx)
            log.info("A1 mlp epoch %d/%d: loss=%.4f", ep + 1, self.epochs, total / max(1, n))
        self.model.eval()
        return self

    def scores(self, X: np.ndarray) -> np.ndarray:
        import torch

        assert self.model is not None
        out = []
        with torch.no_grad():
            for lo in range(0, X.shape[0], 4096):
                xb = torch.as_tensor(np.asarray(X[lo : lo + 4096], dtype=np.float32), device=self.device)
                out.append(torch.sigmoid(self.model(xb)).float().cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.v), dtype=np.float32)

    def state(self) -> dict[str, Any]:
        return {"hidden": self.hidden, "epochs": self.epochs, "lr": self.lr}


class BowInverter:
    """Модель A1: эмбеддинг → скоры наличия V токенов; порог подбирается по micro-F1 на валидации.

    backend: "auto" (torch, если доступен, иначе ridge) | "torch" | "ridge"; decision: "auto" | "per_token" | "global".
    """

    def __init__(self, vocab: Vocab, backend: str = "auto", ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
                 mlp_hidden: int = 1024, mlp_epochs: int = 10, mlp_lr: float = 1e-3, mlp_batch: int = 256,
                 device: str | None = None, seed: int = 0, decision: str = "auto", thr_mem: float = DEFAULT_THR_MEM) -> None:
        self.vocab = vocab
        if decision not in DECISIONS:
            raise ValueError(f"unknown decision rule {decision!r}; known: {DECISIONS}")
        self.decision = decision
        self.val_f1_rules: dict[str, float] = {}
        self.thr_mem = float(thr_mem)
        self.thresholds: np.ndarray | None = None
        self._tr_idx: np.ndarray = np.zeros(0, dtype=np.int64)
        if backend == "auto":
            backend = "torch" if torch_available() else "ridge"
        if backend not in ("torch", "ridge"):
            raise ValueError(f"unknown backend {backend!r}")
        if backend == "torch" and not torch_available():
            raise ImportError("backend='torch' requires torch (requirements-gpu.txt)")
        self.backend = backend
        self.head: Any = _TorchMLP(mlp_hidden, mlp_epochs, mlp_lr, mlp_batch, device=device, seed=seed) if backend == "torch" else _RidgeHead(ridge_alpha)
        self.threshold: float = 0.5
        self.val_f1: float | None = None
        self.n_train: int = 0

    def fit(self, X: np.ndarray, Y: sparse.csr_matrix, val_fraction: float = DEFAULT_VAL_FRACTION, val_max: int = DEFAULT_VAL_MAX,
            rng: random.Random | None = None) -> "BowInverter":
        """Обучение на (X, Y) с отложенной валидацией для порога; возвращает self."""
        n = X.shape[0]
        if n < 4:
            raise ValueError("too few training examples for A1")
        order = as_generator(rng or random.Random(0)).permutation(n)
        n_val = int(min(max(1, round(n * val_fraction)), val_max, n - 2))
        val_idx, tr_idx = np.sort(order[:n_val]), np.sort(order[n_val:])
        self.head.fit(X[tr_idx], Y[tr_idx])
        self.n_train = int(tr_idx.size)
        self._tr_idx = tr_idx
        Xv, Yv = X[val_idx], Y[val_idx]
        sv = self.scores(Xv[: min(len(val_idx), 512)])
        lo_hi = (float(np.percentile(sv, 50)), float(np.percentile(sv, 99.9))) if sv.size else (0.0, 1.0)
        self.threshold, self.val_f1 = best_threshold(lambda lo, hi: self.scores(Xv[lo:hi]), Yv, lo_hi)
        self.thresholds = self._fit_thresholds(X, Y, rng)
        n_fire = int(np.isfinite(self.thresholds).sum())
        self.val_f1_rules = {rule: bow_metrics(lambda lo, hi, r=rule: self.predict(Xv[lo:hi], decision=r), Yv, self.vocab)["f1"]
                             for rule in ("global", "per_token")}
        if self.decision == "auto":
            self.decision = max(self.val_f1_rules, key=lambda r: self.val_f1_rules[r])
        log.info("A1 %s: trained on %d, global threshold=%.4f, per-token thresholds for %d/%d tokens; val F1 %s → rule %s",
                 self.backend, self.n_train, self.threshold, n_fire, len(self.vocab),
                 {k: round(v, 3) for k, v in self.val_f1_rules.items()}, self.decision)
        return self

    def _fit_thresholds(self, X: np.ndarray, Y: sparse.csr_matrix, rng: random.Random | None) -> np.ndarray:
        """Пороги на токен по вневыборочным скорам всех обучающих строк: ridge — точный leave-one-out через рычаги
        (по столбцам, без материализации); MLP — 2-кратный cross-fitting (две дополнительные модели на половинах),
        матрица скоров float16 (при превышении бюджета памяти — случайная подвыборка строк)."""
        n, v = Y.shape
        if self.backend == "ridge":
            h = np.zeros(n, dtype=np.float32)  # строки валидации уже вне выборки: h = 0 → обычные скоры
            h[self._tr_idx] = self.head.leverage(X[self._tr_idx])
            Yc = Y.tocsc()
            return per_token_thresholds(lambda lo, hi: self.head.loo_scores_cols(X, Yc, h, lo, hi), Y)
        g = as_generator(rng or 0)
        rows = np.arange(n)
        max_rows = int(self.thr_mem // max(1, 2 * v))
        if n > max_rows:
            rows = np.sort(g.choice(n, size=max_rows, replace=False))
            log.warning("A1: per-token thresholds on a subsample of %d/%d rows (memory budget)", max_rows, n)
        perm = g.permutation(n)
        folds = (perm[: n // 2], perm[n // 2 :])
        S16 = np.zeros((rows.size, v), dtype=np.float16)
        pos = {int(r): i for i, r in enumerate(rows)}
        for k, hold in enumerate(folds):
            train = folds[1 - k]
            head_k = _TorchMLP(self.head.hidden, self.head.epochs, self.head.lr, self.head.batch, self.head.dropout, self.head.device,
                               self.head.seed + 1 + k).fit(X[train], Y[train])
            sel = np.array([r for r in hold if r in pos], dtype=np.int64)
            for lo in range(0, sel.size, 4096):
                idx = sel[lo : lo + 4096]
                S16[[pos[int(r)] for r in idx]] = head_k.scores(X[idx]).astype(np.float16)
            log.info("A1 mlp cross-fit fold %d/2 done (%d rows scored)", k + 1, sel.size)
        return per_token_thresholds(lambda lo, hi: S16[:, lo:hi].astype(np.float32), Y[rows])

    def scores(self, X: np.ndarray) -> np.ndarray:
        """Скоры (M, V) float32 (ridge — линейные, torch — сигмоида)."""
        return self.head.scores(X)

    def predict(self, X: np.ndarray, decision: str | None = None) -> np.ndarray:
        """Булева матрица предсказанных токенов (M, V) по правилу decision (по умолчанию — правило модели)."""
        rule = decision or self.decision
        s = self.scores(X)
        if rule == "per_token" and self.thresholds is not None:
            return s > self.thresholds[None, :]
        return s > self.threshold

    def evaluate(self, X: np.ndarray, Y: sparse.csr_matrix) -> dict[str, Any]:
        """Метрики на (X, Y): f1/precision/recall по правилу модели (decision), f1_global и f1_per_token — по обоим;
        rare_id_recall/precision/f1 — по правилу per_token (глобальный порог редкие токены не предсказывает)."""
        rules = {rule: bow_metrics(lambda lo, hi, r=rule: self.predict(X[lo:hi], decision=r), Y, self.vocab) for rule in ("global", "per_token")}
        chosen = self.decision if self.decision in rules else "global"
        m = dict(rules[chosen])
        pt = rules["per_token"]
        m.update({
            "f1_global": rules["global"]["f1"], "f1_per_token": pt["f1"],
            "precision_global": rules["global"]["precision"], "recall_global": rules["global"]["recall"],
            "precision_per_token": pt["precision"], "recall_per_token": pt["recall"],
            "rare_id_recall": pt["rare_id_recall"], "rare_id_precision": pt["rare_id_precision"], "rare_id_f1": pt["rare_id_f1"],
            "rare_id_rule": "per_token", f"rare_id_recall_{chosen}": rules[chosen]["rare_id_recall"],
            "decision": chosen, "threshold": float(self.threshold), "backend": self.backend,
        })
        return m


# ----------------------------------------------------------------------------- высокоуровневый прогон


def a1_params(cfg: dict[str, Any]) -> dict[str, Any]:
    """Параметры A1 из cfg.privacy (с умолчаниями)."""
    p = cfg.get("privacy", {}) or {}
    return {
        "vocab": int(p.get("bow_vocab", DEFAULT_VOCAB)), "min_df": int(p.get("bow_min_df", DEFAULT_MIN_DF)),
        "rare_df": int(p.get("bow_rare_df", DEFAULT_RARE_DF)), "rare_fraction": float(p.get("bow_rare_fraction", DEFAULT_RARE_FRACTION)),
        "backend": str(p.get("a1_backend", "auto")), "ridge_alpha": float(p.get("a1_ridge_alpha", DEFAULT_RIDGE_ALPHA)),
        "mlp_hidden": int(p.get("a1_hidden", 1024)), "mlp_epochs": int(p.get("a1_epochs", 10)), "mlp_lr": float(p.get("a1_lr", 1e-3)),
        "mlp_batch": int(p.get("a1_batch", 256)), "decision": str(p.get("a1_decision", "auto")),
        "thr_mem": float(p.get("a1_thr_mem", DEFAULT_THR_MEM)),
    }


def train_bow_attack(train_emb: np.ndarray, train_bags: Sequence[set[str]], cfg: dict[str, Any], rng: random.Random | None = None,
                     backend: str | None = None, vocab: Vocab | None = None) -> tuple[BowInverter, Vocab, sparse.csr_matrix]:
    """Строит словарь (если не задан), цели и обучает BowInverter на plain-эмбеддингах public_train.
    Возвращает (модель, словарь, Y_train)."""
    p = a1_params(cfg)
    rng = rng or random.Random(int(cfg.get("seed", 0)))
    if vocab is None:
        vocab = build_vocab(train_bags, p["vocab"], p["min_df"], p["rare_df"], p["rare_fraction"], rng=random.Random(rng.random()))
    Y = targets(train_bags, vocab)
    model = BowInverter(vocab, backend or p["backend"], ridge_alpha=p["ridge_alpha"], mlp_hidden=p["mlp_hidden"],
                        mlp_epochs=p["mlp_epochs"], mlp_lr=p["mlp_lr"], mlp_batch=p["mlp_batch"], seed=int(cfg.get("seed", 0)),
                        decision=p["decision"], thr_mem=p["thr_mem"])
    model.fit(np.asarray(train_emb, dtype=np.float32), Y, rng=random.Random(rng.random()))
    return model, vocab, Y


def run_attack_bow(train_emb: np.ndarray, train_codes: Sequence[str], train_langs: Sequence[str], eval_emb: np.ndarray,
                   eval_codes: Sequence[str], eval_langs: Sequence[str], cfg: dict[str, Any], rng: random.Random | None = None,
                   backend: str | None = None) -> tuple[dict[str, Any], BowInverter, Vocab]:
    """A1 целиком: словарь и обучение на public_train, оценка на protected (plain-эмбеддинги).
    Возвращает (метрики с базовой линией, модель, словарь)."""
    train_bags = identifier_bags(train_codes, train_langs, show_progress=True)
    eval_bags = identifier_bags(eval_codes, eval_langs, show_progress=True)
    model, vocab, Y_train = train_bow_attack(train_emb, train_bags, cfg, rng=rng, backend=backend)
    Y_eval = targets(eval_bags, vocab)
    metrics = model.evaluate(np.asarray(eval_emb, dtype=np.float32), Y_eval)
    n_val = min(Y_train.shape[0] // 10 + 1, DEFAULT_VAL_MAX)
    metrics["prior_baseline"] = prior_baseline(Y_train[n_val:], Y_train[:n_val], Y_eval, vocab)
    metrics["vocab_size"] = len(vocab)
    metrics["n_rare_vocab"] = vocab.n_rare
    return metrics, model, vocab
