"""Логистический комбинатор признаков кандидата для гибрида M4 (DESIGN.md §6).

Признаки кандидата: [cos, overlap, lcr_norm, log_n_tokens] —
косинус эмбеддингов, доля winnowing-отпечатков запроса у кандидата, длина самой длинной общей
цепочки отпечатков / число отпечатков запроса, log(1 + число токенов запроса).
Обучение: sklearn LogisticRegression (pickle). Абляция без обучения — правило двух порогов
(two_threshold): score = max(cos_rescaled, overlap_rescaled), где каждая компонента проходит 0.5
ровно на своём пороге (t_cos, t_overlap), т.е. решение «score ≥ 0.5» ⇔ «cos ≥ t_cos ∨ overlap ≥ t_ov».
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np

log = logging.getLogger(__name__)

FEATURE_NAMES: tuple[str, ...] = ("cos", "overlap", "lcr_norm", "log_n_tokens")
N_FEATURES = len(FEATURE_NAMES)
RULES: tuple[str, ...] = ("logistic", "two_threshold")
DEFAULT_THRESHOLDS: dict[str, float] = {"cos": 0.8, "overlap": 0.2}
DEFAULT_C = 1.0


def make_features(cos: float, overlap: float, lcr_norm: float, n_tokens: int | float) -> np.ndarray:
    """Вектор признаков одного кандидата (float32, N_FEATURES)."""
    return np.array([float(cos), float(overlap), float(lcr_norm), float(np.log1p(max(0.0, float(n_tokens))))], dtype=np.float32)


def _rescale(x: np.ndarray, t: float) -> np.ndarray:
    """Монотонное отображение [0, 1] → [0, 1], равное 0.5 в точке t."""
    t = float(min(max(t, 1e-6), 1 - 1e-6))
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return np.where(x < t, 0.5 * x / t, 0.5 + 0.5 * (x - t) / (1.0 - t))


def two_threshold_score(X: np.ndarray, thresholds: dict[str, float] | None = None) -> np.ndarray:
    """Правило двух порогов по матрице признаков (n, N_FEATURES) → score (n,) ∈ [0, 1]."""
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    X = np.asarray(X, dtype=np.float32).reshape(-1, N_FEATURES)
    return np.maximum(_rescale(X[:, 0], th["cos"]), _rescale(X[:, 1], th["overlap"])).astype(np.float32)


class Combiner:
    """Комбинатор признаков → вероятность членства кандидата. rule ∈ {logistic, two_threshold}."""

    def __init__(self, rule: str = "logistic", C: float = DEFAULT_C, thresholds: dict[str, float] | None = None,
                 class_weight: str | dict | None = "balanced") -> None:
        if rule not in RULES:
            raise ValueError(f"unknown rule {rule!r}; known: {RULES}")
        self.rule = rule
        self.C = float(C)
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.class_weight = class_weight
        self.model: Any = None
        self.n_train = 0
        self.train_stats: dict[str, Any] = {}

    @property
    def fitted(self) -> bool:
        return self.model is not None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "Combiner":
        """Обучает логистическую регрессию (y ∈ {0, 1}); для two_threshold только запоминает статистику."""
        X = np.asarray(X, dtype=np.float32).reshape(-1, N_FEATURES)
        y = np.asarray(y).astype(int).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y length mismatch")
        self.n_train = int(X.shape[0])
        self.train_stats = {"n": self.n_train, "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}
        if self.rule != "logistic":
            return self
        if len(np.unique(y)) < 2:
            raise ValueError("need both classes to fit the logistic combiner")
        from sklearn.linear_model import LogisticRegression

        self.model = LogisticRegression(C=self.C, max_iter=2000, class_weight=self.class_weight)
        self.model.fit(X, y, sample_weight=sample_weight)
        self.train_stats["coef"] = dict(zip(FEATURE_NAMES, [float(c) for c in self.model.coef_[0]]))
        self.train_stats["intercept"] = float(self.model.intercept_[0])
        try:
            from sklearn.metrics import roc_auc_score

            self.train_stats["train_auc"] = float(roc_auc_score(y, self.model.predict_proba(X)[:, 1]))
        except Exception:  # noqa: BLE001
            pass
        log.info("combiner fitted on %d rows (%d pos): %s", self.n_train, self.train_stats["n_pos"], self.train_stats.get("coef"))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Вероятность класса 1 (n,) по правилу комбинатора; необученный logistic → two_threshold."""
        X = np.asarray(X, dtype=np.float32).reshape(-1, N_FEATURES)
        if X.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        if self.rule == "logistic" and self.model is not None:
            return self.model.predict_proba(X)[:, 1].astype(np.float32)
        if self.rule == "logistic":
            log.warning("combiner not fitted; falling back to two_threshold rule")
        return two_threshold_score(X, self.thresholds)

    score = predict_proba

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Combiner":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"{path}: not a Combiner")
        return obj

    def describe(self) -> dict[str, Any]:
        return {"rule": self.rule, "C": self.C, "thresholds": dict(self.thresholds), "fitted": self.fitted,
                "n_train": self.n_train, **{k: v for k, v in self.train_stats.items() if k != "n"}}


def candidate_labels(query_labels: Sequence[int], query_source_ids: Sequence[str | None], cand_ids: Sequence[Sequence[str | None]]) -> np.ndarray:
    """Метки кандидатов: 1 ⇔ запрос — член и кандидат — его исходная функция; иначе 0. Форма (M, k)."""
    out = []
    for lab, src, cands in zip(query_labels, query_source_ids, cand_ids):
        out.append([1 if (int(lab) == 1 and c is not None and c == src) else 0 for c in cands])
    return np.asarray(out, dtype=np.int64) if out else np.zeros((0, 0), dtype=np.int64)
