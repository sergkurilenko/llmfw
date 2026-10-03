"""Калибровка порога (DESIGN.md §8): split conformal по негативам, ДИ Клоппера–Пирсона, доменная калибровка.

Порог: τ_α = ⌈(n+1)(1−α)⌉-я порядковая статистика скоров калибровочных негативов S = {s_1..s_n}.
Гарантия (при обменяемости нового негатива с S): P(score_new > τ_α) ≤ α.

Правило решения во всём модуле оценки — строгое неравенство ``score > τ`` (``decide``): для непрерывных
скоров оно совпадает с ``score ≥ τ`` из DESIGN.md §1, а при массовых совпадениях скоров (например, нулевые
скоры отпечатков у большинства негативов) только строгое правило сохраняет конечновыборочную границу
FPR ≤ α без искусственного сдвига порога.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

log = logging.getLogger(__name__)

INF = float("inf")


# ----------------------------------------------------------------------------- порог


def _as_scores(scores: Iterable[float] | np.ndarray) -> np.ndarray:
    """float64-массив без NaN (NaN отбрасываются с предупреждением)."""
    s = np.asarray(list(scores) if not isinstance(scores, np.ndarray) else scores, dtype=np.float64).ravel()
    bad = np.isnan(s)
    if bad.any():
        log.warning("calibration: %d NaN-скоров отброшено", int(bad.sum()))
        s = s[~bad]
    return s


def conformal_level(n: int, alpha: float) -> int:
    """Ранг k = ⌈(n+1)(1−α)⌉ порядковой статистики (с защитой от ошибок округления); k > n → порог +inf."""
    if n <= 0:
        return 1
    return max(1, int(math.ceil((n + 1) * (1.0 - alpha) - 1e-9)))


def conformal_threshold(neg_scores: Iterable[float] | np.ndarray, alpha: float) -> float:
    """Split-conformal порог τ_α по скорам негативов: k-я порядковая статистика, k = ⌈(n+1)(1−α)⌉.

    Возвращает +inf, если k > n (слишком мало калибровочных негативов для уровня α: нужно n ≥ 1/α − 1).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    s = _as_scores(neg_scores)
    n = s.size
    if n == 0:
        return INF
    k = conformal_level(n, alpha)
    if k > n:
        return INF
    return float(np.partition(s, k - 1)[k - 1])


def min_calibration_size(alpha: float) -> int:
    """Минимальное n, при котором τ_α конечен: ⌈(n+1)(1−α)⌉ ≤ n ⇔ n ≥ (1−α)/α."""
    return int(math.ceil((1.0 - alpha) / alpha - 1e-9))


def decide(scores: Iterable[float] | np.ndarray, tau: float) -> np.ndarray:
    """Булев вектор решений «член»: score > τ (строгое неравенство, см. докстринг модуля)."""
    s = np.asarray(scores, dtype=np.float64)
    if tau == INF:
        return np.zeros(s.shape, dtype=bool)
    return s > tau


def threshold_at_fpr(neg_scores: Iterable[float] | np.ndarray, alpha: float) -> float:
    """«Оракульный» порог: минимальное τ из значений скоров, при котором доля негативов с score > τ ≤ α.

    Используется для верхней границы достижимого TPR на public_test (порог подсмотрен на тесте).
    """
    s = np.sort(_as_scores(neg_scores))
    n = s.size
    if n == 0:
        return INF
    u = np.unique(s)
    n_greater = n - np.searchsorted(s, u, side="right")  # число скоров строго больше u_j
    ok = n_greater / n <= alpha + 1e-12
    j = int(np.argmax(ok)) if ok.any() else u.size - 1
    return float(u[j])


def rate(scores: Iterable[float] | np.ndarray, tau: float) -> float:
    """Доля скоров, превышающих порог (TPR для членов, FPR для негативов); 0 для пустого входа."""
    s = _as_scores(scores)
    return float(decide(s, tau).mean()) if s.size else 0.0


# ----------------------------------------------------------------------------- доверительные интервалы


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Точный интервал Клоппера–Пирсона для биномиальной доли k/n (через квантили бета-распределения)."""
    from scipy.stats import beta

    if n <= 0:
        return 0.0, 1.0
    if not 0 <= k <= n:
        raise ValueError(f"k must be in [0, n]; got k={k}, n={n}")
    a = 1.0 - conf
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - a / 2.0, k + 1, n - k))
    return lo, hi


def rate_with_ci(scores: Iterable[float] | np.ndarray, tau: float, conf: float = 0.95) -> dict[str, Any]:
    """{'value', 'ci': [lo, hi], 'n', 'k'} — доля score > τ с интервалом Клоппера–Пирсона."""
    s = _as_scores(scores)
    n = int(s.size)
    k = int(decide(s, tau).sum()) if n else 0
    lo, hi = clopper_pearson(k, n, conf) if n else (0.0, 1.0)
    return {"value": (k / n) if n else None, "ci": [lo, hi], "n": n, "k": k}


# ----------------------------------------------------------------------------- доменная калибровка


@dataclass
class DomainCalibration:
    """Результат доменной калибровки: порог и разбиение репозиториев hard_neg."""

    tau: float
    alpha: float
    calib_repos: list[str]
    test_repos: list[str]
    n_calib: int
    test_mask: np.ndarray = field(repr=False)  # маска «тестовых» запросов hard_neg (bool, по входному порядку)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tau": self.tau, "alpha": self.alpha, "n_calib": self.n_calib,
            "calib_repos": list(self.calib_repos), "test_repos": list(self.test_repos),
            "n_test_queries": int(self.test_mask.sum()),
        }


def split_repos(repos: Iterable[str], fraction: float, rng: random.Random) -> tuple[list[str], list[str]]:
    """Детерминированное разбиение множества репозиториев: ⌈fraction·R⌉ в калибровку, остальные — в тест
    (при R ≥ 2 обе части непусты)."""
    uniq = sorted(set(repos))
    r = len(uniq)
    if r == 0:
        return [], []
    order = list(uniq)
    rng.shuffle(order)
    n_cal = int(math.ceil(fraction * r))
    if r >= 2:
        n_cal = min(max(n_cal, 1), r - 1)
    return sorted(order[:n_cal]), sorted(order[n_cal:])


def domain_calibration(
    calib_scores: Iterable[float] | np.ndarray,
    hard_scores: Iterable[float] | np.ndarray,
    hard_repos: Sequence[str],
    alpha: float,
    fraction: float = 0.5,
    rng: random.Random | None = None,
    calib_repos: Iterable[str] | None = None,
) -> DomainCalibration:
    """Калибровка «по домену» (DESIGN.md §8): часть репозиториев hard_neg добавляется к калибровочным
    негативам (public_calib), порог τ_α считается по объединению, остальные репозитории hard_neg — тест.

    calib_repos — заранее заданное множество калибровочных репозиториев (для согласованности между методами);
    иначе разбиение делается split_repos(fraction, rng).
    """
    hard = _as_scores(hard_scores)
    repos = np.asarray(list(hard_repos), dtype=object)
    if hard.size != repos.size:
        raise ValueError("hard_scores and hard_repos must have the same length")
    if calib_repos is None:
        cal, test = split_repos(repos.tolist(), fraction, rng or random.Random(0))
    else:
        cal = sorted(set(calib_repos))
        test = sorted(set(repos.tolist()) - set(cal))
    cal_set = set(cal)
    in_cal = np.fromiter((r in cal_set for r in repos), dtype=bool, count=repos.size)
    pooled = np.concatenate([_as_scores(calib_scores), hard[in_cal]])
    tau = conformal_threshold(pooled, alpha)
    return DomainCalibration(tau=tau, alpha=alpha, calib_repos=cal, test_repos=test, n_calib=int(pooled.size),
                             test_mask=~in_cal)
