"""Ядро алгоритма winnowing (Schleimer, Wilkinson, Aiken, SIGMOD 2003) над абстрактными токенами.

Гарантия: любая общая подстрока длиной ≥ w + k − 1 токенов даёт ≥ 1 общий отпечаток;
подстроки короче k не обнаруживаются. Хэш k-граммы — HMAC-SHA256(key, k-грамма),
усечённый до 64 бит (без ключа — SHA-256, только для тестов). См. DESIGN.md §6 (M1).
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Iterable, Sequence

Fingerprint = tuple[int, int]  # (hash64, позиция первого токена k-граммы)


def kgram_hash(tokens: Sequence[str], key: bytes | None = None) -> int:
    data = "\x1f".join(tokens).encode("utf-8", "replace")
    digest = hmac.new(key, data, hashlib.sha256).digest() if key else hashlib.sha256(data).digest()
    return int.from_bytes(digest[:8], "big")


def kgram_hashes(tokens: Sequence[str], k: int, key: bytes | None = None) -> list[int]:
    """Хэши всех k-грамм по порядку; пустой список, если токенов < k."""
    n = len(tokens)
    if n < k:
        return []
    return [kgram_hash(tokens[i : i + k], key) for i in range(n - k + 1)]


def winnow(hashes: Sequence[int], w: int) -> list[Fingerprint]:
    """Выбор отпечатков: минимум в каждом окне из w хэшей (при равенстве — самый правый),
    без повторной записи одного и того же выбранного элемента."""
    if not hashes:
        return []
    if w <= 1 or len(hashes) <= w:
        if len(hashes) <= w:
            # окно шире последовательности: один минимум на всю последовательность
            m = min(range(len(hashes)), key=lambda i: (hashes[i], -i))
            return [(hashes[m], m)]
    out: list[Fingerprint] = []
    last_sel = -1
    for start in range(0, len(hashes) - w + 1):
        end = start + w
        # самый правый минимум в окне [start, end)
        m = end - 1
        for i in range(end - 2, start - 1, -1):
            if hashes[i] < hashes[m]:
                m = i
        if m != last_sel:
            out.append((hashes[m], m))
            last_sel = m
    return out


def winnow_fingerprints(tokens: Sequence[str], k: int, w: int, key: bytes | None = None) -> list[Fingerprint]:
    """Полный конвейер: k-граммы → хэши → winnowing. Токены — abstract_tokens(mode='full')."""
    return winnow(kgram_hashes(tokens, k, key), w)


def fingerprint_set(tokens: Sequence[str], k: int, w: int, key: bytes | None = None) -> set[int]:
    return {h for h, _ in winnow_fingerprints(tokens, k, w, key)}


def overlap(query_fps: Iterable[int], cand_fps: Iterable[int]) -> float:
    """Доля отпечатков запроса, присутствующих у кандидата (0..1). Для пустого запроса — 0."""
    q = set(query_fps)
    if not q:
        return 0.0
    return len(q & set(cand_fps)) / len(q)


def longest_common_run(q_fps: Sequence[Fingerprint], c_fps: Sequence[Fingerprint]) -> int:
    """Длина (в отпечатках) самой длинной цепочки общих отпечатков, идущих подряд в обоих списках."""
    if not q_fps or not c_fps:
        return 0
    c_pos: dict[int, list[int]] = {}
    for j, (h, _) in enumerate(c_fps):
        c_pos.setdefault(h, []).append(j)
    best = 0
    prev: dict[int, int] = {}  # j → длина цепочки, заканчивающейся на j
    for h, _ in q_fps:
        cur: dict[int, int] = {}
        for j in c_pos.get(h, ()):
            cur[j] = prev.get(j - 1, 0) + 1
            best = max(best, cur[j])
        prev = cur
    return best
