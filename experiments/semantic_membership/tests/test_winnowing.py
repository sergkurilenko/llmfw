"""Тесты ядра winnowing (smcode.fingerprint.winnowing): гарантия обнаружения, плотность, HMAC, устойчивость."""

import random

import pytest

from smcode.fingerprint.winnowing import (
    fingerprint_set,
    kgram_hashes,
    longest_common_run,
    overlap,
    winnow,
    winnow_fingerprints,
)


def _rand_tokens(rng, n, alphabet="abcdefgh"):
    return [rng.choice(alphabet) for _ in range(n)]


@pytest.mark.parametrize("k,w", [(5, 4), (25, 25), (10, 50)])
def test_guarantee_shared_substring(k, w):
    """Общая подстрока длиной ≥ w + k − 1 обязана дать общий отпечаток (для 50 случайных пар)."""
    rng = random.Random(1)
    for _ in range(50):
        base = _rand_tokens(rng, 400)
        start = rng.randrange(0, 400 - (w + k - 1))
        sub = base[start : start + w + k - 1]
        other = _rand_tokens(rng, 60, "xyz") + sub + _rand_tokens(rng, 60, "xyz")
        assert fingerprint_set(base, k, w) & fingerprint_set(other, k, w)


def test_density_close_to_theory():
    rng = random.Random(0)
    toks = _rand_tokens(rng, 5000, "abcdefghijklmnop")
    k, w = 5, 4
    fps = winnow_fingerprints(toks, k, w)
    density = len(fps) / (len(toks) - k + 1)
    assert abs(density - 2 / (w + 1)) < 0.05


def test_short_inputs():
    assert kgram_hashes(["a", "b"], 5) == []
    assert winnow([], 4) == []
    assert winnow([7], 4) == [(7, 0)]
    assert winnow([5, 3, 4], 10) == [(3, 1)]
    assert winnow_fingerprints(["a"] * 3, 5, 4) == []


def test_rightmost_minimum_on_ties():
    # все хэши равны: выбирается самый правый в окне, записывается один раз на окно
    fps = winnow([1, 1, 1, 1, 1, 1], 3)
    positions = [p for _, p in fps]
    assert positions == sorted(set(positions)) and positions[0] == 2


def test_hmac_key_changes_hashes_but_not_overlap():
    rng = random.Random(2)
    toks = _rand_tokens(rng, 300)
    a = fingerprint_set(toks, 5, 4, key=b"k1")
    b = fingerprint_set(toks, 5, 4, key=b"k2")
    c = fingerprint_set(toks, 5, 4, key=None)
    assert a != b and a != c
    assert overlap(a, fingerprint_set(toks, 5, 4, key=b"k1")) == 1.0


def test_overlap_and_longest_run():
    rng = random.Random(3)
    toks = _rand_tokens(rng, 300)
    fps = winnow_fingerprints(toks, 5, 4)
    assert overlap([h for h, _ in fps], [h for h, _ in fps]) == 1.0
    assert overlap([], [1, 2]) == 0.0
    assert longest_common_run(fps, fps) == len(fps)
    # половина последовательности → цепочка не длиннее числа отпечатков половины
    half = winnow_fingerprints(toks[:150], 5, 4)
    run = longest_common_run(half, fps)
    assert 0 < run <= len(half)
    unrelated = winnow_fingerprints(_rand_tokens(rng, 300, "xyz"), 5, 4)
    assert longest_common_run(unrelated, fps) == 0


def test_dead_code_insertion_reduces_but_rarely_zeroes_overlap():
    """Вставка мусора каждые 3 токена ломает k-граммы (k=5) — проверка, что метрика это отражает."""
    rng = random.Random(4)
    toks = _rand_tokens(rng, 300)
    noisy = []
    for i, t in enumerate(toks):
        noisy.append(t)
        if i % 3 == 2:
            noisy.append("JUNK")
    o = overlap(fingerprint_set(noisy, 5, 4), fingerprint_set(toks, 5, 4))
    assert o < 0.2
