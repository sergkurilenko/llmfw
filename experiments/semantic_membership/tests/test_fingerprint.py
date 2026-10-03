"""Тесты модуля fingerprint: реестр, M0 exact, M1 winnowing, M2 minhash, скрипт 04 (синтетика, CPU)."""

from __future__ import annotations

import importlib.util
import json
import random
import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from smcode.config import load_config
from smcode.fingerprint.exact import ExactIndex, normalized_lines, window_hashes
from smcode.fingerprint.index import (
    REGISTRY,
    BaseIndex,
    InvertedIndex,
    build_index,
    deep_sizeof,
    load_index,
    make_index,
    score_hashes,
)
from smcode.fingerprint.minhash import MinHashIndex, jaccard, shingle_set
from smcode.fingerprint.winnowing_index import WinnowingIndex
from smcode.types import FunctionRecord, MembershipIndex, QueryResult, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
PY = "python"

# ----------------------------------------------------------------------------- синтетика

_OPS = ["+", "-", "*", "//", "%", "<<", ">>", "&", "|", "^"]
_FUNCS = ["len", "abs", "min", "max", "sum", "int", "str", "round", "sorted", "list"]


def _expr(rng: random.Random, ids: list[str], depth: int = 0) -> str:
    r = rng.random()
    if depth > 2 or r < 0.3:
        return rng.choice(ids) if rng.random() < 0.7 else str(rng.randint(0, 999))
    if r < 0.55:
        return f"({_expr(rng, ids, depth + 1)} {rng.choice(_OPS)} {_expr(rng, ids, depth + 1)})"
    if r < 0.75:
        return f"{rng.choice(_FUNCS)}({_expr(rng, ids, depth + 1)})"
    return f"{_expr(rng, ids, depth + 1)} {rng.choice(_OPS)} {_expr(rng, ids, depth + 1)}"


def make_fn(rng: random.Random, i: int, n_stmts: int = 12) -> str:
    """Случайная Python-функция с разнообразной структурой выражений (≈ 200–300 токенов)."""
    ids = [f"a{i}_{j}" for j in range(5)]
    lines = [f"def fn_{i}({', '.join(ids[:3])}):"]
    for _ in range(n_stmts):
        t = rng.random()
        v = rng.choice(ids)
        if t < 0.5:
            lines.append(f"    {v} = {_expr(rng, ids)}")
        elif t < 0.65:
            lines.append(f"    if {_expr(rng, ids)} > {_expr(rng, ids)}:")
            lines.append(f"        {v} = {_expr(rng, ids)}")
        elif t < 0.8:
            lines.append(f"    for k{i} in range({_expr(rng, ids)}):")
            lines.append(f"        {v} += {_expr(rng, ids)}")
        else:
            lines.append(f"    {v} = [{_expr(rng, ids)} for q{i} in {rng.choice(ids)}]")
    lines.append(f"    return {_expr(rng, ids)}")
    return "\n".join(lines) + "\n"


def rename(code: str) -> str:
    """Согласованное переименование идентификаторов простым regex."""
    return re.sub(r"\b([a-z]+\d+_\d+|k\d+|q\d+|fn_\d+)\b", lambda m: "zz_" + m.group(1), code)


def partial(code: str, lo: int = 2, hi: int = 8) -> str:
    """Фрагмент из подряд идущих строк тела (≥ 30 токенов, ≥ 5 строк)."""
    return "\n".join(code.splitlines()[lo:hi]) + "\n"


def make_records(n: int = 40, seed: int = 0, split: str = "protected", prefix: str = "r") -> list[FunctionRecord]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        code = make_fn(rng, i + seed * 1000)
        out.append(
            FunctionRecord(
                id=f"{split}/{prefix}/f{i}.py:1-{code.count(chr(10))}", split=split, repo=prefix, path=f"src/f{i // 5}.py",
                lang=PY, code=code, start_line=1, end_line=code.count("\n"), n_lines=code.count("\n"), n_tokens=0, sha="",
            )
        )
    return out


@pytest.fixture()
def cfg(tmp_path: Path) -> dict:
    return load_config(
        overrides={
            "paths": {
                "raw_repos": str(tmp_path / "raw"), "functions": str(tmp_path / "functions"), "queries": str(tmp_path / "queries"),
                "indexes": str(tmp_path / "indexes"), "results": str(tmp_path / "results"), "figures": str(tmp_path / "figures"),
            },
            "fingerprint": {"k": 8, "w": 6, "common_df": 50, "exact_window": 5, "minhash_perm": 128, "minhash_shingle": 5, "workers": 1},
        }
    )


@pytest.fixture()
def records() -> list[FunctionRecord]:
    return make_records(40)


@pytest.fixture()
def nonmember() -> str:
    return make_fn(random.Random(4242), 7777)


# ----------------------------------------------------------------------------- index.py


def test_registry_and_make_index(cfg):
    assert set(REGISTRY) == {"exact", "winnowing", "minhash", "semantic", "hybrid"}
    for name, cls in [("exact", ExactIndex), ("winnowing", WinnowingIndex), ("minhash", MinHashIndex)]:
        idx = make_index(name, cfg)
        assert isinstance(idx, cls) and idx.name == name
        assert isinstance(idx, MembershipIndex)
    with pytest.raises(KeyError):
        make_index("nope", cfg)


def test_inverted_index_roundtrip_and_without():
    hashes = np.array([5, 3, 5, 9, 3, 5, 2**64 - 1], dtype=np.uint64)
    ids = np.array([1, 0, 1, 2, 2, 0, 3], dtype=np.int32)
    inv = InvertedIndex.from_pairs(hashes, ids)
    assert inv.keys.tolist() == [3, 5, 9, 2**64 - 1]
    assert inv.postings_for(0).tolist() == [0, 2]
    assert inv.postings_for(1).tolist() == [0, 1]  # дубликат пары (5,1) схлопнут
    assert inv.df().tolist() == [2, 2, 1, 1]
    assert inv.find(np.array([5, 4, 9], dtype=np.uint64)).tolist() == [1, -1, 2]
    found, post = inv.gather(np.array([3, 4, 9], dtype=np.uint64))
    assert found.tolist() == [True, False, True] and post.tolist() == [0, 2, 2]
    sub = inv.without(np.array([5], dtype=np.uint64))
    assert sub.keys.tolist() == [3, 9, 2**64 - 1] and sub.postings.tolist() == [0, 2, 2, 3]
    back = InvertedIndex.from_arrays(sub.arrays())
    assert np.array_equal(back.keys, sub.keys) and np.array_equal(back.postings, sub.postings)
    score, best, d = score_hashes(inv, np.array([3, 5, 7], dtype=np.uint64))
    assert score == pytest.approx(2 / 3) and best == 0 and d["best_shared"] == 2 and d["n_candidates"] == 3
    assert score_hashes(InvertedIndex(), np.array([1], dtype=np.uint64))[0] == 0.0


def test_base_query_batch_latency():
    class Const(BaseIndex):
        name = "const"

        def query(self, code, lang):
            return QueryResult(0.5, "x")

    res = Const({}).query_batch([("a", PY), ("b", PY)])
    assert [r.score for r in res] == [0.5, 0.5]
    assert all(r.latency_ms is not None and r.latency_ms >= 0.0 for r in res)


def test_deep_sizeof():
    assert deep_sizeof(np.zeros(100, dtype=np.uint64)) >= 800
    assert deep_sizeof(["abc"] * 3) > deep_sizeof([])


# ----------------------------------------------------------------------------- M0 exact


def test_exact_lines_and_windows():
    code = "def f(x):\n    # comment\n    y = x + 1  # trailing\n\n    return y\n"
    assert normalized_lines(code, PY) == ["def f ( x ) :", "y = x + 1", "return y"]
    assert len(window_hashes(code, PY, window=5)) == 1  # короче окна — одно окно
    assert len(window_hashes("def f():\n" + "    x = 1\n" * 9, PY, window=5)) == 6
    assert window_hashes("", PY) == []
    assert window_hashes(code, PY, key=b"k1") != window_hashes(code, PY, key=b"k2")


def test_exact_scores(cfg, records, nonmember):
    idx = ExactIndex(cfg)
    idx.build(records, cfg)
    q = records[3].code
    assert idx.query(q, PY).score == 1.0 and idx.query(q, PY).best_id == records[3].id
    # переформатирование отступов/пробелов и комментарии не мешают
    reformatted = q.replace("    ", "\t").replace(" = ", "=") + "# tail\n"
    assert idx.query(reformatted, PY).score == 1.0
    r = idx.query(partial(q), PY)
    assert r.score == 1.0 and r.best_id == records[3].id
    assert idx.query(rename(q), PY).score == 0.0  # лексический метод: переименование рушит окна
    assert idx.query(nonmember, PY).score == 0.0
    r = idx.query("", PY)
    assert r.score == 0.0 and r.details["reason"] == "too_short"
    assert idx.memory_bytes() > 0 and idx.meta["n_keys"] > 0


# ----------------------------------------------------------------------------- M1 winnowing


def test_winnowing_member_queries(cfg, records, nonmember):
    idx = WinnowingIndex(cfg)
    idx.build(records, cfg)
    q = records[5].code
    for variant in (q, rename(q), partial(q)):
        r = idx.query(variant, PY)
        assert r.score >= 0.95, (variant, r)
        assert r.best_id == records[5].id
        assert r.details["longest_common_run"] >= 3 and r.details["n_fps"] > 0
    r = idx.query(nonmember, PY)
    assert r.score < 0.2, r
    # слишком короткий запрос
    r = idx.query("x = 1\n", PY)
    assert r.score == 0.0 and r.best_id is None and r.details["reason"] == "too_short"
    # пакетный запрос заполняет латентность
    batch = idx.query_batch([(q, PY), (nonmember, PY)])
    assert all(b.latency_ms is not None and b.latency_ms > 0 for b in batch)
    assert batch[0].score >= 0.95 > batch[1].score
    # вспомогательные методы для гибрида
    qh = idx.query_hashes(q, PY)
    assert qh.size > 0 and idx.overlap(qh, idx.idx_of(records[5].id)) == 1.0
    assert idx.overlap(qh, idx.idx_of(records[6].id)) < 0.2


def test_winnowing_common_filter(cfg, records):
    """Отпечатки, частые в public_train, исключаются из индекса и из запроса."""
    shared = make_fn(random.Random(77), 999)  # «общий» код, который есть и в protected, и в 60 public-функциях
    prot = records[:10] + [FunctionRecord(**{**records[0].to_dict(), "id": "protected/r/shared.py:1-5", "path": "src/shared.py", "code": shared})]
    public = [FunctionRecord(**{**r.to_dict(), "id": f"public_train/p/{i}", "split": "public_train", "repo": "p", "code": shared}) for i, r in enumerate(records[:60] * 2)]
    plain = WinnowingIndex(cfg)
    plain.build(prot, cfg)
    assert plain.query(shared, PY).score == 1.0
    filt = WinnowingIndex(cfg)
    n_common = filt.fit_common_filter(public)
    assert n_common > 0
    filt.build(prot, cfg)
    r = filt.query(shared, PY)
    assert r.score == 0.0 and r.details["reason"] == "all_common"
    assert len(filt.inv) < len(plain.inv)
    # членство остальных не страдает
    assert filt.query(records[2].code, PY).score >= 0.95
    # фильтр, подогнанный после build, тоже применяется
    late = WinnowingIndex(cfg)
    late.build(prot, cfg)
    late.fit_common_filter(public)
    assert late.query(shared, PY).score == 0.0 and len(late.inv) == len(filt.inv)


def test_winnowing_protected_file_filter(cfg):
    """Отпечаток в > 5 % файлов protected (и ≥ common_min_files) — общий."""
    rng = random.Random(3)
    boiler = make_fn(rng, 500)
    recs = []
    for i in range(60):  # 60 файлов; шаблон boiler присутствует в 6 из них (10 %)
        code = boiler if i % 10 == 0 else make_fn(rng, 600 + i)
        recs.append(FunctionRecord(id=f"protected/r/{i}", split="protected", repo="r", path=f"src/{i}.py", lang=PY, code=code,
                                   start_line=1, end_line=10, n_lines=10, n_tokens=0, sha=""))
    idx = WinnowingIndex(cfg)
    idx.build(recs, cfg)
    assert idx.protected_common.size > 0 and idx.meta["n_common_protected"] > 0
    assert idx.query(boiler, PY).score == 0.0
    assert idx.query(recs[1].code, PY).score >= 0.95


def test_winnowing_hmac_key(cfg, records, monkeypatch):
    q = records[7].code
    monkeypatch.setenv("SMCODE_INDEX_KEY", "key-one")
    a = WinnowingIndex(cfg)
    a.build(records, cfg)
    monkeypatch.setenv("SMCODE_INDEX_KEY", "key-two")
    b = WinnowingIndex(cfg)
    b.build(records, cfg)
    assert a.key == b"key-one" and b.key == b"key-two"
    assert not np.intersect1d(a.inv.keys, b.inv.keys).size  # хэши различны
    for variant in (q, rename(q), partial(q)):
        assert a.query(variant, PY).score == pytest.approx(b.query(variant, PY).score)
    # без ключа индекс с ключом не отвечает
    monkeypatch.delenv("SMCODE_INDEX_KEY")
    c = WinnowingIndex(cfg)
    c.build(records, cfg)
    assert c.key is None and c.query(q, PY).score == 1.0
    assert not np.intersect1d(a.inv.keys, c.inv.keys).size


# ----------------------------------------------------------------------------- M2 minhash


def test_shingles_and_jaccard():
    toks = list("abcdefgh")
    sh = shingle_set(toks, 3)
    assert sh.size == 6 and np.all(np.diff(sh.astype(np.float64)) > 0)
    assert shingle_set(toks[:2], 3).size == 0
    assert jaccard(sh, sh) == (1.0, 6)
    j, inter = jaccard(sh, shingle_set(list("abcdxyz"), 3))
    assert inter == 2 and j == pytest.approx(2 / (6 + 5 - 2))
    assert jaccard(sh, np.zeros(0, dtype=np.uint64)) == (0.0, 0)
    assert not np.intersect1d(sh, shingle_set(toks, 3, key=b"k")).size


def test_minhash_scores(cfg, records, nonmember):
    idx = MinHashIndex(cfg)
    idx.build(records, cfg)
    assert idx.lsh is not None and idx.hashvalues.shape == (len(records), 128)
    q = records[9].code
    for variant in (q, rename(q)):
        r = idx.query(variant, PY)
        assert r.score == 1.0 and r.best_id == records[9].id and r.details["containment"] == 1.0
    # половина функции: Жаккар ≈ доля, кандидат находится
    half = partial(q, 0, 9)
    r = idx.query(half, PY)
    assert r.best_id == records[9].id and 0.3 < r.score < 1.0 and r.details["containment"] == 1.0
    r = idx.query(nonmember, PY)
    assert r.score < 0.3, r
    r = idx.query("x = 1\n", PY)
    assert r.score == 0.0 and r.details["reason"] == "too_short"
    assert idx.memory_bytes() > idx.hashvalues.nbytes


# ----------------------------------------------------------------------------- save / load


@pytest.mark.parametrize("name", ["exact", "winnowing", "minhash"])
def test_save_load_roundtrip(cfg, records, nonmember, tmp_path, name):
    idx = make_index(name, cfg)
    idx.build(records, cfg)
    out = tmp_path / "idx" / name
    idx.save(out)
    assert (out / "meta.json").exists() and (out / "state.pkl").exists() and (out / "arrays.npz").exists()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["name"] == name and meta["n_records"] == len(records)
    loaded = make_index(name, cfg)
    loaded.load(out)
    assert loaded.ids == idx.ids
    queries = [records[1].code, rename(records[1].code), partial(records[1].code), nonmember]
    for qq in queries:
        a, b = idx.query(qq, PY), loaded.query(qq, PY)
        assert a.score == pytest.approx(b.score) and a.best_id == b.best_id
    assert abs(loaded.memory_bytes() - idx.memory_bytes()) / max(idx.memory_bytes(), 1) < 0.5


def test_build_index_and_load_index(cfg, records):
    idx, stats = build_index("exact", cfg, [r.to_dict() for r in records], tmp_dir := cfg["paths"]["indexes"] + "/exact")
    assert stats["n_records"] == len(records) and stats["memory_bytes"] > 0 and stats["disk_bytes"] > 0
    assert Path(tmp_dir).exists()
    loaded = load_index("exact", cfg)
    assert loaded.query(records[0].code, PY).score == 1.0


# ----------------------------------------------------------------------------- scripts/04_build_indexes.py


def _load_script():
    spec = importlib.util.spec_from_file_location("build_indexes_04", ROOT / "scripts" / "04_build_indexes.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_script_build_indexes(cfg, records, tmp_path, caplog):
    fdir = Path(cfg["paths"]["functions"])
    write_jsonl(fdir / "protected.jsonl", records)
    windows = [FunctionRecord(**{**r.to_dict(), "id": r.id + "#w", "kind": "window", "code": partial(r.code, 0, 9)}) for r in records[:5]]
    write_jsonl(fdir / "protected_windows.jsonl", windows)
    write_jsonl(fdir / "public_train.jsonl", make_records(30, seed=5, split="public_train", prefix="pub"))
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}), encoding="utf-8")
    mod = _load_script()
    assert mod.main(["--config", str(cfg_path), "--with-windows"]) == 0
    stats_path = Path(cfg["paths"]["results"]) / "index_stats.json"
    stats = json.loads(stats_path.read_text())
    assert set(stats) == {"exact", "winnowing", "minhash"}
    for m in ("exact", "winnowing", "minhash"):
        assert (Path(cfg["paths"]["indexes"]) / m / "meta.json").exists()
        assert stats[m]["memory_bytes"] > 0 and stats[m]["build_seconds"] >= 0 and not stats[m]["skipped"]
    assert stats["winnowing"]["n_records"] == len(records) + len(windows)
    assert stats["exact"]["n_records"] == len(records)
    assert stats["winnowing"]["meta"]["n_public_fitted"] == 30
    # идемпотентность: повторный запуск пропускает готовые индексы
    assert mod.main(["--config", str(cfg_path), "--methods", "exact,winnowing"]) == 0
    stats2 = json.loads(stats_path.read_text())
    assert stats2["exact"]["skipped"] and stats2["winnowing"]["skipped"] and not stats2["minhash"]["skipped"]
    # загрузка и запрос
    w = load_index("winnowing", cfg)
    assert w.query(records[2].code, PY).best_id == records[2].id
    # неизвестный метод — ошибка аргументов; GPU-метод без torch — сообщение, код 0
    with pytest.raises(SystemExit):
        mod.main(["--config", str(cfg_path), "--methods", "bogus"])
    rc = mod.main(["--config", str(cfg_path), "--methods", "semantic"])
    assert rc == 0
