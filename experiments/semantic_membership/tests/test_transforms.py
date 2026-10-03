"""Тесты преобразований (smcode.transforms) и построения запросов (smcode.eval.build_queries).
Синтетические данные, CPU, без сети и без torch."""

from __future__ import annotations

import json
import re
import random
import subprocess
import sys
from pathlib import Path

import pytest

from smcode.normalize import abstract_tokens, count_tokens, tokenize
from smcode.transforms import llm_paraphrase
from smcode.transforms.programmatic import parse_code, parses_ok
from smcode.transforms.registry import LLM_TRANSFORMS, TRANSFORMS, apply_transform, list_transforms
from smcode.types import FunctionRecord, read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
LANGS = ["python", "c", "cpp", "go", "java", "javascript"]

SAMPLES: dict[str, str] = {
    "python": '''def checksum(data, seed=7):
    """Compute a checksum."""
    # running total
    total = seed
    count = 0
    for b in data:
        total = (total * 31 + b) % 65521
        count += 1
    if count == 0:
        return 0
    name = "sum"
    return total ^ count
''',
    "c": '''static unsigned int checksum(const unsigned char *data, size_t n, unsigned int seed) {
    /* running total */
    unsigned int total = seed;
    size_t count = 0;
    for (size_t i = 0; i < n; i++) {
        total = (total * 31 + data[i]) % 65521;
        count++;
    }
    if (count == 0) {
        return 0;
    }
    return total ^ (unsigned int)count;
}
''',
    "cpp": '''uint32_t checksum(const std::vector<uint8_t>& data, uint32_t seed) {
    // running total
    uint32_t total = seed;
    size_t count = 0;
    for (auto b : data) {
        total = (total * 31 + b) % 65521;
        count++;
    }
    if (count == 0) {
        return 0;
    }
    return total ^ static_cast<uint32_t>(count);
}
''',
    "go": '''func checksum(data []byte, seed uint32) uint32 {
\t// running total
\ttotal := seed
\tcount := 0
\tfor _, b := range data {
\t\ttotal = (total*31 + uint32(b)) % 65521
\t\tcount++
\t}
\tif count == 0 {
\t\treturn 0
\t}
\treturn total ^ uint32(count)
}
''',
    "java": '''public static int checksum(byte[] data, int seed) {
    // running total
    int total = seed;
    int count = 0;
    for (byte b : data) {
        total = (total * 31 + b) % 65521;
        count++;
    }
    if (count == 0) {
        return 0;
    }
    return total ^ count;
}
''',
    "javascript": '''function checksum(data, seed) {
  // running total
  let total = seed;
  let count = 0;
  for (const b of data) {
    total = (total * 31 + b) % 65521;
    count += 1;
  }
  if (count === 0) {
    return 0;
  }
  return total ^ count;
}
''',
}

ALWAYS_APPLICABLE = ("identity", "reformat", "strip_comments", "rename_ids", "change_literals", "insert_deadcode", "combo")


def _abstract(code: str, lang: str) -> list[str]:
    return abstract_tokens(tokenize(code, lang), mode="full")


def _lexical(code: str, lang: str) -> list[str]:
    return abstract_tokens(tokenize(code, lang), mode="lexical")


# --------------------------------------------------------------------------- реестр


def test_registry_names():
    names = list_transforms()
    for n in ("identity", "reformat", "strip_comments", "rename_ids", "change_literals", "insert_deadcode",
              "reorder_stmts", "combo", "partial"):
        assert n in names and n in TRANSFORMS
    assert set(LLM_TRANSFORMS) <= set(list_transforms(include_llm=True))
    assert "paraphrase" not in names


def test_unknown_and_llm_names():
    with pytest.raises(KeyError):
        apply_transform("nope", "x = 1\n", "python", random.Random(0))
    r = apply_transform("paraphrase", "x = 1\n", "python", random.Random(0))
    assert not r.ok and r.code is None
    assert not apply_transform("rename_ids", "   ", "python", random.Random(0)).ok


@pytest.mark.parametrize("lang", LANGS)
def test_samples_parse(lang):
    assert parses_ok(SAMPLES[lang], lang)


# --------------------------------------------------------------------------- каждое преобразование


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("name", list(TRANSFORMS))
def test_transform_returns_parseable_code(lang, name):
    code = SAMPLES[lang]
    for seed in range(3):
        kw = {"L": 4} if name == "partial" else {}
        r = apply_transform(name, code, lang, random.Random(seed), **kw)
        assert r.name == name
        if name in ALWAYS_APPLICABLE or name == "partial":
            assert r.ok, (lang, name, seed, r.params)
        if r.ok:
            assert r.code and parses_ok(r.code, lang), (lang, name, r.code)
            if name != "identity":
                assert r.code != code
        else:
            assert r.code is None


@pytest.mark.parametrize("lang", LANGS)
def test_determinism(lang):
    for name in ("reformat", "rename_ids", "change_literals", "insert_deadcode", "combo"):
        a = apply_transform(name, SAMPLES[lang], lang, random.Random(42))
        b = apply_transform(name, SAMPLES[lang], lang, random.Random(42))
        assert a.ok and a.code == b.code and a.params == b.params


@pytest.mark.parametrize("lang", LANGS)
def test_rename_ids_keeps_abstract_stream(lang):
    code = SAMPLES[lang]
    for style in ("vn", "snake", "camel", "short"):
        r = apply_transform("rename_ids", code, lang, random.Random(7), style=style)
        assert r.ok and r.params["style"] == style
        assert _abstract(r.code, lang) == _abstract(code, lang), (lang, style)
        assert _lexical(r.code, lang) != _lexical(code, lang)
        mapping = r.params["mapping"]
        assert {"checksum", "data", "seed", "total", "count"} <= set(mapping)
        assert "checksum" not in _lexical(r.code, lang)
        # ключевые слова и внешние имена не переименованы
        for kw in ("return", "for", "if"):
            assert kw in _lexical(r.code, lang)


def test_rename_ids_python_scoping():
    code = (
        "def f(self, items, key=None, **kw):\n"
        "    s = self.x\n"
        "    t, u = 1, 2\n"
        "    for k, v in enumerate(items):\n"
        "        s += k + v\n"
        "    self.helper(key=t, u=u)\n"
        "    return lambda z: z + s + len(kw)\n"
    )
    r = apply_transform("rename_ids", code, "python", random.Random(3), style="vn")
    assert r.ok and parses_ok(r.code, "python")
    m = r.params["mapping"]
    assert {"f", "items", "key", "kw", "s", "t", "u", "k", "v", "z"} <= set(m)
    assert "self" not in m
    assert "self.x" in r.code and "self.helper(key=" in r.code and "u=" in r.code  # member/keyword args сохранены
    assert "enumerate(" in r.code and "len(" in r.code
    assert _abstract(r.code, "python") == _abstract(code, "python")


def test_rename_ids_c_struct_fields_and_calls():
    code = "int foo(int a, char *s) {\n    struct point pt;\n    pt.x = a;\n    s[0] = 'c';\n    return bar(a) + pt.x + MACRO(s);\n}\n"
    r = apply_transform("rename_ids", code, "c", random.Random(1), style="snake")
    assert r.ok and "pt.x" not in r.code and ".x" in r.code and "bar(" in r.code and "MACRO(" in r.code
    assert _abstract(r.code, "c") == _abstract(code, "c")


def test_rename_ids_go_struct_literal_and_fields():
    code = "func (r *T) Foo(a int) int {\n\tx := Point{Name: a}\n\treturn r.base + x.Name + a\n}\n"
    r = apply_transform("rename_ids", code, "go", random.Random(1), style="camel")
    assert r.ok and "Point{Name:" in r.code and "r.base" not in r.code and ".base" in r.code and ".Name" in r.code
    assert "Foo" not in _lexical(r.code, "go")
    assert _abstract(r.code, "go") == _abstract(code, "go")


def test_rename_ids_inapplicable():
    assert not apply_transform("rename_ids", "return 1;\n", "c", random.Random(0)).ok


@pytest.mark.parametrize("lang", LANGS)
def test_strip_comments(lang):
    r = apply_transform("strip_comments", SAMPLES[lang], lang, random.Random(0))
    assert r.ok and "running total" not in r.code and _abstract(r.code, lang) == _abstract(SAMPLES[lang], lang)
    assert not apply_transform("strip_comments", r.code, lang, random.Random(0)).ok  # уже без комментариев


@pytest.mark.parametrize("lang", LANGS)
def test_change_literals(lang):
    code = SAMPLES[lang]
    r = apply_transform("change_literals", code, lang, random.Random(5))
    assert r.ok and r.params["n_num"] >= 3
    assert "65521" not in r.code
    assert _abstract(r.code, lang) == _abstract(code, lang)
    if lang == "python":
        assert r.params["n_str"] == 1 and '"sum"' not in r.code and '"""Compute a checksum."""' in r.code


@pytest.mark.parametrize("lang", LANGS)
def test_reformat_modes(lang):
    code = SAMPLES[lang]
    for indent in ("2", "4", "tab"):
        for spaces in ("add", "remove", "keep"):
            for braces in ("allman", "knr", "keep"):
                r = apply_transform("reformat", code, lang, random.Random(0), indent=indent, spaces=spaces,
                                    braces=braces, join_blocks=True)
                if r.ok:
                    assert parses_ok(r.code, lang), (lang, indent, spaces, braces)
                    assert _abstract(r.code, lang) == _abstract(code, lang)
    # смена единицы отступа всегда применима (в go образец уже на табах → переводим в 2 пробела)
    indent = "2" if "\t" in code else "tab"
    r = apply_transform("reformat", code, lang, random.Random(0), indent=indent, spaces="keep", braces="keep",
                        join_blocks=False)
    assert r.ok and (("\t" in r.code) if indent == "tab" else ("\t" not in r.code and "  total" in r.code))
    # без изменений (та же единица отступа, всё остальное keep) → неприменимо
    same = "tab" if "\t" in code else ("2" if lang == "javascript" else "4")
    assert not apply_transform("reformat", code, lang, random.Random(0), indent=same, spaces="keep",
                               braces="keep", join_blocks=False).ok


@pytest.mark.parametrize("lang", LANGS)
def test_insert_deadcode_increases_tokens(lang):
    code = SAMPLES[lang]
    for every in (1, 3, 100):
        r = apply_transform("insert_deadcode", code, lang, random.Random(every), every=every)
        assert r.ok and r.params["every"] == every and r.params["n_inserted"] >= 1
        assert count_tokens(r.code, lang) > count_tokens(code, lang)
        assert parses_ok(r.code, lang)
        assert len(r.code.splitlines()) == len(code.splitlines()) + r.params["n_inserted"]
    r1 = apply_transform("insert_deadcode", code, lang, random.Random(0), every=1)
    r3 = apply_transform("insert_deadcode", code, lang, random.Random(0), every=3)
    assert r1.params["n_inserted"] >= r3.params["n_inserted"]


@pytest.mark.parametrize("lang", ["c", "cpp", "go", "java", "javascript", "python"])
def test_reorder_stmts_swaps_independent(lang):
    code = SAMPLES[lang]
    r = apply_transform("reorder_stmts", code, lang, random.Random(0))
    assert r.ok and r.params["n_swaps"] >= 1
    assert sorted(_lexical(r.code, lang)) == sorted(_lexical(code, lang))
    assert _lexical(r.code, lang) != _lexical(code, lang)
    # переставляются только целые строки: мультимножество строк сохраняется, порядок меняется
    assert sorted(r.code.splitlines()) == sorted(code.splitlines()) and r.code != code


def test_reorder_stmts_conservative():
    code = "int f(int a) {\n    int x = g(a);\n    int y = x + 1;\n    int z = h();\n    return y + z;\n}\n"
    assert not apply_transform("reorder_stmts", code, "c", random.Random(0)).ok


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("L", [3, 5])
def test_partial_returns_L_lines(lang, L):
    code = SAMPLES[lang]
    r = apply_transform("partial", code, lang, random.Random(L), L=L)
    assert r.ok and r.params["L"] == L
    assert len([ln for ln in r.code.splitlines() if ln.strip()]) == L
    assert r.params["parses"] is True and parses_ok(r.code, lang)
    assert all(ln.strip() in code for ln in r.code.splitlines() if ln.strip())  # строки взяты из тела
    assert not r.code.startswith((" ", "\t"))


def test_partial_too_long_fails():
    assert not apply_transform("partial", SAMPLES["python"], "python", random.Random(0), L=200).ok


def test_combo_params():
    r = apply_transform("combo", SAMPLES["c"], "c", random.Random(2))
    assert r.ok and set(r.params["steps"]) == {"reformat", "strip_comments", "rename_ids", "change_literals"}
    assert "running total" not in r.code and "checksum" not in r.code


def test_wrapped_parse_for_method_bodies():
    js = "foo(a, b) {\n  return a + b;\n}\n"
    assert parses_ok(js, "javascript")
    p = parse_code(js, "javascript")
    assert p.wrapper == "class" and p.unwrap() == js
    r = apply_transform("rename_ids", js, "javascript", random.Random(0), style="vn")
    assert r.ok and "foo" not in r.code and r.code.startswith("v")
    java = "Foo(int a) {\n  this.a = a;\n}\n"
    r = apply_transform("insert_deadcode", java, "java", random.Random(0), every=1)
    assert r.ok and parses_ok(r.code, "java")


# --------------------------------------------------------------------------- build_queries


def _tiny_cfg(tmp_path: Path) -> dict:
    return {
        "seed": 1,
        "paths": {"functions": str(tmp_path / "functions"), "queries": str(tmp_path / "queries")},
        "languages": ["python", "c", "cpp", "go", "java", "javascript"],
        "transforms": {
            "programmatic": ["identity", "reformat", "strip_comments", "rename_ids", "change_literals",
                             "insert_deadcode", "reorder_stmts", "combo"],
            "partial_lines": [3, 5, 40],
            "deadcode_every": 3,
            "max_per_split": 50,
        },
        "eval": {"length_bins_tokens": [[0, 32], [32, 64], [64, 128], [128, 256], [256, 100000]]},
    }


def _rec(split: str, i: int, lang: str, kind: str = "function") -> FunctionRecord:
    code = SAMPLES[lang]
    return FunctionRecord(id=f"{split}/repo{i}/src/f{i}.{lang}:1-{code.count(chr(10))}", split=split,
                          repo=f"org/repo{i}", path=f"src/f{i}.{lang}", lang=lang, code=code, start_line=1,
                          end_line=code.count("\n"), n_lines=code.count("\n"), n_tokens=count_tokens(code, lang),
                          sha=f"sha{i}", kind=kind)


def test_build_queries_tiny(tmp_path):
    from smcode.eval.build_queries import ALL_SETS, build_queries, sample_records

    cfg = _tiny_cfg(tmp_path)
    fdir = tmp_path / "functions"
    # раскладка модуля данных: окна лежат в отдельном файле protected_windows.jsonl (kind == window)
    write_jsonl(fdir / "protected.jsonl", [_rec("protected", i, l) for i, l in enumerate(LANGS)])
    write_jsonl(fdir / "protected_windows.jsonl", [_rec("protected", 99, "python", kind="window"),
                                                   _rec("protected", 98, "c", kind="window")])
    write_jsonl(fdir / "public_calib.jsonl", [_rec("public_calib", i, l) for i, l in enumerate(LANGS[:2])])
    res = build_queries(cfg)
    assert set(res) == set(ALL_SETS)
    assert res["hard_neg"] == 0 and res["public_test"] == 0  # нет файлов функций
    assert res["protected"] > 0 and res["public_calib"] > 0 and res["protected_windows"] > 0
    assert not list((tmp_path / "queries").glob("*.tmp"))  # атомарная запись не оставляет временных файлов

    rows = list(read_jsonl(tmp_path / "queries" / "protected.jsonl"))
    assert len(rows) == res["protected"]
    by_src: dict[str, list[dict]] = {}
    for r in rows:
        assert set(r) == {"qid", "source_id", "label", "set", "transform", "params", "lang", "repo", "code",
                          "n_lines", "n_tokens"}
        assert r["label"] == 1 and r["set"] == "protected"
        src, tname, idx = r["qid"].rsplit("#", 2)
        assert src == r["source_id"] and tname == r["transform"] and idx.isdigit()
        assert r["n_tokens"] == count_tokens(r["code"], r["lang"]) > 0
        assert r["n_lines"] == len([ln for ln in r["code"].splitlines() if ln.strip()])
        assert "mapping" not in r["params"]
        if tname == "partial":
            assert r["params"]["L"] in (3, 5) and r["n_lines"] == r["params"]["L"]
        by_src.setdefault(src, []).append(r)
    assert len(by_src) == 6  # все 6 функций (окно сюда не попадает)
    for src, rs in by_src.items():
        names = {r["transform"] for r in rs}
        assert {"identity", "reformat", "strip_comments", "rename_ids", "change_literals", "insert_deadcode",
                "combo", "partial"} <= names
        assert sum(1 for r in rs if r["transform"] == "partial") == 2  # L=40 неприменим
        ident = [r for r in rs if r["transform"] == "identity"]
        assert len(ident) == 1 and ident[0]["qid"].endswith("#identity#0") and ident[0]["code"] in SAMPLES.values()

    calib = list(read_jsonl(tmp_path / "queries" / "public_calib.jsonl"))
    assert calib and all(r["label"] == 0 and r["set"] == "public_calib" for r in calib)
    win = list(read_jsonl(tmp_path / "queries" / "protected_windows.jsonl"))
    assert win and all(r["label"] == 1 and r["set"] == "protected_windows" for r in win)
    assert {r["transform"] for r in win} == {"identity", "partial"}
    assert {r["source_id"] for r in win} == {_rec("protected", 99, "python").id, _rec("protected", 98, "c").id}
    stats = json.loads((tmp_path / "queries" / "protected.stats.json").read_text())
    assert stats["n_queries"] == res["protected"] and stats["per_transform"]["identity"]["ok"] == 6

    # идемпотентность и --force
    again = build_queries(cfg)
    assert again == {"protected": -1, "hard_neg": 0, "public_calib": -1, "public_test": 0, "protected_windows": -1}
    res2 = build_queries(cfg, splits=["protected"])
    assert res2 == {"protected": -1}
    res3 = build_queries(cfg, splits=["protected"], force=True)
    assert res3["protected"] == res["protected"]
    assert list(read_jsonl(tmp_path / "queries" / "protected.jsonl")) == rows  # детерминизм

    # стратифицированная выборка
    recs = [_rec("x", i, LANGS[i % 6]) for i in range(60)]
    s = sample_records(recs, 12, random.Random(0), cfg["eval"]["length_bins_tokens"])
    assert len(s) == 12 and len({r.lang for r in s}) == 6 and s == sorted(s, key=lambda r: r.id)


def test_script_help_runs():
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "03_build_queries.py"), "--help"],
                         capture_output=True, text=True, cwd=ROOT, timeout=60)
    assert out.returncode == 0 and "--llm" in out.stdout and "--force" in out.stdout


# --------------------------------------------------------------------------- llm_paraphrase (чистые части)


def test_llm_extract_and_validate():
    text = "Sure:\n```python\ndef g(a):\n    return a + 1\n```\nDone."
    assert llm_paraphrase.extract_code(text) == "def g(a):\n    return a + 1\n"
    assert llm_paraphrase.extract_code("def g(a):\n    return a\n") == "def g(a):\n    return a\n"
    assert llm_paraphrase.extract_code("```c\nint f(void) { return 1; }") == "int f(void) { return 1; }\n"
    assert llm_paraphrase.extract_code("") == ""
    assert llm_paraphrase.validate_output("def g(a):\n    return a + 1\n", "python", original=SAMPLES["python"])
    assert not llm_paraphrase.validate_output("def g(a:\n    return\n", "python")
    assert not llm_paraphrase.validate_output(SAMPLES["python"], "python", original=SAMPLES["python"])
    assert not llm_paraphrase.validate_output("", "go")
    rng = random.Random(0)
    for _ in range(10):
        assert llm_paraphrase.pick_dst_lang("c", LANGS, rng) != "c"
    msgs = llm_paraphrase.translate_messages(SAMPLES["c"], "c", "go")
    assert msgs[0]["role"] == "system" and "Go" in msgs[1]["content"] and "```go" in msgs[1]["content"]
    msgs = llm_paraphrase.paraphrase_messages(SAMPLES["java"], "java")
    assert "Java" in msgs[1]["content"] and SAMPLES["java"].rstrip() in msgs[1]["content"]


# --------------------------------------------------------------------------- регрессии после ревью


def test_windows_fallback_to_protected_jsonl(tmp_path):
    """Без protected_windows.jsonl окна берутся из protected.jsonl (kind == window)."""
    from smcode.eval.build_queries import build_queries, functions_path

    cfg = _tiny_cfg(tmp_path)
    fdir = tmp_path / "functions"
    write_jsonl(fdir / "protected.jsonl", [_rec("protected", 0, "python"), _rec("protected", 99, "go", kind="window")])
    assert functions_path(cfg, "protected_windows") == fdir / "protected.jsonl"
    res = build_queries(cfg, splits=["protected", "protected_windows"])
    assert res["protected"] > 0 and res["protected_windows"] > 0
    win = list(read_jsonl(tmp_path / "queries" / "protected_windows.jsonl"))
    assert {r["source_id"] for r in win} == {_rec("protected", 99, "go").id}
    prot = list(read_jsonl(tmp_path / "queries" / "protected.jsonl"))
    assert {r["source_id"] for r in prot} == {_rec("protected", 0, "python").id}  # окно в protected не попадает
    write_jsonl(fdir / "protected_windows.jsonl", [_rec("protected", 98, "c", kind="window")])
    assert functions_path(cfg, "protected_windows") == fdir / "protected_windows.jsonl"


def test_build_set_treats_empty_file_as_missing(tmp_path):
    from smcode.eval.build_queries import build_queries

    cfg = _tiny_cfg(tmp_path)
    write_jsonl(tmp_path / "functions" / "public_calib.jsonl", [_rec("public_calib", 0, "c")])
    res = build_queries(cfg, splits=["public_calib"])
    assert res["public_calib"] > 0
    out = tmp_path / "queries" / "public_calib.jsonl"
    out.write_text("")  # прерванная запись
    assert build_queries(cfg, splits=["public_calib"])["public_calib"] == res["public_calib"]
    assert out.stat().st_size > 0
    assert build_queries(cfg, splits=["public_calib"]) == {"public_calib": -1}
    stats = json.loads((tmp_path / "queries" / "public_calib.stats.json").read_text())
    assert "partial@3" in stats["per_transform"]


class _FakeEngine:
    """Подмена LLMEngine без vllm: paraphrase → rename_ids исходника, translate → образец на целевом языке."""

    model, seed, temperature = "fake", 1, 0.0

    def __init__(self, echo: bool = False) -> None:
        self.echo = echo
        self.n_calls = 0

    def generate(self, conversations):
        self.n_calls += 1
        out = []
        for conv in conversations:
            user = conv[-1]["content"]
            code = llm_paraphrase.extract_code(user)
            tag = re.search(r"in a ```(\w+) block", user).group(1)
            if self.echo:
                out.append(f"```{tag}\n{code}```")
            elif user.startswith("Translate"):
                out.append(f"```{tag}\n{SAMPLES[tag]}```")
            else:
                r = apply_transform("rename_ids", code, tag, random.Random(0), style="vn")
                out.append(f"Sure:\n```{tag}\n{r.code}```")
        return out


def test_llm_queries_subset_of_programmatic_sample(tmp_path):
    from smcode.eval.build_queries import build_queries
    from smcode.transforms.llm_paraphrase import build_llm_queries

    cfg = _tiny_cfg(tmp_path)
    cfg["transforms"]["llm"] = ["paraphrase", "translate"]
    recs = [_rec("protected", i, LANGS[i % 6]) for i in range(12)]
    write_jsonl(tmp_path / "functions" / "protected.jsonl", recs)
    # без программных строк — отказ (0), файл не создаётся
    eng = _FakeEngine()
    assert build_llm_queries(cfg, splits=["protected"], sample=2, engine=eng) == {"protected": 0}
    assert not (tmp_path / "queries" / "protected.jsonl").exists() and eng.n_calls == 0

    assert build_queries(cfg, splits=["protected"], max_per_split=5)["protected"] > 0
    prog_ids = {r["source_id"] for r in read_jsonl(tmp_path / "queries" / "protected.jsonl")}
    assert len(prog_ids) == 5
    res = build_llm_queries(cfg, splits=["protected"], sample=3, engine=eng)
    assert res["protected"] == 6  # 3 функции × (paraphrase + translate)
    rows = list(read_jsonl(tmp_path / "queries" / "protected.jsonl"))
    llm = [r for r in rows if r["transform"] in ("paraphrase", "translate")]
    assert len(llm) == 6 and {r["source_id"] for r in llm} <= prog_ids
    assert len({r["source_id"] for r in llm}) == 3
    assert len([r for r in rows if r["transform"] not in ("paraphrase", "translate")]) == len(rows) - 6
    for r in llm:
        assert r["label"] == 1 and r["qid"] == f"{r['source_id']}#{r['transform']}#0"
        assert r["params"]["model"] == "fake" and r["params"]["prompt_id"].startswith(r["transform"])
        if r["transform"] == "translate":
            assert r["lang"] == r["params"]["dst_lang"] != r["params"]["src_lang"]
            assert llm_paraphrase.lang_family(r["lang"]) != llm_paraphrase.lang_family(r["params"]["src_lang"])
        else:
            assert r["lang"] == r["params"]["src_lang"]
        assert parses_ok(r["code"], r["lang"]) and r["n_tokens"] == count_tokens(r["code"], r["lang"])
    stats = json.loads((tmp_path / "queries" / "protected.llm_stats.json").read_text())
    assert stats["paraphrase_ok"] == 3 and stats["translate_ok"] == 3 and stats["n_programmatic_sources"] == 5
    assert sum(stats["translate_pairs"].values()) == 3
    # идемпотентность и детерминизм
    assert build_llm_queries(cfg, splits=["protected"], sample=3, engine=eng) == {"protected": -1}
    assert build_llm_queries(cfg, splits=["protected"], sample=3, engine=eng, force=True) == {"protected": 6}
    assert list(read_jsonl(tmp_path / "queries" / "protected.jsonl")) == rows
    assert not list((tmp_path / "queries").glob("*.tmp"))


def test_translate_rejects_echo_and_same_family():
    cfg = {"transforms": {}}
    rng = random.Random(0)
    for src in ("c", "cpp"):
        assert {llm_paraphrase.pick_dst_lang(src, LANGS, rng) for _ in range(30)} == {"python", "go", "java", "javascript"}
    assert {llm_paraphrase.pick_dst_lang("go", ["go", "c"], rng) for _ in range(5)} == {"c"}
    # эхо исходника (C → C++ один в один / только переименование) не принимается
    res = llm_paraphrase.translate_batch([SAMPLES["c"]], ["c"], ["cpp"], cfg, engine=_FakeEngine(echo=True))
    assert len(res) == 1 and not res[0].ok and res[0].params["dst_lang"] == "cpp"
    renamed = apply_transform("rename_ids", SAMPLES["c"], "c", random.Random(1), style="vn").code
    assert not llm_paraphrase.validate_output(renamed, "cpp", original=SAMPLES["c"], original_lang="c",
                                              reject_same_abstract=True)
    assert llm_paraphrase.validate_output(renamed, "c", original=SAMPLES["c"])  # для paraphrase переименование — ок
    res = llm_paraphrase.translate_batch([SAMPLES["c"]], ["c"], ["go"], cfg, engine=_FakeEngine())
    assert res[0].ok and res[0].code == SAMPLES["go"] and res[0].params["src_lang"] == "c"
    res = llm_paraphrase.paraphrase_batch([SAMPLES["java"]], ["java"], cfg, engine=_FakeEngine())
    assert res[0].ok and res[0].code != SAMPLES["java"] and _abstract(res[0].code, "java") == _abstract(SAMPLES["java"], "java")


def test_rename_ids_go_keyed_element_keys():
    """Ключи структурного литерала — имена полей (не переименовываются); ключи map-литерала — выражения."""
    code = ("func newConn(fd int, name string) *conn {\n"
            "\tc := &conn{fd: fd, name: name}\n"
            "\tm := map[string]int{name: fd}\n"
            "\ts := []conn{{fd: fd}}\n"
            "\tmm := map[string]conn{name: {fd: fd}}\n"
            "\treturn c\n}\n")
    r = apply_transform("rename_ids", code, "go", random.Random(0), style="vn")
    assert r.ok and parses_ok(r.code, "go")
    m = r.params["mapping"]
    fd, name = m["fd"], m["name"]
    assert f"&conn{{fd: {fd}, name: {name}}}" in r.code
    assert f"map[string]int{{{name}: {fd}}}" in r.code
    assert f"[]conn{{{{fd: {fd}}}}}" in r.code
    assert f"map[string]conn{{{name}: {{fd: {fd}}}}}" in r.code
    assert _abstract(r.code, "go") == _abstract(code, "go")


def test_python_empty_blocks_rejected():
    from smcode.transforms.programmatic import parses_ok as ok

    assert not ok("def f(a):\n", "python") and not ok("x = 1\ntry:\n", "python")
    assert not ok("try:\n    x = 1\nexcept (TypeError, ValueError):\n", "python")
    assert not ok("def f(a):\n    def g():\n    return g\n", "python")
    assert not ok("    x = 0\nreturn x\n", "python")  # рассогласованный отступ (ast)
    assert ok("x = 0\nreturn x\n", "python") and ok("def f(a):\n    return a\n", "python")
    # функция только с docstring: strip_comments дал бы пустое тело → неприменимо; combo пропускает шаг
    code = 'def _warn(self, n):\n    """Only a docstring."""\n'
    assert not apply_transform("strip_comments", code, "python", random.Random(0)).ok
    r = apply_transform("combo", code, "python", random.Random(0))
    assert r.ok and "strip_comments" not in r.params["steps"] and "Only a docstring" in r.code and parses_ok(r.code, "python")
    code2 = 'def f(a):\n    """Doc."""\n    return a\n'
    assert apply_transform("strip_comments", code2, "python", random.Random(0)).code == "def f(a):\n    return a\n"


def test_partial_unparsed_window_is_accepted_with_flag():
    """Все окна L=3 заканчиваются/начинаются висячим заголовком → фрагмент принимается с parses=False."""
    code = "def f(a):\n    try:\n        x = a\n    except (A, B):\n        x = 0\n    y = 1\n    return x + y\n"
    r = apply_transform("partial", code, "python", random.Random(1), L=3)
    assert r.ok and r.params["parses"] is False and r.params["L"] == 3
    assert len([ln for ln in r.code.splitlines() if ln.strip()]) == 3
    # при наличии разбираемого окна (L=2: 'y = 1 / return', L=4: весь try/except) выбирается оно
    for L in (2, 4):
        r2 = apply_transform("partial", code, "python", random.Random(1), L=L)
        assert r2.ok and r2.params["parses"] is True and parses_ok(r2.code, "python"), (L, r2)


def test_reorder_same_type_declarations():
    code = "int f(int a, int b) {\n    int x = a;\n    int y = b;\n    return x + y;\n}\n"
    r = apply_transform("reorder_stmts", code, "c", random.Random(0))
    assert r.ok and r.params["n_swaps"] == 1 and r.code.index("int y = b") < r.code.index("int x = a")
    for lang, code in (
        ("cpp", "void f() {\n    uint32_t x = a;\n    uint32_t y = b;\n    g(x, y);\n}\n"),
        ("go", "func f(a, b int) int {\n\tvar x int = a\n\tvar y int = b\n\treturn x + y\n}\n"),
        ("python", "def f(a, b):\n    x = True\n    y = True\n    return x, y\n"),
        ("java", "int f(int a, int b) {\n    int x = a;\n    int y = b;\n    return x + y;\n}\n"),
    ):
        r = apply_transform("reorder_stmts", code, lang, random.Random(0))
        assert r.ok and r.params["n_swaps"] == 1 and parses_ok(r.code, lang), lang
    # зависимость через идентификатор по-прежнему блокирует
    assert not apply_transform("reorder_stmts", "int f(int a) {\n    int x = a;\n    int y = x;\n    return y;\n}\n",
                               "c", random.Random(0)).ok


def test_change_literals_chars_and_case_labels():
    code = ("int f(char c, int a) {\n    switch (a) {\n        case 1: return 'x';\n        case 2: return 7;\n"
            "        default: break;\n    }\n    if (c == '/' || c == '\\n') {\n        return 100;\n    }\n    return 0;\n}\n")
    r = apply_transform("change_literals", code, "c", random.Random(0))
    assert r.ok and "case 1:" in r.code and "case 2:" in r.code and "return 7" not in r.code
    assert re.search(r"c == '[A-Za-z0-9]'", r.code) and "'/'" not in r.code and "'\\n'" not in r.code
    chars = [t.text for t in tokenize(r.code, "c") if t.kind == "str" and t.text.startswith("'")]
    assert len(chars) == 3 and all(re.fullmatch(r"'[A-Za-z0-9]'", t) for t in chars), chars  # нет многосимвольных char
    assert _abstract(r.code, "c") == _abstract(code, "c") and parses_ok(r.code, "c")
    java = "static int f(char c) {\n    if (c == '\\u0041') {\n        return 1;\n    }\n    return 2;\n}\n"
    r = apply_transform("change_literals", java, "java", random.Random(0))
    assert r.ok and re.search(r"c == '[A-Za-z0-9]'", r.code) and parses_ok(r.code, "java")
    go = "func f(c rune, s string) int {\n\tif c == 'x' && s == \"ab\" {\n\t\treturn 1\n\t}\n\treturn 2\n}\n"
    r = apply_transform("change_literals", go, "go", random.Random(0))
    assert r.ok and re.search(r"c == '[A-Za-z0-9]'", r.code) and '"ab"' not in r.code and parses_ok(r.code, "go")


def test_insert_deadcode_cpp_goto_and_java_super():
    cpp = "int f(int a){\n    int r=0;\n    if (a<0) goto out;\n    r=a*2;\n    r+=1;\nout:\n    return r;\n}\n"
    r = apply_transform("insert_deadcode", cpp, "cpp", random.Random(0), every=1)
    assert r.ok and parses_ok(r.code, "cpp") and r.params["n_inserted"] >= 3
    for ln in r.code.splitlines():
        if "_tmp_" in ln:  # объявление только в собственном блоке: goto через инициализацию ill-formed
            assert ln.strip().startswith("{") and ln.strip().endswith("}"), ln
    java = "Foo(int a) {\n    super(a);\n    this.a = a;\n    this.b = 2;\n}\n"
    r = apply_transform("insert_deadcode", java, "java", random.Random(0), every=1)
    assert r.ok and parses_ok(r.code, "java") and r.params["n_inserted"] == 2
    assert r.code.splitlines()[1].strip() == "super(a);"  # super(...) остаётся первым оператором


def test_reformat_keeps_multiline_string_contents():
    py = 's = """line1\n    line2\n        line3"""\ndef f(x):\n    y = x\n    return y\n'
    r = apply_transform("reformat", py, "python", random.Random(0), indent="2", spaces="keep", braces="keep",
                        join_blocks=False)
    assert r.ok and '"""line1\n    line2\n        line3"""' in r.code and "\n  y = x\n" in r.code
    go = "func f() string {\n\ts := `raw\n\t\tline2\n`\n\treturn s\n}\n"
    r = apply_transform("reformat", go, "go", random.Random(0), indent="2", spaces="keep", braces="keep",
                        join_blocks=False)
    assert r.ok and "`raw\n\t\tline2\n`" in r.code and "\n  return s" in r.code
    java = 'String f() {\n    String t = """\n        hello\n        """;\n    return t;\n}\n'
    r = apply_transform("reformat", java, "java", random.Random(0), indent="2", spaces="keep", braces="keep",
                        join_blocks=False)
    assert r.ok and '"""\n        hello\n        """' in r.code and "\n  return t;" in r.code
