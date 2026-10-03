"""Тесты нормализации кода (smcode.normalize) на шести языках."""

import pytest

from smcode.normalize import (
    abstract_tokens,
    canon_lang,
    count_tokens,
    has_syntax_errors,
    normalized_sha,
    strip_comments,
    tokenize,
)

SAMPLES = {
    "python": 'def add(a, b):\n    """Docstring."""\n    # comment\n    total = a + b  # trailing\n    s = "str"\n    return total * 2\n',
    "c": '/* block */\nint add(int a, int b) {\n    int total = a + b; // line\n    const char *s = "str";\n    return total * 2;\n}\n',
    "cpp": '// comment\nnamespace x { int add(int a, int b) { auto t = a + b; std::string s = "q"; return t * 2; } }',
    "go": 'package m\n// c\nfunc add(a int, b int) int {\n\ttotal := a + b\n\ts := "str"\n\t_ = s\n\treturn total * 2\n}\n',
    "java": 'class A { /** doc */ int add(int a, int b) { int total = a + b; String s = "x"; return total * 2; } }',
    "javascript": 'function add(a, b) { // c\n const total = a + b; const s = `t${a}`; return total * 2; }',
}


@pytest.mark.parametrize("lang", sorted(SAMPLES))
def test_tokenize_kinds(lang):
    toks = tokenize(SAMPLES[lang], lang)
    kinds = {t.kind for t in toks}
    assert {"id", "kw", "num", "str"} <= kinds
    assert "comment" in kinds
    # токены упорядочены и не пересекаются
    for a, b in zip(toks, toks[1:]):
        assert a.end <= b.start


@pytest.mark.parametrize("lang", sorted(SAMPLES))
def test_abstract_full_has_no_identifiers_or_comments(lang):
    full = abstract_tokens(tokenize(SAMPLES[lang], lang), "full")
    assert "add" not in full and "total" not in full
    assert "ID" in full and "NUM" in full and "STR" in full
    assert not any(tok.startswith("//") or tok.startswith("/*") or tok.startswith("#") for tok in full)


def test_abstract_indexed_is_consistent():
    toks = tokenize("def f(x, y):\n    z = x + y\n    return z + x\n", "python")
    idx = abstract_tokens(toks, "indexed")
    assert idx.count("ID_1") == 3  # x встречается трижды
    assert idx.index("ID_0") < idx.index("ID_1") < idx.index("ID_2")


def test_rename_invariance_of_sha():
    a = normalized_sha("def f(x):\n    return x + 1\n", "python")
    b = normalized_sha("def g(y):\n    return y + 1\n", "python")
    c = normalized_sha("def g(y):\n    return y - 1\n", "python")
    assert a == b and a != c


def test_strip_comments_and_docstring():
    out = strip_comments(SAMPLES["python"], "python")
    assert "Docstring" not in out and "# comment" not in out and "trailing" not in out
    assert "total = a + b" in out and "return total * 2" in out
    out_c = strip_comments(SAMPLES["c"], "c")
    assert "block" not in out_c and "// line" not in out_c and "int total = a + b;" in out_c


def test_strip_comments_without_comments_is_identity():
    code = "int f(int a) { return a; }"
    assert strip_comments(code, "c") == code


def test_syntax_error_detection():
    assert has_syntax_errors("def f(:\n  pass", "python")
    assert has_syntax_errors("int f( { return 1; }", "c")
    assert not has_syntax_errors(SAMPLES["go"], "go")


def test_count_tokens_and_lang_aliases():
    assert count_tokens("def f():\n    return 1\n", "py") == 7
    assert canon_lang("JS") == "javascript" and canon_lang(".cpp") == "cpp"
    with pytest.raises(ValueError):
        canon_lang("rust")


def test_empty_and_unicode_input():
    assert tokenize("", "python") == []
    toks = tokenize('s = "привет"  # комментарий\n', "python")
    assert [t.kind for t in toks] == ["id", "op", "str", "comment"]
