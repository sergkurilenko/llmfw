"""Тесты модуля данных (smcode.data): списки репозиториев и сплиты, извлечение tree-sitter,
файловые окна, дедупликация. Синтетические мини-репозитории в tmp_path; без сети и torch."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from smcode.config import ROOT, load_config
from smcode.data import dedup_split, extract, list_repos, windows
from smcode.normalize import normalized_sha
from smcode.types import SPLITS, read_functions, write_jsonl

PY = sys.executable
LANGS = {"python", "c", "cpp", "go", "java", "javascript"}

# ----------------------------------------------------------------------------- synthetic sources

PY_PROTECTED = '''\
import os


class Cipher:
    """Простой шифр."""

    def __init__(self, key):
        self.key = key
        self.rounds = 10

    def encrypt_block(self, block):
        state = list(block)
        for r in range(self.rounds):
            state = [(b ^ self.key[r % len(self.key)]) & 0xFF for b in state]
            state = state[1:] + state[:1]
        return bytes(state)


def derive_key(password, salt, iterations=1000):
    key = bytearray(32)
    for i in range(iterations):
        for j in range(32):
            mixed = password[j % len(password)] + salt[j % len(salt)] + i
            key[j] = (key[j] + mixed) & 0xFF
        if i % 100 == 0:
            key = bytearray(reversed(key))
    checksum = 0
    for b in key:
        checksum = (checksum * 31 + b) & 0xFFFFFFFF
    return bytes(key), checksum


def tiny():
    return 1
'''

# Почти-копия derive_key: другие имена + одна лишняя строка в конце (sha отличается, отпечатки общие).
PY_NEAR_COPY = '''\
def kdf_variant(pw, nacl, rounds=1000):
    buf = bytearray(32)
    for a in range(rounds):
        for c in range(32):
            m = pw[c % len(pw)] + nacl[c % len(nacl)] + a
            buf[c] = (buf[c] + m) & 0xFF
        if a % 100 == 0:
            buf = bytearray(reversed(buf))
    crc = 0
    for q in buf:
        crc = (crc * 31 + q) & 0xFFFFFFFF
    crc ^= 0x5A5A
    return bytes(buf), crc
'''

PY_EXACT_COPY = '''\
"""Вендоренная копия защищённой функции."""


def derive_key(password, salt, iterations=1000):
    key = bytearray(32)
    for i in range(iterations):
        for j in range(32):
            mixed = password[j % len(password)] + salt[j % len(salt)] + i
            key[j] = (key[j] + mixed) & 0xFF
        if i % 100 == 0:
            key = bytearray(reversed(key))
    checksum = 0
    for b in key:
        checksum = (checksum * 31 + b) & 0xFFFFFFFF
    return bytes(key), checksum
'''

C_PROTECTED = '''\
#include <stdint.h>
#include <string.h>

int tunnel_checksum(const uint8_t *buf, size_t len);

int tunnel_checksum(const uint8_t *buf, size_t len) {
    uint32_t sum = 0;
    for (size_t i = 0; i < len; i++) {
        sum += buf[i];
        sum = (sum & 0xffff) + (sum >> 16);
    }
    return (int)(~sum & 0xffff);
}

static void tunnel_reset(struct tunnel *t) {
    t->state = 0;
    t->retries = 3;
    memset(t->buf, 0, sizeof(t->buf));
}
'''

C_UNRELATED = '''\
#include <stdio.h>

char *trim_spaces(char *s) {
    char *end;
    while (*s == ' ') {
        s++;
    }
    end = s + strlen(s) - 1;
    while (end > s && *end == ' ') {
        *end-- = '\\0';
    }
    return s;
}
'''

C_HEADER = '''\
#ifndef UTIL_H
#define UTIL_H
/* class of small helpers */
static inline int add3(int a, int b, int c) {
    int s = a + b;
    s += c;
    return s;
}
#endif
'''

CPP_HEADER = '''\
#pragma once
namespace util {
template <class T>
class Box {
public:
    Box(T v) : v_(v) {}
    T get() const {
        return v_;
    }
    void set(T v) {
        v_ = v;
        n_++;
    }
private:
    T v_;
    int n_ = 0;
};
inline int plain_c_like(int a, int b) {
    int c = a + b;
    c *= 2;
    return c;
}
}
'''

GO_PROTECTED = '''\
package vpn

func Handshake(peer *Peer, nonce []byte) error {
	if len(nonce) < 12 {
		return errInvalidNonce
	}
	peer.state = stateHandshake
	peer.nonce = append([]byte(nil), nonce...)
	return nil
}

func (p *Peer) Reset() {
	p.state = stateIdle
	p.nonce = nil
	p.retries = 0
}

var helper = func(x int) int {
	return x + 1
}
'''

JS_PUBLIC = '''\
function renderPage(title, items) {
    const parts = [];
    for (const it of items) {
        parts.push('<li>' + it + '</li>');
    }
    return '<h1>' + title + '</h1><ul>' + parts.join('') + '</ul>';
}

const shortArrow = (x) => x * 2;

const handler = (req, res) => {
    const body = renderPage('Home', req.items);
    res.setHeader('Content-Type', 'text/html');
    res.end(body);
};

class Router {
    add(path, fn) {
        this.routes = this.routes || {};
        this.routes[path] = fn;
        return this;
    }
}
'''

JS_UTIL = '''\
function clamp(value, lo, hi) {
    if (value < lo) {
        return lo;
    }
    if (value > hi) {
        return hi;
    }
    return value;
}
'''

JAVA_PUBLIC = '''\
package tools;

public class Formatter {
    private final int width;

    public Formatter(int width) {
        this.width = width;
        this.pad = ' ';
    }

    public String pad(String s) {
        StringBuilder sb = new StringBuilder(s);
        while (sb.length() < width) {
            sb.append(' ');
        }
        return sb.toString();
    }

    abstract void nothing(int x);
}
'''

CPP_PUBLIC = '''\
#include <vector>

namespace util {

class Stack {
public:
    void push(int v) {
        data_.push_back(v);
        size_++;
        total_ += v;
    }
    int pop();
private:
    std::vector<int> data_;
    int size_ = 0;
    int total_ = 0;
};

int Stack::pop() {
    int v = data_.back();
    data_.pop_back();
    size_--;
    return v;
}

}  // namespace util
'''


def _w(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_repos(raw: Path) -> None:
    """Мини-репозитории: protected/acme__vpn, hard_neg/other__sec, public_train/foo__web, public_calib/bar__tools."""
    p = raw / "protected" / "acme__vpn"
    _w(p / "src" / "crypto.py", PY_PROTECTED)
    _w(p / "src" / "tunnel.c", C_PROTECTED)
    _w(p / "pkg" / "handshake.go", GO_PROTECTED)
    _w(p / "tests" / "test_crypto.py", PY_PROTECTED)  # skip_dirs: tests
    _w(p / ".cloned", "")

    h = raw / "hard_neg" / "other__sec"
    _w(h / "src" / "kdf.py", PY_NEAR_COPY)
    _w(h / "src" / "strutil.c", C_UNRELATED)
    _w(h / "include" / "util.h", C_HEADER)  # репозиторий без C++ -> .h = c
    _w(h / "src" / "test_kdf.py", PY_NEAR_COPY)  # тестовый файл по имени — пропуск

    pub = raw / "public_train" / "foo__web"
    _w(pub / "lib" / "app.js", JS_PUBLIC)
    _w(pub / "lib" / "util.js", JS_UTIL)
    _w(pub / "lib" / "util_copy.js", JS_UTIL)  # точный дубликат внутри сплита
    _w(pub / "lib" / "vendored_derive.py", PY_EXACT_COPY)  # вендоренная копия protected вне vendor/
    _w(pub / "lib" / "crypto_utils.py", PY_NEAR_COPY)  # домен protected в пути -> пропуск в public_*
    _w(pub / "vendor" / "acme" / "tunnel.c", C_PROTECTED)  # skip_dirs: vendor
    _w(pub / "lib" / "bundle.min.js", JS_UTIL)  # минифицированный — пропуск
    _w(pub / "lib" / "gen.pb.go", GO_PROTECTED)  # сгенерированный — пропуск
    _w(pub / "lib" / "big.py", "def big():\n    x = 1\n" + "    x += 1\n" * 120000)  # > 1 МБ — пропуск
    _w(pub / "README.md", "# foo\n")

    cal = raw / "public_calib" / "bar__tools"
    _w(cal / "src" / "Formatter.java", JAVA_PUBLIC)
    _w(cal / "src" / "stack.cpp", CPP_PUBLIC)
    _w(cal / "include" / "box.h", CPP_HEADER)  # есть .cpp -> .h = cpp


def _cfg_with_paths(tmp_path: Path, overrides: dict | None = None) -> dict:
    ov = {
        "paths": {
            "raw_repos": str(tmp_path / "raw"),
            "functions": str(tmp_path / "functions"),
            "queries": str(tmp_path / "queries"),
            "indexes": str(tmp_path / "indexes"),
            "results": str(tmp_path / "results"),
            "figures": str(tmp_path / "figures"),
        }
    }
    if overrides:
        ov.update(overrides)
    return load_config(overrides=ov)


@pytest.fixture()
def cfg(tmp_path: Path) -> dict:
    c = _cfg_with_paths(tmp_path)
    make_repos(tmp_path / "raw")
    return c


# ----------------------------------------------------------------------------- repos.yaml / splits


def test_repos_yaml_sanity():
    data = list_repos.load_repos_yaml()
    pool = data["protected_pool"]
    public = data["public"]
    assert len(pool) >= 30
    names = {e["repo"] for e in pool}
    for must in ("OpenVPN/openvpn", "strongswan/strongswan", "jedisct1/libsodium", "pyca/cryptography", "netty/netty"):
        assert must in names
    # каждый язык представлен в пуле минимум дважды (нужно для стратификации protected/hard_neg)
    for lang in LANGS:
        assert sum(1 for e in pool if e["lang"] == lang) >= 2, lang
    assert len(public) >= 150
    assert {e["lang"] for e in public} == LANGS
    all_names = [e["repo"].lower() for e in public]
    assert len(all_names) == len(set(all_names)), "дубликаты в public"
    assert not ({n.lower() for n in names} & set(all_names))
    assert "thealgorithms/java" not in all_names  # учебные реализации шифров — домен protected
    kws = data["domain_keywords"]
    bad = [(e["repo"], list_repos.matches_domain(list_repos.entry_text(e), kws)) for e in public]
    bad = [b for b in bad if b[1]]
    assert not bad, f"public содержит репозитории домена protected: {bad}"
    # составные ключевые слова ловят большинство protected-репозиториев уже по одному имени
    hits = sum(1 for e in pool if list_repos.matches_domain(list_repos.entry_text(e), kws))
    assert hits >= 20
    for name in ("OpenVPN/openvpn", "libssh2/libssh2", "apache/mina-sshd", "mscdex/ssh2"):
        assert list_repos.matches_domain(name, kws), name


def test_matches_domain():
    kws = ["crypto", "ids", "ssl", "firewall"]
    assert list_repos.matches_domain("pyca/cryptography", kws) == "crypto"
    assert list_repos.matches_domain("openssl/openssl", kws) is None  # 'ssl' (3 буквы) — только как отдельное слово
    assert list_repos.matches_domain("openssl/openssl", kws + ["openssl"]) == "openssl"
    assert list_repos.matches_domain("nice/grids", kws) is None  # 'ids' только как отдельное слово
    assert list_repos.matches_domain("snort/ids-engine", kws) == "ids"
    assert list_repos.matches_domain("pallets/flask web framework", kws) is None


def test_protected_quotas():
    q = list_repos.protected_quotas({"c": 8, "cpp": 4, "go": 4, "python": 5, "java": 7, "javascript": 8}, 12)
    assert sum(q.values()) == 12 and all(v >= 1 for v in q.values())
    q = list_repos.protected_quotas({"c": 10, "js": 1}, 3)
    assert q == {"c": 2, "js": 1}  # единственный js уходит в protected
    q = list_repos.protected_quotas({"a": 2, "b": 2, "c": 2}, 2)
    assert sum(q.values()) == 2 and max(q.values()) == 1  # языков больше, чем мест
    q = list_repos.protected_quotas({"a": 3}, 10)
    assert q == {"a": 3}


def test_split_repos_deterministic(cfg):
    data = list_repos.load_repos_yaml()
    s1 = list_repos.split_repos(cfg, data, persist=False)
    s2 = list_repos.split_repos(cfg, data, persist=False)
    assert s1 == s2
    assert set(s1) == set(SPLITS)
    assert len(s1["protected"]) == 12
    assert len(s1["hard_neg"]) == len(data["protected_pool"]) - 12
    pool = {e["repo"] for e in data["protected_pool"]}
    assert {e["repo"] for e in s1["protected"]} | {e["repo"] for e in s1["hard_neg"]} == pool
    pub = [e["repo"] for s in list_repos.PUBLIC_SPLITS for e in s1[s]]
    assert len(pub) == len(set(pub)) == len(data["public"])
    n = len(pub)
    assert abs(len(s1["public_train"]) / n - 0.6) < 0.08
    assert abs(len(s1["public_calib"]) / n - 0.2) < 0.08
    assert len(s1["public_test"]) > 0
    # стратификация: в каждом сплите (включая protected и hard_neg) есть все языки
    for s in SPLITS:
        assert {e["lang"] for e in s1[s]} == LANGS, s
    # ... при любом seed
    for seed in range(30):
        s = list_repos.split_repos(dict(cfg, seed=seed), data, persist=False)
        assert len(s["protected"]) == 12
        assert {e["lang"] for e in s["protected"]} == LANGS and {e["lang"] for e in s["hard_neg"]} == LANGS, seed
    # другой seed даёт другой выбор protected
    s3 = list_repos.split_repos(dict(cfg, seed=1), data, persist=False)
    assert s3["protected"] != s1["protected"]
    # n_protected из cfg.splits имеет приоритет
    s4 = list_repos.split_repos(dict(cfg, splits=dict(cfg["splits"], n_protected=15)), data, persist=False)
    assert len(s4["protected"]) == 15


def test_split_repos_persist_and_stale(cfg, tmp_path):
    data = list_repos.load_repos_yaml()
    s1 = list_repos.split_repos(cfg, data, persist=True)
    path = tmp_path / "functions" / "repo_splits.json"
    assert path.exists()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["seed"] == cfg["seed"] and saved["repos_sha1"] == list_repos.repos_digest(data)
    assert saved["splits"] == s1
    assert list_repos.split_repos(cfg, data, persist=True) == s1  # те же параметры -> сохранённое
    assert list_repos.load_repo_splits(cfg) == s1
    # изменился YAML -> ошибка без force
    data2 = dict(data, public=data["public"][:10])
    with pytest.raises(RuntimeError, match="repos.yaml"):
        list_repos.split_repos(cfg, data2, persist=True)
    # изменился seed -> ошибка без force
    with pytest.raises(RuntimeError, match="seed"):
        list_repos.split_repos(dict(cfg, seed=1), data, persist=True)
    assert json.loads(path.read_text(encoding="utf-8"))["splits"] == s1  # файл не тронут
    # force пересчитывает и перезаписывает
    s3 = list_repos.split_repos(cfg, data2, persist=True, force=True)
    assert len(s3["public_train"]) + len(s3["public_calib"]) + len(s3["public_test"]) == 10
    assert list_repos.split_repos(cfg, data2, persist=True) == s3
    lines = list(list_repos.iter_split_repo_lines(s3))
    assert all(len(ln.split(" ")) == 2 and ln.split(" ")[0] in SPLITS for ln in lines)
    assert list_repos.repo_dir_name("OpenVPN/openvpn") == "OpenVPN__openvpn"
    assert list_repos.repo_from_dir_name("OpenVPN__openvpn") == "OpenVPN/openvpn"
    # CLI: сохранённый файл построен по data2, а repos.yaml на диске другой -> код возврата 3;
    # --force пересчитывает и печатает строки "<split> <owner>/<name>"
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}, allow_unicode=True), encoding="utf-8")
    assert list_repos.main(["--config", str(cfg_path), "--print-splits"]) == 3
    assert list_repos.main(["--config", str(cfg_path), "--print-splits", "--force"]) == 0
    assert list_repos.load_repo_splits(cfg) == list_repos.split_repos(cfg, data, persist=False)


# ----------------------------------------------------------------------------- windows


def test_file_windows():
    assert windows.file_windows(0) == []
    assert windows.file_windows(5) == [(1, 5)]
    assert windows.file_windows(20) == [(1, 20)]
    assert windows.file_windows(25) == [(1, 20), (6, 25)]
    assert windows.file_windows(40) == [(1, 20), (11, 30), (21, 40)]
    assert windows.file_windows(45) == [(1, 20), (11, 30), (21, 40), (26, 45)]
    assert windows.file_windows(7, size=3, stride=2) == [(1, 3), (3, 5), (5, 7)]


def test_windows_from_source(cfg):
    recs = windows.windows_from_source(PY_PROTECTED, "python", "protected", "acme/vpn", "acme__vpn", "src/crypto.py", cfg)
    n = PY_PROTECTED.count("\n")
    expected = windows.file_windows(n)
    assert [(r.start_line, r.end_line) for r in recs] == expected
    r0 = recs[0]
    assert r0.kind == "window" and r0.id == "protected/acme__vpn/src/crypto.py:w1-20"
    assert r0.n_lines == 20 and r0.n_tokens > 0 and r0.sha == normalized_sha(r0.code, "python")


# ----------------------------------------------------------------------------- extraction per language


def _ex(text: str, lang: str, cfg: dict, name: str | None = None):
    return extract.extract_records_from_source(text, lang, "public_train", "o/n", "o__n", name or f"f.{lang}", cfg)


def test_extract_python(cfg):
    recs = _ex(PY_PROTECTED, "python", cfg)
    names = [r.code.split("(")[0] for r in recs]
    assert names == ["def __init__", "def encrypt_block", "def derive_key"]  # tiny() < min_lines
    enc = recs[1]
    assert enc.code.startswith("def encrypt_block(self, block):\n    state")  # dedent
    assert (enc.start_line, enc.end_line) == (11, 16) and enc.n_lines == 6
    assert enc.id == f"public_train/o__n/f.python:{enc.start_line}-{enc.end_line}"
    assert enc.sha == normalized_sha(enc.code, "python") and enc.n_tokens > 10


def test_extract_c_cpp(cfg):
    recs = _ex(C_PROTECTED, "c", cfg)
    assert [r.code.split("(")[0].split()[-1] for r in recs] == ["tunnel_checksum", "tunnel_reset"]  # прототип исключён
    assert recs[0].start_line == 6 and recs[0].end_line == 13
    recs = _ex(CPP_PUBLIC, "cpp", cfg)
    heads = [r.code.splitlines()[0] for r in recs]
    assert heads == ["void push(int v) {", "int Stack::pop() {"]  # объявление pop() без тела исключено


def test_extract_go_java_js(cfg):
    recs = _ex(GO_PROTECTED, "go", cfg)
    assert [r.code.splitlines()[0] for r in recs] == ["func Handshake(peer *Peer, nonce []byte) error {", "func (p *Peer) Reset() {"]
    recs = _ex(JAVA_PUBLIC, "java", cfg)
    assert [r.code.splitlines()[0] for r in recs] == ["public Formatter(int width) {", "public String pad(String s) {"]
    recs = _ex(JS_PUBLIC, "javascript", cfg)
    heads = [r.code.splitlines()[0] for r in recs]
    assert heads == ["function renderPage(title, items) {", "const handler = (req, res) => {", "add(path, fn) {"]
    assert recs[1].code.rstrip().endswith("};")


def test_extract_bounds_and_crlf(cfg):
    cfg2 = dict(cfg, extract=dict(cfg["extract"], max_lines=7))
    recs = _ex(C_PROTECTED, "c", cfg2)
    assert [r.n_lines for r in recs] == [5]  # tunnel_checksum (8 строк) отсечён по max_lines
    cfg3 = dict(cfg, extract=dict(cfg["extract"], max_tokens=20))
    assert all(r.n_tokens <= 20 for r in _ex(PY_PROTECTED, "python", cfg3))
    crlf = C_PROTECTED.replace("\n", "\r\n")
    a = _ex(crlf, "c", cfg)
    b = _ex(C_PROTECTED, "c", cfg)
    assert [(r.start_line, r.end_line, r.code) for r in a] == [(r.start_line, r.end_line, r.code) for r in b]


def test_cpp_header_parsed_as_c_has_no_namespace_blob(cfg):
    """Регрессия: C++-заголовок под C-грамматикой давал одну «функцию» на весь namespace."""
    recs_c = _ex(CPP_HEADER, "c", cfg, "box.h")
    assert recs_c and not any(r.code.startswith("namespace") for r in recs_c)
    assert all(r.n_lines <= 6 for r in recs_c)
    recs_cpp = _ex(CPP_HEADER, "cpp", cfg, "box.h")
    heads = [r.code.splitlines()[0] for r in recs_cpp]
    assert heads == ["T get() const {", "void set(T v) {", "inline int plain_c_like(int a, int b) {"]
    assert all(r.lang == "cpp" for r in recs_cpp)
    assert extract.looks_like_cpp(CPP_HEADER.encode()) and not extract.looks_like_cpp(C_HEADER.encode())


def test_header_language_per_repo(cfg, tmp_path):
    cpp_repo = tmp_path / "raw" / "public_calib" / "bar__tools"
    files = dict(extract.iter_source_files(cpp_repo, cfg))
    assert files["include/box.h"] == "cpp" and files["src/stack.cpp"] == "cpp"
    c_repo = tmp_path / "raw" / "hard_neg" / "other__sec"
    st: dict = {}
    files = dict(extract.iter_source_files(c_repo, cfg, split="hard_neg", stats=st))
    assert files["include/util.h"] == "c" and "src/test_kdf.py" not in files
    assert st == {"skipped_test_name": 1}
    assert dict(extract.iter_source_files(cpp_repo, cfg, header_lang="c"))["include/box.h"] == "c"
    # C-репозиторий с C++-содержимым в .h: определяется по содержимому в воркере
    task = ("public_test", "o/n", "o__n", str(cpp_repo), "include/box.h", "c", extract._worker_cfg(cfg), False)
    funcs, _, info = extract._extract_file_task(task)
    assert funcs and all(r["lang"] == "cpp" for r in funcs) and info["headers_sniffed_cpp"] == 1
    task = ("hard_neg", "o/s", "o__s", str(c_repo), "include/util.h", "c", extract._worker_cfg(cfg), False)
    funcs, _, info = extract._extract_file_task(task)
    assert [r["lang"] for r in funcs] == ["c"] and info["headers_sniffed_cpp"] == 0


def test_duplicate_span_ids_unique(cfg):
    cfg1 = dict(cfg, extract=dict(cfg["extract"], min_lines=1))
    src = "int x;\nint a1(void){return 1;} int a2(void){return 2;}\n"
    recs = _ex(src, "c", cfg1, "f.c")
    assert [r.id for r in recs] == ["public_train/o__n/f.c:2-2", "public_train/o__n/f.c:2-2#2"]
    assert [r.code for r in recs] == ["int a1(void){return 1;}", "int a2(void){return 2;}"]
    assert extract.function_id("s", "o__n", "f.c", 2, 2, 3) == "s/o__n/f.c:2-2#3"


def test_dedent_by_column(cfg):
    src = (
        "class A:\n"
        "    def q(self):\n"
        '        s = """\n'
        "SELECT *\n"
        "FROM t\n"
        '"""\n'
        "        return s\n"
        "\n"
        "    def r(self):\n"
        "        x = 1\n"
        "        return x\n"
    )
    recs = _ex(src, "python", cfg, "a.py")
    assert recs[0].code == 'def q(self):\n    s = """\nSELECT *\nFROM t\n"""\n    return s'
    assert recs[1].code == "def r(self):\n    x = 1\n    return x"
    assert extract.dedent_by_prefix("  a\n\n    b\nc", "  ") == "a\n\n  b\nc"


def test_name_filters():
    gen = ("lex.yy.c", "a.pb.go", "a.pb.gw.go", "x_pb2.py", "x_pb2_grpc.py", "foo.min.js", "parser.tab.c",
           "zz_generated.deepcopy.go", "api_generated.go", "my_wrap.cxx", "types.d.ts")
    assert all(extract.is_generated_name(n) for n in gen), [n for n in gen if not extract.is_generated_name(n)]
    hand = ("json_lexer.c", "swigutil.c", "utils_string.go", "tokenizer_gen.go", "generated_at.py", "wrapper.c", "minify.js")
    assert not any(extract.is_generated_name(n) for n in hand), [n for n in hand if extract.is_generated_name(n)]
    tests = ("a_test.go", "test_x.py", "x_test.py", "x_tests.py", "FooTest.java", "FooTests.java", "a.test.js", "b.spec.mjs", "conftest.py")
    assert all(extract.is_test_name(n) for n in tests)
    assert not any(extract.is_test_name(n) for n in ("attest.go", "latest.py", "Contest.java", "testing.c", "protest.js"))
    assert extract.is_generated_head(b"/* Generated by re2c 3.0 */\nint x;\n")
    assert extract.is_generated_head(b"// Code generated by protoc-gen-go. DO NOT EDIT.\npackage x\n")
    assert not extract.is_generated_head(b"// handles events\n// from the kernel\nint x;\n")
    hits = {p: extract.domain_path_match(p) for p in ("libavutil/aes.c", "src/crypt.c", "transports/ssh.c", "streams/openssl.c",
                                                    "src/tls.c", "common/sha2.c", "io/SslContext.java", "core/crypto.js",
                                                    "ciphers/AES.java", "modules/ssh/init.go", "ext/openssl/xp_ssl.c")}
    assert all(hits.values()), hits
    assert not any(extract.domain_path_match(p) for p in ("lib/universal.js", "src/aesthetic.c", "src/describe.c",
                                                           "net/http/server.go", "lib/app.js", "design/destroy.py"))


def test_iter_source_files_filters(cfg, tmp_path):
    repo = tmp_path / "raw" / "public_train" / "foo__web"
    st: dict = {}
    files = dict(extract.iter_source_files(repo, cfg, split="public_train", stats=st))
    assert "lib/app.js" in files and files["lib/app.js"] == "javascript"
    assert "lib/vendored_derive.py" in files
    assert "lib/crypto_utils.py" not in files  # домен protected в пути (только public_*)
    assert not any(p.startswith("vendor/") for p in files)
    assert "lib/bundle.min.js" not in files and "lib/gen.pb.go" not in files
    assert "lib/big.py" not in files and "README.md" not in files
    assert st == {"skipped_domain_path": 1, "skipped_generated_name": 2, "skipped_big": 1}
    # для hard_neg и при выключенном фильтре файл остаётся
    assert "lib/crypto_utils.py" in dict(extract.iter_source_files(repo, cfg, split="hard_neg"))
    cfg_off = dict(cfg, extract=dict(cfg["extract"], domain_path_filter=False))
    assert "lib/crypto_utils.py" in dict(extract.iter_source_files(repo, cfg_off, split="public_train"))
    assert extract.lang_for_path("x.hpp") == "cpp" and extract.lang_for_path("x.txt") is None


def test_non_utf8_filename_does_not_crash(tmp_path):
    cfg = _cfg_with_paths(tmp_path)
    repo = tmp_path / "raw" / "protected" / "o__n"
    _w(repo / "ok.c", C_UNRELATED)
    try:
        with open(os.path.join(os.fsencode(repo), b"caf\xe9.c"), "wb") as f:
            f.write(C_PROTECTED.encode())
    except (OSError, ValueError):
        pytest.skip("файловая система не допускает имена не в UTF-8")
    st: dict = {}
    files = list(extract.iter_source_files(repo, cfg, stats=st))
    assert files == [("ok.c", "c")] and st == {"skipped_bad_name": 1}
    out = extract.run_extract(cfg, splits=["protected"], workers=1)
    fdir = tmp_path / "functions"
    assert out["protected"] == fdir / "protected.jsonl" and (fdir / "protected_windows.jsonl").exists()
    assert not list(fdir.glob("*.tmp"))
    recs = list(read_functions(fdir / "protected.jsonl"))
    assert [r.path for r in recs] == ["ok.c"]
    stats = json.loads((fdir / "extract_stats.json").read_text(encoding="utf-8"))
    assert stats["protected"]["skipped_bad_name"] == 1 and stats["protected"]["errors"] == 0


def test_measure_repo(cfg, tmp_path, capsys):
    repo = tmp_path / "raw" / "public_train" / "foo__web"
    raw, eff = extract.measure_repo(repo, cfg)
    big = (repo / "lib" / "big.py").stat().st_size
    vend = (repo / "vendor" / "acme" / "tunnel.c").stat().st_size
    assert big > extract.MAX_FILE_BYTES
    assert raw == sum(p.stat().st_size for p in repo.rglob("*") if p.is_file())
    assert eff == raw - big - vend
    assert extract.main(["--measure", str(repo)]) == 0
    out = capsys.readouterr().out.split()
    assert [int(x) for x in out] == [-(-raw // (1 << 20)), -(-eff // (1 << 20))]


# ----------------------------------------------------------------------------- end-to-end: extract + dedup


def test_pipeline_extract_dedup(cfg, tmp_path):
    fdir = tmp_path / "functions"
    outputs = extract.run_extract(cfg, workers=1)
    assert set(outputs) == {"protected", "hard_neg", "public_train", "public_calib", "protected_windows"}
    assert not (fdir / "public_test.jsonl").exists()  # клонов этого сплита нет

    prot = list(read_functions(fdir / "protected.jsonl"))
    prot_paths = {r.path for r in prot}
    assert prot_paths == {"src/crypto.py", "src/tunnel.c", "pkg/handshake.go"}  # tests/ пропущен
    assert {r.lang for r in prot} == {"python", "c", "go"}
    assert all(r.split == "protected" and r.repo == "acme/vpn" and r.kind == "function" for r in prot)
    assert all(r.id.startswith("protected/acme__vpn/") and r.id.endswith(f":{r.start_line}-{r.end_line}") for r in prot)
    assert sum(1 for r in prot if r.lang == "python") == 3
    assert sum(1 for r in prot if r.lang == "c") == 2
    assert sum(1 for r in prot if r.lang == "go") == 2  # func_literal не извлекается
    ids = [r.id for r in prot]
    assert len(ids) == len(set(ids))
    # порядок детерминирован: пути отсортированы, функции по позиции
    keys = [(r.path, r.start_line) for r in prot]
    assert keys == sorted(keys)

    wins = list(read_functions(fdir / "protected_windows.jsonl"))
    assert wins and all(r.kind == "window" and ":w" in r.id for r in wins)
    assert {r.path for r in wins} == prot_paths

    pub = list(read_functions(fdir / "public_train.jsonl"))
    pub_paths = sorted({r.path for r in pub})
    assert pub_paths == ["lib/app.js", "lib/util.js", "lib/util_copy.js", "lib/vendored_derive.py"]
    assert sum(1 for r in pub if r.path == "lib/app.js") == 3
    hard = list(read_functions(fdir / "hard_neg.jsonl"))
    assert {r.path for r in hard} == {"src/kdf.py", "src/strutil.c", "include/util.h"}
    assert {r.lang for r in hard if r.path == "include/util.h"} == {"c"}
    calib = list(read_functions(fdir / "public_calib.jsonl"))
    assert sorted(r.lang for r in calib) == ["cpp"] * 5 + ["java"] * 2
    assert {r.lang for r in calib if r.path == "include/box.h"} == {"cpp"}
    xs = json.loads((fdir / "extract_stats.json").read_text(encoding="utf-8"))
    assert xs["public_train"]["skipped_domain_path"] == 1 and xs["public_calib"]["headers_as_cpp"] == 1
    assert xs["hard_neg"]["skipped_test_name"] == 1 and xs["hard_neg"]["skipped_domain_path"] == 0

    # идемпотентность: повторный запуск не переписывает файлы
    mtime = (fdir / "protected.jsonl").stat().st_mtime_ns
    extract.run_extract(cfg, workers=1)
    assert (fdir / "protected.jsonl").stat().st_mtime_ns == mtime

    # ---- dedup
    stats = dedup_split.run_dedup(cfg, workers=1)
    assert (fdir / "stats.json").exists() and (fdir / "dedup_removed.jsonl").exists()
    assert stats["config"]["exact_dedup_key"] == "normalized_sha" and stats["config"]["sha_protected_min_tokens"] == cfg["dedup"]["k"]
    sp = stats["splits"]
    assert sp["protected"]["n_raw"] == 7 and sp["protected"]["n_final"] == 7 and sp["protected"]["removed_near_dup"] == 0
    assert stats["protected_fingerprints"] > 0
    # public_train: точный дубликат clamp (util_copy.js) + вендоренная копия derive_key
    assert sp["public_train"]["removed_exact"] == 1
    assert sp["public_train"]["removed_near_dup"] == 1 and sp["public_train"]["removed_sha_protected"] == 1
    assert sp["public_train"]["removed_by_repo"] == {"foo/web": 2}
    pub2 = list(read_functions(fdir / "public_train.jsonl"))
    assert {r.path for r in pub2} == {"lib/app.js", "lib/util.js"}
    assert len(pub2) == 4
    # hard_neg: почти-копия удалена по отпечаткам (sha другой), несвязанные функции остались
    assert sp["hard_neg"]["removed_near_dup"] == 1 and sp["hard_neg"]["removed_exact"] == 0
    hard2 = list(read_functions(fdir / "hard_neg.jsonl"))
    assert [r.path for r in hard2] == ["include/util.h", "src/strutil.c"]
    removed = [json.loads(l) for l in (fdir / "dedup_removed.jsonl").read_text(encoding="utf-8").splitlines()]
    reasons = {(r["split"], r["reason"]) for r in removed}
    assert ("hard_neg", "near_dup") in reasons and ("public_train", "sha_protected") in reasons and ("public_train", "exact") in reasons
    near = [r for r in removed if r["reason"] == "near_dup"][0]
    assert near["id"].startswith("hard_neg/other__sec/src/kdf.py:") and 0.3 <= near["overlap"] <= 1.0
    # статистика по языкам и бинам длины
    assert sp["public_calib"]["by_lang"] == {"cpp": 5, "java": 2}
    assert sum(sp["protected"]["by_length_bin"].values()) == 7
    assert list(sp["protected"]["by_length_bin"]) == ["0-32", "32-64", "64-128", "128-256", "256-100000"]
    assert stats["windows"]["n_final"] == len(wins) - stats["windows"]["removed_exact"]
    assert stats["totals"]["n_final"] == 7 + 2 + 4 + 7
    assert set(stats["inputs"]) == {"protected.jsonl", "hard_neg.jsonl", "public_train.jsonl", "public_calib.jsonl", "protected_windows.jsonl"}
    # идемпотентность dedup: повторный вызов возвращает сохранённую статистику без изменений
    stats2 = dedup_split.run_dedup(cfg, workers=1)
    assert stats2 == stats
    prot2 = list(read_functions(fdir / "protected.jsonl"))
    assert [r.id for r in prot2] == [r.id for r in prot]

    # ---- устаревший stats.json: 01 --force удаляет маркер, 02 пересчитывает
    extract.run_extract(cfg, splits=["hard_neg"], workers=1, force=True)
    assert not (fdir / "stats.json").exists() and not (fdir / "dedup_removed.jsonl").exists()
    stats3 = dedup_split.run_dedup(cfg, workers=1)
    assert stats3["splits"]["hard_neg"]["n_raw"] == 3 and stats3["splits"]["hard_neg"]["n_final"] == 2
    assert dedup_split.stats_fresh(fdir) == (True, "")
    # входной файл новее stats.json (например, скопирован извне) -> не fresh
    time.sleep(0.01)
    os.utime(fdir / "public_calib.jsonl")
    fresh, why = dedup_split.stats_fresh(fdir)
    assert not fresh and "public_calib.jsonl" in why
    stats4 = dedup_split.run_dedup(cfg, workers=1)
    assert stats4["splits"]["public_calib"]["n_raw"] == 7 and dedup_split.stats_fresh(fdir)[0]


def test_dedup_short_type2_clones_kept(tmp_path):
    """Короткие структурные клоны (одинаковый normalized sha, n_tokens < k) остаются в негативах;
    длинные совпадения по sha удаляются как sha_protected."""
    cfg = _cfg_with_paths(tmp_path)
    fdir = tmp_path / "functions"
    short_p = "int f(void) {\n    return 1;\n}\n"
    short_h = "int h(void) {\n    return 2;\n}\n"
    long_p = C_UNRELATED
    long_h = C_UNRELATED.replace("trim_spaces", "strip_ws").replace("end", "tail")
    prot = extract.extract_records_from_source(short_p + "\n" + long_p, "c", "protected", "a/p", "a__p", "p.c", cfg)
    hard = extract.extract_records_from_source(short_h + "\n" + long_h, "c", "hard_neg", "b/h", "b__h", "h.c", cfg)
    assert len(prot) == 2 and len(hard) == 2
    assert prot[0].sha == hard[0].sha and prot[1].sha == hard[1].sha  # тип-2 клоны
    assert hard[0].n_tokens < cfg["dedup"]["k"] <= hard[1].n_tokens
    write_jsonl(fdir / "protected.jsonl", prot)
    write_jsonl(fdir / "hard_neg.jsonl", hard)
    stats = dedup_split.run_dedup(cfg, workers=1)
    hs = stats["splits"]["hard_neg"]
    assert hs["n_final"] == 1 and hs["removed_sha_protected"] == 1 and hs["kept_short_sha_protected"] == 1
    kept = list(read_functions(fdir / "hard_neg.jsonl"))
    assert [r.code.splitlines()[0] for r in kept] == ["int h(void) {"]
    removed = [json.loads(l) for l in (fdir / "dedup_removed.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(r["reason"], r["id"]) for r in removed] == [("sha_protected", hard[1].id)]


def test_run_extract_multiprocessing(cfg, tmp_path):
    outputs = extract.run_extract(cfg, splits=["protected", "public_train"], workers=2, with_windows=False)
    assert set(outputs) == {"protected", "public_train"}
    seq_dir = tmp_path / "functions_seq"
    cfg2 = dict(cfg, paths=dict(cfg["paths"], functions=str(seq_dir)))
    extract.run_extract(cfg2, splits=["protected", "public_train"], workers=1, with_windows=False)
    for s in ("protected", "public_train"):
        a = (tmp_path / "functions" / f"{s}.jsonl").read_text(encoding="utf-8")
        b = (seq_dir / f"{s}.jsonl").read_text(encoding="utf-8")
        assert a == b


def test_length_bin_label():
    bins = [[0, 32], [32, 64], [64, 100000]]
    assert dedup_split.length_bin_label(0, bins) == "0-32"
    assert dedup_split.length_bin_label(32, bins) == "32-64"
    assert dedup_split.length_bin_label(10**6, bins) == "64-100000"


# ----------------------------------------------------------------------------- scripts


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash недоступен")
def test_clone_script_dry_run(cfg, tmp_path):
    """00_clone.sh --dry-run: список репозиториев по сплитам без сети; repo_splits.json сохраняется."""
    cfg_path = tmp_path / "cfg.yaml"
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    cfg_path.write_text(yaml.safe_dump(clean, allow_unicode=True), encoding="utf-8")
    env = dict(os.environ, PYTHON=PY)
    res = subprocess.run(["bash", str(ROOT / "scripts" / "00_clone.sh"), "--config", str(cfg_path), "--dry-run"],
                         capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=120)
    assert res.returncode == 0, res.stderr
    lines = [ln for ln in res.stdout.splitlines() if " " in ln and ln.split(" ")[0] in SPLITS]
    assert len({ln.split(" ")[0] for ln in lines}) == 5
    assert (tmp_path / "raw" / "repo_list.txt").read_text(encoding="utf-8").splitlines() == lines
    assert (tmp_path / "functions" / "repo_splits.json").exists()
    # повторный прогон с другим seed без --force завершается ошибкой (устаревший repo_splits.json)
    cfg_path.write_text(yaml.safe_dump(dict(clean, seed=7), allow_unicode=True), encoding="utf-8")
    res2 = subprocess.run(["bash", str(ROOT / "scripts" / "00_clone.sh"), "--config", str(cfg_path), "--dry-run"],
                          capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=120)
    assert res2.returncode != 0 and "устарел" in (res2.stderr + res2.stdout)


def test_scripts_cli(cfg, tmp_path):
    """Тонкие обёртки 01/02 через main() с временным конфигом."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib

    s01 = importlib.import_module("01_extract")
    s02 = importlib.import_module("02_dedup_split")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}, allow_unicode=True), encoding="utf-8")
    assert s01.main(["--config", str(cfg_path), "--workers", "1", "--no-domain-filter"]) == 0
    pub = list(read_functions(tmp_path / "functions" / "public_train.jsonl"))
    assert "lib/crypto_utils.py" in {r.path for r in pub}  # фильтр домена выключен флагом
    assert s02.main(["--config", str(cfg_path), "--workers", "1"]) == 0
    assert (tmp_path / "functions" / "stats.json").exists()
