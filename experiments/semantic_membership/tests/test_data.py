"""Тесты модуля данных (smcode.data): списки репозиториев и сплиты, извлечение tree-sitter,
файловые окна, дедупликация. Синтетические мини-репозитории в tmp_path; без сети и torch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smcode.config import load_config
from smcode.data import dedup_split, extract, list_repos, windows
from smcode.normalize import normalized_sha
from smcode.types import SPLITS, read_functions

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

    pub = raw / "public_train" / "foo__web"
    _w(pub / "lib" / "app.js", JS_PUBLIC)
    _w(pub / "lib" / "util.js", JS_UTIL)
    _w(pub / "lib" / "util_copy.js", JS_UTIL)  # точный дубликат внутри сплита
    _w(pub / "lib" / "vendored_crypto.py", PY_EXACT_COPY)  # вендоренная копия protected вне vendor/
    _w(pub / "vendor" / "acme" / "tunnel.c", C_PROTECTED)  # skip_dirs: vendor
    _w(pub / "lib" / "bundle.min.js", JS_UTIL)  # минифицированный — пропуск
    _w(pub / "lib" / "gen.pb.go", GO_PROTECTED)  # сгенерированный — пропуск
    _w(pub / "lib" / "big.py", "def big():\n    x = 1\n" + "    x += 1\n" * 120000)  # > 1 МБ — пропуск
    _w(pub / "README.md", "# foo\n")

    cal = raw / "public_calib" / "bar__tools"
    _w(cal / "src" / "Formatter.java", JAVA_PUBLIC)
    _w(cal / "src" / "stack.cpp", CPP_PUBLIC)


@pytest.fixture()
def cfg(tmp_path: Path) -> dict:
    c = load_config(
        overrides={
            "paths": {
                "raw_repos": str(tmp_path / "raw"),
                "functions": str(tmp_path / "functions"),
                "queries": str(tmp_path / "queries"),
                "indexes": str(tmp_path / "indexes"),
                "results": str(tmp_path / "results"),
                "figures": str(tmp_path / "figures"),
            }
        }
    )
    make_repos(tmp_path / "raw")
    return c


# ----------------------------------------------------------------------------- repos.yaml / splits


def test_repos_yaml_sanity():
    data = list_repos.load_repos_yaml()
    pool = data["protected_pool"]
    public = data["public"]
    assert len(pool) >= 20
    names = {e["repo"] for e in pool}
    for must in ("OpenVPN/openvpn", "strongswan/strongswan", "jedisct1/libsodium", "pyca/cryptography", "netty/netty"):
        assert must in names
    assert len(public) >= 150
    langs = {e["lang"] for e in public}
    assert langs == {"python", "c", "cpp", "go", "java", "javascript"}
    all_names = [e["repo"].lower() for e in public]
    assert len(all_names) == len(set(all_names)), "дубликаты в public"
    assert not ({n.lower() for n in names} & set(all_names))
    kws = data["domain_keywords"]
    bad = [(e["repo"], list_repos.matches_domain(list_repos.entry_text(e), kws)) for e in public]
    bad = [b for b in bad if b[1]]
    assert not bad, f"public содержит репозитории домена protected: {bad}"
    # ключевые слова ловят заметную часть protected-репозиториев уже по одному имени
    # (в GitHub-поиске дополнительно используются описание и темы)
    hits = sum(1 for e in pool if list_repos.matches_domain(list_repos.entry_text(e), kws))
    assert hits >= 8


def test_matches_domain():
    kws = ["crypto", "ids", "ssl", "firewall"]
    assert list_repos.matches_domain("pyca/cryptography", kws) == "crypto"
    assert list_repos.matches_domain("openssl/openssl", kws) is None  # 'ssl' (3 буквы) — только как отдельное слово
    assert list_repos.matches_domain("openssl/openssl", kws + ["openssl"]) == "openssl"
    assert list_repos.matches_domain("nice/grids", kws) is None  # 'ids' только как отдельное слово
    assert list_repos.matches_domain("snort/ids-engine", kws) == "ids"
    assert list_repos.matches_domain("pallets/flask web framework", kws) is None


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
    # стратификация: в каждом публичном сплите есть все языки
    for s in list_repos.PUBLIC_SPLITS:
        assert {e["lang"] for e in s1[s]} == {"python", "c", "cpp", "go", "java", "javascript"}
    # другой seed даёт другой выбор protected
    cfg2 = dict(cfg, seed=1)
    s3 = list_repos.split_repos(cfg2, data, persist=False)
    assert s3["protected"] != s1["protected"]


def test_split_repos_persist(cfg, tmp_path):
    data = list_repos.load_repos_yaml()
    s1 = list_repos.split_repos(cfg, data, persist=True)
    path = tmp_path / "functions" / "repo_splits.json"
    assert path.exists()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["seed"] == cfg["seed"]
    assert saved["splits"] == s1
    # без force возвращается сохранённое, даже если YAML поменялся
    data2 = dict(data, public=data["public"][:10])
    s2 = list_repos.split_repos(cfg, data2, persist=True)
    assert s2 == s1
    s3 = list_repos.split_repos(cfg, data2, persist=True, force=True)
    assert len(s3["public_train"]) + len(s3["public_calib"]) + len(s3["public_test"]) == 10
    lines = list(list_repos.iter_split_repo_lines(s3))
    assert all(len(ln.split(" ")) == 2 and ln.split(" ")[0] in SPLITS for ln in lines)
    assert list_repos.repo_dir_name("OpenVPN/openvpn") == "OpenVPN__openvpn"
    assert list_repos.repo_from_dir_name("OpenVPN__openvpn") == "OpenVPN/openvpn"


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


def _ex(text: str, lang: str, cfg: dict):
    return extract.extract_records_from_source(text, lang, "public_train", "o/n", "o__n", f"f.{lang}", cfg)


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


def test_iter_source_files_filters(cfg, tmp_path):
    repo = tmp_path / "raw" / "public_train" / "foo__web"
    files = dict(extract.iter_source_files(repo, cfg))
    assert "lib/app.js" in files and files["lib/app.js"] == "javascript"
    assert "lib/vendored_crypto.py" in files
    assert not any(p.startswith("vendor/") for p in files)
    assert "lib/bundle.min.js" not in files and "lib/gen.pb.go" not in files
    assert "lib/big.py" not in files and "README.md" not in files
    assert extract.lang_for_path("x.hpp") == "cpp" and extract.lang_for_path("x.txt") is None
    assert extract.is_generated_head(b"// Code generated by protoc-gen-go. DO NOT EDIT.\npackage x\n")


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
    # порядок детерминирован: пути отсортированы, функции по позиции
    keys = [(r.path, r.start_line) for r in prot]
    assert keys == sorted(keys)

    wins = list(read_functions(fdir / "protected_windows.jsonl"))
    assert wins and all(r.kind == "window" and ":w" in r.id for r in wins)
    assert {r.path for r in wins} == prot_paths

    pub = list(read_functions(fdir / "public_train.jsonl"))
    pub_paths = sorted({r.path for r in pub})
    assert pub_paths == ["lib/app.js", "lib/util.js", "lib/util_copy.js", "lib/vendored_crypto.py"]
    assert sum(1 for r in pub if r.path == "lib/app.js") == 3
    hard = list(read_functions(fdir / "hard_neg.jsonl"))
    assert {r.path for r in hard} == {"src/kdf.py", "src/strutil.c"}
    calib = list(read_functions(fdir / "public_calib.jsonl"))
    assert sorted(r.lang for r in calib) == ["cpp", "cpp", "java", "java"]

    # идемпотентность: повторный запуск не переписывает файлы
    mtime = (fdir / "protected.jsonl").stat().st_mtime_ns
    extract.run_extract(cfg, workers=1)
    assert (fdir / "protected.jsonl").stat().st_mtime_ns == mtime
    assert (fdir / "extract_stats.json").exists()

    # ---- dedup
    stats = dedup_split.run_dedup(cfg, workers=1)
    assert (fdir / "stats.json").exists() and (fdir / "dedup_removed.jsonl").exists()
    sp = stats["splits"]
    assert sp["protected"]["n_raw"] == 7 and sp["protected"]["n_final"] == 7 and sp["protected"]["removed_near_dup"] == 0
    assert stats["protected_fingerprints"] > 0
    # public_train: точный дубликат clamp (util_copy.js) + вендоренная копия derive_key
    assert sp["public_train"]["removed_exact"] == 1
    assert sp["public_train"]["removed_near_dup"] == 1
    assert sp["public_train"]["removed_by_repo"] == {"foo/web": 2}
    pub2 = list(read_functions(fdir / "public_train.jsonl"))
    assert {r.path for r in pub2} == {"lib/app.js", "lib/util.js"}
    assert len(pub2) == 4
    # hard_neg: почти-копия удалена по отпечаткам (sha другой), несвязанная функция осталась
    assert sp["hard_neg"]["removed_near_dup"] == 1 and sp["hard_neg"]["removed_exact"] == 0
    hard2 = list(read_functions(fdir / "hard_neg.jsonl"))
    assert [r.path for r in hard2] == ["src/strutil.c"]
    removed = [json.loads(l) for l in (fdir / "dedup_removed.jsonl").read_text(encoding="utf-8").splitlines()]
    reasons = {(r["split"], r["reason"]) for r in removed}
    assert ("hard_neg", "near_dup") in reasons and ("public_train", "sha_protected") in reasons and ("public_train", "exact") in reasons
    near = [r for r in removed if r["reason"] == "near_dup"][0]
    assert near["id"].startswith("hard_neg/other__sec/src/kdf.py:") and 0.3 <= near["overlap"] <= 1.0
    # статистика по языкам и бинам длины
    assert sp["public_calib"]["by_lang"] == {"cpp": 2, "java": 2}
    assert sum(sp["protected"]["by_length_bin"].values()) == 7
    assert list(sp["protected"]["by_length_bin"]) == ["0-32", "32-64", "64-128", "128-256", "256-100000"]
    assert stats["windows"]["n_final"] == len(wins) - stats["windows"]["removed_exact"]
    assert stats["totals"]["n_final"] == 7 + 1 + 4 + 4
    # идемпотентность dedup: повторный вызов возвращает сохранённую статистику без изменений
    stats2 = dedup_split.run_dedup(cfg, workers=1)
    assert stats2 == stats
    # protected не изменился (кроме порядка — тот же)
    prot2 = list(read_functions(fdir / "protected.jsonl"))
    assert [r.id for r in prot2] == [r.id for r in prot]


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
