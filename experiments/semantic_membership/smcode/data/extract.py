"""Извлечение функций из клонов репозиториев с помощью tree-sitter (DESIGN.md §2.2, §2.4).

Обход data/raw/<split>/<owner>__<name>/, фильтрация файлов (расширение -> язык через
normalize.EXT_TO_LANG, пропуск extract.skip_dirs и EXTRA_SKIP_DIRS, сгенерированных,
минифицированных и тестовых файлов по имени, файлов > 1 МБ, имён не в UTF-8), извлечение
функций по типам узлов:
  python:      function_definition (включая методы классов; вложенные def — внутри родителя)
  c/cpp:       function_definition с телом compound_statement и function_declarator
  java:        method_declaration (с телом), constructor_declaration
  go:          function_declaration, method_declaration
  javascript:  function_declaration, method_definition, arrow_function с телом statement_block
Извлекаются внешние (не вложенные в другую извлечённую функцию) узлы; кандидаты с ERROR в
декларации (артефакты восстановления парсера) отбрасываются, но внутрь них спускаемся.
Заголовки .h: cpp, если в репозитории есть C++-исходники (.cc/.cpp/.cxx/.hpp/.hh) или
содержимое выглядит как C++ (template/namespace/class), иначе c.
Для public_* дополнительно пропускаются файлы, путь которых указывает на домен
protected (crypto/tls/ssh/...; extract.domain_path_filter, по умолчанию включено).
Границы: extract.min_lines <= n_lines <= extract.max_lines, n_tokens <= extract.max_tokens.
id = "<split>/<owner>__<name>/<path>:<start>-<end>" (строки 1-based, включительно); при
повторе диапазона в файле добавляется суффикс "#2", "#3", ...
Выход: data/functions/<split>.jsonl, protected_windows.jsonl (smcode.data.windows),
extract_stats.json. Запись любого сплита удаляет маркеры шага 02 (stats.json,
dedup_removed.jsonl), чтобы дедупликация выполнилась заново.

Публичные функции: run_extract(cfg, ...), extract_records_from_source(...), iter_source_files(...),
measure_repo(...). CLI: python -m smcode.data.extract --config CFG --measure DIR.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterable, Iterator

from tqdm import tqdm

from smcode.config import load_config, resolve_path
from smcode.data.list_repos import repo_from_dir_name
from smcode.data.windows import windows_from_source
from smcode.normalize import EXT_TO_LANG, canon_lang, count_tokens, normalized_sha, parse
from smcode.types import SPLITS, FunctionRecord, write_jsonl

log = logging.getLogger(__name__)

MAX_FILE_BYTES = 1_000_000
WINDOWS_SPLITS = ("protected",)  # по умолчанию; cfg.windows.splits расширяет (public_calib/public_test — негативы формы окна)
WINDOWS_FILE = "protected_windows.jsonl"  # = windows_file("protected")


def windows_file(split: str) -> str:
    """Имя файла окон сплита: data/functions/<split>_windows.jsonl."""
    return f"{split}_windows.jsonl"


def windows_splits(cfg: dict[str, Any]) -> list[str]:
    """Сплиты, для которых строятся файловые окна (cfg.windows.splits; по умолчанию только protected)."""
    raw = (cfg.get("windows") or {}).get("splits")
    out = [str(s) for s in (raw if raw else WINDOWS_SPLITS)]
    unknown = [s for s in out if s not in SPLITS]
    if unknown:
        raise ValueError(f"windows.splits contains unknown splits {unknown}; known: {SPLITS}")
    return out
EXTRACT_STATS_FILE = "extract_stats.json"
DEDUP_STATS_FILE = "stats.json"  # маркер шага 02 (smcode.data.dedup_split.STATS_FILE)
DEDUP_REMOVED_FILE = "dedup_removed.jsonl"
DOMAIN_FILTER_SPLITS = ("public_train", "public_calib", "public_test")
CPP_EXTS = {".cc", ".cpp", ".cxx", ".hpp", ".hh"}

# Типы узлов функций по языкам (DESIGN.md §2.2).
FUNCTION_NODE_TYPES: dict[str, set[str]] = {
    "python": {"function_definition"},
    "c": {"function_definition"},
    "cpp": {"function_definition"},
    "java": {"method_declaration", "constructor_declaration"},
    "go": {"function_declaration", "method_declaration"},
    "javascript": {"function_declaration", "method_definition", "arrow_function"},
}
# Требуемый тип тела (field "body"); None — тело не проверяется.
BODY_TYPES: dict[str, dict[str, set[str] | None]] = {
    "python": {"function_definition": {"block"}},
    "c": {"function_definition": {"compound_statement"}},
    "cpp": {"function_definition": {"compound_statement"}},
    "java": {"method_declaration": {"block"}, "constructor_declaration": {"constructor_body"}},
    "go": {"function_declaration": {"block"}, "method_declaration": {"block"}},
    "javascript": {
        "function_declaration": {"statement_block"},
        "method_definition": {"statement_block"},
        "arrow_function": {"statement_block"},
    },
}

# Имена сгенерированных/минифицированных файлов (регулярные выражения по базовому имени).
GENERATED_NAME_PATTERNS = (
    r"\.min\.[mc]?js$", r"\.bundle\.js$", r"\.pb(\.gw)?\.go$", r"\.pb\.(cc|h)$", r"\.pb-c\.[ch]$",
    r"_pb2(_grpc)?\.py$", r"[._]generated\.", r"^zz_generated", r"\.tab\.(c|h|cc|cpp|hh|hpp)$",
    r"^lex\.yy\.", r"_wrap\.(c|cxx|cpp)$", r"_generated$", r"\.d\.ts$",
)
GENERATED_NAME_RES = tuple(re.compile(p, re.IGNORECASE) for p in GENERATED_NAME_PATTERNS)
GENERATED_HEAD_MARKERS = (
    "do not edit", "@generated", "autogenerated", "auto-generated", "automatically generated",
    "generated by", "code generated", "generated from", "generated file", "file is generated",
    "file was generated", "generated automatically", "generated code",
)
# Тестовые файлы по имени (базовое имя).
TEST_NAME_PATTERNS = (
    r"_test\.go$", r"^test_.*\.py$", r"_tests?\.py$", r"^conftest\.py$", r"(Test|Tests|TestCase)\.java$",
    r"\.(test|spec)\.[mc]?jsx?$", r"_test\.(c|cc|cpp|cxx|h|hpp)$", r"^test_.*\.(c|cc|cpp|cxx)$",
    r"_unittest\.(cc|cpp|py)$", r"^test\.(c|cc|cpp|go|js|py)$",
)
TEST_NAME_RES = tuple(re.compile(p) for p in TEST_NAME_PATTERNS)
SKIP_ALWAYS_DIRS = {".git", ".hg", ".svn", "__pycache__", ".tox", ".venv", "venv", ".idea", ".vscode"}
# Каталоги тестов/сторонних/примеров сверх extract.skip_dirs (см. открытые вопросы: default.yaml).
EXTRA_SKIP_DIRS = {
    "__tests__", "__mocks__", "__snapshots__", "fixtures", "fixture", "3rdparty", "third-party", "thirdparty",
    "testdir", "testing", "testsuite", "unittest", "unittests", "integration_tests", "e2e", "samples", "sample",
    "docs", "doc", "t", "golden", "snapshots",
}

# Маркеры домена protected (crypto/tls/ssh/...) в пути файла — для public_* (DESIGN §2.1, RQ2).
# exact: точное совпадение токена пути; prefix: токен начинается с; substr: подстрока токена.
DOMAIN_PATH_EXACT = {
    "aes", "des", "rsa", "dsa", "dh", "ecdh", "ecdsa", "md4", "md5", "rc4", "pgp", "gpg", "jwt", "jwe", "jws",
    "pcap", "x509", "hmac", "tls", "ssl", "ssh", "sshd", "ike", "ipsec", "vpn", "pki", "csr", "crl", "ocsp",
    "kdf", "otp", "totp", "hotp", "srp", "sasl", "ntlm", "cmac", "gcm", "ccm", "ctr", "ecb", "cbc", "xts",
}
DOMAIN_PATH_PREFIX = {
    "ssl", "tls", "ssh", "crypt", "cipher", "hmac", "x509", "pcap", "sha1", "sha2", "sha3", "sha224", "sha256",
    "sha384", "sha512", "md5", "rsa", "ecdsa", "ecdh", "pkcs", "pbkdf", "scrypt", "bcrypt", "argon2", "scram",
    "kerberos", "gssapi", "firewall", "iptables", "netfilter", "blowfish", "twofish", "chacha", "salsa",
    "poly1305", "curve25519", "ed25519", "x25519", "secp", "keccak", "whirlpool", "ripemd", "camellia",
    "openssl", "wolfssl", "mbedtls", "boringssl", "libressl", "libssh", "sodium", "certificate", "certs",
}
DOMAIN_PATH_SUBSTR = {
    "crypt", "cipher", "blowfish", "chacha", "openssl", "wolfssl", "mbedtls", "boringssl", "libsodium", "sha256",
    "sha512", "keccak", "poly1305", "curve25519", "pkcs", "x509", "kerberos", "firewall", "ipsec", "wireguard",
}
_PATH_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CPP_HEADER_RE = re.compile(
    r"^\s*(template\s*<|namespace\s+[A-Za-z_]\w*\s*\{|using\s+namespace\b|extern\s+\"C\+\+\"|"
    r"class\s+[A-Za-z_]\w*\s*(:|\{|final\b)|(public|private|protected)\s*:)",
    re.MULTILINE,
)


# ----------------------------------------------------------------------------- файлы


def _skip_dirs(cfg: dict[str, Any]) -> set[str]:
    return {str(d).lower() for d in (cfg.get("extract", {}).get("skip_dirs") or [])} | SKIP_ALWAYS_DIRS | EXTRA_SKIP_DIRS


def is_generated_name(name: str) -> bool:
    """Сгенерированный/минифицированный файл по базовому имени (якорные регулярные выражения)."""
    base = os.path.basename(name)
    return any(r.search(base) for r in GENERATED_NAME_RES)


def is_test_name(name: str) -> bool:
    """Тестовый файл по базовому имени (*_test.go, test_*.py, *Test.java, *.spec.js, ...)."""
    base = os.path.basename(name)
    return any(r.search(base) for r in TEST_NAME_RES)


def is_generated_head(head: bytes) -> bool:
    """Маркер «generated» в первых строках файла."""
    text = head[:2048].decode("utf-8", "replace").lower()
    first = "\n".join(text.split("\n")[:5])
    return any(m in first for m in GENERATED_HEAD_MARKERS)


def domain_path_match(rel_path: str) -> str | None:
    """Маркер домена protected в относительном пути (crypto/tls/ssh/...) или None."""
    for tok in _PATH_TOKEN_RE.findall(rel_path.lower()):
        if tok in DOMAIN_PATH_EXACT:
            return tok
        for p in DOMAIN_PATH_PREFIX:
            if tok.startswith(p):
                return p
        for s in DOMAIN_PATH_SUBSTR:
            if s in tok:
                return s
    return None


def looks_like_cpp(data: bytes) -> bool:
    """Заголовок .h похож на C++ (template/namespace/class/using namespace/public:)."""
    text = data[:262144].decode("utf-8", "replace")
    return _CPP_HEADER_RE.search(text) is not None


def lang_for_path(path: str | Path, languages: Iterable[str] | None = None) -> str | None:
    """Язык по расширению (normalize.EXT_TO_LANG) или None; languages — допустимые языки."""
    ext = Path(path).suffix.lower()
    lang = EXT_TO_LANG.get(ext)
    if lang is None:
        return None
    if languages is not None and lang not in set(languages):
        return None
    return lang


def _bump(stats: dict[str, int] | None, key: str) -> None:
    if stats is not None:
        stats[key] = stats.get(key, 0) + 1


def iter_source_files(
    repo_dir: Path,
    cfg: dict[str, Any],
    max_bytes: int = MAX_FILE_BYTES,
    split: str | None = None,
    stats: dict[str, int] | None = None,
    header_lang: str | None = None,
) -> Iterator[tuple[str, str]]:
    """Пары (относительный путь, язык) исходных файлов репозитория в детерминированном порядке.

    Пропускает каталоги skip_dirs, симлинки, имена не в UTF-8, сгенерированные и тестовые файлы,
    файлы > max_bytes; для split in DOMAIN_FILTER_SPLITS (при extract.domain_path_filter) — файлы с
    маркером домена protected в пути. Заголовки .h получают язык cpp, если в репозитории есть
    C++-исходники (header_lang=None — автоопределение; "c"/"cpp" — принудительно).
    Счётчики пропусков накапливаются в stats (skipped_*, headers_as_cpp)."""
    skip = _skip_dirs(cfg)
    langs = cfg.get("languages")
    ext_cfg = cfg.get("extract") or {}
    domain_filter = bool(ext_cfg.get("domain_path_filter", True)) and split in DOMAIN_FILTER_SPLITS
    skip_tests = bool(ext_cfg.get("skip_test_files", True))
    repo_dir = Path(repo_dir)
    out: list[tuple[str, str]] = []
    has_cpp = False
    for root, dirs, files in os.walk(repo_dir, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d.lower() not in skip and not os.path.islink(os.path.join(root, d)))
        for fn in sorted(files):
            if Path(fn).suffix.lower() in CPP_EXTS:
                has_cpp = True
            lang = lang_for_path(fn, langs)
            if lang is None:
                continue
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, repo_dir).replace(os.sep, "/")
            try:
                rel.encode("utf-8")
            except UnicodeEncodeError:
                _bump(stats, "skipped_bad_name")
                log.warning("имя файла не в UTF-8, пропуск: %r (%s)", rel, repo_dir.name)
                continue
            if is_generated_name(fn):
                _bump(stats, "skipped_generated_name")
                continue
            if skip_tests and is_test_name(fn):
                _bump(stats, "skipped_test_name")
                continue
            if domain_filter:
                kw = domain_path_match(rel)
                if kw:
                    _bump(stats, "skipped_domain_path")
                    log.debug("домен protected в пути (%s), пропуск: %s/%s", kw, repo_dir.name, rel)
                    continue
            if os.path.islink(full):
                continue
            try:
                size = os.path.getsize(full)
            except OSError:
                continue
            if size == 0:
                continue
            if size > max_bytes:
                _bump(stats, "skipped_big")
                continue
            out.append((rel, lang))
    if header_lang is None:
        header_lang = "cpp" if has_cpp else "c"
    if header_lang == "cpp" and (langs is None or "cpp" in set(langs)):
        remapped: list[tuple[str, str]] = []
        for rel, lang in out:
            if lang == "c" and rel.lower().endswith(".h"):
                lang = "cpp"
                _bump(stats, "headers_as_cpp")
            remapped.append((rel, lang))
        out = remapped
    out.sort()
    yield from out


def discover_repos(raw_dir: Path, splits: Iterable[str] | None = None) -> list[tuple[str, str, Path]]:
    """(split, repo, каталог) для всех клонов в data/raw/<split>/<owner>__<name>/."""
    res: list[tuple[str, str, Path]] = []
    for split in splits or SPLITS:
        d = Path(raw_dir) / split
        if not d.is_dir():
            continue
        for sub in sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith(".")):
            res.append((split, repo_from_dir_name(sub.name), sub))
    return res


def measure_repo(repo_dir: Path, cfg: dict[str, Any], max_bytes: int = MAX_FILE_BYTES) -> tuple[int, int]:
    """(raw_bytes, effective_bytes): размер всего дерева и размер файлов вне skip_dirs (и симлинков)
    не больше max_bytes — то, что реально читает извлечение. Для лимита в scripts/00_clone.sh."""
    skip = _skip_dirs(cfg)
    repo_dir = Path(repo_dir)
    raw = eff = 0
    for root, dirs, files in os.walk(repo_dir, followlinks=False):
        rel_root = os.path.relpath(root, repo_dir)
        in_skip = rel_root != "." and any(p.lower() in skip for p in rel_root.split(os.sep))
        for fn in files:
            full = os.path.join(root, fn)
            try:
                st = os.lstat(full)
            except OSError:
                continue
            raw += st.st_size
            if not in_skip and not os.path.islink(full) and st.st_size <= max_bytes:
                eff += st.st_size
    return raw, eff


# ----------------------------------------------------------------------------- tree-sitter


def _body_ok(node, lang: str) -> bool:
    need = BODY_TYPES.get(lang, {}).get(node.type)
    if need is None:
        return True
    body = node.child_by_field_name("body")
    return body is not None and body.type in need


def _subtree_has(node, pred) -> bool:
    stack = [node]
    while stack:
        n = stack.pop()
        if pred(n):
            return True
        if n.child_count:
            stack.extend(n.children)
    return False


def _has_error(node) -> bool:
    return _subtree_has(node, lambda n: n.type == "ERROR" or n.is_missing)


def _declarator_ok(node, lang: str) -> bool:
    """c/cpp: declarator без ERROR и с function_declarator; остальные: поле name без ERROR."""
    if lang in ("c", "cpp"):
        decl = node.child_by_field_name("declarator")
        if decl is None or _has_error(decl):
            return False
        return _subtree_has(decl, lambda n: n.type == "function_declarator")
    name = node.child_by_field_name("name")
    return name is None or not _has_error(name)


def _candidate_ok(node, lang: str) -> bool:
    """Узел — настоящая функция: тело нужного типа, корректная декларация, без вложенных
    function_definition в теле при ошибках парсинга (артефакт восстановления в C/C++)."""
    if node.type == "ERROR" or not _body_ok(node, lang) or not _declarator_ok(node, lang):
        return False
    if lang in ("c", "cpp") and node.has_error:
        body = node.child_by_field_name("body")
        if body is not None and any(c.type == "function_definition" for c in body.children):
            return False
    return True


def find_function_nodes(root, lang: str) -> list:
    """Внешние узлы функций (в узлы принятых функций не спускаемся; в отклонённые — спускаемся)."""
    types = FUNCTION_NODE_TYPES[lang]
    out: list = []
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type in types and _candidate_ok(n, lang):
            out.append(n)
            continue  # тело без вложенных функций
        if n.child_count:
            stack.extend(reversed(n.children))
    out.sort(key=lambda x: (x.start_byte, x.end_byte))
    return out


def _span_node(node, lang: str):
    """Узел, определяющий границы фрагмента: для JS arrow_function — объявление
    `const f = (...) => {...};` или выражение присваивания целиком."""
    if lang == "javascript" and node.type == "arrow_function":
        p = node.parent
        if p is not None and p.type == "variable_declarator":
            gp = p.parent
            if gp is not None and gp.type in ("lexical_declaration", "variable_declaration"):
                return gp
            return p
        if p is not None and p.type == "assignment_expression":
            gp = p.parent
            if gp is not None and gp.type == "expression_statement":
                return gp
            return p
    return node


def _node_lines(node) -> tuple[int, int]:
    """(start_row, end_row), 0-based; узел, заканчивающийся в начале строки, не захватывает её."""
    sr = node.start_point[0]
    er = node.end_point[0]
    if node.end_point[1] == 0 and er > sr:
        er -= 1
    return sr, er


def dedent_by_prefix(text: str, indent: str) -> str:
    """Снимает ровно отступ `indent` со строк, начинающихся с него; прочие непустые строки (напр.
    содержимое многострочных строк с колонки 0) не трогает; пустые строки нормализует в ''."""
    if not indent:
        return text.rstrip("\n")
    out: list[str] = []
    for ln in text.split("\n"):
        if ln.startswith(indent):
            out.append(ln[len(indent):])
        elif ln.strip() == "":
            out.append("")
        else:
            out.append(ln)
    return "\n".join(out).rstrip("\n")


def _slice_code(src: bytes, lines: list[bytes], line_offsets: list[int], node) -> tuple[str, int, int]:
    """Текст фрагмента и строки (1-based). Если на первой строке перед узлом и на последней после
    узла только пробелы (и ';'), берутся строки целиком, иначе — байты узла. Отступ снимается по
    колонке первой строки узла (dedent_by_prefix)."""
    sr, er = _node_lines(node)
    prefix = src[line_offsets[sr] : node.start_byte]
    suffix = src[node.end_byte : line_offsets[er] + len(lines[er])]
    first = lines[sr]
    indent = first[: len(first) - len(first.lstrip())]
    if prefix.strip() == b"" and suffix.strip().strip(b";") == b"":
        raw = b"\n".join(lines[sr : er + 1])
    else:
        raw = src[node.start_byte : node.end_byte]
    text = dedent_by_prefix(raw.decode("utf-8", "replace"), indent.decode("utf-8", "replace"))
    return text, sr + 1, er + 1


def extract_spans(src: bytes, lang: str) -> list[tuple[int, int, str]]:
    """(start_line, end_line, code) всех внешних функций в исходном тексте (байты, '\\n')."""
    lang = canon_lang(lang)
    tree = parse(src, lang)
    lines = src.split(b"\n")
    offsets: list[int] = []
    pos = 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1
    out: list[tuple[int, int, str]] = []
    for node in find_function_nodes(tree.root_node, lang):
        span = _span_node(node, lang)
        code, s, e = _slice_code(src, lines, offsets, span)
        if code.strip():
            out.append((s, e, code))
    return out


def normalize_newlines(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def function_id(split: str, repo_dir: str, rel_path: str, start: int, end: int, ordinal: int = 1) -> str:
    """id функции; ordinal > 1 — повтор диапазона строк в файле (суффикс '#<ordinal>')."""
    base = f"{split}/{repo_dir}/{rel_path}:{start}-{end}"
    return base if ordinal <= 1 else f"{base}#{ordinal}"


def extract_records_from_source(
    text: str | bytes,
    lang: str,
    split: str,
    repo: str,
    repo_dir: str,
    rel_path: str,
    cfg: dict[str, Any],
) -> list[FunctionRecord]:
    """Записи функций (FunctionRecord) из одного файла с проверкой границ cfg.extract; id уникальны."""
    ext = cfg.get("extract") or {}
    min_lines = int(ext.get("min_lines", 1))
    max_lines = int(ext.get("max_lines", 10**9))
    max_tokens = int(ext.get("max_tokens", 10**9))
    src = text.encode("utf-8", "replace") if isinstance(text, str) else text
    src = normalize_newlines(src)
    lang = canon_lang(lang)
    out: list[FunctionRecord] = []
    seen_spans: dict[tuple[int, int], int] = {}
    for start, end, code in extract_spans(src, lang):
        n_lines = end - start + 1
        if n_lines < min_lines or n_lines > max_lines:
            continue
        try:
            n_tokens = count_tokens(code, lang)
            if n_tokens == 0 or n_tokens > max_tokens:
                continue
            sha = normalized_sha(code, lang)
        except Exception as exc:  # noqa: BLE001
            log.debug("%s:%d-%d skipped: %s", rel_path, start, end, exc)
            continue
        ordinal = seen_spans.get((start, end), 0) + 1
        seen_spans[(start, end)] = ordinal
        out.append(
            FunctionRecord(
                id=function_id(split, repo_dir, rel_path, start, end, ordinal),
                split=split,
                repo=repo,
                path=rel_path,
                lang=lang,
                code=code,
                start_line=start,
                end_line=end,
                n_lines=n_lines,
                n_tokens=n_tokens,
                sha=sha,
                kind="function",
            )
        )
    return out


# ----------------------------------------------------------------------------- worker


def _extract_file_task(task: tuple[str, str, str, str, str, str, dict[str, Any], bool]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Рабочая функция пула: (split, repo, repo_dir_name, repo_root, rel_path, lang, cfg, with_windows)
    -> (функции, окна, счётчики)."""
    split, repo, repo_dir, repo_root, rel_path, lang, cfg, with_windows = task
    info = {"files": 1, "skipped_binary": 0, "skipped_generated": 0, "errors": 0, "headers_sniffed_cpp": 0}
    full = os.path.join(repo_root, rel_path)
    try:
        with open(full, "rb") as f:
            data = f.read()
    except OSError:
        info["errors"] += 1
        return [], [], info
    head = data[:8192]
    if b"\x00" in head:
        info["skipped_binary"] += 1
        return [], [], info
    if is_generated_head(head):
        info["skipped_generated"] += 1
        return [], [], info
    data = normalize_newlines(data)
    langs = cfg.get("languages")
    if lang == "c" and rel_path.lower().endswith(".h") and (not langs or "cpp" in langs) and looks_like_cpp(data):
        lang = "cpp"
        info["headers_sniffed_cpp"] += 1
    try:
        funcs = extract_records_from_source(data, lang, split, repo, repo_dir, rel_path, cfg)
    except Exception as exc:  # noqa: BLE001 — один плохой файл не должен ронять этап
        log.warning("extract failed for %s/%s: %s", repo, rel_path, exc)
        info["errors"] += 1
        funcs = []
    wins: list[FunctionRecord] = []
    if with_windows:
        try:
            wins = windows_from_source(data.decode("utf-8", "replace"), lang, split, repo, repo_dir, rel_path, cfg)
        except Exception as exc:  # noqa: BLE001
            log.warning("windows failed for %s/%s: %s", repo, rel_path, exc)
            info["errors"] += 1
    return [r.to_dict() for r in funcs], [r.to_dict() for r in wins], info


def _worker_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Минимальный срез конфига для воркеров (без путей)."""
    return {"extract": dict(cfg.get("extract") or {}), "windows": dict(cfg.get("windows") or {}), "languages": list(cfg.get("languages") or [])}


def _write_record(f_out, r: dict[str, Any]) -> bool:
    """Одна строка JSONL; False (с предупреждением), если запись не сериализуется/кодируется."""
    try:
        f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
        return True
    except (UnicodeEncodeError, TypeError, ValueError) as exc:
        log.warning("запись %s пропущена: %s", r.get("id", "?").encode("utf-8", "replace"), exc)
        return False


def invalidate_dedup(out_dir: Path) -> list[str]:
    """Удаляет маркеры шага 02 (stats.json, dedup_removed.jsonl): после перезаписи сплита
    дедупликация должна выполниться заново. Возвращает имена удалённых файлов."""
    removed: list[str] = []
    for name in (DEDUP_STATS_FILE, DEDUP_REMOVED_FILE):
        p = Path(out_dir) / name
        if p.exists():
            p.unlink()
            removed.append(name)
    if removed:
        log.info("удалены маркеры дедупликации %s — запустите scripts/02_dedup_split.py заново", ", ".join(removed))
    return removed


# ----------------------------------------------------------------------------- run


def run_extract(
    cfg: dict[str, Any],
    splits: Iterable[str] | None = None,
    workers: int = 4,
    force: bool = False,
    with_windows: bool = True,
    max_file_bytes: int = MAX_FILE_BYTES,
) -> dict[str, Path]:
    """Извлечение функций для всех сплитов из data/raw -> data/functions/<split>.jsonl
    (+ <split>_windows.jsonl для сплитов из cfg.windows.splits, по умолчанию protected_windows.jsonl).
    Идемпотентно: готовые файлы пропускаются без force. Возвращает {split|"<split>_windows": путь}.
    Порядок записей детерминирован (репозитории и файлы отсортированы, функции по позиции).
    Перезапись сплита удаляет stats.json шага 02."""
    raw_dir = resolve_path(cfg, "raw_repos")
    out_dir = resolve_path(cfg, "functions")
    splits = list(splits or SPLITS)
    win_splits = windows_splits(cfg)
    wcfg = _worker_cfg(cfg)
    outputs: dict[str, Path] = {}
    stats: dict[str, Any] = {}
    stats_path = out_dir / EXTRACT_STATS_FILE
    if stats_path.exists():
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
        except (OSError, json.JSONDecodeError):
            stats = {}

    pool: Pool | None = None
    try:
        for split in splits:
            out_path = out_dir / f"{split}.jsonl"
            win_path = out_dir / windows_file(split)
            win_key = f"{split}_windows"
            make_windows = with_windows and split in win_splits
            done = out_path.exists() and (not make_windows or win_path.exists())
            if done and not force:
                log.info("[%s] уже извлечено: %s — пропуск (--force для пересчёта)", split, out_path)
                outputs[split] = out_path
                if make_windows:
                    outputs[win_key] = win_path
                continue
            repos = discover_repos(raw_dir, [split])
            if not repos:
                log.warning("[%s] нет клонов в %s — сплит пропущен", split, raw_dir / split)
                continue
            # задачи по репозиториям (память ограничена одним репозиторием)
            repo_tasks: list[tuple[str, str, Path, list[tuple[str, str]]]] = []
            file_stats: dict[str, int] = {}
            n_files = 0
            for _, repo, rdir in repos:
                files = list(iter_source_files(rdir, cfg, max_bytes=max_file_bytes, split=split, stats=file_stats))
                repo_tasks.append((split, repo, rdir, files))
                n_files += len(files)
            log.info("[%s] %d репозиториев, %d файлов; пропущено по имени: %s", split, len(repos), n_files,
                     json.dumps(file_stats, ensure_ascii=False, sort_keys=True))
            if workers > 1 and pool is None and n_files > 0:
                pool = Pool(processes=workers)

            counts: dict[str, Any] = {"repos": len(repos), "files": 0, "functions": 0, "windows": 0, "skipped_binary": 0,
                                      "skipped_generated": 0, "errors": 0, "headers_sniffed_cpp": 0, "by_lang": {}, "by_repo": {}}
            for key in ("skipped_bad_name", "skipped_generated_name", "skipped_test_name", "skipped_domain_path",
                        "skipped_big", "headers_as_cpp"):
                counts[key] = file_stats.get(key, 0)
            t0 = time.perf_counter()
            tmp_path = out_path.with_suffix(".jsonl.tmp")
            tmp_win = win_path.with_suffix(".jsonl.tmp")
            try:
                with open(tmp_path, "w", encoding="utf-8") as f_out, \
                     (open(tmp_win, "w", encoding="utf-8") if make_windows else _NullFile()) as f_win, \
                     tqdm(total=n_files, desc=f"extract {split}", unit="file", disable=n_files == 0) as bar:
                    for _, repo, rdir, files in repo_tasks:
                        repo_dir = rdir.name
                        tasks = [(split, repo, repo_dir, str(rdir), rel, lang, wcfg, make_windows) for rel, lang in files]
                        if pool is not None:
                            results = pool.map(_extract_file_task, tasks, chunksize=8)
                        else:
                            results = [_extract_file_task(t) for t in tasks]
                        n_repo = 0
                        for funcs, wins, info in results:
                            for r in funcs:
                                if not _write_record(f_out, r):
                                    counts["errors"] += 1
                                    continue
                                counts["by_lang"][r["lang"]] = counts["by_lang"].get(r["lang"], 0) + 1
                                counts["functions"] += 1
                                n_repo += 1
                            for r in wins:
                                if _write_record(f_win, r):
                                    counts["windows"] += 1
                                else:
                                    counts["errors"] += 1
                            for k in ("files", "skipped_binary", "skipped_generated", "errors", "headers_sniffed_cpp"):
                                counts[k] += info.get(k, 0)
                        counts["by_repo"][repo] = n_repo
                        bar.update(len(files))
            except BaseException:
                for p in (tmp_path, tmp_win):
                    if p.exists():
                        p.unlink()
                raise
            invalidate_dedup(out_dir)
            os.replace(tmp_path, out_path)
            outputs[split] = out_path
            if make_windows:
                os.replace(tmp_win, win_path)
                outputs[win_key] = win_path
            counts["seconds"] = round(time.perf_counter() - t0, 1)
            stats[split] = counts
            log.info("[%s] функций: %d, окон: %d, файлов: %d (бинарных пропущено %d, generated %d, домен %d, "
                     "тестовых %d, .h как cpp %d+%d, ошибок %d) за %.1f c",
                     split, counts["functions"], counts["windows"], counts["files"], counts["skipped_binary"],
                     counts["skipped_generated"], counts["skipped_domain_path"], counts["skipped_test_name"],
                     counts["headers_as_cpp"], counts["headers_sniffed_cpp"], counts["errors"], counts["seconds"])
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=1)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return outputs


class _NullFile:
    """Заглушка файла для контекстного менеджера, когда окна не пишутся."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def write(self, _s: str) -> int:
        return 0


def write_records(path: str | Path, records: Iterable[FunctionRecord]) -> int:
    """Утилита: запись списка FunctionRecord в JSONL (для тестов и ручных прогонов)."""
    return write_jsonl(path, records)


# ----------------------------------------------------------------------------- CLI (вспомогательный)


def main(argv: list[str] | None = None) -> int:
    """`--measure DIR`: печатает "raw_mb eff_mb" (для лимита размера в scripts/00_clone.sh)."""
    ap = argparse.ArgumentParser(description="Вспомогательные команды извлечения")
    ap.add_argument("--config", default=None)
    ap.add_argument("--measure", default=None, help="каталог клона: напечатать 'raw_mb eff_mb'")
    ap.add_argument("--max-bytes", type=int, default=MAX_FILE_BYTES)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.measure:
        raw, eff = measure_repo(Path(args.measure), cfg, max_bytes=args.max_bytes)
        print(f"{-(-raw // (1 << 20))} {-(-eff // (1 << 20))}")
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
