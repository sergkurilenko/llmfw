"""Нормализация кода: токенизация tree-sitter, абстракция токенов, удаление комментариев (DESIGN.md §3).

Поддерживаемые языки: python, c, cpp, go, java, javascript (алиасы в LANG_ALIASES).
Все функции чистые и детерминированные; парсеры кэшируются.
"""

from __future__ import annotations

import hashlib
import keyword as _py_keyword
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

LANG_ALIASES = {
    "py": "python", "python": "python",
    "c": "c", "h": "c",
    "cc": "cpp", "cpp": "cpp", "cxx": "cpp", "hpp": "cpp", "hh": "cpp", "c++": "cpp",
    "go": "go", "golang": "go",
    "java": "java",
    "js": "javascript", "jsx": "javascript", "mjs": "javascript", "cjs": "javascript", "javascript": "javascript",
}

EXT_TO_LANG = {
    ".py": "python", ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp",
    ".go": "go", ".java": "java", ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
}

_C_KW = {
    "auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else", "enum", "extern", "float",
    "for", "goto", "if", "inline", "int", "long", "register", "restrict", "return", "short", "signed", "sizeof", "static",
    "struct", "switch", "typedef", "union", "unsigned", "void", "volatile", "while", "_Bool", "_Complex", "NULL", "true",
    "false", "bool",
}
_CPP_KW = _C_KW | {
    "alignas", "alignof", "and", "catch", "class", "constexpr", "const_cast", "decltype", "delete", "dynamic_cast",
    "explicit", "export", "friend", "mutable", "namespace", "new", "noexcept", "not", "nullptr", "operator", "or",
    "private", "protected", "public", "reinterpret_cast", "static_assert", "static_cast", "template", "this", "throw",
    "try", "typeid", "typename", "using", "virtual", "xor", "override", "final", "std",
}
_GO_KW = {
    "break", "case", "chan", "const", "continue", "default", "defer", "else", "fallthrough", "for", "func", "go", "goto",
    "if", "import", "interface", "map", "package", "range", "return", "select", "struct", "switch", "type", "var", "nil",
    "true", "false", "iota", "int", "int8", "int16", "int32", "int64", "uint", "uint8", "uint16", "uint32", "uint64",
    "byte", "rune", "string", "bool", "error", "float32", "float64", "make", "new", "len", "cap", "append", "panic",
}
_JAVA_KW = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class", "const", "continue", "default",
    "do", "double", "else", "enum", "extends", "final", "finally", "float", "for", "goto", "if", "implements", "import",
    "instanceof", "int", "interface", "long", "native", "new", "package", "private", "protected", "public", "return",
    "short", "static", "strictfp", "super", "switch", "synchronized", "this", "throw", "throws", "transient", "try",
    "void", "volatile", "while", "true", "false", "null", "var", "record", "sealed", "yield",
}
_JS_KW = {
    "async", "await", "break", "case", "catch", "class", "const", "continue", "debugger", "default", "delete", "do",
    "else", "export", "extends", "finally", "for", "function", "if", "import", "in", "instanceof", "let", "new", "of",
    "return", "static", "super", "switch", "this", "throw", "try", "typeof", "var", "void", "while", "with", "yield",
    "true", "false", "null", "undefined", "get", "set",
}
KEYWORDS: dict[str, set[str]] = {
    "python": set(_py_keyword.kwlist) | {"self", "cls", "print", "True", "False", "None", "match", "case"},
    "c": _C_KW, "cpp": _CPP_KW, "go": _GO_KW, "java": _JAVA_KW, "javascript": _JS_KW,
}

_ID_TYPES = {
    "identifier", "type_identifier", "field_identifier", "property_identifier", "shorthand_property_identifier",
    "shorthand_property_identifier_pattern", "statement_identifier", "package_identifier", "label_name",
    "namespace_identifier", "primitive_type", "predefined_type", "private_property_identifier",
}
_COMMENT_TYPES = {"comment", "line_comment", "block_comment", "documentation_comment"}
_NUM_TYPES = {
    "integer", "float", "number", "number_literal", "int_literal", "float_literal", "imaginary_literal", "rune_literal",
    "decimal_integer_literal", "hex_integer_literal", "octal_integer_literal", "binary_integer_literal",
    "decimal_floating_point_literal", "hex_floating_point_literal", "character_literal", "char_literal",
}
_STR_SUBSTR = ("string", "char", "template", "raw_string", "concatenated_string", "heredoc")


@dataclass(frozen=True)
class Token:
    text: str
    kind: str  # id | kw | op | punct | num | str | comment
    start: int  # байтовое смещение
    end: int


def canon_lang(lang: str) -> str:
    """Канонизирует имя языка; ValueError для неподдерживаемого."""
    key = lang.lower().lstrip(".")
    if key not in LANG_ALIASES:
        raise ValueError(f"unsupported language: {lang}")
    return LANG_ALIASES[key]


@lru_cache(maxsize=16)
def get_parser(lang: str):
    from tree_sitter_language_pack import get_parser as _gp

    return _gp(canon_lang(lang))


def parse(code: str | bytes, lang: str):
    """Возвращает дерево tree-sitter для кода."""
    src = code if isinstance(code, bytes) else code.encode("utf-8", errors="replace")
    return get_parser(lang).parse(src)


def has_syntax_errors(code: str, lang: str) -> bool:
    """True, если в дереве есть ERROR/MISSING узлы (используется для валидации преобразований)."""
    tree = parse(code, lang)
    stack = [tree.root_node]
    while stack:
        n = stack.pop()
        if n.type == "ERROR" or n.is_missing or n.has_error and n.child_count == 0:
            return True
        stack.extend(n.children)
    return False


def _is_string_node(node) -> bool:
    t = node.type
    return any(s in t for s in _STR_SUBSTR) and t not in ("string_type",)


def _is_py_docstring(node, lang: str) -> bool:
    """Строка-оператор в Python (docstring или «висячая» строка): родитель — block/module/expression_statement."""
    if lang != "python" or not _is_string_node(node):
        return False
    p = node.parent
    return p is not None and p.type in ("expression_statement", "block", "module")


def tokenize(code: str, lang: str) -> list[Token]:
    """Листовые токены tree-sitter с классификацией вида (DESIGN.md §3). Docstring Python → comment."""
    lang = canon_lang(lang)
    src = code.encode("utf-8", errors="replace")
    tree = get_parser(lang).parse(src)
    kws = KEYWORDS[lang]
    out: list[Token] = []
    stack = [tree.root_node]
    while stack:
        n = stack.pop()
        t = n.type
        if t in _COMMENT_TYPES:
            out.append(Token(src[n.start_byte:n.end_byte].decode("utf-8", "replace"), "comment", n.start_byte, n.end_byte))
            continue
        if _is_string_node(n) and n.is_named:
            kind = "comment" if _is_py_docstring(n, lang) else "str"
            out.append(Token(src[n.start_byte:n.end_byte].decode("utf-8", "replace"), kind, n.start_byte, n.end_byte))
            continue
        if n.child_count > 0:
            stack.extend(reversed(n.children))
            continue
        if n.end_byte <= n.start_byte:
            continue
        text = src[n.start_byte:n.end_byte].decode("utf-8", "replace")
        if t in _NUM_TYPES:
            kind = "num"
        elif t in _ID_TYPES or (n.is_named and (text[:1].isalpha() or text[:1] == "_")):
            kind = "kw" if text in kws else "id"
        elif text in kws or (not n.is_named and (text[:1].isalpha() or text[:1] == "_")):
            kind = "kw"
        elif text.strip() == "":
            continue
        elif text[:1].isalnum() or text[:1] == "_":
            kind = "id"
        else:
            kind = "op" if any(ch in text for ch in "+-*/%=<>!&|^~?:.") else "punct"
        out.append(Token(text, kind, n.start_byte, n.end_byte))
    out.sort(key=lambda x: x.start)
    return out


def abstract_tokens(tokens: Iterable[Token], mode: str = "full") -> list[str]:
    """Абстрактный поток токенов.

    mode="full":    id→"ID", num→"NUM", str→"STR", комментарии удалены (тип-2 клоны ловятся).
    mode="indexed": идентификаторы → "ID_k" по порядку первого появления.
    mode="lexical": тексты как есть, комментарии удалены.
    """
    out: list[str] = []
    seen: dict[str, int] = {}
    for tok in tokens:
        if tok.kind == "comment":
            continue
        if mode == "lexical":
            out.append(tok.text)
        elif tok.kind == "id":
            if mode == "indexed":
                idx = seen.setdefault(tok.text, len(seen))
                out.append(f"ID_{idx}")
            else:
                out.append("ID")
        elif tok.kind == "num":
            out.append("NUM")
        elif tok.kind == "str":
            out.append("STR")
        else:
            out.append(tok.text)
    return out


def strip_comments(code: str, lang: str) -> str:
    """Удаляет комментарии и docstring, схлопывает получившиеся пустые строки."""
    src = code.encode("utf-8", errors="replace")
    toks = [t for t in tokenize(code, lang) if t.kind == "comment"]
    if not toks:
        return code
    parts: list[bytes] = []
    pos = 0
    for t in toks:
        parts.append(src[pos:t.start])
        pos = t.end
    parts.append(src[pos:])
    text = b"".join(parts).decode("utf-8", "replace")
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip() != ""]
    return "\n".join(lines) + ("\n" if code.endswith("\n") else "")


def normalized_sha(code: str, lang: str) -> str:
    """SHA-1 по abstract_tokens(mode='full') — для дедупликации (DESIGN.md §2.3)."""
    toks = abstract_tokens(tokenize(code, lang), mode="full")
    return hashlib.sha1(" ".join(toks).encode("utf-8")).hexdigest()


def count_tokens(code: str, lang: str) -> int:
    return sum(1 for t in tokenize(code, lang) if t.kind != "comment")
