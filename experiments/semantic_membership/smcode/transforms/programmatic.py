"""Программные преобразования кода (DESIGN.md §4): identity, reformat, strip_comments, rename_ids,
change_literals, combo — и общие помощники разбора для остальных модулей пакета transforms.

Все функции имеют сигнатуру ``t(code, lang, rng, **params) -> TransformResult``; при неприменимости
возвращается ``TransformResult.failed``. Синтаксическая валидация результата выполняется в
``registry.apply_transform`` (здесь — только в составных шагах).
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from smcode import normalize
from smcode.normalize import KEYWORDS, canon_lang, has_syntax_errors, tokenize
from smcode.types import TransformResult

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- разбор с обёрткой

# Обёртки для фрагментов, которые сами по себе не разбираются на верхнем уровне:
# метод класса (java/javascript), последовательность операторов (go/c/java).
_CLASS_WRAP: dict[str, tuple[str, str]] = {
    "java": ("class _W {\n", "\n}\n"),
    "javascript": ("class _W {\n", "\n}\n"),
    "cpp": ("struct _W {\n", "\n};\n"),
}
_FUNC_WRAP: dict[str, tuple[str, str]] = {
    "python": ("def _w():\n", "\n"),  # тело дополнительно индентируется
    "c": ("void _w(void) {\n", "\n}\n"),
    "cpp": ("void _w(void) {\n", "\n}\n"),
    "go": ("package _w\nfunc _w() {\n", "\n}\n"),
    "java": ("class _W { void _w() {\n", "\n} }\n"),
    "javascript": ("function _w() {\n", "\n}\n"),
}

FUNCTION_NODE_TYPES: dict[str, set[str]] = {
    "python": {"function_definition"},
    "c": {"function_definition"},
    "cpp": {"function_definition"},
    "go": {"function_declaration", "method_declaration", "func_literal"},
    "java": {"method_declaration", "constructor_declaration"},
    "javascript": {
        "function_declaration", "method_definition", "arrow_function", "function_expression", "function",
        "generator_function_declaration", "generator_function",
    },
}
BLOCK_NODE_TYPES: dict[str, set[str]] = {
    "python": {"block"},
    "c": {"compound_statement"},
    "cpp": {"compound_statement"},
    "go": {"statement_list"},
    "java": {"block", "constructor_body"},
    "javascript": {"statement_block"},
}
# Операторы (statement-узлы), перед которыми можно вставлять код и которые можно переставлять.
STATEMENT_NODE_TYPES: dict[str, set[str]] = {
    "python": {
        "expression_statement", "return_statement", "pass_statement", "break_statement", "continue_statement",
        "if_statement", "for_statement", "while_statement", "try_statement", "with_statement", "raise_statement",
        "assert_statement", "delete_statement", "function_definition", "class_definition", "import_statement",
        "import_from_statement", "global_statement", "nonlocal_statement", "match_statement",
        "decorated_definition", "print_statement", "exec_statement",
        # в некоторых сборках грамматики expression_statement скрыт: выражения лежат в block напрямую
        "assignment", "augmented_assignment", "call", "await", "yield", "string", "identifier",
        "named_expression", "list_comprehension", "attribute", "subscript", "binary_operator",
    },
    "c": {
        "expression_statement", "declaration", "return_statement", "if_statement", "for_statement",
        "while_statement", "do_statement", "switch_statement", "break_statement", "continue_statement",
        "goto_statement", "compound_statement", "labeled_statement",
    },
    "cpp": {
        "expression_statement", "declaration", "return_statement", "if_statement", "for_statement",
        "for_range_loop", "while_statement", "do_statement", "switch_statement", "break_statement",
        "continue_statement", "goto_statement", "compound_statement", "labeled_statement", "try_statement",
        "throw_statement",
    },
    "go": {
        "expression_statement", "short_var_declaration", "var_declaration", "const_declaration",
        "assignment_statement", "inc_statement", "dec_statement", "return_statement", "if_statement",
        "for_statement", "expression_switch_statement", "type_switch_statement", "select_statement",
        "go_statement", "defer_statement", "send_statement", "break_statement", "continue_statement",
        "block", "labeled_statement", "fallthrough_statement", "goto_statement",
    },
    "java": {
        "expression_statement", "local_variable_declaration", "return_statement", "if_statement",
        "for_statement", "enhanced_for_statement", "while_statement", "do_statement", "switch_expression",
        "break_statement", "continue_statement", "throw_statement", "try_statement",
        "try_with_resources_statement", "block", "synchronized_statement", "labeled_statement",
        "yield_statement", "assert_statement", "explicit_constructor_invocation",
    },
    "javascript": {
        "expression_statement", "lexical_declaration", "variable_declaration", "return_statement",
        "if_statement", "for_statement", "for_in_statement", "while_statement", "do_statement",
        "switch_statement", "break_statement", "continue_statement", "throw_statement", "try_statement",
        "statement_block", "labeled_statement", "function_declaration", "class_declaration",
    },
}
# Узлы-«атомы», внутрь которых токенизация для reformat не заглядывает.
_ATOMIC_SUBSTR = ("string", "char", "template", "regex", "heredoc", "comment", "preproc", "raw_string")


@dataclass
class Parsed:
    """Разобранный код (возможно, с обёрткой). Смещения узлов — в ``src``; оригинал = src[prefix:len-suffix]."""

    lang: str
    src: bytes
    tree: Any
    prefix: int
    suffix: int
    wrapper: str  # "" | "class" | "func"

    @property
    def root(self):
        return self.tree.root_node

    def text(self, node) -> str:
        return self.src[node.start_byte : node.end_byte].decode("utf-8", "replace")

    def in_original(self, node) -> bool:
        return node.start_byte >= self.prefix and node.end_byte <= len(self.src) - self.suffix

    def unwrap(self, src: bytes | None = None) -> str:
        """Снимает обёртку с (возможно отредактированного) исходника той же структуры."""
        s = self.src if src is None else src
        body = s[self.prefix : len(s) - self.suffix] if self.suffix else s[self.prefix :]
        text = body.decode("utf-8", "replace")
        if self.wrapper == "func" and self.lang == "python":
            text = "\n".join(ln[4:] if ln.startswith("    ") else ln for ln in text.split("\n"))
        return text

    def walk(self) -> Iterator[Any]:
        """Обход всех узлов в порядке pre-order."""
        stack = [self.root]
        while stack:
            n = stack.pop()
            yield n
            stack.extend(reversed(n.children))


def _wrap(code: str, lang: str, kind: str) -> tuple[str, int, int]:
    pre, post = (_CLASS_WRAP if kind == "class" else _FUNC_WRAP)[lang]
    body = code
    if kind == "func" and lang == "python":
        body = "\n".join(("    " + ln) if ln.strip() else ln for ln in code.split("\n"))
    return pre + body + post, len(pre.encode("utf-8")), len(post.encode("utf-8"))


def _tree_errors(tree) -> int:
    n_err = 0
    stack = [tree.root_node]
    while stack:
        n = stack.pop()
        if n.type == "ERROR" or n.is_missing:
            n_err += 1
            continue
        if n.has_error:
            stack.extend(n.children)
    return n_err


def parse_code(code: str, lang: str) -> Parsed:
    """Разбирает код; если на верхнем уровне есть ошибки — пробует обёртки (класс, функция).
    Возвращает вариант с наименьшим числом ошибок."""
    lang = canon_lang(lang)
    best: Parsed | None = None
    best_err = None
    variants: list[tuple[str, str, int, int]] = [("", code, 0, 0)]
    if lang in _CLASS_WRAP:
        variants.append(("class",) + _wrap(code, lang, "class"))
    if lang in _FUNC_WRAP:
        variants.append(("func",) + _wrap(code, lang, "func"))
    for kind, text, pre, post in variants:
        src = text.encode("utf-8", "replace")
        tree = normalize.get_parser(lang).parse(src)
        n_err = _tree_errors(tree)
        if best is None or n_err < best_err:
            best, best_err = Parsed(lang, src, tree, pre, post, kind), n_err
        if n_err == 0:
            break
    assert best is not None
    return best


def parses_ok(code: str, lang: str) -> bool:
    """True, если код разбирается без ERROR/MISSING сам по себе либо внутри стандартной обёртки."""
    lang = canon_lang(lang)
    if not code.strip():
        return False
    if not has_syntax_errors(code, lang):
        return True
    for kind in ("class", "func"):
        table = _CLASS_WRAP if kind == "class" else _FUNC_WRAP
        if lang in table:
            text, _, _ = _wrap(code, lang, kind)
            if not has_syntax_errors(text, lang):
                return True
    return False


def apply_edits(src: bytes, edits: list[tuple[int, int, bytes]]) -> bytes:
    """Применяет непересекающиеся замены (start, end, new) к байтовой строке."""
    out: list[bytes] = []
    pos = 0
    for start, end, new in sorted(edits, key=lambda e: e[0]):
        if start < pos:
            continue  # пересечение — пропускаем
        out.append(src[pos:start])
        out.append(new)
        pos = end
    out.append(src[pos:])
    return b"".join(out)


def find_function_node(p: Parsed):
    """Первый функциональный узел (в оригинальной части кода) или None."""
    types = FUNCTION_NODE_TYPES[p.lang]
    for n in p.walk():
        if n.type in types and p.in_original(n):
            return n
    return None


def body_node(p: Parsed, fn=None):
    """Узел тела функции (block/compound_statement/...); для go — statement_list внутри block."""
    fn = fn if fn is not None else find_function_node(p)
    if fn is None:
        return None
    body = fn.child_by_field_name("body")
    if body is None:
        return None
    if p.lang == "go":
        for c in body.children:
            if c.type == "statement_list":
                return c
    return body


def block_children(p: Parsed, block) -> list[Any]:
    """Операторы блока: именованные дети, кроме комментариев и скобок."""
    return [c for c in block.children if c.is_named and "comment" not in c.type and c.type not in ("{", "}")]


def count_lines(code: str) -> int:
    """Число непустых строк."""
    return sum(1 for ln in code.splitlines() if ln.strip())


def _is_atomic(node) -> bool:
    t = node.type
    return any(s in t for s in _ATOMIC_SUBSTR) and t not in ("string_type",)


def leaves(p: Parsed) -> list[Any]:
    """Листовые узлы (строки/комментарии/regex — целиком), по порядку."""
    out: list[Any] = []
    stack = [p.root]
    while stack:
        n = stack.pop()
        if _is_atomic(n) and n.is_named:
            out.append(n)
            continue
        if n.child_count:
            stack.extend(reversed(n.children))
        elif n.end_byte > n.start_byte:
            out.append(n)
    out.sort(key=lambda n: n.start_byte)
    return out


# --------------------------------------------------------------------------- identity


def identity(code: str, lang: str, rng: random.Random, **params: Any) -> TransformResult:
    """Тождественное преобразование."""
    return TransformResult(code=code, name="identity", params={}, ok=True)


# --------------------------------------------------------------------------- strip_comments


def strip_comments(code: str, lang: str, rng: random.Random, **params: Any) -> TransformResult:
    """Удаление комментариев и docstring (normalize.strip_comments); failed, если комментариев нет."""
    out = normalize.strip_comments(code, canon_lang(lang))
    if out == code or not out.strip():
        return TransformResult.failed("strip_comments")
    return TransformResult(code=out, name="strip_comments", params={}, ok=True)


# --------------------------------------------------------------------------- reformat

_INDENT_UNITS = ("2", "4", "tab")


def _detect_indent(lines: list[str]) -> str | None:
    """Определяет единицу отступа: '2' | '4' | 'tab' | None (нет отступов)."""
    widths: list[int] = []
    tabs = 0
    for ln in lines:
        if not ln.strip():
            continue
        lead = ln[: len(ln) - len(ln.lstrip())]
        if not lead:
            continue
        if lead.startswith("\t"):
            tabs += 1
        else:
            widths.append(len(lead))
    if not widths and not tabs:
        return None
    if tabs >= max(1, len(widths)):
        return "tab"
    for u in (8, 4, 2):
        ok = sum(1 for w in widths if w % u == 0)
        if ok >= 0.9 * len(widths) and any(w == u for w in widths):
            return "4" if u == 8 else str(u)
    return "4" if min(widths) >= 4 else "2"


def _reindent(code: str, unit_from: str, unit_to: str) -> str:
    unit_w = 1 if unit_from == "tab" else int(unit_from)
    new = "\t" if unit_to == "tab" else " " * int(unit_to)
    out: list[str] = []
    for ln in code.split("\n"):
        if not ln.strip():
            out.append(ln.rstrip())
            continue
        lead = ln[: len(ln) - len(ln.lstrip())]
        rest = ln[len(lead):]
        if unit_from == "tab":
            level = len(lead) - len(lead.lstrip("\t"))
            rem = lead[level:].replace("\t", " ")
        else:
            w = len(lead.replace("\t", " " * 4))
            level, r = divmod(w, unit_w)
            rem = " " * r
        out.append(new * level + rem + rest)
    return "\n".join(out)


def _space_ops(code: str, lang: str, mode: str) -> str:
    """Расставляет (add) или убирает (remove) пробелы вокруг операторов между токенами одной строки."""
    p = parse_code(code, lang)
    lv = leaves(p)
    src = p.src
    edits: list[tuple[int, int, bytes]] = []
    kws = KEYWORDS[lang]

    def kind(n) -> str:
        if _is_atomic(n):
            return "atom"
        t = p.text(n)
        if n.type in normalize._NUM_TYPES:
            return "num"
        if t in kws:
            return "kw"
        if t[:1].isalpha() or t[:1] == "_":
            return "id"
        if t[:1].isdigit():
            return "num"
        if any(ch in t for ch in "+-*/%=<>!&|^~?:."):
            return "op"
        return "punct"

    for i in range(len(lv) - 1):
        a, b = lv[i], lv[i + 1]
        if a.end_point[0] != b.start_point[0]:
            continue  # разные строки — не трогаем
        if not (p.in_original(a) and p.in_original(b)):
            continue
        ka, kb = kind(a), kind(b)
        gap = src[a.end_byte : b.start_byte]
        if b"\n" in gap:
            continue
        if ka == "op" and kb == "op":
            continue
        if ka != "op" and kb != "op":
            continue
        op, other = (a, b) if ka == "op" else (b, a)
        kother = kb if ka == "op" else ka
        if p.text(op) in ("...", "->", "=>", "::", "@"):
            continue
        if mode == "add":
            if gap == b"":
                edits.append((a.end_byte, b.start_byte, b" "))
        else:  # remove
            if gap and gap.strip() == b"" and kother in ("id", "num", "punct"):
                ot = p.text(other)
                if ot[:1] in "([{" or ot[-1:] in ")]}":
                    continue
                if p.text(op) in (".",) and kother == "num":
                    continue
                edits.append((a.end_byte, b.start_byte, b""))
    if not edits:
        return code
    return p.unwrap(apply_edits(src, edits))


def _brace_style(code: str, lang: str, style: str) -> str:
    """Перенос открывающей скобки блока: K&R ('knr') ↔ Allman ('allman'); не для go/python."""
    if lang in ("python", "go"):
        return code
    p = parse_code(code, lang)
    src = p.src
    blocks = BLOCK_NODE_TYPES[lang] | {"class_body", "enum_body", "switch_block", "constructor_body"}
    lv = leaves(p)
    idx_by_start = {n.start_byte: i for i, n in enumerate(lv)}
    edits: list[tuple[int, int, bytes]] = []
    for n in p.walk():
        if n.type not in blocks or not p.in_original(n) or not n.children:
            continue
        brace = n.children[0]
        if p.text(brace) != "{":
            continue
        i = idx_by_start.get(brace.start_byte)
        if i is None or i == 0:
            continue
        prev = lv[i - 1]
        if not p.in_original(prev) or _is_atomic(prev):
            continue
        gap = src[prev.end_byte : brace.start_byte]
        if gap.strip() != b"":
            continue
        if style == "allman" and b"\n" not in gap:
            # отступ строки, где начинается оператор-владелец блока
            line_start = src.rfind(b"\n", 0, n.parent.start_byte if n.parent else prev.start_byte) + 1
            lead = src[line_start:]
            lead = lead[: len(lead) - len(lead.lstrip())]
            edits.append((prev.end_byte, brace.start_byte, b"\n" + lead))
        elif style == "knr" and b"\n" in gap:
            edits.append((prev.end_byte, brace.start_byte, b" "))
    if not edits:
        return code
    return p.unwrap(apply_edits(src, edits))


_PY_SIMPLE = {"return_statement", "pass_statement", "break_statement", "continue_statement", "expression_statement",
              "raise_statement", "assert_statement", "delete_statement", "global_statement", "nonlocal_statement"}


def _join_blocks(code: str, lang: str, rng: random.Random) -> tuple[str, int]:
    """Безопасное слияние строк: блок из одного простого оператора без комментариев → одна строка."""
    p = parse_code(code, lang)
    src = p.src
    edits: list[tuple[int, int, bytes]] = []
    if lang == "python":
        for n in p.walk():
            if n.type != "block" or not p.in_original(n):
                continue
            named = [c for c in n.children if c.is_named]
            if len(named) != 1 or named[0].type not in _PY_SIMPLE or n.parent is None:
                continue
            stmt = named[0]
            if stmt.start_point[0] != stmt.end_point[0]:
                continue
            # двоеточие заголовка — последний лист перед блоком
            colon_end = None
            for c in n.parent.children:
                if c.type == ":" and c.end_byte <= stmt.start_byte:
                    colon_end = c.end_byte
            if colon_end is None or src[colon_end : stmt.start_byte].strip() != b"":
                continue
            if b"\n" not in src[colon_end : stmt.start_byte]:
                continue
            if rng.random() < 0.7:
                edits.append((colon_end, stmt.start_byte, b" "))
    else:
        for n in p.walk():
            if n.type not in BLOCK_NODE_TYPES[lang] or not p.in_original(n) or n.type == "statement_list":
                continue
            kids = n.children
            if len(kids) != 3 or p.text(kids[0]) != "{" or p.text(kids[2]) != "}":
                continue
            stmt = kids[1]
            if not stmt.is_named or stmt.type in BLOCK_NODE_TYPES[lang] or "comment" in stmt.type:
                continue
            if stmt.start_point[0] != stmt.end_point[0] or stmt.start_point[0] == kids[0].start_point[0]:
                continue
            if stmt.type not in STATEMENT_NODE_TYPES[lang] or stmt.type.endswith("_statement") and stmt.type in (
                "if_statement", "for_statement", "while_statement", "do_statement", "switch_statement",
                "try_statement", "for_in_statement", "for_range_loop", "enhanced_for_statement",
            ):
                continue
            if rng.random() < 0.7:
                edits.append((kids[0].end_byte, stmt.start_byte, b" "))
                edits.append((stmt.end_byte, kids[2].start_byte, b" "))
    if not edits:
        return code, 0
    return p.unwrap(apply_edits(src, edits)), len(edits)


def reformat(code: str, lang: str, rng: random.Random, **params: Any) -> TransformResult:
    """Изменение форматирования: отступы (2/4/tab), пробелы вокруг операторов, перенос скобок, слияние строк."""
    lang = canon_lang(lang)
    lines = code.split("\n")
    cur = _detect_indent(lines)
    choices = [u for u in _INDENT_UNITS if u != cur]
    indent_to = params.get("indent") or rng.choice(choices)
    spaces = params.get("spaces") or rng.choice(["add", "remove", "keep"])
    braces = params.get("braces") or (rng.choice(["allman", "knr", "keep"]) if lang not in ("python", "go") else "keep")
    join = params.get("join_blocks", rng.random() < 0.5)
    out = code
    applied: dict[str, Any] = {"indent": indent_to, "spaces": spaces, "braces": braces, "join_blocks": bool(join)}
    try:
        n_join = 0
        if join:
            out, n_join = _join_blocks(out, lang, rng)
            applied["n_joined"] = n_join
        if braces != "keep":
            out = _brace_style(out, lang, braces)
        if spaces != "keep":
            out = _space_ops(out, lang, spaces)
        cur2 = _detect_indent(out.split("\n"))
        if cur2 is not None:
            out = _reindent(out, cur2, indent_to)
        if not code.endswith("\n") and out.endswith("\n"):
            out = out.rstrip("\n")
        elif code.endswith("\n") and not out.endswith("\n"):
            out += "\n"
    except Exception as e:  # pragma: no cover - защитный барьер
        log.debug("reformat failed: %s", e)
        return TransformResult.failed("reformat", applied)
    if out == code:
        return TransformResult.failed("reformat", applied)
    return TransformResult(code=out, name="reformat", params=applied, ok=True)


# --------------------------------------------------------------------------- rename_ids

_WORDS = (
    "data", "value", "count", "item", "node", "buf", "size", "index", "result", "total", "key", "entry", "state",
    "flag", "pos", "tmp", "ptr", "len", "acc", "cur", "next", "prev", "src", "dst", "ctx", "arg", "ret", "out", "inp",
    "sum", "num", "elem", "list", "name", "text", "line", "word", "code", "head", "tail", "left", "right", "step",
    "limit", "mask", "bits", "rec", "obj", "ref", "cfg", "opt", "val", "idx", "cnt", "res",
)
_SHORT = ("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "m", "n", "p", "q", "r", "s", "t", "u", "v", "w",
          "x", "y", "z", "aa", "bb", "cc", "ii", "jj", "kk", "nn", "tmp", "val", "res", "idx", "cnt", "ptr", "buf")
RENAME_STYLES = ("vn", "snake", "camel", "short")

_ID_TYPE = "identifier"


def _collect_declared(p: Parsed) -> tuple[list[str], list[Any]]:
    """Имена, объявленные внутри функции (по языку), и узлы «имя функции», которые надо переименовать явно."""
    lang = p.lang
    names: list[str] = []
    fn_name_nodes: list[Any] = []
    seen: set[str] = set()

    def add(node) -> None:
        if node is None:
            return
        if node.type == _ID_TYPE:
            t = p.text(node)
            if t and t not in seen:
                seen.add(t)
                names.append(t)
        elif node.type in ("pattern_list", "tuple_pattern", "list_pattern", "expression_list", "array_pattern",
                           "parenthesized_expression"):
            # контейнеры паттернов: все идентификаторы внутри (рекурсивно)
            for c in node.children:
                if c.is_named:
                    add(c)
        elif node.type in ("parenthesized_declarator", "pointer_declarator", "reference_declarator",
                           "array_declarator", "init_declarator", "variable_declarator", "inferred_parameters",
                           "lambda_parameters", "default_parameter", "typed_parameter", "typed_default_parameter",
                           "list_splat_pattern", "dictionary_splat_pattern", "rest_pattern", "assignment_pattern",
                           "attributed_declarator", "variadic_parameter_declaration"):
            # вложенные деклараторы/паттерны: имя — поле name/declarator/left или первый identifier
            inner = node.child_by_field_name("name") or node.child_by_field_name("declarator") \
                or node.child_by_field_name("left")
            if inner is not None and inner is not node:
                add(inner)
            else:
                for c in node.children:
                    if c.type == _ID_TYPE or c.type in ("pointer_declarator", "parenthesized_declarator",
                                                         "reference_declarator", "array_declarator"):
                        add(c)
                        if c.type == _ID_TYPE:
                            break

    fn_types = FUNCTION_NODE_TYPES[lang]
    for n in p.walk():
        if not p.in_original(n):
            continue
        t = n.type
        if t in fn_types:
            name = n.child_by_field_name("name")
            if name is not None:
                if name.type == _ID_TYPE:
                    add(name)
                elif name.type in ("field_identifier", "property_identifier"):
                    fn_name_nodes.append(name)
            if lang in ("c", "cpp"):
                d = n.child_by_field_name("declarator")
                while d is not None and d.type != "function_declarator":
                    d = d.child_by_field_name("declarator") or next((c for c in d.children if c.is_named), None)
                if d is not None:
                    ident = d.child_by_field_name("declarator")
                    if ident is not None and ident.type == "qualified_identifier":
                        ident = ident.child_by_field_name("name")
                    if ident is not None and ident.type == _ID_TYPE:
                        add(ident)
        if lang == "python":
            if t in ("parameters", "lambda_parameters"):
                for c in n.children:
                    if c.is_named:
                        add(c)
            elif t in ("assignment", "augmented_assignment"):
                add(n.child_by_field_name("left"))
            elif t in ("for_statement", "for_in_clause"):
                add(n.child_by_field_name("left"))
            elif t == "named_expression":
                add(n.child_by_field_name("name"))
            elif t == "as_pattern":
                alias = n.child_by_field_name("alias")
                if alias is not None:
                    for c in alias.children if alias.child_count else [alias]:
                        add(c)
                    if alias.type == _ID_TYPE:
                        add(alias)
            elif t in ("global_statement", "nonlocal_statement"):
                for c in n.children:
                    if c.type == _ID_TYPE:
                        seen.add(p.text(c))  # внешние имена — не переименовывать
        elif lang in ("c", "cpp"):
            if t in ("parameter_declaration", "declaration", "optional_parameter_declaration",
                     "variadic_parameter_declaration"):
                for i, c in enumerate(n.children):
                    if n.field_name_for_child(i) == "declarator":
                        add(c)
            elif t == "for_range_loop":
                add(n.child_by_field_name("declarator"))
            elif t == "init_declarator" and n.parent is not None and n.parent.type == "condition_clause":
                add(n.child_by_field_name("declarator"))
        elif lang == "go":
            if t in ("parameter_declaration", "variadic_parameter_declaration", "var_spec", "const_spec"):
                for i, c in enumerate(n.children):
                    if n.field_name_for_child(i) == "name":
                        add(c)
            elif t in ("short_var_declaration", "range_clause"):
                left = n.child_by_field_name("left")
                if left is not None and (t == "short_var_declaration" or any(p.text(c) == ":=" for c in n.children)):
                    for c in left.children:
                        add(c)
            elif t == "type_switch_statement":
                alias = n.child_by_field_name("alias")
                if alias is not None:
                    for c in alias.children:
                        add(c)
            elif t == "receive_statement":
                left = n.child_by_field_name("left")
                if left is not None and any(p.text(c) == ":=" for c in n.children):
                    for c in left.children:
                        add(c)
        elif lang == "java":
            if t in ("formal_parameter", "catch_formal_parameter", "enhanced_for_statement", "resource",
                     "variable_declarator", "instanceof_expression", "type_pattern", "record_pattern"):
                add(n.child_by_field_name("name"))
            elif t == "spread_parameter":
                for c in n.children:
                    if c.type == "variable_declarator":
                        add(c)
            elif t == "inferred_parameters":
                for c in n.children:
                    add(c)
            elif t == "lambda_expression":
                prm = n.child_by_field_name("parameters")
                if prm is not None and prm.type == _ID_TYPE:
                    add(prm)
        elif lang == "javascript":
            if t == "formal_parameters":
                for c in n.children:
                    if c.is_named:
                        add(c)
            elif t == "variable_declarator":
                add(n.child_by_field_name("name"))
            elif t in ("for_in_statement",):
                add(n.child_by_field_name("left"))
            elif t == "catch_clause":
                add(n.child_by_field_name("parameter"))
            elif t == "arrow_function":
                add(n.child_by_field_name("parameter"))
            elif t == "class_declaration":
                add(n.child_by_field_name("name"))
    kws = KEYWORDS[lang]
    names = [x for x in names if x not in kws and x != "_" and not (x.startswith("__") and x.endswith("__"))]
    return names, fn_name_nodes


def _excluded_use(p: Parsed, node) -> bool:
    """Позиции, где identifier не является ссылкой на локальное имя (обращения к членам, именованные аргументы...)."""
    par = node.parent
    if par is None:
        return False
    lang = p.lang
    pt = par.type
    idx = par.children.index(node)
    field = par.field_name_for_child(idx)
    if lang == "python":
        if pt == "attribute" and field == "attribute":
            return True
        if pt == "keyword_argument" and field == "name":
            return True
        if pt in ("import_statement", "import_from_statement", "dotted_name", "aliased_import", "decorator"):
            return True
    elif lang in ("c", "cpp"):
        if pt == "qualified_identifier":
            gp = par.parent
            # Foo::bar в объявлении функции — переименовываем; остальные квалифицированные имена — нет
            return not (field == "name" and gp is not None and gp.type == "function_declarator")
        if pt in ("preproc_def", "preproc_function_def", "preproc_ifdef"):
            return True
        if pt == "field_designator" or pt == "designated_initializer":
            return True
    elif lang == "go":
        if pt == "literal_element" and par.parent is not None and par.parent.type == "keyed_element" \
                and par.parent.children and par.parent.children[0] is par:
            return True
        if pt in ("package_clause", "import_spec", "qualified_type"):
            return True
    elif lang == "java":
        if pt == "field_access" and field == "field":
            return True
        if pt == "method_invocation" and field == "name" and par.child_by_field_name("object") is not None:
            return True
        if pt in ("scoped_identifier", "method_reference", "annotation", "marker_annotation",
                  "element_value_pair", "labeled_statement", "break_statement", "continue_statement"):
            return True
    elif lang == "javascript":
        if pt == "member_expression" and field == "property":
            return True
        if pt in ("pair", "labeled_statement", "break_statement", "continue_statement") and field in ("key", "label"):
            return True
    return False


def _gen_names(n: int, style: str, taken: set[str], rng: random.Random) -> list[str]:
    out: list[str] = []
    used = set(taken)
    order = list(range(n))
    rng.shuffle(order)
    for i in range(n):
        for attempt in range(1000):
            if style == "vn":
                cand = f"v{order[i] + 1}" if attempt == 0 else f"v{order[i] + 1}_{attempt}"
            elif style == "snake":
                cand = f"{rng.choice(_WORDS)}_{rng.choice(_WORDS)}"
            elif style == "camel":
                a, b = rng.choice(_WORDS), rng.choice(_WORDS)
                cand = a + b[:1].upper() + b[1:]
            else:
                cand = rng.choice(_SHORT) if attempt < 3 else f"{rng.choice(_SHORT)}{rng.randint(1, 99)}"
            if cand not in used:
                used.add(cand)
                out.append(cand)
                break
        else:  # pragma: no cover
            cand = f"v_{i}_{rng.randint(1000, 9999)}"
            used.add(cand)
            out.append(cand)
    return out


def rename_ids(code: str, lang: str, rng: random.Random, **params: Any) -> TransformResult:
    """Согласованное переименование идентификаторов, объявленных внутри функции (параметры, локальные,
    имя функции). Стили: vn | snake | camel | short. Внешние имена, ключевые слова, поля не трогаются."""
    lang = canon_lang(lang)
    style = params.get("style") or rng.choice(RENAME_STYLES)
    p = parse_code(code, lang)
    declared, fn_name_nodes = _collect_declared(p)
    fn_extra = [p.text(n) for n in fn_name_nodes]
    all_targets = list(dict.fromkeys(declared + [t for t in fn_extra if t not in KEYWORDS[lang]]))
    if not all_targets:
        return TransformResult.failed("rename_ids", {"style": style})
    # все идентификаторы кода и все ключевые слова всех языков — заняты
    taken: set[str] = set()
    for tok in tokenize(code, lang):
        if tok.kind in ("id", "kw"):
            taken.add(tok.text)
    for kw in KEYWORDS.values():
        taken |= kw
    frac = float(params.get("fraction", 1.0))
    targets = all_targets
    if frac < 1.0 and len(all_targets) > 1:
        k = max(1, int(round(frac * len(all_targets))))
        targets = sorted(rng.sample(all_targets, k), key=all_targets.index)
    new_names = _gen_names(len(targets), style, taken, rng)
    mapping = dict(zip(targets, new_names))
    edits: list[tuple[int, int, bytes]] = []
    for n in p.walk():
        if not p.in_original(n):
            continue
        if n.type == _ID_TYPE:
            t = p.text(n)
            if t in mapping and not _excluded_use(p, n):
                edits.append((n.start_byte, n.end_byte, mapping[t].encode("utf-8")))
        elif lang == "javascript" and n.type == "shorthand_property_identifier":
            t = p.text(n)  # {a} → {a: v1}: сохраняем имя свойства, подставляем новое имя переменной
            if t in mapping:
                edits.append((n.start_byte, n.end_byte, f"{t}: {mapping[t]}".encode("utf-8")))
    for n in fn_name_nodes:
        t = p.text(n)
        if t in mapping:
            edits.append((n.start_byte, n.end_byte, mapping[t].encode("utf-8")))
    if not edits:
        return TransformResult.failed("rename_ids", {"style": style})
    out = p.unwrap(apply_edits(p.src, edits))
    return TransformResult(code=out, name="rename_ids",
                           params={"style": style, "n_renamed": len(mapping), "mapping": mapping}, ok=True)


# --------------------------------------------------------------------------- change_literals

_INT_RE = re.compile(r"^(0[xX][0-9a-fA-F]+|0[bB][01]+|[1-9][0-9]*|0)([uUlLnN]{0,3})$")
_FLOAT_RE = re.compile(r"^([0-9]*)(\.?)([0-9]*)([eE][+-]?[0-9]+)?([fFlLdD]?)$")
_STR_RE = re.compile(r"^([A-Za-z]{0,3})('''|\"\"\"|'|\"|`)(.*)(\2)$", re.S)
_DELTAS = (1, 2, 3, 5, 7, 10, 16, 100)


def _new_int(text: str, rng: random.Random) -> str | None:
    m = _INT_RE.match(text)
    if not m:
        return None
    num, suf = m.group(1), m.group(2)
    low = num.lower()
    if low.startswith("0x"):
        v = int(num, 16)
        v2 = v + rng.choice(_DELTAS)
        return ("0X" if num[1] == "X" else "0x") + format(v2, "x" if num[2:].islower() or num[2:].isdigit() else "X") + suf
    if low.startswith("0b"):
        v = int(num, 2)
        return num[:2] + bin(v + rng.choice(_DELTAS))[2:] + suf
    v = int(num)
    d = rng.choice(_DELTAS)
    v2 = v + d if (v == 0 or rng.random() < 0.6 or v - d <= 0) else v - d
    if v2 == v:
        v2 = v + 1
    return str(v2) + suf


def _new_float(text: str, rng: random.Random) -> str | None:
    m = _FLOAT_RE.match(text)
    if not m or not any(ch.isdigit() for ch in text) or (not m.group(2) and not m.group(4)):
        return None
    ip, dot, fp, ex, suf = m.groups()
    digits = ip + fp
    if not digits:
        return None
    # меняем последнюю цифру мантиссы
    last = int(digits[-1])
    new_last = str((last + rng.randint(1, 8)) % 10)
    if fp:
        fp = fp[:-1] + new_last
    else:
        ip = ip[:-1] + new_last
        if ip == "0" and dot == "":
            ip = "1"
    out = ip + dot + fp + (ex or "") + (suf or "")
    return None if out == text else out


def _new_string(text: str, rng: random.Random) -> str | None:
    m = _STR_RE.match(text)
    if not m:
        return None
    pre, q, content, _ = m.groups()
    if len(content) == 0:
        words = [rng.choice(_WORDS)]
    else:
        n = max(1, min(len(content) // 4 + 1, 6))
        words = [rng.choice(_WORDS) for _ in range(n)]
    new_content = " ".join(words)
    out = pre + q + new_content + q
    return None if out == text else out


def change_literals(code: str, lang: str, rng: random.Random, **params: Any) -> TransformResult:
    """Замена числовых и строковых литералов (docstring/комментарии не трогаются)."""
    lang = canon_lang(lang)
    frac = float(params.get("fraction", 1.0))
    src = code.encode("utf-8", "replace")
    edits: list[tuple[int, int, bytes]] = []
    n_num = n_str = 0
    for tok in tokenize(code, lang):
        if tok.kind not in ("num", "str"):
            continue
        if frac < 1.0 and rng.random() > frac:
            continue
        new = None
        if tok.kind == "num":
            if "_" in tok.text or "'" in tok.text:
                continue
            new = _new_int(tok.text, rng) or _new_float(tok.text, rng)
            if new:
                n_num += 1
        else:
            if lang == "python" and tok.text[:1].lower() in ("f",) or tok.text[:2].lower() in ("fr", "rf"):
                new = None  # f-строки: содержимое может быть выражением
            else:
                new = _new_string(tok.text, rng)
            if new:
                n_str += 1
        if new:
            edits.append((tok.start, tok.end, new.encode("utf-8")))
    if not edits:
        return TransformResult.failed("change_literals", {})
    out = apply_edits(src, edits).decode("utf-8", "replace")
    return TransformResult(code=out, name="change_literals", params={"n_num": n_num, "n_str": n_str}, ok=True)


# --------------------------------------------------------------------------- combo

COMBO_STEPS: tuple[str, ...] = ("reformat", "strip_comments", "rename_ids", "change_literals")


def combo(code: str, lang: str, rng: random.Random, **params: Any) -> TransformResult:
    """reformat + strip_comments + rename_ids + change_literals (неприменимые шаги пропускаются)."""
    lang = canon_lang(lang)
    steps: dict[str, Callable[..., TransformResult]] = {
        "reformat": reformat, "strip_comments": strip_comments, "rename_ids": rename_ids,
        "change_literals": change_literals,
    }
    cur = code
    applied: list[str] = []
    sub: dict[str, Any] = {}
    orig_ok = parses_ok(code, lang)
    for name in COMBO_STEPS:
        try:
            r = steps[name](cur, lang, rng)
        except Exception as e:  # pragma: no cover
            log.debug("combo step %s failed: %s", name, e)
            continue
        if not r.ok or not r.code:
            continue
        if orig_ok and not parses_ok(r.code, lang):
            continue
        cur = r.code
        applied.append(name)
        sub[name] = {k: v for k, v in r.params.items() if k != "mapping"}
    if not applied or cur == code:
        return TransformResult.failed("combo", {"steps": applied})
    return TransformResult(code=cur, name="combo", params={"steps": applied, **sub}, ok=True)
