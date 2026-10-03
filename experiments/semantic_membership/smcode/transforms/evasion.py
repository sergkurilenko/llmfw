"""Преобразования-уклонения (DESIGN.md §4): insert_deadcode (атака Mossad) и reorder_stmts."""

from __future__ import annotations

import logging
import random
from typing import Any

from smcode.normalize import canon_lang
from smcode.transforms.programmatic import (
    BLOCK_NODE_TYPES,
    Parsed,
    apply_edits,
    block_children,
    parse_code,
)
from smcode.types import TransformResult

log = logging.getLogger(__name__)

# Шаблоны no-op операторов; {k} — уникальный номер вставки.
DEADCODE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "python": ("_tmp_{k} = 0", "_tmp_{k} = None", "_tmp_{k} = []"),
    "c": ("int _tmp_{k} = 0; (void)_tmp_{k};", "do {{ }} while (0);", "(void)0;"),
    "cpp": ("int _tmp_{k} = 0; (void)_tmp_{k};", "do {{ }} while (0);", "(void)0;"),
    "go": ("_ = 0", "var _tmp_{k} int; _ = _tmp_{k}", "_ = \"\""),
    "java": ("int _tmp_{k} = 0;", "long _tmp_{k} = 0L;", "boolean _tmp_{k} = false;"),
    "javascript": ("void 0;", "let _tmp_{k} = 0;", "const _tmp_{k} = null;"),
}

# Узлы, внутри которых переставлять операторы небезопасно (вызовы, побочные эффекты).
_UNSAFE_TYPES = {
    "call", "call_expression", "method_invocation", "object_creation_expression", "new_expression", "await",
    "await_expression", "yield", "yield_expression", "update_expression", "inc_statement", "dec_statement",
    "lambda", "lambda_expression", "arrow_function", "func_literal", "function", "function_expression",
    "generator_expression", "list_comprehension", "dictionary_comprehension", "set_comprehension",
    "conditional_expression", "ternary_expression", "delete_expression", "unary_expression", "pointer_expression",
    "ERROR",
}
_UNSAFE_TOKENS = {"<-", "++", "--"}
# «Простые» операторы для перестановки.
_SIMPLE_TYPES: dict[str, set[str]] = {
    "python": {"expression_statement", "assignment", "augmented_assignment"},
    "c": {"expression_statement", "declaration"},
    "cpp": {"expression_statement", "declaration"},
    "go": {"short_var_declaration", "assignment_statement", "var_declaration", "const_declaration"},
    "java": {"expression_statement", "local_variable_declaration"},
    "javascript": {"expression_statement", "lexical_declaration", "variable_declaration"},
}
_ASSIGN_TYPES = {"assignment", "augmented_assignment", "assignment_expression", "augmented_assignment_expression"}
_ID_LIKE = {
    "identifier", "field_identifier", "property_identifier", "type_identifier", "shorthand_property_identifier",
    "shorthand_property_identifier_pattern", "statement_identifier", "namespace_identifier", "label_name",
    "this", "self", "super",
}


def _first_on_line(p: Parsed, node) -> str | None:
    """Если узел — первое непробельное на своей строке, возвращает ведущий отступ строки, иначе None."""
    line_start = p.src.rfind(b"\n", 0, node.start_byte) + 1
    lead = p.src[line_start : node.start_byte]
    if lead.strip() != b"":
        return None
    return lead.decode("utf-8", "replace")


def _block_statements(p: Parsed, block) -> list[Any]:
    return block_children(p, block)


def _iter_blocks(p: Parsed):
    blocks = BLOCK_NODE_TYPES[p.lang]
    for n in p.walk():
        if n.type in blocks and p.in_original(n):
            yield n


def insert_deadcode(code: str, lang: str, rng: random.Random, **params: Any) -> TransformResult:
    """Вставка синтаксически корректных no-op операторов каждые ``every`` строк тела (минимум одна)."""
    lang = canon_lang(lang)
    every = int(params.get("every", 3))
    p = parse_code(code, lang)
    candidates: list[tuple[int, int, str]] = []  # (row, byte, indent)
    for block in _iter_blocks(p):
        for st in _block_statements(p, block):
            if st.type in ("case_statement",):
                continue
            indent = _first_on_line(p, st)
            if indent is None:
                continue
            candidates.append((st.start_point[0], st.start_byte, indent))
    if not candidates:
        return TransformResult.failed("insert_deadcode", {"every": every})
    candidates.sort()
    # дедупликация по байту
    uniq: list[tuple[int, int, str]] = []
    seen_b: set[int] = set()
    for c in candidates:
        if c[1] not in seen_b:
            seen_b.add(c[1])
            uniq.append(c)
    first_row = uniq[0][0]
    chosen: list[tuple[int, int, str]] = []
    last_row = first_row - every + rng.randint(0, max(0, every - 1))
    for row, b, ind in uniq:
        if row - last_row >= every:
            chosen.append((row, b, ind))
            last_row = row
    if not chosen:
        chosen = [rng.choice(uniq)]
    templates = DEADCODE_TEMPLATES[lang]
    edits: list[tuple[int, int, bytes]] = []
    for k, (_, b, ind) in enumerate(chosen, start=1):
        stmt = rng.choice(templates).format(k=k)
        edits.append((b, b, (stmt + "\n" + ind).encode("utf-8")))
    out = p.unwrap(apply_edits(p.src, edits))
    return TransformResult(code=out, name="insert_deadcode",
                           params={"every": every, "n_inserted": len(chosen)}, ok=True)


def _ids_of(p: Parsed, node) -> set[str]:
    out: set[str] = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type in _ID_LIKE or (n.child_count == 0 and n.is_named and p.text(n)[:1].isalpha()):
            out.add(p.text(n))
        stack.extend(n.children)
    return out


def _is_simple_safe(p: Parsed, node) -> bool:
    if node.type not in _SIMPLE_TYPES[p.lang]:
        return False
    if node.start_point[0] != node.end_point[0]:
        return False
    if node.type == "expression_statement":
        inner = [c for c in node.children if c.is_named]
        if len(inner) != 1 or inner[0].type not in _ASSIGN_TYPES:
            return False
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type in _UNSAFE_TYPES or n.has_error:
            return False
        if n.child_count == 0 and p.text(n) in _UNSAFE_TOKENS:
            return False
        stack.extend(n.children)
    return True


def reorder_stmts(code: str, lang: str, rng: random.Random, **params: Any) -> TransformResult:
    """Перестановка соседних независимых простых операторов одного блока (нет общих идентификаторов,
    нет вызовов, между ними нет комментариев)."""
    lang = canon_lang(lang)
    p = parse_code(code, lang)
    edits: list[tuple[int, int, bytes]] = []
    n_swaps = 0
    for block in _iter_blocks(p):
        stmts = block_children(p, block)
        i = rng.randint(0, 1) if len(stmts) > 2 else 0
        while i + 1 < len(stmts):
            a, b = stmts[i], stmts[i + 1]
            if _is_simple_safe(p, a) and _is_simple_safe(p, b) and _first_on_line(p, a) is not None \
                    and _first_on_line(p, b) is not None and not (_ids_of(p, a) & _ids_of(p, b)):
                gap = p.src[a.end_byte : b.start_byte]
                if gap.strip() == b"" and b"\n" in gap:
                    edits.append((a.start_byte, a.end_byte, p.src[b.start_byte : b.end_byte]))
                    edits.append((b.start_byte, b.end_byte, p.src[a.start_byte : a.end_byte]))
                    n_swaps += 1
                    i += 2
                    continue
            i += 1
    if not edits:
        return TransformResult.failed("reorder_stmts", {})
    out = p.unwrap(apply_edits(p.src, edits))
    if out == code:
        return TransformResult.failed("reorder_stmts", {})
    return TransformResult(code=out, name="reorder_stmts", params={"n_swaps": n_swaps}, ok=True)
