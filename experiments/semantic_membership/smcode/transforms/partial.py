"""Частичное извлечение (DESIGN.md §4, `partial`): окно из L подряд идущих строк тела функции,
по возможности выровненное по границам операторов и разбираемое без ошибок."""

from __future__ import annotations

import logging
import random
import textwrap
from typing import Any

from smcode.normalize import canon_lang
from smcode.transforms.programmatic import (
    BLOCK_NODE_TYPES,
    Parsed,
    block_children,
    body_node,
    find_function_node,
    parse_code,
    parses_ok,
)
from smcode.types import TransformResult

log = logging.getLogger(__name__)

_MAX_TRIES = 12


def _statement_rows(p: Parsed, body) -> tuple[set[int], set[int]]:
    """Строки начала и конца операторов (любой вложенности) внутри тела: дети блоков и корня."""
    starts: set[int] = set()
    ends: set[int] = set()
    blocks = BLOCK_NODE_TYPES[p.lang]
    stack = [body]
    while stack:
        n = stack.pop()
        if n.type in blocks or n is p.root:
            for c in block_children(p, n):
                starts.add(c.start_point[0])
                ends.add(c.end_point[0])
        stack.extend(n.children)
    return starts, ends


def _window_text(lines: list[str], start: int, L: int) -> str:
    win = lines[start : start + L]
    return textwrap.dedent("\n".join(win)).rstrip() + "\n"


def partial(code: str, lang: str, rng: random.Random, L: int = 5, **params: Any) -> TransformResult:
    """Окно из L подряд идущих непустых строк тела. Кандидаты: выровненные по началу и концу оператора,
    затем по началу, затем любые; среди них предпочитается разбираемый фрагмент."""
    lang = canon_lang(lang)
    L = int(L)
    if L <= 0:
        return TransformResult.failed("partial", {"L": L})
    p = parse_code(code, lang)
    # строки исходника (в системе координат обёрнутого src) → индексы в оригинале
    src_lines = p.src.decode("utf-8", "replace").split("\n")
    pre_rows = p.src[: p.prefix].count(b"\n")
    orig_lines = code.split("\n")
    n_orig = len(orig_lines)
    fn = find_function_node(p)
    body = body_node(p, fn)
    if body is not None and body.end_point[0] > body.start_point[0]:
        b0, b1 = body.start_point[0], body.end_point[0]
        # исключаем строки самих скобок тела
        first_tok = body.children[0] if body.children else None
        if first_tok is not None and p.text(first_tok) == "{" and b0 == first_tok.start_point[0]:
            b0 += 1
        last_tok = body.children[-1] if body.children else None
        if last_tok is not None and p.text(last_tok) == "}" and b1 == last_tok.start_point[0]:
            b1 -= 1
        starts, ends = _statement_rows(p, body)
    else:
        b0, b1 = pre_rows, pre_rows + n_orig - 1
        starts, ends = _statement_rows(p, p.root)
    # индексы непустых строк тела в оригинальных координатах
    rows = [r for r in range(b0, b1 + 1) if 0 <= r - pre_rows < n_orig and orig_lines[r - pre_rows].strip()]
    if len(rows) < L:
        # запасной вариант: все непустые строки функции
        rows = [r for r in range(pre_rows, pre_rows + n_orig) if orig_lines[r - pre_rows].strip()]
        if len(rows) < L or len(rows) == L and L >= n_orig:
            return TransformResult.failed("partial", {"L": L})
    n_windows = len(rows) - L + 1
    both = [i for i in range(n_windows) if rows[i] in starts and rows[i + L - 1] in ends]
    start_only = [i for i in range(n_windows) if rows[i] in starts and i not in both]
    rest = [i for i in range(n_windows) if i not in both and i not in start_only]
    for lst in (both, start_only, rest):
        rng.shuffle(lst)
    order = both + start_only + rest
    text_lines = [orig_lines[r - pre_rows] for r in rows]
    chosen = None
    chosen_parses = False
    tries = 0
    for i in order:
        txt = _window_text(text_lines, i, L)
        tries += 1
        if parses_ok(txt, lang):
            chosen, chosen_parses = i, True
            break
        if chosen is None:
            chosen = i
        if tries >= _MAX_TRIES:
            break
    if chosen is None:
        return TransformResult.failed("partial", {"L": L})
    out = _window_text(text_lines, chosen, L)
    if out.strip() == code.strip():
        return TransformResult.failed("partial", {"L": L})
    aligned = "both" if chosen in both else ("start" if chosen in start_only else "none")
    return TransformResult(code=out, name="partial",
                           params={"L": L, "start_line": int(rows[chosen] - pre_rows), "aligned": aligned,
                                   "parses": chosen_parses}, ok=True)
