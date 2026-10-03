"""Реестр преобразований кода (DESIGN.md §4): ``apply_transform(name, code, lang, rng, **params)``.

Каждое преобразование возвращает ``TransformResult``; результат принимается только если он не пуст,
отличается от исходного кода (кроме identity) и разбирается tree-sitter без ошибок
(``programmatic.parses_ok``), при условии что исходный код разбирался.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Callable

from smcode.normalize import canon_lang
from smcode.transforms import evasion, partial as _partial_mod, programmatic
from smcode.types import TransformResult

log = logging.getLogger(__name__)

TransformFn = Callable[..., TransformResult]

TRANSFORMS: dict[str, TransformFn] = {
    "identity": programmatic.identity,
    "reformat": programmatic.reformat,
    "strip_comments": programmatic.strip_comments,
    "rename_ids": programmatic.rename_ids,
    "change_literals": programmatic.change_literals,
    "insert_deadcode": evasion.insert_deadcode,
    "reorder_stmts": evasion.reorder_stmts,
    "combo": programmatic.combo,
    "partial": _partial_mod.partial,
}
# LLM-преобразования работают батчами на GPU (transforms/llm_paraphrase.py); через apply_transform недоступны.
LLM_TRANSFORMS: tuple[str, ...] = ("paraphrase", "translate")


def list_transforms(include_llm: bool = False) -> list[str]:
    """Имена доступных преобразований (программные; с include_llm — плюс LLM-имена)."""
    names = list(TRANSFORMS)
    if include_llm:
        names += list(LLM_TRANSFORMS)
    return names


def apply_transform(name: str, code: str, lang: str, rng: random.Random, **params: Any) -> TransformResult:
    """Применяет преобразование ``name``; при неприменимости/поломке синтаксиса — ``TransformResult.failed``."""
    if name in LLM_TRANSFORMS:
        log.debug("transform %s is batch/GPU-only; use llm_paraphrase", name)
        return TransformResult.failed(name, dict(params))
    fn = TRANSFORMS.get(name)
    if fn is None:
        raise KeyError(f"unknown transform: {name!r}; known: {list_transforms(include_llm=True)}")
    lang = canon_lang(lang)
    if not code or not code.strip():
        return TransformResult.failed(name, dict(params))
    try:
        res = fn(code, lang, rng, **params)
    except Exception as e:  # защитный барьер: любое исключение → неприменимо
        log.warning("transform %s raised %s: %s", name, type(e).__name__, e)
        return TransformResult.failed(name, dict(params))
    if res is None or not res.ok or not res.code or not res.code.strip():
        return TransformResult.failed(name, (res.params if res is not None else None) or dict(params))
    if name != "identity" and res.code == code:
        return TransformResult.failed(name, res.params)
    if name != "identity" and not programmatic.parses_ok(res.code, lang) and programmatic.parses_ok(code, lang):
        log.debug("transform %s broke syntax (%s); rejected", name, lang)
        return TransformResult.failed(name, res.params)
    res.name = name
    return res
