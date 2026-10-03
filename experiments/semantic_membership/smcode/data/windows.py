"""Файловые окна для сценария IDE (DESIGN.md §2.2): скользящие окна по 20 строк с шагом 10
по файлам protected, kind="window", id с суффиксом ":w<start>-<end>".

Параметры читаются из cfg["windows"] (size, stride; по умолчанию 20/10) и cfg["extract"]
(min_lines, max_tokens).
"""

from __future__ import annotations

import logging
import textwrap
from typing import Any

from smcode.normalize import count_tokens, normalized_sha
from smcode.types import FunctionRecord

log = logging.getLogger(__name__)

DEFAULT_WINDOW_SIZE = 20
DEFAULT_WINDOW_STRIDE = 10


def window_params(cfg: dict[str, Any]) -> tuple[int, int]:
    """(size, stride) из cfg['windows'] с значениями по умолчанию 20/10."""
    w = cfg.get("windows") or {}
    size = int(w.get("size", DEFAULT_WINDOW_SIZE))
    stride = int(w.get("stride", DEFAULT_WINDOW_STRIDE))
    if size <= 0 or stride <= 0:
        raise ValueError(f"window size/stride must be positive: {size}/{stride}")
    return size, stride


def file_windows(n_lines: int, size: int = DEFAULT_WINDOW_SIZE, stride: int = DEFAULT_WINDOW_STRIDE) -> list[tuple[int, int]]:
    """Границы окон (1-based, включительно) для файла из n_lines строк.

    Если файл короче окна — одно окно на весь файл. Последнее окно выравнивается по концу файла,
    чтобы хвост не остался непокрытым и не возникло «обрезков» короче окна.
    """
    if n_lines <= 0:
        return []
    if n_lines <= size:
        return [(1, n_lines)]
    out: list[tuple[int, int]] = []
    start = 1
    while start + size - 1 <= n_lines:
        out.append((start, start + size - 1))
        start += stride
    last_end = out[-1][1]
    if last_end < n_lines:
        out.append((n_lines - size + 1, n_lines))
    return out


def window_id(split: str, repo_dir: str, rel_path: str, start: int, end: int) -> str:
    return f"{split}/{repo_dir}/{rel_path}:w{start}-{end}"


def windows_from_source(
    text: str,
    lang: str,
    split: str,
    repo: str,
    repo_dir: str,
    rel_path: str,
    cfg: dict[str, Any],
) -> list[FunctionRecord]:
    """Записи-окна для одного файла. Окна без токенов (только пустые строки/комментарии),
    короче extract.min_lines или длиннее extract.max_tokens пропускаются."""
    size, stride = window_params(cfg)
    ext = cfg.get("extract") or {}
    min_lines = int(ext.get("min_lines", 1))
    max_tokens = int(ext.get("max_tokens", 10**9))
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    out: list[FunctionRecord] = []
    for start, end in file_windows(len(lines), size, stride):
        chunk = lines[start - 1 : end]
        if len(chunk) < min_lines:
            continue
        code = textwrap.dedent("\n".join(chunk))
        if not code.strip():
            continue
        try:
            n_tokens = count_tokens(code, lang)
            if n_tokens == 0 or n_tokens > max_tokens:
                continue
            sha = normalized_sha(code, lang)
        except Exception as exc:  # noqa: BLE001 — окно с непарсируемым текстом пропускаем
            log.debug("window %s:%d-%d skipped: %s", rel_path, start, end, exc)
            continue
        out.append(
            FunctionRecord(
                id=window_id(split, repo_dir, rel_path, start, end),
                split=split,
                repo=repo,
                path=rel_path,
                lang=lang,
                code=code,
                start_line=start,
                end_line=end,
                n_lines=len(chunk),
                n_tokens=n_tokens,
                sha=sha,
                kind="window",
            )
        )
    return out
