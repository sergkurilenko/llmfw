"""Общие типы записей и утилиты ввода-вывода (контракты из docs/semantic_membership/DESIGN.md, §2.4, §5, §6)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol, runtime_checkable

SPLITS = ("protected", "hard_neg", "public_train", "public_calib", "public_test")


@dataclass
class FunctionRecord:
    """Одна функция (или файловое окно) корпуса. См. DESIGN.md §2.4."""

    id: str
    split: str
    repo: str
    path: str
    lang: str
    code: str
    start_line: int
    end_line: int
    n_lines: int
    n_tokens: int
    sha: str
    kind: str = "function"  # function | window

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FunctionRecord":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class QueryRecord:
    """Запрос к детектору членства. См. DESIGN.md §5. label=1 только для set=protected."""

    qid: str
    source_id: str | None
    label: int
    set: str
    transform: str
    params: dict[str, Any]
    lang: str
    repo: str
    code: str
    n_lines: int
    n_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QueryRecord":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class QueryResult:
    """Ответ индекса на запрос: score ∈ [0, 1] (больше — вероятнее член)."""

    score: float
    best_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None


@dataclass
class TransformResult:
    """Результат применения преобразования к коду."""

    code: str | None
    name: str
    params: dict[str, Any]
    ok: bool

    @classmethod
    def failed(cls, name: str, params: dict[str, Any] | None = None) -> "TransformResult":
        return cls(code=None, name=name, params=params or {}, ok=False)


@runtime_checkable
class MembershipIndex(Protocol):
    """Единый API индекса членства (DESIGN.md §6). Реализации: exact, winnowing, minhash, semantic, hybrid."""

    name: str

    def build(self, records: Iterable[FunctionRecord], cfg: dict[str, Any]) -> None: ...

    def query(self, code: str, lang: str) -> QueryResult: ...

    def query_batch(self, items: list[tuple[str, str]]) -> list[QueryResult]: ...

    def save(self, path: str | Path) -> None: ...

    def load(self, path: str | Path) -> None: ...

    def memory_bytes(self) -> int: ...


# ----------------------------------------------------------------------------- jsonl


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Ленивое чтение JSONL. Пустые строки пропускаются."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any] | Any]) -> int:
    """Запись JSONL; объекты с методом to_dict сериализуются через него. Возвращает число строк."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            if hasattr(row, "to_dict"):
                row = row.to_dict()
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_functions(path: str | Path) -> Iterator[FunctionRecord]:
    for d in read_jsonl(path):
        yield FunctionRecord.from_dict(d)


def read_queries(path: str | Path) -> Iterator[QueryRecord]:
    for d in read_jsonl(path):
        yield QueryRecord.from_dict(d)
