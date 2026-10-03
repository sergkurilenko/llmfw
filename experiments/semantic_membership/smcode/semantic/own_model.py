"""«Своя» модель с нуля (DESIGN.md §7, own_small_model): небольшой трансформер-энкодер (torch.nn) над
словарём лексических токенов tree-sitter (normalize.tokenize) с байтовым запасным вариантом.

Словарь (CodeVocab) строится по public_train: идентификаторы дробятся на snake/camel-части (в нижнем
регистре), строковые литералы → <STR>, длинные числа → <NUM>; части вне словаря кодируются байтами
UTF-8 (256 зарезервированных токенов). Часть словаря чистая (без torch) и покрывается CPU-тестами;
torch импортируется лениво внутри OwnEncoder / _torch_classes().
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from smcode.normalize import canon_lang, tokenize

log = logging.getLogger(__name__)

PAD_ID, UNK_ID, CLS_ID, SEP_ID = 0, 1, 2, 3
N_SPECIAL = 4
N_BYTES = 256
FIRST_WORD_ID = N_SPECIAL + N_BYTES  # 260
STR_PIECE, NUM_PIECE = "<STR>", "<NUM>"
MAX_BYTES_PER_PIECE = 16
MAX_NUM_CHARS = 6
VOCAB_FILE = "vocab.json"
WEIGHTS_FILE = "model.pt"
ENCODER_META = "encoder_meta.json"
DEFAULT_HPARAMS = {"layers": 6, "hidden": 384, "heads": 6, "vocab": 32000, "dropout": 0.1}

_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_FALLBACK_RE = re.compile(r"\w+|[^\w\s]")


def split_identifier(text: str) -> list[str]:
    """Части идентификатора: snake_case и camelCase → строчные части (getHTTPResponse → get, http, response)."""
    parts: list[str] = []
    for chunk in text.split("_"):
        if not chunk:
            continue
        found = _CAMEL_RE.findall(chunk)
        parts.extend(p.lower() for p in found) if found else parts.append(chunk.lower())
    return parts or [text.lower()]


def lexical_pieces(code: str, lang: str | None) -> list[str]:
    """Последовательность словарных частей кода (комментарии удалены). При сбое парсера — regex-разбиение."""
    try:
        toks = tokenize(code, canon_lang(lang or ""))
    except Exception:  # noqa: BLE001 — неподдерживаемый язык/ошибка парсера
        return [w.lower() if w[:1].isalpha() or w[:1] == "_" else w for w in _FALLBACK_RE.findall(code or "")]
    out: list[str] = []
    for t in toks:
        if t.kind == "comment":
            continue
        if t.kind == "str":
            out.append(STR_PIECE)
        elif t.kind == "num":
            out.append(t.text if len(t.text) <= MAX_NUM_CHARS else NUM_PIECE)
        elif t.kind == "id":
            out.extend(split_identifier(t.text))
        else:
            out.append(t.text)
    return out


class CodeVocab:
    """Словарь частей: специальные (4) + байты (256) + слова. Кодирует код в список id с байтовым запасом."""

    def __init__(self, words: Sequence[str] = ()) -> None:
        self.words: list[str] = list(words)
        self.index: dict[str, int] = {w: FIRST_WORD_ID + i for i, w in enumerate(self.words)}

    def __len__(self) -> int:
        return FIRST_WORD_ID + len(self.words)

    @classmethod
    def build(cls, items: Iterable[tuple[str, str]], max_size: int = 32000, min_count: int = 2) -> "CodeVocab":
        """Строит словарь по (code, lang): самые частые части (≥ min_count), не более max_size − 260 слов."""
        counter: Counter[str] = Counter()
        n = 0
        for code, lang in items:
            counter.update(lexical_pieces(code, lang))
            n += 1
        budget = max(0, int(max_size) - FIRST_WORD_ID)
        words = [w for w, c in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])) if c >= min_count][:budget]
        log.info("own-model vocab: %d words from %d functions (%d distinct pieces)", len(words), n, len(counter))
        return cls(words)

    def piece_ids(self, piece: str) -> list[int]:
        wid = self.index.get(piece)
        if wid is not None:
            return [wid]
        raw = piece.encode("utf-8", "replace")[:MAX_BYTES_PER_PIECE]
        return [N_SPECIAL + b for b in raw] or [UNK_ID]

    def encode(self, code: str, lang: str | None, max_length: int = 512) -> list[int]:
        """[CLS] части... [SEP], усечение до max_length с сохранением [SEP]."""
        ids: list[int] = [CLS_ID]
        limit = max(2, int(max_length)) - 1
        for p in lexical_pieces(code, lang):
            ids.extend(self.piece_ids(p))
            if len(ids) >= limit:
                break
        ids = ids[:limit]
        ids.append(SEP_ID)
        return ids

    def encode_batch(self, codes: Sequence[str], langs: Sequence[str] | None, max_length: int = 512) -> tuple[np.ndarray, np.ndarray]:
        """(input_ids (B, L) int64, attention_mask (B, L) int64) с паддингом PAD_ID."""
        seqs = [self.encode(c, (langs[i] if langs is not None else None), max_length) for i, c in enumerate(codes)]
        L = max((len(s) for s in seqs), default=2)
        ids = np.full((len(seqs), L), PAD_ID, dtype=np.int64)
        mask = np.zeros((len(seqs), L), dtype=np.int64)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = s
            mask[i, : len(s)] = 1
        return ids, mask

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"words": self.words, "first_word_id": FIRST_WORD_ID}, f, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CodeVocab":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["words"])


def build_vocab_from_records(records: Iterable[Any], max_size: int = 32000, min_count: int = 2) -> CodeVocab:
    """CodeVocab по FunctionRecord/dict-записям (code, lang)."""
    def items():
        for r in records:
            if isinstance(r, dict):
                yield r["code"], r.get("lang", "")
            else:
                yield r.code, r.lang
    return CodeVocab.build(items(), max_size=max_size, min_count=min_count)


# ----------------------------------------------------------------------------- torch-часть (ленивая)


@lru_cache(maxsize=1)
def _torch_classes() -> dict[str, Any]:
    """Определяет nn.Module-классы при первом обращении (torch не нужен для импорта модуля)."""
    import torch
    from torch import nn
    from torch.utils.checkpoint import checkpoint as _ckpt

    class OwnTransformer(nn.Module):
        """Pre-LN трансформер-энкодер: эмбеддинги токенов + позиций → N слоёв → mean pooling → L2."""

        def __init__(self, vocab_size: int, hidden: int, layers: int, heads: int, max_len: int,
                     dropout: float = 0.1, pad_id: int = PAD_ID) -> None:
            super().__init__()
            self.pad_id = pad_id
            self.max_len = int(max_len)
            self.tok = nn.Embedding(vocab_size, hidden, padding_idx=pad_id)
            self.pos = nn.Embedding(self.max_len, hidden)
            self.emb_norm = nn.LayerNorm(hidden)
            self.drop = nn.Dropout(dropout)
            self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=hidden, nhead=heads, dim_feedforward=4 * hidden,
                                                                    dropout=dropout, activation="gelu", batch_first=True,
                                                                    norm_first=True) for _ in range(layers)])
            self.final_norm = nn.LayerNorm(hidden)
            self.gradient_checkpointing = False
            self.apply(self._init)

        @staticmethod
        def _init(m: nn.Module) -> None:
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.zeros_(m.bias)

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            B, L = input_ids.shape
            pos = torch.arange(L, device=input_ids.device).clamp(max=self.max_len - 1)
            x = self.tok(input_ids) + self.pos(pos)[None]
            x = self.drop(self.emb_norm(x))
            pad_mask = attention_mask == 0
            for layer in self.layers:
                if self.gradient_checkpointing and self.training:
                    x = _ckpt(layer, x, None, pad_mask, use_reentrant=False)
                else:
                    x = layer(x, src_key_padding_mask=pad_mask)
            x = self.final_norm(x).float()
            m = attention_mask.unsqueeze(-1).to(x.dtype)
            pooled = (x * m).sum(1) / torch.clamp(m.sum(1), min=1.0)
            return nn.functional.normalize(pooled, p=2, dim=-1)

    return {"OwnTransformer": OwnTransformer}


class OwnEncoder:
    """Энкодер на своей модели: тот же интерфейс, что у CodeEncoder (tokenize, forward_batch, encode, save)."""

    kind = "own"

    def __init__(self, cfg: dict[str, Any] | None, vocab: CodeVocab, device: str | None = None,
                 hparams: dict[str, Any] | None = None, max_length: int | None = None) -> None:
        import torch

        self.cfg = cfg or {}
        scfg = self.cfg.get("semantic", {}) or {}
        hp = {**DEFAULT_HPARAMS, **(scfg.get("own_small_model", {}) or {}), **(hparams or {})}
        hp.pop("enabled", None)
        self.hparams: dict[str, Any] = hp
        self.vocab = vocab
        self.max_length = int(max_length or scfg.get("max_length", 512))
        self.model_name = "own_small"
        self.base_model = "own_small"
        self.pooling = "mean"
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cls = _torch_classes()["OwnTransformer"]
        self.model = cls(vocab_size=len(vocab), hidden=int(hp["hidden"]), layers=int(hp["layers"]), heads=int(hp["heads"]),
                         max_len=self.max_length, dropout=float(hp.get("dropout", 0.1)))
        self.model.to(self._device)
        self.model.eval()
        self.dim = int(hp["hidden"])

    @property
    def device(self) -> str:
        return self._device

    def to(self, device: str) -> "OwnEncoder":
        self._device = device
        self.model.to(device)
        return self

    def parameters(self):
        return self.model.parameters()

    def train_mode(self, flag: bool = True) -> None:
        self.model.train(flag)

    def enable_gradient_checkpointing(self) -> bool:
        self.model.gradient_checkpointing = True
        return True

    def tokenize(self, codes: Sequence[str], langs: Sequence[str] | None = None) -> dict[str, Any]:
        import torch

        ids, mask = self.vocab.encode_batch(codes, langs, self.max_length)
        return {"input_ids": torch.from_numpy(ids), "attention_mask": torch.from_numpy(mask)}

    def forward_batch(self, batch: dict[str, Any]) -> Any:
        return self.model(batch["input_ids"].to(self._device), batch["attention_mask"].to(self._device))

    def encode(self, codes: Sequence[str], langs: Sequence[str] | None = None, batch_size: int = 32,
               show_progress: bool = False) -> np.ndarray:
        import torch

        n = len(codes)
        if n == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        was_training = self.model.training
        self.model.eval()
        out = np.zeros((n, self.dim), dtype=np.float32)
        order = sorted(range(n), key=lambda i: -len(codes[i]))
        batches = [order[i : i + max(1, int(batch_size))] for i in range(0, n, max(1, int(batch_size)))]
        it: Iterable[list[int]] = batches
        if show_progress and len(batches) > 20:
            try:
                from tqdm import tqdm

                it = tqdm(batches, desc="encode(own)", unit="batch")
            except Exception:  # pragma: no cover
                it = batches
        with torch.inference_mode():
            for idx in it:
                batch = self.tokenize([codes[i] for i in idx], [langs[i] for i in idx] if langs is not None else None)
                out[idx] = self.forward_batch(batch).float().cpu().numpy()
        if was_training:
            self.model.train()
        return out

    def save(self, path: str | Path) -> Path:
        import torch

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / WEIGHTS_FILE)
        self.vocab.save(path / VOCAB_FILE)
        meta = {"kind": self.kind, "base_model": "own_small", "pooling": "mean", "max_length": self.max_length,
                "dim": self.dim, "hparams": self.hparams, "vocab_size": len(self.vocab)}
        with open(path / ENCODER_META, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        return path

    @classmethod
    def load(cls, cfg: dict[str, Any] | None, path: str | Path, device: str | None = None) -> "OwnEncoder":
        import torch

        path = Path(path)
        with open(path / ENCODER_META, "r", encoding="utf-8") as f:
            meta = json.load(f)
        vocab = CodeVocab.load(path / VOCAB_FILE)
        enc = cls(cfg, vocab, device=device, hparams=meta.get("hparams"), max_length=meta.get("max_length"))
        state = torch.load(path / WEIGHTS_FILE, map_location=enc.device)
        enc.model.load_state_dict(state)
        enc.model.eval()
        enc.model_name = str(path)
        return enc
