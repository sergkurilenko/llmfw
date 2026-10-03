"""Обёртка над HF-энкодером кода (DESIGN.md §7): mean pooling по маске внимания, L2-нормировка,
max_length 512, fp16 на CUDA, вход — код без комментариев (normalize.strip_comments).

torch/transformers импортируются лениво (внутри методов), поэтому модуль импортируется на CPU без них.
Чекпойнт после дообучения — каталог save_pretrained + encoder_meta.json; load_encoder различает
HF-модель и «свою» модель (own_model.OwnEncoder) по полю kind в encoder_meta.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from smcode.config import ROOT
from smcode.normalize import canon_lang, strip_comments

log = logging.getLogger(__name__)

ENCODER_META = "encoder_meta.json"
DEFAULT_BASE_MODEL = "microsoft/unixcoder-base"
DEFAULT_MAX_LENGTH = 512
MAX_CHARS = 20000  # ограничение длины входа до токенизации (512 токенов ≪ 20k символов)


def preprocess(code: str, lang: str | None = None) -> str:
    """Код без комментариев/docstring, без хвостовых пробелов, обрезанный до MAX_CHARS символов."""
    text = code or ""
    if lang:
        try:
            text = strip_comments(text, canon_lang(lang))
        except Exception as exc:  # noqa: BLE001 — неподдерживаемый язык или сбой парсера
            log.debug("strip_comments failed (%s): %s", lang, exc)
    text = "\n".join(ln.rstrip() for ln in text.splitlines()).strip("\n")
    return text[:MAX_CHARS]


def preprocess_many(codes: Sequence[str], langs: Sequence[str] | None) -> list[str]:
    if langs is None:
        return [preprocess(c, None) for c in codes]
    return [preprocess(c, l) for c, l in zip(codes, langs)]


def pick_device(device: str | None = None) -> str:
    """cuda, если доступна, иначе cpu (ленивый импорт torch)."""
    if device:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def read_encoder_meta(path: str | Path) -> dict[str, Any] | None:
    p = Path(path) / ENCODER_META
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _length_sorted_batches(texts: Sequence[str], batch_size: int) -> Iterable[list[int]]:
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    for i in range(0, len(order), batch_size):
        yield order[i : i + batch_size]


class CodeEncoder:
    """HF-энкодер (AutoModel) с mean/cls pooling и L2-нормировкой; encode() → np.ndarray float32 (N, d)."""

    kind = "hf"

    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        model_name_or_path: str | None = None,
        device: str | None = None,
        max_length: int | None = None,
        pooling: str | None = None,
        fp16: bool | None = None,
        trust_remote_code: bool | None = None,
        lazy: bool = True,
    ) -> None:
        self.cfg = cfg or {}
        scfg = self.cfg.get("semantic", {}) or {}
        self.model_name: str = str(model_name_or_path or scfg.get("checkpoint") or scfg.get("base_model") or DEFAULT_BASE_MODEL)
        self.base_model: str = self.model_name
        self.max_length = int(max_length or scfg.get("max_length", DEFAULT_MAX_LENGTH))
        self.pooling = str(pooling or scfg.get("pooling", "mean"))
        self.fp16 = bool(scfg.get("fp16", True)) if fp16 is None else bool(fp16)
        self.trust_remote_code = bool(scfg.get("trust_remote_code", False)) if trust_remote_code is None else bool(trust_remote_code)
        self._device: str | None = device
        self.tokenizer: Any = None
        self.model: Any = None
        self.dim: int | None = None
        meta = read_encoder_meta(self.model_name) if Path(self.model_name).is_dir() else None
        if meta:
            self.base_model = str(meta.get("base_model", self.model_name))
            self.pooling = str(meta.get("pooling", self.pooling))
            self.max_length = int(meta.get("max_length", self.max_length))
        if not lazy:
            self._ensure()

    # --- загрузка

    def _ensure(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._device = pick_device(self._device)
        log.info("loading encoder %s on %s (pooling=%s, max_length=%d)", self.model_name, self._device, self.pooling, self.max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
        self.model.to(self._device)
        self.model.eval()
        if self.fp16 and str(self._device).startswith("cuda"):
            self.model.half()
        hidden = getattr(self.model.config, "hidden_size", None) or getattr(self.model.config, "d_model", None)
        self.dim = int(hidden) if hidden else None

    @property
    def device(self) -> str:
        return self._device or "cpu"

    def to(self, device: str) -> "CodeEncoder":
        """Перенос модели на устройство (fp16 только на cuda)."""
        self._ensure()
        self._device = device
        self.model.to(device)
        if str(device).startswith("cuda"):
            if self.fp16:
                self.model.half()
        else:
            self.model.float()
        return self

    def parameters(self):
        self._ensure()
        return self.model.parameters()

    def train_mode(self, flag: bool = True) -> None:
        self._ensure()
        self.model.train(flag)

    def enable_gradient_checkpointing(self) -> bool:
        self._ensure()
        if hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable()
                return True
            except Exception as exc:  # noqa: BLE001
                log.warning("gradient checkpointing unavailable: %s", exc)
        return False

    # --- прямой проход

    def tokenize(self, codes: Sequence[str], langs: Sequence[str] | None = None) -> dict[str, Any]:
        """Предобработка + токенизация с паддингом/усечением → dict тензоров (на CPU)."""
        self._ensure()
        texts = preprocess_many(codes, langs)
        texts = [t if t else " " for t in texts]
        return dict(self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"))

    def pool(self, hidden: Any, mask: Any) -> Any:
        """mean (по маске) или cls; результат float32."""
        import torch

        hidden = hidden.float()
        if self.pooling == "cls":
            return hidden[:, 0]
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(1) / torch.clamp(m.sum(1), min=1.0)

    def forward_batch(self, batch: dict[str, Any]) -> Any:
        """Эмбеддинги батча (B, d), L2-нормированные; с градиентом (для обучения)."""
        import torch

        self._ensure()
        batch = {k: v.to(self.device) for k, v in batch.items() if hasattr(v, "to")}
        out = self.model(**batch)
        if hasattr(out, "last_hidden_state"):
            emb = self.pool(out.last_hidden_state, batch["attention_mask"])
        elif torch.is_tensor(out):  # модели-эмбеддеры (codet5p-110m-embedding) отдают вектор
            emb = out.float() if out.dim() == 2 else self.pool(out, batch["attention_mask"])
        else:
            first = out[0]
            emb = self.pool(first, batch["attention_mask"]) if first.dim() == 3 else first.float()
        return torch.nn.functional.normalize(emb, p=2, dim=-1)

    def encode(self, codes: Sequence[str], langs: Sequence[str] | None = None, batch_size: int = 32,
               show_progress: bool = False) -> np.ndarray:
        """Кодирует список функций → float32 (N, d), строки L2-нормированы. Батчи сортируются по длине."""
        import torch

        self._ensure()
        n = len(codes)
        if n == 0:
            return np.zeros((0, self.dim or 0), dtype=np.float32)
        texts = preprocess_many(codes, langs)
        was_training = self.model.training
        self.model.eval()
        out: np.ndarray | None = None
        batches = list(_length_sorted_batches(texts, max(1, int(batch_size))))
        it: Iterable[list[int]] = batches
        if show_progress and len(batches) > 20:
            try:
                from tqdm import tqdm

                it = tqdm(batches, desc="encode", unit="batch")
            except Exception:  # pragma: no cover
                it = batches
        with torch.inference_mode():
            for idx in it:
                sub = [texts[i] if texts[i] else " " for i in idx]
                batch = dict(self.tokenizer(sub, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"))
                emb = self.forward_batch(batch).float().cpu().numpy()
                if out is None:
                    out = np.zeros((n, emb.shape[1]), dtype=np.float32)
                    self.dim = int(emb.shape[1])
                out[idx] = emb
        if was_training:
            self.model.train()
        assert out is not None
        return out

    # --- сохранение

    def save(self, path: str | Path) -> Path:
        """save_pretrained модели и токенизатора + encoder_meta.json."""
        self._ensure()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)
        meta = {"kind": self.kind, "base_model": self.base_model, "pooling": self.pooling, "max_length": self.max_length,
                "dim": self.dim}
        with open(path / ENCODER_META, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        return path


def load_encoder(cfg: dict[str, Any], path_or_name: str | None = None, device: str | None = None,
                 own_model: bool = False, fp16: bool | None = None) -> Any:
    """Энкодер по чекпойнту/имени: каталог с encoder_meta.json kind=own → OwnEncoder, иначе CodeEncoder.
    Порядок по умолчанию: аргумент → cfg.semantic.checkpoint → cfg.semantic.base_model."""
    scfg = cfg.get("semantic", {}) or {}
    name = str(path_or_name or scfg.get("checkpoint") or scfg.get("base_model") or DEFAULT_BASE_MODEL)
    cand = Path(name)
    if not cand.is_absolute() and not cand.is_dir() and (ROOT / cand).is_dir():  # runs/<name>/best относительно корня стенда
        name = str(ROOT / cand)
    meta = read_encoder_meta(name) if Path(name).is_dir() else None
    if (meta and meta.get("kind") == "own") or (own_model and meta is None and Path(str(name)).is_dir()):
        from smcode.semantic.own_model import OwnEncoder

        return OwnEncoder.load(cfg, name, device=device)
    if own_model:
        raise FileNotFoundError(f"own-model checkpoint not found at {name!r} (train it with scripts/05_train_encoder.py --own_model)")
    return CodeEncoder(cfg, str(name), device=device, fp16=fp16)
