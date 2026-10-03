"""Обёртка над HF-энкодером кода (DESIGN.md §7): mean pooling по маске внимания, L2-нормировка,
max_length 512, fp16 на CUDA, вход — код без комментариев (normalize.strip_comments).

torch/transformers импортируются лениво (внутри методов), поэтому модуль импортируется на CPU без них.
Чекпойнт после дообучения — каталог save_pretrained + encoder_meta.json; load_encoder различает
HF-модель и «свою» модель (own_model.OwnEncoder) по полю kind в encoder_meta.json.

Идентичность модели: model_key(name) — каноническое имя (чекпойнт внутри стенда → путь относительно ROOT,
HF-имя → как есть); им помечаются эмбеддинги (meta.model), индексы и комбинатор, и по нему проверяется,
что переиспользуемые артефакты получены той же моделью. model_tag(name) — файловый тег для каталогов.
Вход UniXcoder: формат encoder-only ([CLS] <encoder-only> [SEP] токены [SEP], как в официальном README)
через HFBatchTokenizer; общий для инференса и обучения (train.Collator). cfg.semantic.input_prefix
переопределяет префикс ("" — обычный вход токенизатора).
"""

from __future__ import annotations

import json
import logging
import re
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
UNIXCODER_PREFIX = "<encoder-only>"
_TAG_RE = re.compile(r"[^A-Za-z0-9._-]+")


# ----------------------------------------------------------------------------- идентичность модели


def model_key(name: str | Path | None) -> str:
    """Каноническое имя модели: каталог чекпойнта внутри стенда → путь относительно ROOT (posix),
    другой каталог → абсолютный путь, HF-имя → как есть; пустое → ''."""
    if name is None:
        return ""
    s = str(name).strip().rstrip("/")
    if not s:
        return ""
    p = Path(s)
    try:
        root = ROOT.resolve()
        if p.is_absolute():
            rp = p.resolve()
            try:
                return rp.relative_to(root).as_posix()
            except ValueError:
                return rp.as_posix()
        if (ROOT / p).is_dir():
            return (ROOT / p).resolve().relative_to(root).as_posix()
        if p.is_dir():
            rp = p.resolve()
            try:
                return rp.relative_to(root).as_posix()
            except ValueError:
                return rp.as_posix()
    except OSError:
        pass
    return Path(s).as_posix() if ("\\" in s) else s


def model_tag(name: str | Path | None) -> str:
    """Файловый тег модели (microsoft__unixcoder-base, runs__unixcoder-base_ft__best); 'unknown' для пустого."""
    key = model_key(name) or "unknown"
    return _TAG_RE.sub("__", key).strip("_") or "unknown"


def resolve_model_name(cfg: dict[str, Any], path_or_name: str | None = None) -> str:
    """Имя/путь модели по правилам load_encoder: аргумент → cfg.semantic.checkpoint → cfg.semantic.base_model;
    относительный каталог чекпойнта разрешается от корня стенда."""
    scfg = (cfg or {}).get("semantic", {}) or {}
    name = str(path_or_name or scfg.get("checkpoint") or scfg.get("base_model") or DEFAULT_BASE_MODEL)
    cand = Path(name)
    if not cand.is_absolute():
        if (ROOT / cand).is_dir():  # runs/<name>/best относительно корня стенда, независимо от cwd
            name = str(ROOT / cand)
        elif cand.is_dir():
            name = str(cand.resolve())
    return name


def expected_model_key(cfg: dict[str, Any], path_or_name: str | None = None) -> str:
    """model_key модели, которую load_encoder загрузит для данного cfg (без torch)."""
    return model_key(resolve_model_name(cfg, path_or_name))


def encoder_model_key(encoder: Any) -> str:
    """model_key энкодера (по атрибуту model_name)."""
    return model_key(getattr(encoder, "model_name", None))


def default_input_prefix(base_model: str | None) -> str:
    """Префикс режима по базовой модели: UniXcoder → '<encoder-only>', иначе ''."""
    return UNIXCODER_PREFIX if "unixcoder" in str(base_model or "").lower() else ""


class HFBatchTokenizer:
    """Токенизация батча текстов для HF-энкодера. При непустом prefix — формат UniXcoder
    [CLS] prefix [SEP] токены[:max_length−4] [SEP]; иначе обычный вызов токенизатора.
    Лёгкий picklable объект (только токенизатор), общий для encode() и обучения (train.Collator)."""

    def __init__(self, tokenizer: Any, max_length: int, prefix: str = "") -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.prefix = str(prefix or "")
        self.prefix_id: int | None = None
        if self.prefix:
            try:
                pid = tokenizer.convert_tokens_to_ids(self.prefix)
            except Exception:  # noqa: BLE001
                pid = None
            unk = getattr(tokenizer, "unk_token_id", None)
            cls_id, sep_id = getattr(tokenizer, "cls_token_id", None), getattr(tokenizer, "sep_token_id", None)
            if pid is None or pid == unk or cls_id is None or sep_id is None:
                log.warning("input prefix %r is not a special token of the tokenizer; using plain inputs", self.prefix)
                self.prefix = ""
            else:
                self.prefix_id = int(pid)

    def __call__(self, texts: Sequence[str]) -> dict[str, Any]:
        texts = [t if t else " " for t in texts]
        tok = self.tokenizer
        if not self.prefix:
            return dict(tok(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"))
        body = tok(list(texts), add_special_tokens=False, truncation=True, max_length=max(1, self.max_length - 4))["input_ids"]
        head = [int(tok.cls_token_id), int(self.prefix_id), int(tok.sep_token_id)]  # type: ignore[arg-type]
        ids = [head + list(b) + [int(tok.sep_token_id)] for b in body]
        return dict(tok.pad({"input_ids": ids}, padding=True, return_tensors="pt"))


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
        input_prefix: str | None = None,
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
        self.batch_tokenizer: HFBatchTokenizer | None = None
        self.model: Any = None
        self.dim: int | None = None
        meta = read_encoder_meta(self.model_name) if Path(self.model_name).is_dir() else None
        if meta:
            self.base_model = str(meta.get("base_model", self.model_name))
            self.pooling = str(meta.get("pooling", self.pooling))
            self.max_length = int(meta.get("max_length", self.max_length))
        # префикс режима: аргумент → encoder_meta.json чекпойнта → cfg.semantic.input_prefix → по базовой модели
        if input_prefix is not None:
            self.input_prefix = str(input_prefix)
        elif meta and "input_prefix" in meta:
            self.input_prefix = str(meta.get("input_prefix") or "")
        elif scfg.get("input_prefix") is not None:
            self.input_prefix = str(scfg.get("input_prefix") or "")
        else:
            self.input_prefix = default_input_prefix(self.base_model)
        if not lazy:
            self._ensure()

    @property
    def key(self) -> str:
        """Каноническое имя модели (model_key) для meta эмбеддингов/индексов."""
        return model_key(self.model_name)

    # --- загрузка

    def _ensure(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._device = pick_device(self._device)
        log.info("loading encoder %s on %s (pooling=%s, max_length=%d)", self.model_name, self._device, self.pooling, self.max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
        self.batch_tokenizer = HFBatchTokenizer(self.tokenizer, self.max_length, self.input_prefix)
        self.input_prefix = self.batch_tokenizer.prefix  # '' если токенизатор не знает префикс
        if self.input_prefix:
            log.info("encoder input format: [CLS] %s [SEP] tokens [SEP]", self.input_prefix)
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
        """Предобработка + токенизация (HFBatchTokenizer: паддинг/усечение, префикс режима) → dict тензоров (на CPU)."""
        self._ensure()
        assert self.batch_tokenizer is not None
        return self.batch_tokenizer(preprocess_many(codes, langs))

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
        assert self.batch_tokenizer is not None
        with torch.inference_mode():
            for idx in it:
                batch = self.batch_tokenizer([texts[i] for i in idx])
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
                "dim": self.dim, "input_prefix": self.input_prefix}
        with open(path / ENCODER_META, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        return path


def load_encoder(cfg: dict[str, Any], path_or_name: str | None = None, device: str | None = None,
                 own_model: bool = False, fp16: bool | None = None) -> Any:
    """Энкодер по чекпойнту/имени: каталог с encoder_meta.json kind=own → OwnEncoder, иначе CodeEncoder.
    Порядок по умолчанию: аргумент → cfg.semantic.checkpoint → cfg.semantic.base_model (resolve_model_name)."""
    name = resolve_model_name(cfg, path_or_name)
    meta = read_encoder_meta(name) if Path(name).is_dir() else None
    if (meta and meta.get("kind") == "own") or (own_model and meta is None and Path(str(name)).is_dir()):
        from smcode.semantic.own_model import OwnEncoder

        return OwnEncoder.load(cfg, name, device=device)
    if own_model:
        raise FileNotFoundError(f"own-model checkpoint not found at {name!r} (train it with scripts/05_train_encoder.py --own_model)")
    return CodeEncoder(cfg, str(name), device=device, fp16=fp16)
