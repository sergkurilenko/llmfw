"""Контрастивное дообучение энкодера (DESIGN.md §7). Требует torch (GPU); torch импортируется лениво.

Якорь — функция public_train (+ protected при adapt_on_protected); позитив — её случайное программное
преобразование (smcode.transforms.registry.apply_transform; partial с случайным L) либо LLM-преобразование
из data/queries/llm_pairs.jsonl (строки {"source_id", "code", "lang"}; также принимаются LLM-строки
data/queries/public_train.jsonl), если файл есть; негативы — in-batch + hard_negatives_per_anchor функций
того же файла/репозитория. Симметричный InfoNCE (temperature), AdamW, линейный warmup 5 %, bf16 autocast,
gradient checkpointing по флагу. Валидация после каждой эпохи — recall@1 на парах public_calib
(data/queries/public_calib.jsonl, иначе преобразования на лету). Лучший чекпойнт → runs/<name>/best/,
последний → runs/<name>/last/, лог → runs/<name>/log.jsonl.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from smcode.config import ROOT, get_rng, resolve_path
from smcode.normalize import canon_lang
from smcode.semantic.model import ENCODER_META, preprocess_many
from smcode.transforms.registry import LLM_TRANSFORMS, apply_transform
from smcode.types import FunctionRecord, read_functions, read_jsonl

log = logging.getLogger(__name__)

RUNS_DIR_DEFAULT = "runs"
LLM_PAIRS_FILE = "llm_pairs.jsonl"
LOG_FILE = "log.jsonl"
BEST_DIR = "best"
LAST_DIR = "last"
TRAIN_META = "train_meta.json"
DEFAULTS = {"epochs": 2, "batch_size": 128, "lr": 2e-5, "temperature": 0.05, "hard_negatives_per_anchor": 2,
            "max_length": 512, "warmup_frac": 0.05, "weight_decay": 0.01, "own_lr": 3e-4}


def runs_dir(cfg: dict[str, Any]) -> Path:
    cfg.setdefault("paths", {}).setdefault("runs", RUNS_DIR_DEFAULT)
    return resolve_path(cfg, "runs")


def run_dir(cfg: dict[str, Any], name: str) -> Path:
    d = runs_dir(cfg) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class TrainConfig:
    """Параметры обучения; None → значение из cfg.semantic / DEFAULTS."""

    name: str = ""
    base_model: str | None = None
    epochs: int | None = None
    batch_size: int | None = None
    lr: float | None = None
    max_steps: int | None = None
    adapt_on_protected: bool = False
    own_model: bool = False
    hard_negatives: int | None = None
    temperature: float | None = None
    max_length: int | None = None
    warmup_frac: float | None = None
    weight_decay: float | None = None
    grad_checkpointing: bool = False
    amp: str = "auto"  # auto | bf16 | fp16 | none
    num_workers: int = 4
    val_pairs: int = 2000
    log_every: int = 20
    max_anchors: int | None = None
    p_llm: float = 0.5
    seed: int | None = None
    force: bool = False
    device: str | None = None
    grad_clip: float = 1.0

    def resolved(self, cfg: dict[str, Any]) -> "TrainConfig":
        scfg = cfg.get("semantic", {}) or {}
        tc = TrainConfig(**asdict(self))
        tc.base_model = tc.base_model or str(scfg.get("base_model", "microsoft/unixcoder-base"))
        tc.epochs = int(tc.epochs if tc.epochs is not None else scfg.get("epochs", DEFAULTS["epochs"]))
        tc.batch_size = int(tc.batch_size if tc.batch_size is not None else scfg.get("batch_size", DEFAULTS["batch_size"]))
        if tc.lr is None:
            tc.lr = float(scfg.get("own_lr", DEFAULTS["own_lr"]) if tc.own_model else scfg.get("lr", DEFAULTS["lr"]))
        tc.hard_negatives = int(tc.hard_negatives if tc.hard_negatives is not None
                                else scfg.get("hard_negatives_per_anchor", DEFAULTS["hard_negatives_per_anchor"]))
        tc.temperature = float(tc.temperature if tc.temperature is not None else scfg.get("temperature", DEFAULTS["temperature"]))
        tc.max_length = int(tc.max_length if tc.max_length is not None else scfg.get("max_length", DEFAULTS["max_length"]))
        tc.warmup_frac = float(tc.warmup_frac if tc.warmup_frac is not None else scfg.get("warmup_frac", DEFAULTS["warmup_frac"]))
        tc.weight_decay = float(tc.weight_decay if tc.weight_decay is not None else scfg.get("weight_decay", DEFAULTS["weight_decay"]))
        tc.grad_checkpointing = bool(tc.grad_checkpointing or scfg.get("grad_checkpointing", False))
        tc.seed = int(tc.seed if tc.seed is not None else cfg.get("seed", 0))
        if not tc.name:
            base = "own_small" if tc.own_model else Path(str(tc.base_model)).name.replace("/", "_") + "_ft"
            tc.name = base + ("_protected" if tc.adapt_on_protected else "")
        return tc


# ----------------------------------------------------------------------------- данные


def load_records(cfg: dict[str, Any], split: str, kind: str = "function") -> list[FunctionRecord]:
    path = resolve_path(cfg, "functions") / f"{split}.jsonl"
    if not path.exists():
        return []
    return [r for r in read_functions(path) if (r.kind or "function") == kind]


def load_llm_pairs(cfg: dict[str, Any]) -> dict[str, list[tuple[str, str]]]:
    """source_id → [(code, lang)] LLM-позитивов: data/queries/llm_pairs.jsonl и LLM-строки public_train.jsonl."""
    qdir = resolve_path(cfg, "queries")
    out: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for path, only_llm in ((qdir / LLM_PAIRS_FILE, False), (qdir / "public_train.jsonl", True)):
        if not path.exists():
            continue
        n = 0
        for row in read_jsonl(path):
            if only_llm and row.get("transform") not in LLM_TRANSFORMS:
                continue
            src = row.get("source_id") or row.get("anchor_id")
            code = row.get("code") or row.get("positive")
            if not src or not code:
                continue
            out[str(src)].append((str(code), str(row.get("lang") or row.get("dst_lang") or "")))
            n += 1
        log.info("llm positives: %d rows from %s", n, path)
    return dict(out)


class ContrastiveDataset:
    """Map-style датасет: якорь, позитив (преобразование/LLM), hard-негативы того же файла/репозитория.
    ГПСЧ детерминирован по (seed, epoch, id); set_epoch() меняет позитивы между эпохами."""

    def __init__(self, records: Sequence[FunctionRecord], cfg: dict[str, Any], n_hard: int = 2,
                 llm_pairs: dict[str, list[tuple[str, str]]] | None = None, seed: int = 0, p_llm: float = 0.5,
                 transforms: Iterable[str] | None = None, partial_lines: Iterable[int] | None = None) -> None:
        self.records = list(records)
        self.cfg = cfg
        self.n_hard = int(n_hard)
        self.llm = llm_pairs or {}
        self.seed = int(seed)
        self.p_llm = float(p_llm)
        self.epoch = 0
        tcfg = cfg.get("transforms", {}) or {}
        names = list(tcfg.get("programmatic", ["identity"]) if transforms is None else transforms)
        self.transforms = [t for t in names if t != "identity" and t not in LLM_TRANSFORMS] or ["identity"]
        self.partial_lines = [int(L) for L in (tcfg.get("partial_lines", []) if partial_lines is None else partial_lines)]
        self.deadcode_every = int(tcfg.get("deadcode_every", 3))
        by_file: dict[tuple[str, str], list[int]] = defaultdict(list)
        by_repo: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(self.records):
            by_file[(r.repo, r.path)].append(i)
            by_repo[r.repo].append(i)
        self.by_file: dict[tuple[str, str], list[int]] = dict(by_file)
        self.by_repo: dict[str, list[int]] = dict(by_repo)

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self, rec: FunctionRecord, salt: str = "") -> random.Random:
        return random.Random(f"{self.seed}:{self.epoch}:{rec.id}:{salt}")

    def positive(self, rec: FunctionRecord, rng: random.Random) -> tuple[str, str, str]:
        """(code, lang, вид позитива): LLM-пара с вероятностью p_llm, иначе случайное программное преобразование."""
        lang = canon_lang(rec.lang)
        llm = self.llm.get(rec.id)
        if llm and rng.random() < self.p_llm:
            code, plang = rng.choice(llm)
            return code, plang or lang, "llm"
        choices = list(self.transforms) + (["partial"] if self.partial_lines else [])
        rng.shuffle(choices)
        for name in choices[:4]:
            params: dict[str, Any] = {}
            if name == "partial":
                params["L"] = rng.choice(self.partial_lines)
            elif name == "insert_deadcode":
                params["every"] = self.deadcode_every
            res = apply_transform(name, rec.code, lang, rng, **params)
            if res.ok and res.code:
                return res.code, lang, name
        return rec.code, lang, "identity"

    def negatives(self, i: int, rng: random.Random) -> list[int]:
        """n_hard индексов: сначала тот же файл, затем тот же репозиторий, затем случайные (без self и дубликатов по sha)."""
        rec = self.records[i]
        out: list[int] = []
        n_total = len(self.records)
        for pool in (self.by_file.get((rec.repo, rec.path), []), self.by_repo.get(rec.repo, []), None):
            if len(out) >= self.n_hard or n_total < 2:
                break
            size = n_total if pool is None else len(pool)
            if size < 2:
                continue
            tries = 0
            while len(out) < self.n_hard and tries < 4 * self.n_hard + 8:
                tries += 1
                j = rng.randrange(size) if pool is None else pool[rng.randrange(size)]
                if j != i and j not in out and self.records[j].sha != rec.sha:
                    out.append(j)
        return out

    def __getitem__(self, i: int) -> dict[str, Any]:
        rec = self.records[i]
        rng = self._rng(rec)
        pos_code, pos_lang, kind = self.positive(rec, rng)
        negs = self.negatives(i, rng)
        return {"anchor": (rec.code, canon_lang(rec.lang)), "positive": (pos_code, pos_lang), "pos_kind": kind,
                "negatives": [(self.records[j].code, canon_lang(self.records[j].lang)) for j in negs]}


class Collator:
    """Токенизация батча одним вызовом: [якоря | позитивы | негативы] (лёгкий объект для DataLoader-воркеров)."""

    def __init__(self, kind: str, tokenizer: Any, max_length: int) -> None:
        self.kind = kind  # hf | own
        self.tokenizer = tokenizer  # HF tokenizer или CodeVocab
        self.max_length = int(max_length)

    def tokenize(self, codes: Sequence[str], langs: Sequence[str]) -> dict[str, Any]:
        import torch

        texts = [t if t else " " for t in preprocess_many(codes, langs)]
        if self.kind == "own":
            ids, mask = self.tokenizer.encode_batch(texts, langs, self.max_length)
            return {"input_ids": torch.from_numpy(ids), "attention_mask": torch.from_numpy(mask)}
        return dict(self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"))

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        anchors = [it["anchor"] for it in items]
        positives = [it["positive"] for it in items]
        negatives = [n for it in items for n in it["negatives"]]
        all_pairs = anchors + positives + negatives
        batch = self.tokenize([c for c, _ in all_pairs], [l for _, l in all_pairs])
        return {"batch": batch, "n_anchor": len(anchors), "n_neg": len(negatives), "pos_kinds": [it["pos_kind"] for it in items]}


# ----------------------------------------------------------------------------- валидация


def build_val_pairs(cfg: dict[str, Any], n: int, rng: random.Random, fallback_records: Sequence[FunctionRecord] = ()) -> tuple[list[tuple[str, str]], list[tuple[str, str]], str]:
    """Пары (якорь public_calib, его преобразование): из data/queries/public_calib.jsonl, иначе преобразования на лету;
    без public_calib — отложенная часть переданных записей. Возвращает (anchors, positives, источник)."""
    calib = load_records(cfg, "public_calib")
    source = "public_calib"
    if not calib:
        calib = list(fallback_records)
        source = "train_holdout"
    if not calib:
        return [], [], "none"
    by_id = {r.id: r for r in calib}
    qpath = resolve_path(cfg, "queries") / "public_calib.jsonl"
    pairs: list[tuple[tuple[str, str], tuple[str, str]]] = []
    if source == "public_calib" and qpath.exists():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in read_jsonl(qpath):
            if row.get("source_id") in by_id and row.get("transform") != "identity":
                grouped[row["source_id"]].append(row)
        srcs = sorted(grouped)
        rng.shuffle(srcs)
        for sid in srcs[:n]:
            q = rng.choice(grouped[sid])
            rec = by_id[sid]
            pairs.append(((rec.code, canon_lang(rec.lang)), (q["code"], q.get("lang") or canon_lang(rec.lang))))
        source = "public_calib_queries"
    if not pairs:
        recs = sorted(calib, key=lambda r: r.id)
        rng.shuffle(recs)
        ds = ContrastiveDataset(recs[:n], cfg, n_hard=0, seed=rng.randrange(10 ** 9), p_llm=0.0)
        for rec in ds.records:
            code, lang, _ = ds.positive(rec, ds._rng(rec, "val"))
            pairs.append(((rec.code, canon_lang(rec.lang)), (code, lang)))
    anchors = [a for a, _ in pairs]
    positives = [p for _, p in pairs]
    return anchors, positives, source


def evaluate_recall(encoder: Any, anchors: Sequence[tuple[str, str]], positives: Sequence[tuple[str, str]],
                    batch_size: int = 64) -> dict[str, float]:
    """recall@1/@5 позитива среди всех якорей и средний косинус пар."""
    if not anchors:
        return {"recall1": float("nan"), "recall5": float("nan"), "mean_cos": float("nan"), "n": 0}
    ea = np.asarray(encoder.encode([c for c, _ in anchors], [l for _, l in anchors], batch_size=batch_size), dtype=np.float32)
    ep = np.asarray(encoder.encode([c for c, _ in positives], [l for _, l in positives], batch_size=batch_size), dtype=np.float32)
    sims = ep @ ea.T
    diag = np.diag(sims)
    rank = (sims > diag[:, None]).sum(1)
    return {"recall1": float((rank == 0).mean()), "recall5": float((rank < 5).mean()), "mean_cos": float(diag.mean()), "n": int(len(anchors))}


# ----------------------------------------------------------------------------- loss и обучение


def info_nce(a: Any, p: Any, n: Any, temperature: float) -> Any:
    """Симметричный InfoNCE: якорь→[позитивы; негативы] и позитив→[якоря; негативы], метки — диагональ."""
    import torch
    import torch.nn.functional as F

    B = a.shape[0]
    labels = torch.arange(B, device=a.device)
    cands_ap = torch.cat([p, n], 0) if n is not None and n.shape[0] else p
    cands_pa = torch.cat([a, n], 0) if n is not None and n.shape[0] else a
    loss_ap = F.cross_entropy(a @ cands_ap.T / temperature, labels)
    loss_pa = F.cross_entropy(p @ cands_pa.T / temperature, labels)
    return 0.5 * (loss_ap + loss_pa)


def _amp_setup(device: str, mode: str) -> tuple[str, Any, Any]:
    """(режим, autocast-контекст-фабрика, GradScaler|None)."""
    import contextlib

    import torch

    if not str(device).startswith("cuda"):
        return "none", contextlib.nullcontext, None
    if mode == "auto":
        mode = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if mode == "bf16":
        return mode, (lambda: torch.autocast("cuda", dtype=torch.bfloat16)), None
    if mode == "fp16":
        scaler = torch.amp.GradScaler("cuda") if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler()
        return mode, (lambda: torch.autocast("cuda", dtype=torch.float16)), scaler
    return "none", contextlib.nullcontext, None


def build_encoder(cfg: dict[str, Any], tc: TrainConfig, records: Sequence[FunctionRecord], out_dir: Path) -> Any:
    """CodeEncoder (HF, fp32-веса) либо OwnEncoder со словарём по public_train (runs/<name>/vocab.json)."""
    from smcode.semantic.model import CodeEncoder, pick_device

    device = pick_device(tc.device)
    if tc.own_model:
        from smcode.semantic.own_model import VOCAB_FILE, CodeVocab, OwnEncoder, build_vocab_from_records

        hp = (cfg.get("semantic", {}) or {}).get("own_small_model", {}) or {}
        vpath = out_dir / VOCAB_FILE
        if vpath.exists():
            vocab = CodeVocab.load(vpath)
        else:
            vocab = build_vocab_from_records(records, max_size=int(hp.get("vocab", 32000)))
            vocab.save(vpath)
        return OwnEncoder(cfg, vocab, device=device, max_length=tc.max_length)
    return CodeEncoder(cfg, tc.base_model, device=device, max_length=tc.max_length, fp16=False, lazy=False)


def _worker_init(worker_id: int) -> None:
    """Сброс кэша парсеров tree-sitter в DataLoader-воркере (объекты не делятся между процессами)."""
    from smcode.normalize import get_parser

    get_parser.cache_clear()


def _write_log(path: Path, row: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def train(cfg: dict[str, Any], tc: TrainConfig) -> dict[str, Any]:
    """Полный цикл обучения; возвращает сводку (best recall@1, пути). Идемпотентно: best/ готов → пропуск без force."""
    import torch
    from torch.utils.data import DataLoader

    tc = tc.resolved(cfg)
    out = run_dir(cfg, tc.name)
    best_dir, last_dir, log_path = out / BEST_DIR, out / LAST_DIR, out / LOG_FILE
    if (best_dir / ENCODER_META).exists() and not tc.force:
        log.info("run %s: best checkpoint exists at %s, skipping (use --force)", tc.name, best_dir)
        return {"name": tc.name, "skipped": True, "best_dir": str(best_dir)}
    if tc.force and log_path.exists():
        log_path.unlink()
    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed % (2 ** 32))
    rng = get_rng(cfg, f"train:{tc.name}")

    records = load_records(cfg, "public_train")
    if not records:
        raise FileNotFoundError("data/functions/public_train.jsonl is empty or missing (run scripts/01_extract.py, 02_dedup_split.py)")
    if tc.adapt_on_protected:
        prot = load_records(cfg, "protected")
        log.info("ablation adapt_on_protected: +%d protected anchors", len(prot))
        records += prot
    records = sorted(records, key=lambda r: r.id)
    if tc.max_anchors and len(records) > tc.max_anchors:
        rng.shuffle(records)
        records = sorted(records[: tc.max_anchors], key=lambda r: r.id)
    llm_pairs = load_llm_pairs(cfg)
    n_hold = min(len(records) // 10, tc.val_pairs)
    val_anchors, val_positives, val_source = build_val_pairs(cfg, tc.val_pairs, rng, fallback_records=records[-n_hold:] if n_hold else [])
    if val_source == "train_holdout" and val_anchors:
        held = {c for c, _ in val_anchors}
        records = [r for r in records if r.code not in held]

    encoder = build_encoder(cfg, tc, records, out)
    device = encoder.device
    if tc.grad_checkpointing:
        encoder.enable_gradient_checkpointing()
    dataset = ContrastiveDataset(records, cfg, n_hard=tc.hard_negatives, llm_pairs=llm_pairs, seed=tc.seed, p_llm=tc.p_llm)
    collator = Collator(encoder.kind, encoder.vocab if encoder.kind == "own" else encoder.tokenizer, tc.max_length)
    drop_last = len(dataset) >= 2 * tc.batch_size
    gen = torch.Generator().manual_seed(tc.seed)
    loader = DataLoader(dataset, batch_size=tc.batch_size, shuffle=True, collate_fn=collator, num_workers=tc.num_workers,
                        drop_last=drop_last, generator=gen, persistent_workers=False, worker_init_fn=_worker_init)
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * tc.epochs
    if tc.max_steps:
        total_steps = min(total_steps, int(tc.max_steps))
    warmup = max(1, int(round(tc.warmup_frac * total_steps)))
    model = encoder.model
    decay, no_decay = [], []
    for n_, p_ in model.named_parameters():
        if not p_.requires_grad:
            continue
        (no_decay if (p_.ndim < 2 or "bias" in n_ or "norm" in n_.lower()) else decay).append(p_)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": tc.weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=tc.lr)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    amp_mode, autocast, scaler = _amp_setup(device, tc.amp)
    meta = {"name": tc.name, "args": asdict(tc), "n_anchors": len(dataset), "n_llm_sources": len(llm_pairs), "val_pairs": len(val_anchors),
            "val_source": val_source, "steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "warmup_steps": warmup,
            "amp": amp_mode, "device": str(device), "encoder_kind": encoder.kind, "model": getattr(encoder, "model_name", None)}
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    _write_log(log_path, {"type": "start", "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **meta})
    log.info("train %s: %d anchors, %d steps (%d/epoch, warmup %d), device=%s amp=%s", tc.name, len(dataset), total_steps,
             steps_per_epoch, warmup, device, amp_mode)

    best = {"recall1": -1.0, "epoch": -1}
    if val_anchors:
        zs = evaluate_recall(encoder, val_anchors, val_positives)
        _write_log(log_path, {"type": "epoch", "epoch": 0, "step": 0, "zero_shot": True, **{f"val_{k}": v for k, v in zs.items()}})
        log.info("zero-shot: recall@1=%.4f recall@5=%.4f mean_cos=%.4f (n=%d)", zs["recall1"], zs["recall5"], zs["mean_cos"], zs["n"])
    step = 0
    t_start = time.perf_counter()
    done = False
    for epoch in range(1, tc.epochs + 1):
        dataset.set_epoch(epoch)
        encoder.train_mode(True)
        losses: list[float] = []
        kinds: dict[str, int] = defaultdict(int)
        for batch in loader:
            if step >= total_steps:
                done = True
                break
            inputs = {k: v.to(device, non_blocking=True) for k, v in batch["batch"].items()}
            B, n_neg = batch["n_anchor"], batch["n_neg"]
            for k_ in batch["pos_kinds"]:
                kinds[k_] += 1
            with autocast():
                emb = encoder.forward_batch(inputs)
            emb = emb.float()
            a, p, n = emb[:B], emb[B : 2 * B], emb[2 * B : 2 * B + n_neg]
            loss = info_nce(a, p, n, tc.temperature)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
                optimizer.step()
            scheduler.step()
            step += 1
            losses.append(float(loss.item()))
            if step % tc.log_every == 0 or step == total_steps:
                row = {"type": "step", "step": step, "epoch": epoch, "loss": float(np.mean(losses[-tc.log_every:])),
                       "lr": float(scheduler.get_last_lr()[0]), "elapsed_s": round(time.perf_counter() - t_start, 1)}
                _write_log(log_path, row)
                log.info("step %d/%d epoch %d loss=%.4f lr=%.2e", step, total_steps, epoch, row["loss"], row["lr"])
        encoder.train_mode(False)
        val = evaluate_recall(encoder, val_anchors, val_positives) if val_anchors else {"recall1": float("nan"), "recall5": float("nan"), "mean_cos": float("nan"), "n": 0}
        is_best = (not math.isnan(val["recall1"]) and val["recall1"] > best["recall1"]) or (not val_anchors and epoch >= best["epoch"])
        if is_best:
            best = {"recall1": val["recall1"], "epoch": epoch, "step": step}
            encoder.save(best_dir)
            with open(best_dir / TRAIN_META, "w", encoding="utf-8") as f:
                json.dump({"epoch": epoch, "step": step, "val": val, "train_loss": float(np.mean(losses)) if losses else None,
                           "name": tc.name, "args": asdict(tc)}, f, ensure_ascii=False, indent=1)
        row = {"type": "epoch", "epoch": epoch, "step": step, "train_loss": float(np.mean(losses)) if losses else None,
               "pos_kinds": dict(kinds), **{f"val_{k}": v for k, v in val.items()}, "best": is_best,
               "elapsed_s": round(time.perf_counter() - t_start, 1)}
        _write_log(log_path, row)
        log.info("epoch %d: loss=%.4f val recall@1=%.4f recall@5=%.4f%s", epoch, row["train_loss"] or float("nan"), val["recall1"],
                 val["recall5"], " (best)" if is_best else "")
        if done or step >= total_steps:
            break
    encoder.save(last_dir)
    summary = {"name": tc.name, "skipped": False, "best_dir": str(best_dir), "last_dir": str(last_dir), "log": str(log_path),
               "best": best, "steps": step, "elapsed_s": round(time.perf_counter() - t_start, 1), "amp": amp_mode}
    _write_log(log_path, {"type": "done", **summary})
    log.info("train %s done: best recall@1=%.4f (epoch %s) → %s", tc.name, best["recall1"], best["epoch"], best_dir)
    return summary
