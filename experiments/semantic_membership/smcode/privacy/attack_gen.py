"""Атака A3 — генеративная инверсия (DESIGN.md §9; ОПЦИОНАЛЬНО, только GPU).

Небольшой декодер-трансформер (по умолчанию 6 слоёв, d_model 384) обучается восстанавливать лексические
токены функции (normalize.abstract_tokens(mode="lexical")) по её эмбеддингу: эмбеддинг проецируется в
префиксную позицию, далее авторегрессия с teacher forcing (vec2text-подобно). Обучение на public_train,
оценка на protected: BLEU-4 (собственная реализация, сглаживание add-1 для n ≥ 2) и точность
идентификаторов (доля идентификаторов функции, встретившихся в сгенерированном коде; exact — доля функций
с полностью восстановленным множеством идентификаторов).

torch импортируется только внутри функций; CPU-части (словарь, BLEU, метрики) работают без него.
Без torch run_attack_gen поднимает ImportError — вызывающий код трактует A3 как недоступную (null в JSON).
Статус: GPU-часть не проверена на CPU-стенде (нет torch), проверена только py_compile и самопроверка.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from smcode.normalize import abstract_tokens, tokenize
from smcode.privacy.quantize import as_generator

log = logging.getLogger(__name__)

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL: tuple[str, ...] = (PAD, BOS, EOS, UNK)
DEFAULT_GEN = {"layers": 6, "d_model": 384, "heads": 6, "ff": 1536, "dropout": 0.1, "max_len": 256, "vocab": 20000,
               "epochs": 5, "lr": 3e-4, "batch": 64, "train_max": 50000, "eval_max": 1000, "device": None}


# ----------------------------------------------------------------------------- токены и словарь


def lexical_tokens(code: str, lang: str) -> list[str]:
    """Лексические токены без комментариев; [] при ошибке парсера."""
    try:
        return abstract_tokens(tokenize(code, lang), mode="lexical")
    except Exception as exc:  # noqa: BLE001
        log.debug("tokenize failed: %s", exc)
        return []


def identifier_set(code: str, lang: str) -> set[str]:
    try:
        return {t.text for t in tokenize(code, lang) if t.kind == "id"}
    except Exception:  # noqa: BLE001
        return set()


@dataclass
class GenVocab:
    """Словарь токенов декодера: специальные + top-N по частоте."""

    tokens: list[str]
    index: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.index:
            self.index = {t: i for i, t in enumerate(self.tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def pad(self) -> int:
        return self.index[PAD]

    @property
    def bos(self) -> int:
        return self.index[BOS]

    @property
    def eos(self) -> int:
        return self.index[EOS]

    @property
    def unk(self) -> int:
        return self.index[UNK]

    @classmethod
    def build(cls, token_lists: Sequence[Sequence[str]], max_size: int) -> "GenVocab":
        cnt: Counter = Counter()
        for toks in token_lists:
            cnt.update(toks)
        body = [t for t, _ in sorted(cnt.items(), key=lambda x: (-x[1], x[0]))[: max(0, max_size - len(SPECIAL))]]
        return cls(tokens=list(SPECIAL) + body)

    def encode(self, toks: Sequence[str], max_len: int) -> list[int]:
        """[bos] + ids + [eos], обрезка до max_len токенов тела."""
        ids = [self.index.get(t, self.unk) for t in toks[:max_len]]
        return [self.bos] + ids + [self.eos]

    def decode(self, ids: Sequence[int]) -> list[str]:
        out: list[str] = []
        for i in ids:
            if i == self.eos:
                break
            if i in (self.pad, self.bos):
                continue
            out.append(self.tokens[int(i)])
        return out


# ----------------------------------------------------------------------------- BLEU


def _ngrams(toks: Sequence[str], n: int) -> Counter:
    return Counter(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))


def sentence_bleu(hyp: Sequence[str], ref: Sequence[str], max_n: int = 4) -> float:
    """BLEU-N одной пары (одна ссылка): геометрическое среднее модифицированных точностей с add-1 сглаживанием
    для n ≥ 2 и штрафом за краткость; 0 для пустой гипотезы."""
    if not hyp or not ref:
        return 0.0
    log_p = 0.0
    for n in range(1, max_n + 1):
        h, r = _ngrams(hyp, n), _ngrams(ref, n)
        total = max(0, len(hyp) - n + 1)
        match = sum(min(c, r[g]) for g, c in h.items())
        if n == 1:
            if match == 0:
                return 0.0
            p = match / total
        else:
            p = (match + 1.0) / (total + 1.0)
        log_p += math.log(p) / max_n
    bp = 1.0 if len(hyp) > len(ref) else math.exp(1.0 - len(ref) / len(hyp))
    return float(bp * math.exp(log_p))


def corpus_bleu(hyps: Sequence[Sequence[str]], refs: Sequence[Sequence[str]], max_n: int = 4) -> float:
    """Корпусный BLEU-N: суммарные совпадения n-грамм по корпусу, штраф за краткость по суммарным длинам."""
    if len(hyps) != len(refs):
        raise ValueError("hyps and refs must have the same length")
    match = np.zeros(max_n)
    total = np.zeros(max_n)
    hyp_len = ref_len = 0
    for h, r in zip(hyps, refs):
        hyp_len += len(h)
        ref_len += len(r)
        for n in range(1, max_n + 1):
            hn, rn = _ngrams(h, n), _ngrams(r, n)
            total[n - 1] += max(0, len(h) - n + 1)
            match[n - 1] += sum(min(c, rn[g]) for g, c in hn.items())
    if hyp_len == 0 or match[0] == 0:
        return 0.0
    log_p = 0.0
    for n in range(max_n):
        p = match[n] / total[n] if n == 0 else (match[n] + 1.0) / (total[n] + 1.0)
        log_p += math.log(max(p, 1e-12)) / max_n
    bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - ref_len / hyp_len)
    return float(bp * math.exp(log_p))


def identifier_metrics(hyps: Sequence[Sequence[str]], ref_ids: Sequence[set[str]]) -> dict[str, Any]:
    """id_acc — micro-доля идентификаторов ссылок, встретившихся в гипотезе; id_exact — доля функций
    с полностью восстановленным множеством идентификаторов."""
    hit = tot = 0
    exact = 0
    n = 0
    for h, ids in zip(hyps, ref_ids):
        if not ids:
            continue
        n += 1
        hs = set(h)
        got = len(ids & hs)
        hit += got
        tot += len(ids)
        exact += int(got == len(ids))
    return {"id_acc": hit / tot if tot else None, "id_exact": exact / n if n else None, "n_with_ids": n}


def gen_params(cfg: dict[str, Any]) -> dict[str, Any]:
    p = dict(DEFAULT_GEN)
    p.update((cfg.get("privacy", {}) or {}).get("a3", {}) or {})
    return p


# ----------------------------------------------------------------------------- torch-часть (GPU)


def _build_decoder(d_emb: int, vocab_size: int, p: dict[str, Any]) -> Any:
    """nn.Module декодера (класс определяется внутри, чтобы torch не импортировался на уровне модуля)."""
    import torch
    from torch import nn

    class EmbDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dm = int(p["d_model"])
            self.max_len = int(p["max_len"]) + 2  # bos/eos
            self.prefix = nn.Linear(d_emb, dm)
            self.tok = nn.Embedding(vocab_size, dm)
            self.pos = nn.Embedding(self.max_len + 1, dm)
            layer = nn.TransformerEncoderLayer(dm, int(p["heads"]), int(p["ff"]), float(p["dropout"]), activation="gelu",
                                               batch_first=True, norm_first=True)
            self.blocks = nn.TransformerEncoder(layer, int(p["layers"]))
            self.norm = nn.LayerNorm(dm)
            self.out = nn.Linear(dm, vocab_size)

        def forward(self, emb: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
            """emb (B, d_emb), inp (B, T) токены-входы → логиты (B, T+1, V): позиция 0 — префикс."""
            b, t = inp.shape
            x = torch.cat([self.prefix(emb)[:, None, :], self.tok(inp)], dim=1)
            pos = torch.arange(t + 1, device=inp.device)
            x = x + self.pos(pos)[None]
            mask = torch.triu(torch.full((t + 1, t + 1), float("-inf"), device=inp.device), diagonal=1)
            h = self.blocks(x, mask=mask)
            return self.out(self.norm(h))

        @torch.no_grad()
        def generate(self, emb: torch.Tensor, bos: int, eos: int, max_len: int) -> torch.Tensor:
            """Жадная генерация (B, ≤max_len) без KV-кэша."""
            b = emb.shape[0]
            seq = torch.full((b, 1), bos, dtype=torch.long, device=emb.device)
            done = torch.zeros(b, dtype=torch.bool, device=emb.device)
            for _ in range(max_len):
                logits = self.forward(emb, seq)[:, -1, :]
                nxt = logits.argmax(dim=-1)
                nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
                seq = torch.cat([seq, nxt[:, None]], dim=1)
                done |= nxt == eos
                if bool(done.all()):
                    break
            return seq[:, 1:]

    return EmbDecoder()


def _pad_batch(seqs: Sequence[Sequence[int]], pad: int) -> np.ndarray:
    L = max(len(s) for s in seqs)
    out = np.full((len(seqs), L), pad, dtype=np.int64)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
    return out


def train_gen_attack(train_emb: np.ndarray, train_tokens: Sequence[Sequence[str]], vocab: GenVocab, cfg: dict[str, Any],
                     seed: int = 0) -> Any:
    """Обучение декодера (teacher forcing, cross-entropy без pad). Возвращает модель (torch)."""
    import torch
    from torch import nn

    p = gen_params(cfg)
    device = p.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = _build_decoder(int(train_emb.shape[1]), len(vocab), p).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(p["lr"]), weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad)
    enc = [vocab.encode(t, int(p["max_len"])) for t in train_tokens]
    X = torch.as_tensor(np.asarray(train_emb, dtype=np.float32))
    rng = np.random.default_rng(seed)
    n, bs = len(enc), int(p["batch"])
    model.train()
    for ep in range(int(p["epochs"])):
        order = rng.permutation(n)
        total = 0.0
        for lo in range(0, n, bs):
            idx = order[lo : lo + bs]
            seq = torch.as_tensor(_pad_batch([enc[i] for i in idx], vocab.pad), device=device)
            inp, tgt = seq[:, :-1], seq[:, 1:]
            logits = model(X[torch.as_tensor(idx, dtype=torch.long)].to(device), inp)[:, 1:, :]  # позиция 0 (префикс→bos) не учится
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(idx)
        log.info("A3 epoch %d/%d: loss=%.4f", ep + 1, int(p["epochs"]), total / max(1, n))
    model.eval()
    return model


def generate_tokens(model: Any, emb: np.ndarray, vocab: GenVocab, cfg: dict[str, Any], batch: int = 64) -> list[list[str]]:
    """Жадная генерация токенов для каждой строки emb."""
    import torch

    p = gen_params(cfg)
    device = next(model.parameters()).device
    out: list[list[str]] = []
    for lo in range(0, emb.shape[0], batch):
        x = torch.as_tensor(np.asarray(emb[lo : lo + batch], dtype=np.float32), device=device)
        seq = model.generate(x, vocab.bos, vocab.eos, int(p["max_len"]) + 1).cpu().numpy()
        out.extend(vocab.decode(row) for row in seq)
    return out


def evaluate_gen_attack(model: Any, emb: np.ndarray, ref_tokens: Sequence[Sequence[str]], ref_ids: Sequence[set[str]],
                        vocab: GenVocab, cfg: dict[str, Any]) -> dict[str, Any]:
    hyps = generate_tokens(model, emb, vocab, cfg)
    m = {"bleu": corpus_bleu(hyps, ref_tokens), "sentence_bleu_mean": float(np.mean([sentence_bleu(h, r) for h, r in zip(hyps, ref_tokens)])) if hyps else 0.0,
         "n_eval": len(hyps)}
    m.update(identifier_metrics(hyps, ref_ids))
    return m


def eval_subset(n_eval: int, cfg: dict[str, Any], rng: Any = None) -> np.ndarray:
    """Индексы подвыборки оценки A3 (≤ cfg.privacy.a3.eval_max, отсортированы); вызывающий код фиксирует её один раз,
    чтобы BLEU/id_acc были сравнимы между защитами."""
    n_ev = min(int(n_eval), int(gen_params(cfg)["eval_max"]))
    if n_ev >= n_eval:
        return np.arange(int(n_eval))
    return np.sort(as_generator(rng if rng is not None else int(cfg.get("seed", 0))).choice(int(n_eval), size=n_ev, replace=False))


def run_attack_gen(train_emb: np.ndarray, train_codes: Sequence[str], train_langs: Sequence[str], eval_emb: np.ndarray,
                   eval_codes: Sequence[str], eval_langs: Sequence[str], cfg: dict[str, Any], rng: Any = None,
                   model: Any = None, vocab: GenVocab | None = None,
                   eval_idx: np.ndarray | None = None) -> tuple[dict[str, Any], Any, GenVocab]:
    """A3 целиком (нужен torch): словарь и обучение на public_train (если model не задана), оценка на eval_emb[eval_idx]
    (eval_idx — фиксированная подвыборка, см. eval_subset; None → случайная по rng). Возвращает (метрики, модель, словарь)."""
    import torch  # noqa: F401 — ImportError без torch

    p = gen_params(cfg)
    g = as_generator(rng if rng is not None else int(cfg.get("seed", 0)))
    if model is None or vocab is None:
        n_tr = min(len(train_codes), int(p["train_max"]))
        idx = np.sort(g.choice(len(train_codes), size=n_tr, replace=False)) if n_tr < len(train_codes) else np.arange(len(train_codes))
        toks = [lexical_tokens(train_codes[i], train_langs[i]) for i in idx]
        vocab = GenVocab.build(toks, int(p["vocab"]))
        model = train_gen_attack(np.asarray(train_emb)[idx], toks, vocab, cfg, seed=int(cfg.get("seed", 0)))
    eidx = np.asarray(eval_idx, dtype=np.int64) if eval_idx is not None else eval_subset(len(eval_codes), cfg, g)
    refs = [lexical_tokens(eval_codes[i], eval_langs[i]) for i in eidx]
    ids = [identifier_set(eval_codes[i], eval_langs[i]) for i in eidx]
    metrics = evaluate_gen_attack(model, np.asarray(eval_emb)[eidx], refs, ids, vocab, cfg)
    metrics["vocab_size"] = len(vocab)
    return metrics, model, vocab
