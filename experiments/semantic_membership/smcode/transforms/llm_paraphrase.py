"""LLM-преобразования (DESIGN.md §4, только GPU): paraphrase и translate через vLLM.

vllm импортируется лениво (внутри ``LLMEngine``), поэтому модуль импортируется и на CPU: чистые
функции ``extract_code`` / ``validate_output`` / ``pick_dst_lang`` покрываются тестами без torch.
Результат принимается только если tree-sitter разбирает его без ошибок; в params хранятся
prompt_id, текст промпта-шаблона, модель, seed, temperature.

Поведенческая эквивалентность не проверяется (ограничение статьи).
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter
from typing import Any, Iterable, Sequence

from smcode.config import get_rng, resolve_path
from smcode.normalize import abstract_tokens, canon_lang, count_tokens, tokenize
from smcode.transforms.programmatic import count_lines, parses_ok
from smcode.types import FunctionRecord, QueryRecord, TransformResult, read_jsonl

log = logging.getLogger(__name__)

LANG_NAMES = {"python": "Python", "c": "C", "cpp": "C++", "go": "Go", "java": "Java", "javascript": "JavaScript"}
FENCE_TAGS = {"python": "python", "c": "c", "cpp": "cpp", "go": "go", "java": "java", "javascript": "javascript"}
# Семейства языков: перевод внутри семейства (C ↔ C++) переводом не считается — модель может вернуть копию.
LANG_FAMILIES = {"c": "c", "cpp": "c"}

PARAPHRASE_PROMPT_ID = "paraphrase_v1"
TRANSLATE_PROMPT_ID = "translate_v1"
SYSTEM_PROMPT = "You are an expert software engineer. You answer with code only, inside a single fenced code block."
PARAPHRASE_PROMPT = (
    "Rewrite the following {lang} function so that it keeps exactly the same behaviour but looks different: "
    "use different names for the function, its parameters and local variables, restructure the control flow "
    "where possible (loops, conditionals, early returns), reorder independent statements and change the "
    "formatting. Do not add comments or explanations. Output only the rewritten function in a ```{tag} block.\n\n"
    "```{tag}\n{code}\n```"
)
TRANSLATE_PROMPT = (
    "Translate the following {src} function into {dst}. Keep the same behaviour, signature semantics and "
    "structure; use idiomatic {dst} names and standard-library equivalents. Do not add comments or "
    "explanations. Output only the {dst} function in a ```{tag} block.\n\n"
    "```{src_tag}\n{code}\n```"
)
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 2048
LLM_SET_TRANSFORMS = ("paraphrase", "translate")

_FENCE_RE = re.compile(r"```[a-zA-Z0-9+#_.-]*[ \t]*\r?\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    """Извлекает код из ответа модели: первый (самый длинный) fenced-блок либо весь текст без обрамления."""
    if not text:
        return ""
    blocks = _FENCE_RE.findall(text)
    if blocks:
        return max(blocks, key=len).strip("\n") + "\n"
    t = text.strip()
    # незакрытый fence
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
    if t.endswith("```"):
        t = t[:-3]
    return t.strip("\n") + "\n" if t.strip() else ""


def validate_output(code: str, lang: str, original: str | None = None, original_lang: str | None = None,
                    reject_same_abstract: bool = False) -> bool:
    """Код непустой, разбирается без ошибок и (если задан оригинал) отличается от него текстом;
    с ``reject_same_abstract`` — ещё и потоком abstract(full)-токенов (эхо оригинала с другими именами)."""
    if not code or not code.strip():
        return False
    if original is not None and code.strip() == original.strip():
        return False
    try:
        if not parses_ok(code, lang):
            return False
        if count_tokens(code, lang) < 3:
            return False
        if original is not None and reject_same_abstract:
            src_lang = original_lang or lang
            if abstract_tokens(tokenize(code, lang), "full") == abstract_tokens(tokenize(original, src_lang), "full"):
                return False
        return True
    except Exception:  # pragma: no cover
        return False


def lang_family(lang: str) -> str:
    """Семейство языка (c и cpp — одно семейство, остальные — сами по себе)."""
    lang = canon_lang(lang)
    return LANG_FAMILIES.get(lang, lang)


def pick_dst_lang(src_lang: str, languages: Sequence[str], rng: random.Random) -> str:
    """Случайный целевой язык из другого семейства (для c/cpp — не c и не cpp)."""
    fam = lang_family(src_lang)
    cands: list[str] = []
    for l in languages:
        c = canon_lang(l)
        if lang_family(c) != fam and c not in cands:
            cands.append(c)
    if not cands:
        raise ValueError("no destination language available")
    return rng.choice(cands)


def paraphrase_messages(code: str, lang: str) -> list[dict[str, str]]:
    lang = canon_lang(lang)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PARAPHRASE_PROMPT.format(lang=LANG_NAMES[lang], tag=FENCE_TAGS[lang], code=code.rstrip())},
    ]


def translate_messages(code: str, src_lang: str, dst_lang: str) -> list[dict[str, str]]:
    src, dst = canon_lang(src_lang), canon_lang(dst_lang)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": TRANSLATE_PROMPT.format(
            src=LANG_NAMES[src], dst=LANG_NAMES[dst], tag=FENCE_TAGS[dst], src_tag=FENCE_TAGS[src], code=code.rstrip())},
    ]


class LLMEngine:
    """Обёртка над vLLM (ленивый импорт). Один экземпляр на процесс (модель загружается один раз)."""

    def __init__(self, cfg: dict[str, Any], seed: int | None = None, temperature: float | None = None,
                 max_tokens: int | None = None) -> None:
        tcfg = cfg.get("transforms", {})
        self.model: str = str(tcfg.get("llm_model", "Qwen/Qwen2.5-Coder-7B-Instruct"))
        self.seed: int = int(cfg.get("seed", 0) if seed is None else seed)
        self.temperature: float = float(tcfg.get("llm_temperature", DEFAULT_TEMPERATURE) if temperature is None else temperature)
        self.max_tokens: int = int(tcfg.get("llm_max_tokens", DEFAULT_MAX_TOKENS) if max_tokens is None else max_tokens)
        self.max_model_len: int = int(tcfg.get("llm_max_model_len", 8192))
        self.gpu_util: float = float(tcfg.get("llm_gpu_memory_utilization", 0.9))
        self.tensor_parallel: int = int(tcfg.get("llm_tensor_parallel", 1))
        self._llm = None
        self._sp = None

    def _ensure(self) -> None:
        if self._llm is not None:
            return
        from vllm import LLM, SamplingParams  # GPU-only, ленивый импорт

        log.info("loading vLLM model %s (seed=%d)", self.model, self.seed)
        self._llm = LLM(model=self.model, seed=self.seed, dtype="auto", max_model_len=self.max_model_len,
                        gpu_memory_utilization=self.gpu_util, tensor_parallel_size=self.tensor_parallel,
                        trust_remote_code=True)
        self._sp = SamplingParams(temperature=self.temperature, top_p=0.95, max_tokens=self.max_tokens, seed=self.seed)

    def generate(self, conversations: list[list[dict[str, str]]]) -> list[str]:
        """Батчевый chat-инференс; возвращает текст первого сэмпла для каждого диалога."""
        if not conversations:
            return []
        self._ensure()
        assert self._llm is not None and self._sp is not None
        try:
            outputs = self._llm.chat(messages=conversations, sampling_params=self._sp, use_tqdm=True)
        except TypeError:  # старые версии vLLM: chat принимает один диалог
            tok = self._llm.get_tokenizer()
            prompts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in conversations]
            outputs = self._llm.generate(prompts, self._sp, use_tqdm=True)
        return [o.outputs[0].text if o.outputs else "" for o in outputs]


def _base_params(engine: LLMEngine, prompt_id: str, prompt: str) -> dict[str, Any]:
    return {"prompt_id": prompt_id, "prompt": prompt, "model": engine.model, "seed": engine.seed,
            "temperature": engine.temperature}


def paraphrase_batch(codes: Sequence[str], langs: Sequence[str], cfg: dict[str, Any],
                     engine: LLMEngine | None = None, seed: int | None = None) -> list[TransformResult]:
    """Перефраз функций LLM («перепиши, сохранив поведение, другими именами и структурой»)."""
    assert len(codes) == len(langs)
    engine = engine or LLMEngine(cfg, seed=seed)
    convs = [paraphrase_messages(c, l) for c, l in zip(codes, langs)]
    texts = engine.generate(convs)
    out: list[TransformResult] = []
    base = _base_params(engine, PARAPHRASE_PROMPT_ID, PARAPHRASE_PROMPT)
    for code, lang, text in zip(codes, langs, texts):
        new = extract_code(text)
        params = {**base, "src_lang": canon_lang(lang), "dst_lang": canon_lang(lang)}
        if validate_output(new, lang, original=code):
            out.append(TransformResult(code=new, name="paraphrase", params=params, ok=True))
        else:
            out.append(TransformResult.failed("paraphrase", params))
    return out


def translate_batch(codes: Sequence[str], src_langs: Sequence[str], dst_langs: Sequence[str],
                    cfg: dict[str, Any], engine: LLMEngine | None = None,
                    seed: int | None = None) -> list[TransformResult]:
    """Перевод функций на другой язык LLM; результат проверяется парсером целевого языка."""
    assert len(codes) == len(src_langs) == len(dst_langs)
    engine = engine or LLMEngine(cfg, seed=seed)
    convs = [translate_messages(c, s, d) for c, s, d in zip(codes, src_langs, dst_langs)]
    texts = engine.generate(convs)
    out: list[TransformResult] = []
    base = _base_params(engine, TRANSLATE_PROMPT_ID, TRANSLATE_PROMPT)
    for code, src, dst, text in zip(codes, src_langs, dst_langs, texts):
        new = extract_code(text)
        params = {**base, "src_lang": canon_lang(src), "dst_lang": canon_lang(dst)}
        # эхо исходника (текстом или abstract-потоком) переводом не считается
        if validate_output(new, dst, original=code, original_lang=src, reject_same_abstract=True):
            out.append(TransformResult(code=new, name="translate", params=params, ok=True))
        else:
            out.append(TransformResult.failed("translate", params))
    return out


# --------------------------------------------------------------------------- построение LLM-запросов


def _llm_query(rec: FunctionRecord, set_name: str, res: TransformResult, label: int) -> QueryRecord:
    code = res.code or ""
    lang = res.params.get("dst_lang", canon_lang(rec.lang))
    return QueryRecord(
        qid=f"{rec.id}#{res.name}#0", source_id=rec.id, label=label, set=set_name, transform=res.name,
        params=dict(res.params), lang=lang, repo=rec.repo, code=code, n_lines=count_lines(code),
        n_tokens=count_tokens(code, lang),
    )


def build_llm_queries(cfg: dict[str, Any], splits: Iterable[str] | None = None, sample: int | None = None,
                      force: bool = False, engine: LLMEngine | None = None,
                      batch_size: int = 256) -> dict[str, int]:
    """Добавляет LLM-запросы (paraphrase, translate) в data/queries/{set}.jsonl для наборов из
    build_queries.QUERY_SETS. Идемпотентно: если LLM-строки уже есть и не задан force — пропуск (−1).
    Функции берутся ТОЛЬКО из выборки программных запросов (source_id уже имеющихся строк файла), чтобы
    TPR по преобразованиям сравнивались на одних и тех же функциях (DESIGN §5); при llm_sample_per_split
    меньше выборки — стратифицированная подвыборка. Без программных строк набор пропускается с ошибкой (0):
    сначала scripts/03_build_queries.py без --llm-only. Требует GPU и vllm."""
    from smcode.eval.build_queries import (
        QUERY_SETS, _is_complete, _load_records, label_for_set, sample_records, write_jsonl_atomic,
    )

    tcfg = cfg.get("transforms", {})
    wanted = [t for t in tcfg.get("llm", list(LLM_SET_TRANSFORMS)) if t in LLM_SET_TRANSFORMS]
    if not wanted:
        log.warning("transforms.llm is empty; nothing to do")
        return {}
    n_sample = int(tcfg.get("llm_sample_per_split", 5000) if sample is None else sample)
    languages = list(cfg.get("languages", list(LANG_NAMES)))
    bins = cfg.get("eval", {}).get("length_bins_tokens", [[0, 10 ** 9]])
    sets = [s for s in (list(splits) if splits else list(QUERY_SETS)) if s != "protected_windows"]
    out_dir = resolve_path(cfg, "queries")
    engine = engine or LLMEngine(cfg)
    result: dict[str, int] = {}
    for set_name in sets:
        path = out_dir / f"{set_name}.jsonl"
        all_rows: list[dict[str, Any]] = list(read_jsonl(path)) if _is_complete(path) else []
        existing = [r for r in all_rows if r.get("transform") not in LLM_SET_TRANSFORMS]
        if len(existing) < len(all_rows) and not force:
            log.info("skip llm %s: rows exist", set_name)
            result[set_name] = -1
            continue
        if not existing:
            log.error("llm %s: no programmatic queries in %s; build them first (03_build_queries.py without "
                      "--llm-only)", set_name, path)
            result[set_name] = 0
            continue
        prog_ids = {r.get("source_id") for r in existing if r.get("source_id")}
        recs = [r for r in _load_records(cfg, set_name) if r.id in prog_ids]
        if len(recs) < len(prog_ids):
            log.warning("llm %s: %d of %d programmatic source functions not found in functions file",
                        set_name, len(prog_ids) - len(recs), len(prog_ids))
        if not recs:
            result[set_name] = 0
            continue
        sampled = sample_records(recs, n_sample, get_rng(cfg, f"llm_sample:{set_name}"), bins)
        label = label_for_set(set_name)
        new_rows: list[QueryRecord] = []
        stats: Counter = Counter()
        for i in range(0, len(sampled), batch_size):
            chunk = sampled[i : i + batch_size]
            codes = [r.code for r in chunk]
            langs = [canon_lang(r.lang) for r in chunk]
            if "paraphrase" in wanted:
                for rec, res in zip(chunk, paraphrase_batch(codes, langs, cfg, engine=engine)):
                    stats["paraphrase_ok" if res.ok else "paraphrase_failed"] += 1
                    if res.ok:
                        new_rows.append(_llm_query(rec, set_name, res, label))
            if "translate" in wanted:
                dsts = [pick_dst_lang(l, languages, get_rng(cfg, f"llm_dst:{set_name}:{r.id}")) for r, l in zip(chunk, langs)]
                for rec, res in zip(chunk, translate_batch(codes, langs, dsts, cfg, engine=engine)):
                    stats["translate_ok" if res.ok else "translate_failed"] += 1
                    if res.ok:
                        new_rows.append(_llm_query(rec, set_name, res, label))
        n = write_jsonl_atomic(path, [*existing, *new_rows])
        pairs = Counter(f"{r.params.get('src_lang')}->{r.lang}" for r in new_rows if r.transform == "translate")
        with open(out_dir / f"{set_name}.llm_stats.json", "w", encoding="utf-8") as f:
            json.dump({"set": set_name, "n_sampled": len(sampled), "n_programmatic_sources": len(prog_ids),
                       "model": engine.model, **stats, "translate_pairs": dict(sorted(pairs.items()))}, f,
                      ensure_ascii=False, indent=2)
        log.info("llm %s: +%d rows (%s) → %s (%d rows total)", set_name, len(new_rows), dict(stats), path, n)
        result[set_name] = len(new_rows)
    return result


def lexical_overlap(a: str, b: str, lang: str) -> float:
    """Доля лексических токенов a, встречающихся в b (диагностика «насколько переписано»)."""
    ta = {t.text for t in tokenize(a, lang) if t.kind == "id"}
    tb = {t.text for t in tokenize(b, lang) if t.kind == "id"}
    return len(ta & tb) / len(ta) if ta else 0.0
