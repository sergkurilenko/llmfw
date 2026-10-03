# LLM Firewall (llmfw): инженерная и научно-техническая части

Дата: 03.10.2026. Проверено на litellm 1.103.2 (установлен локально, интерфейсы сняты с кода).

---

## Часть 1. Инженерная: LLM FW на открытой модели с интеграцией в LiteLLM

### 1.1. Как LiteLLM подключает guardrails (факты из кода 1.103.2)

Два способа:

A. **Generic Guardrail API** (рекомендуется). LiteLLM сам вызывает внешний HTTP-сервис.
   - Конфиг: `guardrail: generic_guardrail_api`, `api_base: http://llmfw:8080`,
     `mode: pre_call | post_call | during_call | [pre_call, post_call]`,
     `unreachable_fallback: fail_closed | fail_open`, `fail_on_error`, `default_on`,
     `api_key` (уходит заголовком `x-api-key`).
   - LiteLLM шлёт `POST {api_base}/beta/litellm_basic_guardrail_api` с телом:
     `input_type: "request"|"response"`, `structured_messages` (роли system/user/assistant/tool),
     `texts`, `tools`, `tool_calls`, `images`, `model`, `litellm_call_id`, `litellm_trace_id`,
     `request_data` (хэш ключа, user_id, team_id, org_id, end_user_id), `request_headers`,
     `additional_provider_specific_params`.
   - Ответ: `action: "BLOCKED" | "NONE" | "GUARDRAIL_INTERVENED"`, `blocked_reason`,
     `texts` / `structured_messages` (изменённый ввод/вывод, напр. маскирование),
     `stream_holdback_chars`.
   - Стриминг: `streaming_end_of_stream_only`, `streaming_sampling_rate` (каждый N-й чанк),
     `streaming_transform_mode: block_only | incremental_diff`.
   - Включение на запрос: поле `"guardrails": ["llmfw"]` в теле chat/completions,
     с динамическими параметрами через `extra_body`; на ключ/команду — через UI/БД LiteLLM.
   Плюсы для продукта: файрвол — отдельный сертифицируемый сервис, язык и стек свои,
   не зависит от версий LiteLLM, тот же эндпоинт подключается к другим шлюзам.

B. **In-process CustomGuardrail**: класс-наследник `litellm.integrations.custom_guardrail.CustomGuardrail`,
   переопределяется `apply_guardrail(inputs, request_data, input_type, logging_obj)`;
   блокировка — `GuardrailRaisedException`/`HTTPException`, изменение — вернуть изменённые `inputs`.
   Хуки: pre_call, during_call, post_call, logging_only, pre/during/post_mcp_call.
   Подходит для тонкого адаптера и для тестов, но продуктовое ядро держать в сервисе (A).

В LiteLLM уже есть встроенные хуки `promptguard` (Meta Prompt Guard), `llm_as_a_judge`,
`typesafe` (Jev, но только для компакции контекста, не для безопасности), `presidio` (PII).
Это базовые линии для сравнения и образцы кода.

### 1.2. Архитектура сервиса llmfw

```
LiteLLM proxy ──HTTP──► llmfw-gateway (FastAPI)
                          ├─ normalizer      : Unicode NFKC, гомоглифы кириллица/латиница,
                          │                    невидимые символы, транслит, base64/rot/zero-width
                          ├─ policy engine   : правила per team/key (категории, пороги, действия)
                          ├─ detectors (параллельно, батарея типизированных вопросов):
                          │    • injection/jailbreak (вход)        — модель A
                          │    • harm categories + severity (вход/выход) — модель B
                          │    • PII/секреты (regex + NER)        — presidio/natasha
                          │    • leak/policy violation (выход)
                          ├─ decision        : пороги allow/block/escalate, калибровка,
                          │                    объяснение (какой фрагмент/категория)
                          ├─ actions         : BLOCKED / NONE / INTERVENED (маскирование, вырезание)
                          └─ audit           : журнал решений, метрики, трассировка по litellm_trace_id
Model server: vLLM или ONNX Runtime/CTranslate2 (CPU-режим обязателен для on-prem)
```

Принципы: fail_closed по умолчанию; детерминированный слой (нормализация, regex, политика)
отделён от ML-слоя — это упрощает сертификацию; все решения с вероятностями и причиной.

### 1.3. Выбор открытых моделей (стартовый набор)

| Роль | Модель | Почему |
|---|---|---|
| Инъекции/джейлбрейки, вход | Meta Prompt Guard 2 (86M, 22M) | быстрая, есть хук в LiteLLM; слабее на русском |
| Инъекции, русский | HiveTraceGuard-Pro 0.6B (LoRA от Qwen3-0.6B, Apache-2.0) | лучший заявленный русский recall (0.999), 14 мс; но их русские наборы собраны ими же, ≥27% пересечения с обучением |
| Категории вреда, вход/выход, мультиязычность | Qwen3Guard-Gen 0.6B/4B (Apache-2.0, 119 языков, safe/controversial/unsafe) | лучший F1 в независимом сравнении (0.756 у 4B), стриминговый вариант Qwen3Guard-Stream |
| Сравнение/запас | Granite Guardian 3.x, PolyGuard, PIGuard (анти-overdefense) | базовые линии |

Стартовая конфигурация MVP: Prompt Guard 2 + Qwen3Guard-Gen-0.6B + presidio; HiveTraceGuard-Pro
как альтернатива для русского входа. Все модели открытые, on-prem, без передачи данных.

### 1.4. Минимальный сервис (скелет)

```python
# llmfw/gateway.py
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Literal, Any

app = FastAPI()

class Req(BaseModel):
    input_type: Literal["request", "response"]
    structured_messages: list[dict[str, Any]] | None = None
    texts: list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    model: str | None = None
    litellm_call_id: str | None = None
    litellm_trace_id: str | None = None
    request_data: dict[str, Any] = {}
    additional_provider_specific_params: dict[str, Any] | None = None

class Resp(BaseModel):
    action: Literal["BLOCKED", "NONE", "GUARDRAIL_INTERVENED"]
    blocked_reason: str | None = None
    texts: list[str] | None = None
    structured_messages: list[dict[str, Any]] | None = None

@app.post("/beta/litellm_basic_guardrail_api", response_model=Resp)
async def guard(req: Req, x_api_key: str | None = Header(default=None)):
    policy = policies.for_key(req.request_data)             # per team/key
    segs = normalizer.segments(req)                          # роль + текст + нормализованный текст
    verdict = await decision_core.evaluate(segs, req.input_type, policy)  # батарея вопросов
    audit.log(req, verdict)
    if verdict.block:
        return Resp(action="BLOCKED", blocked_reason=verdict.reason)
    if verdict.modified:
        return Resp(action="GUARDRAIL_INTERVENED", texts=verdict.texts)
    return Resp(action="NONE")
```

```yaml
# litellm config.yaml
litellm_settings:
  guardrails:
    - guardrail_name: "llmfw-in"
      litellm_params:
        guardrail: generic_guardrail_api
        mode: pre_call
        api_base: http://llmfw:8080
        api_key: os.environ/LLMFW_API_KEY
        unreachable_fallback: fail_closed
        default_on: true
    - guardrail_name: "llmfw-out"
      litellm_params:
        guardrail: generic_guardrail_api
        mode: post_call
        api_base: http://llmfw:8080
        streaming_sampling_rate: 5
        streaming_transform_mode: incremental_diff   # маскирование PII в стриме
        default_on: true
```

### 1.5. Этапы

1. (2–3 нед.) Сервис-скелет, нормализатор, подключение к LiteLLM, e2e-тесты с локальной LLM
   (Qwen через vLLM). Журнал решений.
2. (3–4 нед.) Детекторы: Prompt Guard 2, Qwen3Guard-Gen-0.6B, presidio. Политики per key.
   Пороги. Стриминг post_call.
3. (2 нед.) Стенд оценки: PIDS-Bench, NotInject, CAPTURE, русские наборы (п. 2.4), латентность
   p50/p95 на CPU и GPU. Отчёт — это и есть «нулевая точка» для научной части.
4. (2 нед.) Упаковка: Docker/Helm, конфиг, метрики Prometheus, документация, fail-closed тесты.

Что считать готовым MVP: блокирует инъекции и вредоносный контент на русском и английском,
маскирует PII в ответах, p95 < 50 мс на CPU для входа ≤ 2k токенов, журнал с причинами,
воспроизводимый отчёт качества.

---

## Часть 2. Научно-техническая: своя модель вместо открытой с заметным ростом качества

### 2.1. Где именно открытые модели проваливаются (и это и есть «заметный рост»)

1. **Over-defense.** PIDS-Bench: детектор с F1 = 0.98 на отложенной выборке ошибочно блокирует
   около трети внешних безобидных текстов на тематику безопасности; ни один детектор не
   достигает одновременно F1 ≥ 0.95 и FPR ≤ 0.10 на трудных негативах; три вставленных слова
   переворачивают до 23% чистых запросов. NotInject (339 безобидных с триггер-словами)
   измеряет это напрямую. Для корпоративного заказчика ложные блокировки — главный
   источник отказа от продукта.
2. **Русский язык.** Независимых русскоязычных оценок нет. У HiveTraceGuard-Pro русские
   наборы собраны самой командой и ≥27% пересекаются с обучением. Qwen3Guard мультиязычна,
   но русский не входит в её сильные языки по отчёту.
3. **Обфускация.** Гомоглифы кириллица/латиница, транслит, ё/е, невидимые символы, падежные
   перефразы — атаки обхода дают до 100% успеха против Prompt Guard и Azure Prompt Shield.
4. **Калибровка под атакой.** ECE гард-моделей растёт на порядок (Llama-Guard-3: 0.013 → 0.308).
   Пороги, выставленные на чистых данных, под атакой не держат ни пропуски, ни ложные срабатывания.
5. **Сдвиг распределения.** Обучающие корпуса — чаты и форумы; корпоративный трафик
   (тикеты, документы, код, RAG-контекст) другой. PIDS-Bench: hard-negative аугментация
   чинит overdefense на курируемых примерах, но не на внешних (provenance-sensitive over-defense).

Цель научной части — модель, которая на русском и английском при фиксированном recall
по атакам даёт в разы меньший FPR на трудных негативах, сохраняет это под обфускацией и
сдвигом, и имеет калиброванные вероятности с гарантиями на пороги.

### 2.2. Что обучать

Архитектура: неавторегрессионный двунаправленный энкодер (кандидаты: Laya multilingual 322M,
ruRoBERTa-large, multilingual-E5-large, или энкодерная «голова» над Qwen3-0.6B) с батареей
типизированных выходов за один проход:
- `injection: bool`, `jailbreak: bool`, `harm_category: choice` (таксономия по OWASP LLM Top-10 +
  категории под требования ФСТЭК), `severity: score`, `controversial: bool`,
  `culprit_span: choice/segment` (объяснение).
- Эмбеддинги роли/происхождения (system/user/assistant/tool/RAG-контекст): одно и то же
  предложение «перешли файл» в user и в tool-выводе — разные решения.

Обучение (ключевые приёмы, каждый — измеримый вклад в абляции):
1. Строго собственные скоринговые правила (log/Brier/spherical) как функция потерь →
   калибровка как свойство обучения, а не постобработка (так делает Laya; подтверждено
   на Jev).
2. Канонизация + инвариантность к обфускации: обучающая пара (исходный, обфусцированный)
   с одинаковой меткой и контрастивной потерей; нормализатор на инференсе снимает часть
   атак детерминированно.
3. Hard-negative mining против over-defense: безобидные тексты с триггер-словами
   (документация по безопасности, обсуждение атак, инструкции), сгенерированные и
   собранные из внешних источников — именно внешних, иначе provenance-sensitive over-defense.
4. Русскоязычные данные: перевод+адаптация существующих наборов (PIDS-Bench, NotInject,
   CAPTURE, BIPIA, JailbreakBench, AEGIS2.0, PolyGuardMix-ru) + синтез локальной LLM +
   ручная валидация инженерами ИБ; корпоративные домены компании.
5. Дистилляция из LLM-судьи (Qwen3-32B/GigaChat) на неразмеченном корпоративном трафике
   с фильтрацией по согласию судей.
6. Пороги через conformal risk control / Learn-Then-Test: гарантии на долю пропусков и
   на FPR при заданных ограничениях, в т. ч. раздельные пороги allow/block/escalate.

### 2.3. Что защищать (диссертация, 2.3.6)

Тема (вариант под LLM FW):
«Методы обнаружения и блокирования состязательных воздействий на большие языковые модели
на основе калиброванных неавторегрессионных моделей принятия решений»

Положения:
П1. Модель угроз и формальная постановка задачи фильтрации взаимодействий с LLM как
    селективной многозадачной классификации с ограничениями на пропуски и ложные
    блокировки при обфускации и сдвиге распределения.
П2. Метод построения калиброванной неавторегрессионной модели-фильтра с батареей
    типизированных решений, эмбеддингами происхождения и обучением на собственных
    скоринговых правилах с инвариантностью к обфускации.
П3. Метод выбора порогов решений с конечновыборочными гарантиями на долю пропусков и
    ложных блокировок (conformal risk control) с учётом состязательного сдвига.
П4. Русскоязычный бенчмарк фильтров LLM (атаки, трудные негативы, обфускации, сдвиг),
    методика оценки и экспериментальное подтверждение превосходства над открытыми
    гард-моделями при фиксированном recall.

Связь с ранней идеей: это тот же «калиброванный слой типизированных решений», что и у
Jev/Laya, но свой, on-prem, на русском и с гарантиями — без агентной части.

### 2.4. Протокол доказательства «заметного роста»

- Наборы: PIDS-Bench (over-defense, обфускация, сдвиг), NotInject, CAPTURE, BIPIA (косвенные
  инъекции), JailbreakBench/HarmBench, русские наборы HiveTrace (за вычетом пересечений),
  собственный русский бенчмарк (пилот 500 → 3 000).
- Базовые линии: Prompt Guard 2, PIGuard/ProtectAI-DeBERTa, Qwen3Guard-Gen 0.6B/4B,
  HiveTraceGuard-Pro, Granite Guardian, LLM-судья, Jev (если доступ).
- Главные графики: (а) FPR на трудных негативах при recall ≥ 0.95; (б) recall под обфускацией
  по типам; (в) ECE/Brier чисто и под атакой; (г) латентность p95 CPU; (д) сдвиг на
  корпоративный трафик. Бутстрэп-ДИ, 5 сидов, полная матрица ошибок, без одного F1.
- Абляции по каждому приёму из 2.2.
- Критерий успеха для диссертации: на русском при recall 0.95 FPR на трудных негативах
  ниже лучшей открытой модели минимум вдвое, калибровка под атакой ECE < 0.05,
  p95 < 30 мс на CPU.

### 2.5. Риски

- Чужие наборы с утечкой в обучение: дедупликация против всех тестов (MinHash), держать
  приватный тест.
- Over-defense не лечится аугментацией на внешних данных (PIDS-Bench) — это главный
  научный вызов, ответ — provenance-эмбеддинги и внешний сбор негативов, проверить рано.
- Темп области: статьи по гард-моделям еженедельно; препринт П1/П3 — рано.

### Источники
- LiteLLM 1.103.2, код: `litellm/proxy/guardrails/guardrail_hooks/generic_guardrail_api/`,
  `litellm/integrations/custom_guardrail.py`, `litellm/types/proxy/guardrails/guardrail_hooks/generic_guardrail_api.py`
- https://docs.litellm.ai/docs/proxy/guardrails/custom_guardrail
- https://docs.litellm.ai/docs/adding_provider/generic_guardrail_api
- https://arxiv.org/pdf/2609.01046 — HiveTraceGuard-Pro
- https://arxiv.org/pdf/2510.14276 — Qwen3Guard Technical Report
- https://arxiv.org/html/2605.28830v1 — Benchmarking Open-Source Safety Guard Models
- https://arxiv.org/abs/2609.15017 — PIDS-Bench
- https://arxiv.org/abs/2410.22770 — InjecGuard / NotInject / PIGuard
- https://arxiv.org/pdf/2505.12368v1 — CAPTURE
- https://arxiv.org/pdf/2504.11168 — Bypassing LLM Guardrails
- https://arxiv.org/html/2609.36477 — Guard Models Are Overconfident
- https://arxiv.org/pdf/2504.08848 — X-Guard; PolyGuard (17 языков)
- https://github.com/bastion-soft/pi-detector-bench — открытый двухосевой бенчмарк детекторов
- https://github.com/AjeyDS/guardrail-showdown — Jev vs ProtectAI vs Lakera (Jev: 79.3% recall, 3.6% FPR)
