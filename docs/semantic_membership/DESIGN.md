# Дизайн исследования: слой «семантическое членство кода» для LLM-файрвола

Статус: источник истины для кода в `experiments/semantic_membership/` и для статьи в `paper/`.
Дата: 2026-10-03. Все модули обязаны следовать контрактам из разделов 2–10.

## 0. Цель и исследовательские вопросы

Слой «семантическое членство» отвечает на один типизированный вопрос файрвола:
**принадлежит ли фрагмент кода в запросе к LLM защищённой (закрытой) кодовой базе**,
даже если фрагмент переформатирован, переименован, вырезан частично, разбавлен
мусорным кодом или переписан. Это второй уровень после детерминированных отпечатков
(winnowing), который ловит преобразования, разрушающие k-граммы.

Исследовательские вопросы:
- RQ1. Насколько преобразования разных классов (форматирование, переименование, частичное
  извлечение, вставка мёртвого кода, переупорядочивание, LLM-перефраз, перевод на другой
  язык) снижают полноту отпечатков winnowing, и восстанавливает ли её семантический слой?
- RQ2. Какова цена семантического слоя по ложным срабатываниям на публичном коде и на
  трудных негативах той же предметной области, и даёт ли конформная калибровка порога
  обещанную границу FPR при сдвиге распределения?
- RQ3. Превосходит ли гибрид (ANN по эмбеддингам → переранжирование отпечатками) каждую
  из компонент по TPR при фиксированном FPR, и как зависит эффект от длины фрагмента?
- RQ4. Сколько стоит слой по латентности и памяти при индексе 10^5–10^6 функций
  (CPU и GPU), и укладывается ли он в бюджеты чата и IDE-автодополнения?
- RQ5. Что утекает из индекса эмбеддингов при компрометации шлюза (атаки инверсии и
  выравнивания), и какие защиты (ключевая ортогональная проекция, квантование, шум)
  сохраняют полноту членства при ограниченной утечке?

## 1. Постановка задачи и модель угроз

- Защищённая база `P = {s_1..s_N}` — множество функций (и файловых окон) из закрытых
  репозиториев. Индекс `I(P)` строится офлайн и развёртывается на шлюзе.
- Запрос `q` — фрагмент кода из промпта. `q` получен из `s ∈ P` преобразованием `t ∈ T`
  (член) либо из кода вне `P` (не-член).
- Детектор `D: q → score ∈ [0,1]`; решение `score ≥ τ`.
- Требования: (a) TPR по классам `T` и длинам; (b) FPR ≤ α на не-членах с конечновыборочной
  гарантией; (c) латентность: чат ≤ 50 мс, IDE ≤ 20 мс на запрос (CPU); (d) приватность:
  из `I(P)` без ключа нельзя восстановить `P` сверх заданной границы.
- Нарушители: небрежный сотрудник (T без умысла), IDE-ассистент (частичные окна, большие
  объёмы), инсайдер с умыслом (вставка мёртвого кода, переименование, перефраз LLM,
  перевод на другой язык), компрометация шлюза (доступ к индексу без ключа; с n
  утёкшими парами «код — вектор»).

Вне рамок статьи: пересказ алгоритма словами, сессионное дробление (другой слой).

## 2. Данные

### 2.1. Источники и роли репозиториев
Закрытой базы компании в публикуемом эксперименте нет; её моделирует набор
репозиториев одной предметной области (сетевая безопасность, VPN, криптография,
обнаружение вторжений), а трудные негативы — другие репозитории той же области.
Публичный корпус — репозитории других областей.

Файл `configs/repos.yaml` (создаёт модуль данных) со списками:
- `protected`: 12–20 репозиториев, 1–20 тыс. звёзд, языки C/C++/Go/Python/Java/JS,
  домен «security/network/crypto». Кандидаты (проверить лицензию и доступность):
  OpenVPN/openvpn, strongswan/strongswan, libssh2/libssh2, wolfSSL/wolfssl,
  jedisct1/libsodium, the-tcpdump-group/libpcap, ntop/nDPI, snort3/snort3,
  OISF/suricata, osquery/osquery, zeek/zeek, slackhq/nebula, cloudflare/cfssl,
  smallstep/certificates, coredns/coredns, scapy/scapy, pyca/cryptography,
  paramiko/paramiko, fortra/impacket, mitmproxy/mitmproxy, weidai11/cryptopp,
  bcgit/bc-java, netty/netty (сетевой слой), nodejs/undici.
  Из них случайно (seed) выбираются `protected` (≈12) и `hard_neg` (остальные).
- `public`: ≥150 репозиториев других областей (веб-фреймворки, данные, игры, утилиты,
  UI, компиляторы): pallets/flask, psf/requests, fastapi/fastapi, django/django,
  gin-gonic/gin, gohugoio/hugo, spf13/cobra, nlohmann/json, fmtlib/fmt, redis/redis,
  sqlite? (не git) → antirez/sds, ggerganov/llama.cpp, apache/commons-lang,
  google/gson, expressjs/express, lodash/lodash, ... Список расширяется скриптом
  через api.github.com (поиск по звёздам и языку, исключая домен protected/hard_neg).
  Делится по репозиториям на `public_train` (0.6) / `public_calib` (0.2) /
  `public_test` (0.2).

### 2.2. Клонирование и извлечение
- `git clone --depth 1`, лимит размера репозитория (по умолчанию 300 МБ), пропуск
  каталогов из `extract.skip_dirs`, бинарных и сгенерированных файлов (`*.min.js`,
  `*.pb.go`, `*_generated.*`), файлов > 1 МБ.
- Функции извлекаются tree-sitter (`tree_sitter_language_pack`):
  python: function_definition; c/cpp: function_definition; java: method_declaration,
  constructor_declaration; go: function_declaration, method_declaration;
  javascript: function_declaration, method_definition, arrow_function с телом-блоком.
- Ограничения: `min_lines ≤ строк ≤ max_lines`, токенов ≤ `max_tokens`.
- Дополнительно файловые окна: скользящие окна по 20 строк с шагом 10 для файлов
  protected (для сценария IDE, `kind: window`).

### 2.3. Дедупликация между ролями (обязательна)
Публичный код часто вендорит те же библиотеки, что и protected. Любая функция из
`public_*` и `hard_neg`, доля winnowing-отпечатков которой, общих с protected,
≥ `dedup.overlap_threshold`, удаляется (с логированием числа удалённых). Внутри
protected точные дубликаты схлопываются (один `sha`).

### 2.4. Контракт записи функции (`data/functions/{split}.jsonl`)
```json
{"id": "protected/openvpn/src/openvpn/crypto.c:120-158",
 "split": "protected", "repo": "OpenVPN/openvpn", "path": "src/openvpn/crypto.c",
 "lang": "c", "kind": "function", "start_line": 120, "end_line": 158,
 "code": "...", "n_lines": 39, "n_tokens": 212, "sha": "<sha1 нормализованного кода>"}
```
`split ∈ {protected, hard_neg, public_train, public_calib, public_test}`.

## 3. Нормализация (`smcode/normalize.py`)

```python
def tokenize(code: str, lang: str) -> list[Token]      # Token(text, kind, start, end); kind ∈ {id, kw, op, punct, num, str, comment}
def abstract_tokens(tokens, mode="full") -> list[str]  # full: id→"ID", num→"NUM", str→"STR", comment удалён; "indexed": ID_k по порядку появления; "lexical": как есть без комментариев
def strip_comments(code: str, lang: str) -> str
def normalized_sha(code: str, lang: str) -> str        # sha1 по abstract_tokens(mode="full")
```
Реализация на tree-sitter (листовые узлы); ключевые слова — по спискам на язык.

## 4. Преобразования (`smcode/transforms/`)

Каждое преобразование — функция `t(code: str, lang: str, rng) -> str | None`
(None — неприменимо). Классы и параметры:
- `identity`.
- `reformat`: изменение отступов (2/4/tab), слияние/разрыв строк где безопасно,
  пробелы вокруг операторов, перенос скобок (для C-подобных).
- `strip_comments`: удаление комментариев и docstring.
- `rename_ids`: согласованное переименование всех локальных идентификаторов (параметры,
  локальные переменные, имя функции) в случайные имена (стили: `v1..vn`, snake,
  camel, короткие слова), без переименования вызовов внешних API и ключевых слов;
  реализация через tree-sitter: идентификаторы, объявленные внутри функции.
- `change_literals`: замена числовых и строковых литералов.
- `insert_deadcode` (атака Mossad): вставка no-op выражений/присваиваний неиспользуемых
  переменных каждые `deadcode_every` строк в теле (синтаксически корректных для языка).
- `reorder_stmts`: перестановка соседних независимых простых операторов (нет общих
  идентификаторов, нет вызовов) — консервативная проверка через множества идентификаторов.
- `partial`: окно из `L` подряд идущих строк тела (`L ∈ partial_lines`), по возможности
  с выровненными границами операторов.
- `combo`: reformat + strip_comments + rename_ids + change_literals.
- LLM (только GPU, `transforms/llm_paraphrase.py`, vLLM, `transforms.llm_model`):
  `paraphrase` («перепиши функцию, сохранив поведение, другими именами и структурой»),
  `translate` («переведи на язык X», X ≠ исходного, из списка languages). Результат
  принимается только если tree-sitter парсит его без `ERROR`-узлов; хранится промпт,
  модель, seed. Поведенческая эквивалентность не проверяется (ограничение статьи).

Контракт: `apply_transform(name, code, lang, rng, **params) -> TransformResult(code, name, params, ok)`.

## 5. Наборы запросов (`smcode/eval/build_queries.py`)

Для каждого набора негативов `N ∈ {public_calib, public_test, hard_neg}` и для
`protected` (члены) строятся запросы: к каждой функции применяется каждое
преобразование (и `partial` для каждого `L`), чтобы артефакты преобразования не стали
признаком класса. Семплирование: до `max_per_split` функций (по умолчанию 5000) с
стратификацией по языку и длине.

Контракт (`data/queries/{set}.jsonl`):
```json
{"qid": "protected/.../crypto.c:120-158#rename_ids#0", "source_id": "...", "label": 1,
 "set": "protected", "transform": "rename_ids", "params": {"style": "camel"},
 "lang": "c", "repo": "OpenVPN/openvpn", "code": "...", "n_lines": 39, "n_tokens": 207}
```
`label = 1` только для `set == protected`. Для `partial` в `params.L`.

## 6. Методы — единый API индекса (`smcode/fingerprint/index.py`, `smcode/semantic/`)

```python
class MembershipIndex(Protocol):
    name: str
    def build(self, records: Iterable[FunctionRecord], cfg) -> None
    def query(self, code: str, lang: str) -> QueryResult   # QueryResult(score: float, best_id: str|None, details: dict)
    def query_batch(self, items: list[tuple[str, str]]) -> list[QueryResult]
    def save(self, path) / load(path)
    def memory_bytes(self) -> int
```
Методы:
- `M0 exact`: sha по нормализованным окнам из 5 строк (lexical без комментариев);
  score = доля окон запроса, найденных в индексе.
- `M1 winnowing`: k-граммы abstract(full)-токенов, окно w, отпечаток = min hash в окне
  (гарантия обнаружения общих подстрок ≥ w+k−1 токенов), хэш = HMAC-SHA256(key, k-грамма)
  усечённый до 64 бит; индекс: dict hash → list[function_id] (compact arrays).
  score = доля отпечатков запроса, найденных в индексе; details: лучший `function_id`
  по числу общих отпечатков, длина максимального непрерывного совпадения.
  Фильтр общности: отпечатки, встречающиеся в > `common_df` (по умолчанию 50) функциях
  public_train или в > 5% файлов protected, исключаются.
- `M2 minhash`: MinHash (perm=128) по shingle'ам из `minhash_shingle` abstract-токенов,
  LSH (datasketch) → кандидаты → точная оценка Жаккара; score = max Jaccard.
- `M3 semantic`: эмбеддинг энкодера (раздел 7), ANN (FAISS IndexFlatIP / HNSW, на CPU —
  numpy brute force при N ≤ 2·10^5) → score = max cosine по top-k.
- `M4 hybrid`: top-k кандидатов M3 → для каждого кандидата признаки
  [cos, overlap_winnowing(q, cand), lcs_tokens_norm, n_tokens_q] → логистический
  комбинатор (обучен на public_train: члены = преобразованные public_train функции против
  индекса public_train, не-члены = другие функции) → score = max по кандидатам.
  Абляция: правило двух порогов без обучения.

## 7. Семантический энкодер (`smcode/semantic/`)

- `model.py`: обёртка над HF-энкодером (`microsoft/unixcoder-base` по умолчанию;
  альтернативы через конфиг: `Salesforce/codet5p-110m-embedding`, `codesage/codesage-small`),
  mean-pooling, L2-нормализация, max_length 512, вход — код без комментариев.
  Вариант `own_small_model` — 6-слойный трансформер с нуля на токенах tree-sitter
  (BPE на abstract(indexed)+lexical) для лицензионно чистого on-prem варианта.
- `train.py` (GPU): контрастивное дообучение. Якорь — функция из public_train; позитив —
  её случайное преобразование (программное; LLM-преобразования — если сгенерированы);
  негативы — in-batch + `hard_negatives_per_anchor` функций из того же файла/репозитория.
  Loss InfoNCE (симметричный), temperature 0.05, batch 128, 2 эпохи, lr 2e-5, AdamW,
  warmup 5%, fp16/bf16, grad checkpointing. Логи: loss, recall@1 на валидации
  (public_calib-пары). Чекпойнт лучшей эпохи. Zero-shot (без дообучения) — базовая линия.
  Протектед-код в обучении не участвует (реалистично: модель общая для всех заказчиков);
  абляция `adapt_on_protected: true` — дообучение на protected (заказчик-специфичная
  адаптация) с отчётом о влиянии.
- `embed.py`: батчевый инференс, сохранение `float16` матрицы + ids; замер латентности
  на CPU (1 и 4 потока) и GPU, batch 1 и 32.
- `ann.py`: FAISS (если установлен) или numpy; `hybrid.py`: раздел 6 M4.

## 8. Калибровка порога (`smcode/calibration.py`)

Split conformal для нулевого класса: по скорам негативов калибровки `S = {s_1..s_n}`
(public_calib с теми же преобразованиями) порог `τ_α = quantile(S, ⌈(n+1)(1−α)⌉/n)`.
Гарантия: при обменяемости новых негативов с S, `P(score_new ≥ τ_α) ≤ α`.
Проверка: эмпирический FPR на public_test (та же популяция) и hard_neg (сдвиг),
с доверительными интервалами Клоппера–Пирсона. Вариант «калибровка по домену»:
часть hard_neg (по репозиториям) идёт в калибровку, остальная — в тест.

## 9. Приватность индекса (`smcode/privacy/`)

- Защиты: `projection.py` — ключевая случайная ортогональная матрица `Q` (seed из
  ключа), `e' = Q e`; `quantize.py` — int8/int4/binary квантование; шум Гаусса σ.
- Атака A1 (`attack_bow.py`): инверсия «мешок токенов»: MLP по эмбеддингу предсказывает
  наличие каждого из `bow_vocab` лексических токенов (идентификаторов); обучение на
  public_train, оценка на protected: F1 по токенам, доля восстановленных редких
  идентификаторов (df ≤ 3 в public).
- Атака A2 (`attack_align.py`): выравнивание (ALGEN-подобное): атакующему известны
  `n` пар (код, защищённый вектор) → МНК-оценка `Q̂` → применяет A1 через `Q̂^{-1}`;
  кривая утечки от `n ∈ attack_leaked_pairs`.
- Атака A3 (`attack_gen.py`, GPU, опционально): генеративная инверсия — небольшой
  декодер (6 слоёв) обучается восстанавливать код по эмбеддингу (vec2text-подобно);
  метрики BLEU/точное совпадение идентификаторов.
- Полезность: TPR@FPR=1% метода M3/M4 при каждой защите (та же калибровка).

## 10. Оценка (`smcode/eval/`)

- `run_eval.py`: для каждого метода и набора запросов → `results/scores/{method}/{set}.jsonl`
  (`qid, score, best_id, latency_ms`).
- `metrics.py`: TPR@FPR=α (порог из калибровки и «оракульный» порог на public_test),
  AUROC (члены vs public_test, члены vs hard_neg), TPR по преобразованию, по языку, по
  бинам длины; FPR по наборам негативов с ДИ; бутстрэп (1000) с кластеризацией по
  исходной функции; латентность p50/p95; память индекса.
- `tables.py`: markdown-таблицы в `results/tables/*.md` (их включает статья).
- `plots.py`: рисунки в `paper/figures/*.png` (300 dpi, подписи на русском):
  F1 TPR по преобразованиям (группы столбцов по методам); F2 TPR vs длина фрагмента;
  F3 ROC члены vs hard_neg; F4 FPR заявленный vs эмпирический по наборам;
  F5 абляция гибрида; F6 кривые приватность–полезность; F7 латентность vs размер индекса.
- Правила: никакого одного F1; полная матрица ошибок доступна в `results/summary.json`.

## 11. Код, CLI и воспроизводимость

```
experiments/semantic_membership/
  configs/default.yaml, configs/repos.yaml
  smcode/ (пакет; python -m smcode.<module>)
  scripts/00_clone.sh 01_extract.py 02_dedup_split.py 03_build_queries.py
          04_build_indexes.py 05_train_encoder.py(GPU) 06_embed.py 07_eval.py
          08_privacy.py 09_report.py  run_all.sh  smoke_test.sh
  tests/ (pytest, синтетические данные, CPU, без сети и без torch)
  README.md (runbook: шаги, время, GPU-требования, что где лежит)
```
Правила реализации: Python ≥ 3.10, type hints, docstring на русском (кратко);
детерминизм по `seed`; все пути из конфига; без сетевых обращений кроме `00_clone.sh`
и загрузки HF-моделей на GPU-машине; импорт torch/transformers/faiss только внутри
функций или модулей GPU-части, чтобы CPU-тесты шли без них; логирование через `logging`;
прогресс через tqdm; каждый скрипт идемпотентен (пропуск готовых артефактов).
Smoke-тест: `scripts/smoke_test.sh` создаёт 3 синтетических «репозитория» в tmp,
прогоняет 01–04, 07 (без M3/M4 если torch нет), 09 и проверяет, что таблицы созданы.

## 12. Карта статьи → результаты

| Раздел статьи | Источник |
|---|---|
| 5.1 Данные | `results/summary.json: data_stats`, таблица T1 |
| 6.1 Полнота по преобразованиям (RQ1) | T2, F1 |
| 6.2 Длина фрагмента (RQ1/RQ3) | F2 |
| 6.3 Ложные срабатывания и гарантия (RQ2) | T3, F4, F3 |
| 6.4 Гибрид и абляции (RQ3) | T4, F5 |
| 6.5 Латентность и память (RQ4) | T5, F7 |
| 6.6 Приватность индекса (RQ5) | T6, F6 |
| 6.7 LLM-преобразования и перевод | T7 (если сгенерированы) |
