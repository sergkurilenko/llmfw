# Стенд «семантическое членство кода» для LLM-файрвола — runbook

Экспериментальный стенд к статье *«Слой семантического членства кода для LLM-файрвола»*.
Единственный источник истины по контрактам данных, методам и метрикам — `docs/semantic_membership/DESIGN.md`
(далее «DESIGN»); этот файл описывает, **как** прогнать эксперимент: что установить, в каком порядке запускать
шаги, сколько они длятся, где лежат результаты и как их потребляет статья.

Вопрос, на который отвечает слой: *принадлежит ли фрагмент кода в промпте защищённой кодовой базе*, даже если он
переформатирован, переименован, вырезан частично, разбавлен мёртвым кодом, перефразирован LLM или переведён на
другой язык. Сравниваются пять методов: M0 `exact` (окна строк), M1 `winnowing`, M2 `minhash`, M3 `semantic`
(эмбеддинги энкодера, ANN), M4 `hybrid` (ANN → переранжирование отпечатками, логистический комбинатор), с
конформной калибровкой порога и анализом приватности индекса эмбеддингов (DESIGN §0–§10).

## 1. Структура каталога

```
experiments/semantic_membership/
  configs/default.yaml      единая конфигурация (пути, языки, k/w отпечатков, модель, α, бутстрэп)
  configs/repos.yaml        пул репозиториев protected/hard_neg (36) и public (372), доменные слова
  smcode/                   пакет (python -m smcode.<module>)
    config.py types.py normalize.py calibration.py
    data/        list_repos.py extract.py windows.py dedup_split.py       шаги 00–02
    transforms/  programmatic.py evasion.py partial.py registry.py llm_paraphrase.py (vLLM)
    fingerprint/ index.py (реестр, BaseIndex) exact.py winnowing.py winnowing_index.py minhash.py
    semantic/    model.py own_model.py train.py embed.py ann.py semantic_index.py hybrid.py combiner.py
    eval/        build_queries.py run_eval.py metrics.py tables.py plots.py report.py
    privacy/     projection.py quantize.py attack_bow.py attack_align.py attack_gen.py run.py
  scripts/      00_clone.sh 01_extract.py 02_dedup_split.py 03_build_queries.py 04_build_indexes.py
                05_train_encoder.py (GPU) 06_embed.py (GPU) 07_eval.py 08_privacy.py 09_report.py
                run_all.sh (весь конвейер)  smoke_test.sh (синтетика, CPU)
  tests/        pytest: синтетические данные, CPU, без сети и torch; tests/synthetic_repos.py — генератор smoke-данных
  data/         raw/ functions/ queries/ indexes/ embeddings/        (создаётся; в git не попадает)
  results/      scores/ summary.json tables/ index_stats.json latency_encoder.json privacy.json
  runs/         чекпойнты дообучения (05)
  ../../paper/figures/   рисунки F1–F7 (cfg.paths.figures)
```

Все пути берутся из `configs/default.yaml: paths` и разрешаются относительно этого каталога; все скрипты
запускаются **из него**: `cd experiments/semantic_membership`. Каждый скрипт — тонкая обёртка над функциями
`smcode/` (`--config`, флаги шага), идемпотентен (готовые артефакты пропускаются, пересчёт — `--force`).

## 2. Требования

| Окружение | Установка | Что доступно |
|---|---|---|
| CPU (разработка, тесты, M0–M2, оценка и отчёт) | `python -m venv venv && venv/bin/pip install -r requirements.txt` | Python ≥ 3.10, numpy/scipy/scikit-learn, tree-sitter + language-pack, pyyaml, tqdm, matplotlib, pytest; `datasketch` нужен только тестам (M2 реализован на numpy) |
| GPU (M3/M4, дообучение, приватность с MLP/A3) | `venv/bin/pip install -r requirements-gpu.txt` | + torch, transformers, faiss-cpu (необязателен: `ann.py` откатывается к numpy) |
| LLM-преобразования (paraphrase/translate) | `python -m venv venv-llm && venv-llm/bin/pip install -r requirements-llm.txt` | + **vllm** — в **отдельном** venv: колёса vllm фиксируют точную версию torch и иначе диктуют torch всему стенду; шаг `03 --llm-only` запускается этим интерпретатором |

`torch`/`transformers`/`faiss`/`vllm` импортируются лениво — CPU-часть (`pytest`, шаги 01–04, 07, 09 для M0–M2)
работает без них. Модель энкодера (`semantic.base_model`, по умолчанию `microsoft/unixcoder-base`) и LLM
(`transforms.llm_model`, `Qwen/Qwen2.5-Coder-7B-Instruct`) скачиваются с Hugging Face при первом запуске на
GPU-машине (`HF_HOME` — куда класть кэш). Для Python < 3.11 нужен `tree-sitter >= 0.23`. Альтернативные энкодеры
(`Salesforce/codet5p-110m-embedding`, `codesage/codesage-small`) требуют `semantic.trust_remote_code: true`.

Ориентир по железу для полного прогона (§6): 8+ ядер CPU, 32 ГБ ОЗУ, одна GPU класса A100/L40S/4090 (24–80 ГБ;
vLLM с 7B-моделью в fp16 требует ≥ 16 ГБ), 50–100 ГБ диска (клоны ограничены 300 МБ «полезного» размера каждый).
Все примеры ниже предполагают интерпретатор venv: либо `source venv/bin/activate`, либо `PYTHON=venv/bin/python`
перед `scripts/*.sh` (без переменной `run_all.sh`/`smoke_test.sh` сами берут `venv/bin/python`, если он есть) и
`venv/bin/python scripts/NN_*.py` для отдельных шагов.

### Ключи

```bash
export SMCODE_INDEX_KEY=$(openssl rand -hex 32)        # HMAC-ключ отпечатков M0/M1/M2/M4 (DESIGN §1(d), §6)
export SMCODE_PROJECTION_KEY=$(openssl rand -hex 32)   # ключ ортогональной проекции §9 (иначе берётся SMCODE_INDEX_KEY)
```

Ключ не хранится в репозитории и в артефактах (в `meta.json` индекса записывается только `key_id = sha256(key)[:16]`).
**Один и тот же `SMCODE_INDEX_KEY` должен быть задан при сборке индексов (04) и при оценке (07/08):**
при несовпадении загрузка индекса завершается `ValueError` (иначе все запросы молча получали бы score 0).
Без ключа шаги 04/07 работают с предупреждением (режим отладки/тестов). Для гарантированной
воспроизводимости удобно сохранить ключи в `~/.smcode_env` и делать `source` перед каждым запуском.

## 3. Быстрый старт: smoke-тест и тесты

```bash
cd experiments/semantic_membership
venv/bin/python -m pytest tests -q                                  # ~40 с, CPU
PYTHON=venv/bin/python scripts/smoke_test.sh                        # ~30 с: синтетика → 01→02→03→04→07 (+ winnowing_lexical)→09
PYTHON=venv/bin/python scripts/smoke_test.sh --keep                 # оставить рабочий каталог (путь печатается)
PYTHON=venv/bin/python scripts/smoke_test.sh --gpu                  # на GPU-машине: + 05→06→04 M3/M4→07→08→09 (нужен torch)
```

`smoke_test.sh` генерирует пять синтетических «репозиториев» (protected, hard_neg, public_train, public_calib,
public_test; файлы на python/c/go/javascript по 10–20 функций; `tests/synthetic_repos.py`), пишет smoke-конфиг
(маленькие k/w, 24 функции на набор, бутстрэп 20, рисунки в tmp), прогоняет шаги и проверяет, что таблицы
T1–T5 и рисунки F1–F4, F7 созданы, M0 находит все `identity`-запросы и не срабатывает на public_test. То же
в одном процессе делает `tests/test_smoke_pipeline.py`. На GPU-машине первым делом запустите smoke с `--gpu`:
он за несколько минут проверяет весь GPU-путь (скачивание модели, обучение 5 шагов, эмбеддинги, M3/M4,
приватность) до запуска многочасового эксперимента.

## 4. Конфигурация

`configs/default.yaml` — единственный конфиг; для вариантов делайте копию (`configs/<name>.yaml`) или
переопределяйте ключи на лету: `scripts/04_build_indexes.py`/`07_eval.py --set section.key=value` (значение — YAML).
Главные ключи:

- `seed` — все случайные решения детерминированы по нему (`smcode.config.get_rng(cfg, salt)`).
- `languages`, `extract.*` — языки и границы функций (3 ≤ строк ≤ 300, ≤ 2048 токенов); `extract.skip_dirs`
  дополняется жёстко заданным списком `EXTRA_SKIP_DIRS` в `extract.py` (`docs`, `fixtures`, `samples`, `t`, …);
  `extract.domain_path_filter` — выбрасывать из public_* файлы с доменными токенами в пути (`crypto/`, `tls`, `ssh`, …;
  число таких файлов — в `data/functions/extract_stats.json: skipped_domain_path`, сообщить в ограничениях статьи).
- `dedup.*` — near-dup фильтр негативов против protected (k = w = 25, порог 0.3; DESIGN §2.3).
- `transforms.*` — программные преобразования, `partial_lines`, `max_per_split` (5000 функций на набор),
  LLM: `llm`, `llm_model`, `llm_sample_per_split`.
- `fingerprint.*` — k/w, `exact_window`, параметры MinHash, фильтр общности M1 (`common_df`, `common_file_frac`,
  `common_min_files`), `workers`, `hmac_key_env`.
- `semantic.*` — `base_model`, `checkpoint` (дообученный энкодер; `null` = zero-shot), гиперпараметры обучения,
  `ann_top_k`, `hybrid_rule` (`logistic` | `two_threshold`), `hybrid_thresholds`, параметры комбинатора, `own_small_model`.
- `calibration.alphas` (0.01, 0.001), `calibration.domain_calib_fraction`; `eval.bootstrap` (1000),
  `eval.latency_repeats` (200), `eval.latency_threads` ([1, 4]), `eval.length_bins_tokens`.
- `privacy.*` — защиты (`quant_bits`, `noise_sigma`, `noise_mode`), `attack_leaked_pairs`, `bow_vocab`, пороги A1.

`configs/repos.yaml` — списки репозиториев: `protected_pool` (36, домен security/network/crypto; из них seed-ом
выбираются `n_protected` = 12 protected, остальные — hard_neg, стратификация по языку) и `public` (372, другие
области; делятся 0.6/0.2/0.2 на public_train/calib/test по репозиториям). Распределение сохраняется в
`data/functions/repo_splits.json`; изменение seed/долей/repos.yaml делает его «устаревшим», и `00_clone.sh`
останавливается до `--force` (который пересчитывает роли и переклонирует всё) — роли уже склонированных
репозиториев не должны дрейфовать молча. Список public можно расширить через GitHub API:
`python -m smcode.data.list_repos --config configs/default.yaml --extend --dry-run` (единственное сетевое
обращение кроме клонирования и загрузки моделей).

## 5. Шаги конвейера

Полный прогон одной командой (проверяет окружение, ключи, torch/CUDA/vllm, пишет лог в `results/run_all.log`):

```bash
PYTHON=venv/bin/python scripts/run_all.sh --config configs/default.yaml --jobs 4 --workers 8   # всё, включая LLM и дообучение
PYTHON=venv/bin/python scripts/run_all.sh --skip-llm                                          # без vLLM
PYTHON=venv/bin/python scripts/run_all.sh --skip-llm --batch-size 32 --grad-accum 4           # GPU 24 ГБ (эффективный батч 128)
PYTHON=venv/bin/python scripts/run_all.sh --skip-llm --skip-train                             # M3/M4 zero-shot
PYTHON=venv/bin/python scripts/run_all.sh --skip-gpu                                          # только M0–M2 (CPU-машина)
PYTHON=venv/bin/python scripts/run_all.sh --checkpoint runs/unixcoder_ft/best --skip-clone --with-a3   # готовый чекпойнт, A3 включена
PYTHON=venv/bin/python scripts/run_all.sh --skip-llm --ablate-evasion                         # + энкодер без insert_deadcode/reorder_stmts
PYTHON=venv/bin/python scripts/run_all.sh --dry-run                                           # показать команды
```

`run_all.sh` останавливается, если шаг 04 не построил запрошенный индекс (04 возвращает ненулевой код, если любой
запрошенный метод не собран; `--allow-missing-gpu` только для CPU-машины без torch), если после 05 нет чекпойнта
`<paths.runs>/<name>/best`, и если после сборки M3/M4 нет `data/indexes/{semantic,hybrid}/meta.json`.

Ниже — те же шаги по отдельности с ожидаемыми выходами и **ориентировочным** временем для ≈150 репозиториев /
≈300 тыс. функций (public_train ≈ 180 тыс., protected ≈ 30 тыс. + окна) на машине с 8 ядрами и одной GPU класса
A100 (измерено на синтетике и экстраполировано линейно: ×10–20 запас по сравнению с реальным кодом не закладывался,
поэтому первые реальные замеры времени стоит сверить и внести сюда).

| Шаг | Команда | Вход → выход | Время |
|---|---|---|---|
| 00 | `PYTHON=venv/bin/python scripts/00_clone.sh --config configs/default.yaml [--jobs 4] [--max-mb 300]` | `configs/repos.yaml` → `data/raw/<split>/<owner>__<name>/` (shallow, без `.git`), `data/raw/{repo_list,skipped,failed}.txt`, `data/functions/repo_splits.json` | 15–60 мин (сеть) |
| 01 | `venv/bin/python scripts/01_extract.py --config configs/default.yaml --workers 8` | клоны → `data/functions/<split>.jsonl` (FunctionRecord, DESIGN §2.4), `<split>_windows.jsonl` (окна 20/10 строк для `windows.splits`: protected + негативы public_calib/public_test), `extract_stats.json` | 5–15 мин |
| 02 | `venv/bin/python scripts/02_dedup_split.py --config configs/default.yaml --workers 8` | дубликаты по sha внутри сплитов, near-dup фильтр негативов (функций и окон) против protected (на месте) → `data/functions/stats.json` (таблица T1), `dedup_removed.jsonl` | 5–15 мин |
| 03 | `venv/bin/python scripts/03_build_queries.py --config configs/default.yaml` | функции → `data/queries/<set>.jsonl` (QueryRecord, DESIGN §5; наборы protected, hard_neg, public_calib, public_test, protected_windows, public_calib_windows, public_test_windows) + `<set>.stats.json` | 15–30 мин |
| 03 LLM | `venv-llm/bin/python scripts/03_build_queries.py --config configs/default.yaml --llm-only` | добавляет строки `paraphrase`/`translate` в те же файлы + `<set>.llm_stats.json` (§6) | 1–3 ч (GPU) |
| 04 | `venv/bin/python scripts/04_build_indexes.py --config configs/default.yaml --methods exact,winnowing,minhash --with-windows --workers 8` | protected (+ окна) → `data/indexes/<method>/{meta.json,state.pkl,arrays.npz}`, `results/index_stats.json` | 5–15 мин |
| 06 | `venv/bin/python scripts/06_embed.py --config configs/default.yaml --no-latency` | zero-shot эмбеддинги сплитов protected, public_train, protected_windows и всех наборов запросов → `data/embeddings/<split>.npz`, `queries/<set>.npz`, кэш `by_model/<tag>/` | 15–40 мин |
| 05 | `venv/bin/python scripts/05_train_encoder.py --config configs/default.yaml --name unixcoder_ft [--batch_size 32 --grad_accum 4]` | public_train (+ LLM-пары, если есть) → `runs/unixcoder_ft/{best,last}/`, `log.jsonl`, `config.json` | 2–4 ч (2 эпохи) |
| 06 | `venv/bin/python scripts/06_embed.py --config configs/unixcoder_ft.yaml` (конфиг с `semantic.checkpoint: runs/unixcoder_ft/best`) | эмбеддинги дообученной модели заменяют активные файлы; zero-shot остаются в `by_model/microsoft__unixcoder-base/`; + `results/latency_encoder.json` | 15–40 мин + 15–20 мин латентность |
| 04 | `venv/bin/python scripts/04_build_indexes.py --config configs/unixcoder_ft.yaml --methods semantic,hybrid --with-windows` | готовые `protected.npz` + `protected_windows.npz` → `data/indexes/{semantic,hybrid}/` (без кодирования), комбинатор `data/indexes/hybrid/combiner_<tag>.pkl` (обучается на public_train, нужен энкодер) | 5–15 мин |
| 07 | `venv/bin/python scripts/07_eval.py --config configs/unixcoder_ft.yaml --bench 20` | все методы × наборы → `results/scores/<method>/<set>.jsonl` (`qid, score, best_id, latency_ms` + `reason` для M0–M2) + `<set>.meta.json` | 30–90 мин |
| 08 | `venv/bin/python scripts/08_privacy.py --config configs/unixcoder_ft.yaml [--a3]` | эмбеддинги (индекс — функции + окна, как у M3) → `results/privacy.json` (защиты × полезность M3/M4 × атаки A1/A2[/A3]) | 30–90 мин (+1–2 ч с A3) |
| 09 | `venv/bin/python scripts/09_report.py --config configs/unixcoder_ft.yaml` | скоры → `results/summary.json` (+ `summary_schema.md`), `results/tables/T1..T7.md`, `paper/figures/F1..F7.png` | 5–15 мин |

Итого без LLM-преобразований ≈ 6–10 ч GPU-машины; LLM-преобразования добавляют 1–3 ч. Самые тяжёлые шаги —
05 (дообучение) и 07 для M0–M2 (чистый Python на ≈1 млн запросов; при желании поднимите `fingerprint.workers`
только для сборки — оценка однопоточная). Для быстрого контрольного прогона: `transforms.max_per_split: 1000`,
`05 --max_anchors 50000 --epochs 1`, `eval.bootstrap: 200`.

Замечания по шагам:

- **00.** Лимит размера — «полезный» размер (файлы вне `skip_dirs` и ≤ 1 МБ) ≤ `--max-mb` (300) и сырой ≤ `--max-raw-mb`
  (2000); превысившие попадают в `skipped.txt`, недоступные — в `failed.txt`. Пул репозиториев составлен по памяти
  и не проверялся онлайн: переименованные клонируются через редирект, отсутствующие окажутся в `failed.txt` —
  проверьте его и при необходимости поправьте `repos.yaml` (затем `--force`).
- **01.** `--force` для подмножества сплитов удаляет маркер шага 02 — повторите 02. Если нужны точные «сырые»
  счётчики для T1, делайте `01 --force` для всех сплитов сразу. `--no-domain-filter`, `--keep-test-files` — отключение фильтров.
- **03.** К каждой отобранной функции применяется каждое преобразование (`identity, reformat, strip_comments,
  rename_ids, change_literals, insert_deadcode, reorder_stmts, combo`) и `partial` для каждого `L` (3/5/10/20/40
  строк; окна, не разбираемые парсером, сохраняются с `params.parses=false` — сценарий IDE). `reorder_stmts`
  консервативен и применим к малой доле функций — размер его подмножества по языкам есть в `<set>.stats.json`.
- **04.** `--with-windows` добавляет окна protected во все индексы (для набора `protected_windows`). Фильтр
  общности M1 подгоняется по `public_train.jsonl` (если есть). Семантические индексы берут готовые эмбеддинги из
  `data/embeddings/` (`protected.npz` для функций, `protected_windows.npz` для окон; только той модели, что задана в
  конфиге: `semantic.checkpoint`/`base_model`); иначе кодируют сами. Индекс пересобирается, если его входы
  (`data/functions/*.jsonl`, `data/embeddings/*.npz`) новее `meta.json`. `fingerprint.token_mode` (`full` | `lexical`)
  задаёт токены отпечатков M1/M2/M4: на `full` (ID/NUM/STR) они по построению инвариантны к reformat/strip_comments/
  rename_ids/change_literals/combo (детекторы клонов типа 2), `lexical` — классический Moss (вариант `winnowing_lexical`, §8).
- **06.** Идемпотентно по модели и содержимому: активные `.npz` другой модели заменяются, но остаются в `by_model/<tag>/`;
  повторный запуск с прежним чекпойнтом восстанавливает их из кэша без кодирования; если файл запросов пополнился
  (например, LLM-строки после `03 --llm-only`), дозаписываются только недостающие строки. По умолчанию кодируются
  сплиты protected, public_train, protected_windows (их читают 04 и 08) и все наборы запросов; остальные — `--splits`.
  Латентность: CPU 1 и 4 потока, GPU, батчи 1 и 32, `eval.latency_repeats` повторов (`--no-latency` — пропустить,
  достаточно одного замера на архитектуру; `--latency-threads 1` — быстрее).
  Проверьте в логе строку `encoder input format: [CLS] <encoder-only> [SEP] ...` — токенизатор UniXcoder должен
  знать спецтокен `<encoder-only>`; иначе будет предупреждение и обычный вход.
- **05.** Контрастивное дообучение (InfoNCE, якорь — функция public_train, позитив — её преобразование, негативы —
  in-batch + 2 из того же репозитория, in-batch пары с одинаковым sha маскируются; DESIGN §7). **Память:** gradient
  checkpointing включён конфигом (`semantic.grad_checkpointing: true`); батч 128 якорей означает 512 последовательностей
  по 512 токенов на шаг, поэтому на 24 ГБ используйте `--batch_size 32 --grad_accum 4`, на 40–48 ГБ — `--batch_size 64
  --grad_accum 2`, 128 — только на 80 ГБ (`semantic.batch_size` — ещё и батч энкодера в 06/04, поэтому задавайте батч
  обучения флагом, а не в YAML; контраст in-batch — внутри микробатча). Чекпойнт лучшей эпохи выбирается по recall@1
  на отложенных репозиториях public_train (`--val_on holdout`, по умолчанию) или на public_calib (`--val_on public_calib`,
  буквально по DESIGN); recall на public_calib логируется всегда (`calib_recall1`). Smoke: `--max_steps 20 --batch_size 8 --max_anchors 500 --val_pairs 100`.
  Позитивы обучения — `transforms.programmatic` минус `semantic.exclude_transforms` (`--exclude_transforms a,b`), стили
  rename_ids — `semantic.rename_styles`; шаблоны insert_deadcode при обучении («train») отличаются от шаблонов в наборах
  оценки («eval», `transforms.deadcode_templates`). Тем не менее TPR дообученного M3/M4 на программных классах —
  результат «в распределении» (тот же генератор преобразований); честные оценки устойчивости — `semantic_zeroshot`
  и абляция «неизвестный класс» (`--exclude_transforms insert_deadcode,reorder_stmts`, `run_all.sh --ablate-evasion`).
  Абляция «адаптация на protected»: `--adapt_on_protected --name unixcoder_adapt` — функции оцениваемой выборки
  (`data/queries/protected.jsonl`) из якорей исключаются (`--adapt_include_evaluated` возвращает их: это обучение на тесте,
  результат — запоминание, а не обобщение; так и подписывается в T4).
- **07.** Для `semantic`/`hybrid` при наличии `data/embeddings/queries/<set>.npz` используется numpy-путь без
  torch (ANN/переранжирование; латентность батча амортизируется и **не** является временем одного запроса), поэтому
  `--bench N` обязателен для сквозной латентности (батч 1, `eval.latency_repeats` повторов) — вместе с
  `results/latency_encoder.json` она попадает в T5/F7. 07 проверяет, что эмбеддинги запросов получены той же
  моделью, что и индекс (`meta.model` в `.npz`), и останавливается при несовпадении; запросы без эмбеддинга — ошибка
  с указанием на `06 --sets <set>`. Файл скоров пересчитывается, если запросы или индекс новее него.
- **09.** `summary.json` пересчитывается, если появились новые каталоги `results/scores/*`, файлы скоров или запросов
  новее него; `--force` — всегда. `--bootstrap N` для быстрого черновика (`eval.bootstrap` = 1000 для статьи).

## 6. LLM-преобразования (paraphrase, translate)

Нужны `vllm` и GPU (≥ 16 ГБ для 7B в fp16). Модель и выборка — из конфига: `transforms.llm_model`
(`Qwen/Qwen2.5-Coder-7B-Instruct`), `transforms.llm_sample_per_split` (5000), `transforms.llm` (`[paraphrase, translate]`).

```bash
python scripts/03_build_queries.py --config configs/default.yaml            # сначала программные строки (обязательно)
python scripts/03_build_queries.py --config configs/default.yaml --llm-only # затем LLM: paraphrase + translate
python scripts/03_build_queries.py --config configs/default.yaml --llm-only --llm-sample 1000 --sets protected hard_neg
```

Функции для LLM берутся **из выборки программных запросов** того же набора (TPR по преобразованиям сравниваются на
одних функциях); `translate` переводит на язык другого семейства из `languages` (C/C++ считаются одним семейством),
результат принимается, только если разбирается tree-sitter без `ERROR` и не повторяет вход; в `params` хранятся
`prompt_id, model, seed, temperature, src_lang, dst_lang`. Статистика принятых генераций — `<set>.llm_stats.json`
(таблица T7). Поведенческая эквивалентность не проверяется (ограничение статьи).

LLM-позитивы для дообучения (опционально, DESIGN §7): `train.py` берёт их из `data/queries/llm_pairs.jsonl`
(строки `{source_id, code, lang}`) или из LLM-строк `data/queries/public_train.jsonl`, которые можно получить так:
`python scripts/03_build_queries.py --config configs/default.yaml --sets public_train --max-per-split 2000 --llm --llm-sample 2000`
(файл `public_train.jsonl` в `data/queries/` не участвует в оценке). Доля LLM-позитивов — `05 --p_llm 0.5`.

## 7. Вариант «своя модель» (own_small_model)

Лицензионно чистый энкодер с нуля: 6-слойный трансформер на токенах tree-sitter (BPE поверх abstract(indexed) +
lexical), словарь строится по public_train (`smcode/semantic/own_model.py`).

```bash
python scripts/05_train_encoder.py --config configs/default.yaml --own_model --name own_small     # runs/own_small/best
python scripts/06_embed.py --config configs/own_small.yaml --own-model                          # эмбеддинги + латентность
python scripts/04_build_indexes.py --config configs/own_small.yaml --methods semantic,hybrid --with-windows
python scripts/07_eval.py --config configs/own_small.yaml --methods semantic --variant semantic_own --index-dir data/indexes_own/semantic --bench 20
python scripts/07_eval.py --config configs/own_small.yaml --methods hybrid --variant hybrid_own --index-dir data/indexes_own/hybrid --bench 20
```

где `configs/own_small.yaml` = `default.yaml` + `semantic.own_small_model.enabled: true`, `semantic.checkpoint: runs/own_small/best`,
`paths.indexes: data/indexes_own`, `paths.embeddings: data/embeddings_own` (свои каталоги, чтобы не затирать индексы и
эмбеддинги основной модели; скоры попадают в `results/scores/semantic_own/`, `hybrid_own/` и в T4/F5 как варианты).
Чекпойнт своей модели распознаётся по `encoder_meta.json` (`kind: own`), поэтому 04/07 загружают его без флагов.
Гиперпараметры — `semantic.own_small_model.{layers, hidden, heads, vocab}`, `semantic.own_lr`.

## 8. Абляции и варианты (DESIGN §6–§7)

Каталог скоров варианта — `results/scores/<variant>/`; 09 подхватывает все каталоги автоматически (T4, F5).
Один метод на `--variant`.

```bash
# M4: правило двух порогов вместо логистического комбинатора (индекс не пересобирается)
python scripts/07_eval.py --methods hybrid --variant hybrid_rule --set semantic.hybrid_rule=two_threshold --bench 20

# M3 zero-shot против дообученного: индекс из кэша zero-shot эмбеддингов (06 без --checkpoint был выполнен до обучения)
python scripts/04_build_indexes.py --config configs/unixcoder_ft.yaml --methods semantic --with-windows \
    --set semantic.checkpoint=null --set paths.indexes=data/indexes_zeroshot --set paths.embeddings=data/embeddings/by_model/microsoft__unixcoder-base
python scripts/07_eval.py --config configs/unixcoder_ft.yaml --methods semantic --variant semantic_zeroshot \
    --index-dir data/indexes_zeroshot/semantic --set semantic.checkpoint=null \
    --set paths.embeddings=data/embeddings/by_model/microsoft__unixcoder-base --bench 20

# M1 без фильтра общности (цена фильтра: пропуски all_common в T2)
python scripts/04_build_indexes.py --methods winnowing --with-windows --set paths.indexes=data/indexes_nofilter \
    --set fingerprint.common_df=1000000000 --set fingerprint.common_file_frac=1.0
python scripts/07_eval.py --methods winnowing --variant winnowing_nofilter --index-dir data/indexes_nofilter/winnowing --bench 20

# M1 на лексических токенах (классический Moss): базовая линия, которую reformat/rename_ids/change_literals действительно ломают
python scripts/04_build_indexes.py --methods winnowing --with-windows --set paths.indexes=data/indexes_lexical --set fingerprint.token_mode=lexical
python scripts/07_eval.py --methods winnowing --variant winnowing_lexical --index-dir data/indexes_lexical/winnowing \
    --set fingerprint.token_mode=lexical --bench 20

# M3 без классов уклонения в обучении (абляция «неизвестный класс»; run_all.sh --ablate-evasion делает то же)
python scripts/05_train_encoder.py --config configs/default.yaml --name unixcoder_ft_noevasion --exclude_transforms insert_deadcode,reorder_stmts
python scripts/06_embed.py --config configs/default.yaml --checkpoint runs/unixcoder_ft_noevasion/best --no-latency --splits protected protected_windows
python scripts/04_build_indexes.py --config configs/unixcoder_ft.yaml --methods semantic --with-windows \
    --set semantic.checkpoint=runs/unixcoder_ft_noevasion/best --set paths.indexes=data/indexes_noevasion \
    --set paths.embeddings=data/embeddings/by_model/runs__unixcoder_ft_noevasion__best
python scripts/07_eval.py --config configs/unixcoder_ft.yaml --methods semantic --variant semantic_noevasion --index-dir data/indexes_noevasion/semantic \
    --set semantic.checkpoint=runs/unixcoder_ft_noevasion/best --set paths.embeddings=data/embeddings/by_model/runs__unixcoder_ft_noevasion__best --bench 20
python scripts/06_embed.py --config configs/unixcoder_ft.yaml --no-latency      # вернуть активные эмбеддинги основной модели (из кэша)
```

Общее правило для любого другого энкодера (адаптация на protected, другая HF-модель, своя модель): `06 --checkpoint X`
кладёт его эмбеддинги в `data/embeddings/by_model/<tag>/` (`tag` — `smcode.semantic.model.model_tag(X)`, например
`runs__unixcoder_adapt__best`); далее 04 и 07 с `--set semantic.checkpoint=X --set paths.embeddings=data/embeddings/by_model/<tag>`
и своим `paths.indexes` / `--index-dir`, 07 — с `--variant <имя>`. `run_all.sh` выполняет `hybrid_rule` и
`semantic_zeroshot` автоматически, а также CPU-варианты `winnowing_nofilter` и `winnowing_lexical` (`--no-ablations` — отключить;
`--ablate-evasion` добавляет `semantic_noevasion`).

## 9. Результаты и их потребление статьёй (DESIGN §12)

| Раздел статьи | Источник | Файл |
|---|---|---|
| 5.1 Данные | `summary.json: data_stats` (из `data/functions/stats.json`, `queries/*.stats.json`) | `results/tables/T1_data_stats.md` |
| 6.1 Полнота по преобразованиям (RQ1) | `methods[m].tpr[α].conformal.by_transform` | `T2_tpr_by_transform.md`, `figures/F1_tpr_by_transform.png` |
| 6.2 Длина фрагмента (RQ1/RQ3) | `tpr[α].conformal.by_length_bin`, `by_partial_L` | `figures/F2_tpr_vs_length.png` |
| 6.3 Ложные срабатывания и гарантия (RQ2) | `thresholds`, `fpr[α][kind][set]` (conformal / conformal_single / mondrian / oracle / domain; `test_above_alpha` — кластерный z-тест и МДЭ), `auroc`, `roc`; сценарий IDE — `tpr_windows[α]` (порог по public_calib_windows, FPR на public_test_windows) | `T3_fpr_guarantee.md`, `figures/F3_roc_hardneg.png`, `F4_fpr_claimed_vs_empirical.png` |
| 6.4 Гибрид и абляции (RQ3) | каталоги `results/scores/<variant>/` | `T4_ablation.md`, `figures/F5_hybrid_ablation.png` |
| 6.5 Латентность и память (RQ4) | `latency_ms` (+ `results/latency_encoder.json`, `index_stats.json`) | `T5_latency_memory.md`, `figures/F7_latency_vs_index.png` |
| 6.6 Приватность индекса (RQ5) | `results/privacy.json` | `T6_privacy.md`, `figures/F6_privacy_utility.png` |
| 6.7 LLM-преобразования и перевод | `data_stats.queries[set].llm_stats`, `by_transform` | `T7_llm.md` (только если LLM-строки сгенерированы) |

- `results/summary.json` — полная матрица результатов (схема — `results/summary_schema.md`): пороги по α и виду
  калибровки, TPR по преобразованию/языку/бину длины/`L` с кластерным бутстрэпом по функции-источнику,
  FPR по наборам негативов (Клоппер–Пирсон + кластерный ДИ), AUROC, латентность p50/p95, память индекса,
  `warnings` (например, неполные скоры после отладочного `--limit`).
- Таблицы — markdown с подписями на русском, вставляются в `paper/` как есть; рисунки — PNG 300 dpi в
  `paper/figures/` (`cfg.paths.figures`). Решающее правило всюду `score > τ` (строгое; DESIGN §1, §8).
- `results/scores/<method>/<set>.meta.json` — режим замера латентности, размер/память индекса, переопределения
  конфига варианта; `results/index_stats.json` — время сборки/память/диск индексов (варианты из `--set paths.indexes`
  записываются как `<method>@<каталог>`).
- `results/latency.json` — необязательный внешний файл (`{method: {cpu1, cpu4, gpu_b1, gpu_b32, components, by_size}}`);
  если есть, имеет приоритет в T5/F7 (сейчас его никто не создаёт).

## 10. Известные ограничения и что проверить на GPU-машине

- GPU-код (`semantic/model.py`, `own_model.py`, `train.py`, `embed.py` с реальным энкодером, `semantic_index.query`,
  `hybrid.query`, `transforms/llm_paraphrase.py` с vLLM, `privacy/attack_gen.py`, MLP-ветка A1) проверен только
  `py_compile` и ревью — первым запуском должен быть `scripts/smoke_test.sh --gpu`.
- M0 по построению даёт TPR = 0 для частичных извлечений короче `exact_window` (3-строчные `partial`); хэш M0 —
  blake2b-64 (не SHA, как в DESIGN §6). M2 использует numpy-LSH, статистически эквивалентный datasketch, но не
  бит-в-бит.
- M1/M2 (и признаки M4) на `fingerprint.token_mode: full` — детекторы клонов типа 2: reformat, strip_comments,
  rename_ids, change_literals и combo не меняют их отпечатки по построению (overlap = 1.0), поэтому для RQ1
  информативны только insert_deadcode, reorder_stmts, partial, paraphrase, translate; «ломающуюся» базовую линию даёт
  `winnowing_lexical`. Гарантия winnowing («общая подстрока ≥ w+k−1 токенов даёт общий отпечаток») условна: фильтр
  общности может удалить все отпечатки (причина `all_common` в T2), цена фильтра — вариант `winnowing_nofilter`.
- Негативы с тем же нормализованным sha, что у protected (тип-2 клоны любой длины), удаляются при дедупликации
  (`dedup.sha_protected_min_tokens: 0`); число удалённых — `stats.json: removed_sha_protected` (T1). Значение > 0
  оставляет короткие клоны негативами с меткой 0 (`kept_short_sha_protected`), что искажает FPR в бине 0–32.
- Фильтр доменных путей для public_* эвристичен (`extract.DOMAIN_PATH_*`); доля отброшенных файлов — в
  `extract_stats.json`. Заголовки `.h` в C-репозиториях распознаются как C++ только по сильным маркерам.
- Приватность (08): индекс собирается так же, как основной индекс M3 (функции + окна, по `data/indexes/semantic/meta.json`),
  поэтому строка «none» воспроизводит TPR M3 из T2/T4; полезность M4 считается с комбинатором основного индекса
  (`privacy.utility_hybrid`; без него — `null` с причиной в `hybrid_utility.reason`). A1/A2 измеряют утечку только
  идентификаторов, известных по public_train (нижняя оценка): доля вхождений protected-only идентификаторов —
  `a1.protected_coverage`; их покрывает лишь A3.
- Конформная гарантия FPR относится к обменяемым негативам; калибровка объединяет все преобразования одной
  функции (`conformal`), строгий вариант «одна функция — один запрос» — `conformal_single` (маргинальная гарантия),
  мондриановская по ячейкам (преобразование, L) с порогом max_t τ_t — `mondrian` (гарантия по классу; ячейки с
  n < ⌈1/α⌉ − 1 функций получают τ = ∞ и в max не входят), доменная — `domain`; все есть в `summary.json` и T3.
  Вывод о превышении α — кластерно-робастный z-тест и МДЭ (при α = 0,1 % и ~5000 функций FPR 0.001 и 0.003
  неразличимы). Для сценария IDE порог калибруется по окнам public_calib_windows, FPR — public_test_windows;
  TPR identity-окон protected_windows — проверка выравнивания индекса, а не результат детекции.
- Семантическая эквивалентность преобразований (`insert_deadcode`, `reorder_stmts`, `change_literals`, LLM)
  не проверяется.
- `repo_splits.json` строгий: любое изменение `repos.yaml`/seed/долей требует `00_clone.sh --force` (переклонирование).
