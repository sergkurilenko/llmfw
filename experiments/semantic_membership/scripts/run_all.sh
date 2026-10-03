#!/usr/bin/env bash
# Полный конвейер стенда «семантическое членство кода» (DESIGN.md §11):
#   00 clone → 01 extract → 02 dedup → 03 queries [+ LLM] → 04 индексы отпечатков
#   → [GPU: 06 zero-shot эмбеддинги → 05 дообучение → 06 эмбеддинги чекпойнта → 04 semantic,hybrid → 08 приватность]
#   → 07 оценка (+ абляции hybrid_rule, semantic_zeroshot) → 09 отчёт.
#
#   scripts/run_all.sh [--config configs/default.yaml] [--python PATH] [--jobs 4] [--workers 8]
#                      [--skip-clone] [--skip-llm] [--skip-train] [--skip-gpu] [--skip-privacy] [--no-ablations] [--ablate-evasion]
#                      [--checkpoint DIR] [--run-name NAME] [--epochs N] [--batch-size N] [--grad-accum N]
#                      [--with-a3] [--allow-unkeyed] [--force] [--dry-run]
#
# Переменные окружения: PYTHON (интерпретатор; по умолчанию venv/bin/python, если есть, иначе python),
# SMCODE_INDEX_KEY (обязателен: HMAC-ключ отпечатков), SMCODE_PROJECTION_KEY (ключ проекции §9; иначе берётся
# SMCODE_INDEX_KEY), HF_HOME/HF_TOKEN (кэш/доступ HF).
# Все шаги идемпотентны: повторный запуск продолжает с первого незавершённого артефакта; --force передаётся всем шагам.
# --skip-llm      не строить LLM-преобразования (paraphrase/translate; нужен vllm и GPU).
# --skip-train    не дообучать энкодер: семантические методы считаются zero-shot (cfg.semantic.base_model), если
#                 не задан --checkpoint DIR (готовый <paths.runs>/<name>/best).
# --skip-gpu      только отпечатки M0–M2 (как на CPU-машине без torch).
# --ablate-evasion дополнительно дообучить энкодер без insert_deadcode/reorder_stmts в позитивах (абляция «неизвестный
#                 класс», вариант semantic_noevasion; ещё 2–4 ч GPU).
# --batch-size/--grad-accum  батч шага 05 (24 ГБ: 32/4, 40–48 ГБ: 64/2, 80 ГБ: 128/1; gradient checkpointing включён конфигом).
# После дообучения шаги 06/04/07/08/09 выполняются с производным конфигом <results>/run_all_config.yaml,
# в котором semantic.checkpoint указывает на чекпойнт (04/07 берут модель из конфига, DESIGN §7).
# Абляции CPU (winnowing_nofilter, winnowing_lexical) и GPU (hybrid_rule, semantic_zeroshot) выполняются при --no-ablations=выкл.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
CONFIG="configs/default.yaml"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x venv/bin/python ]]; then PYTHON=venv/bin/python; else PYTHON=python; fi
fi
JOBS=4
WORKERS=8
SKIP_CLONE=0; SKIP_LLM=0; SKIP_TRAIN=0; SKIP_GPU=0; SKIP_PRIVACY=0; ABLATIONS=1; ABLATE_EVASION=0
CHECKPOINT=""; RUN_NAME="unixcoder_ft"; EPOCHS=""; BATCH_SIZE=""; GRAD_ACCUM=""; WITH_A3=0; ALLOW_UNKEYED=0; FORCE=""; DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --skip-clone) SKIP_CLONE=1; shift ;;
    --skip-llm) SKIP_LLM=1; shift ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    --skip-gpu) SKIP_GPU=1; shift ;;
    --skip-privacy) SKIP_PRIVACY=1; shift ;;
    --no-ablations) ABLATIONS=0; shift ;;
    --ablate-evasion) ABLATE_EVASION=1; shift ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size|--batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --grad-accum|--grad_accum) GRAD_ACCUM="$2"; shift 2 ;;
    --with-a3) WITH_A3=1; shift ;;
    --allow-unkeyed) ALLOW_UNKEYED=1; shift ;;
    --force) FORCE="--force"; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# ----------------------------------------------------------------------------- проверки окружения
say() { echo; echo "### $*"; }
die() { echo "run_all: $*" >&2; exit 1; }
run() {
  echo
  echo ">>> $*"
  if [[ "$DRY" == "1" ]]; then return 0; fi
  "$@"
}

say "окружение"
command -v "$PYTHON" >/dev/null || die "интерпретатор не найден: $PYTHON (задайте PYTHON=/path/to/venv/bin/python)"
"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), sys.version; print("python", sys.version.split()[0], sys.executable)' \
  || die "нужен Python >= 3.10"
"$PYTHON" -c "import numpy, scipy, sklearn, yaml, tqdm, matplotlib, tree_sitter, tree_sitter_language_pack" \
  || die "нет CPU-зависимостей: pip install -r requirements.txt"
test -f "$CONFIG" || die "конфиг не найден: $CONFIG"

HAS_TORCH=$("$PYTHON" -c "import importlib.util as u; print(int(u.find_spec('torch') is not None and u.find_spec('transformers') is not None))")
HAS_CUDA=0
if [[ "$HAS_TORCH" == "1" ]]; then
  HAS_CUDA=$("$PYTHON" -c "import torch; print(int(torch.cuda.is_available()))" 2>/dev/null || echo 0)
fi
HAS_VLLM=$("$PYTHON" -c "import importlib.util as u; print(int(u.find_spec('vllm') is not None))")
echo "torch/transformers: $HAS_TORCH; CUDA: $HAS_CUDA; vllm: $HAS_VLLM"
if [[ "$HAS_TORCH" == "1" && "$HAS_CUDA" == "1" ]]; then
  "$PYTHON" -c "import torch; p=torch.cuda.get_device_properties(0); print('GPU:', p.name, round(p.total_memory/2**30, 1), 'GiB')"
fi
if [[ "$HAS_TORCH" == "0" && "$SKIP_GPU" == "0" ]]; then
  echo "ПРЕДУПРЕЖДЕНИЕ: torch/transformers не установлены (requirements-gpu.txt) — GPU-часть (M3/M4, приватность) пропускается"
  SKIP_GPU=1
fi
if [[ "$SKIP_GPU" == "0" && "$HAS_CUDA" == "0" ]]; then
  echo "ПРЕДУПРЕЖДЕНИЕ: CUDA недоступна — дообучение и эмбеддинги пойдут на CPU (очень медленно)"
fi
if [[ "$SKIP_LLM" == "0" && ( "$HAS_VLLM" == "0" || "$HAS_CUDA" == "0" ) ]]; then
  echo "ПРЕДУПРЕЖДЕНИЕ: vllm/CUDA недоступны — LLM-преобразования пропускаются (--skip-llm)"
  SKIP_LLM=1
fi
if [[ -z "${SMCODE_INDEX_KEY:-}" ]]; then
  if [[ "$ALLOW_UNKEYED" == "1" ]]; then
    echo "ПРЕДУПРЕЖДЕНИЕ: SMCODE_INDEX_KEY не задан — отпечатки без HMAC-ключа (только для отладки)"
  else
    die "задайте SMCODE_INDEX_KEY (HMAC-ключ отпечатков, DESIGN §6): export SMCODE_INDEX_KEY=\$(openssl rand -hex 32); или --allow-unkeyed"
  fi
fi
if [[ -z "${SMCODE_PROJECTION_KEY:-}" ]]; then
  echo "SMCODE_PROJECTION_KEY не задан — ключ проекции (§9) берётся из SMCODE_INDEX_KEY"
fi
if [[ "$SKIP_CLONE" == "0" ]]; then
  command -v git >/dev/null || die "нужен git для шага 00 (или --skip-clone при готовых клонах)"
fi
export TOKENIZERS_PARALLELISM=false

# каталоги из конфига (результаты, индексы, эмбеддинги, чекпойнты) и тег базовой модели
PATHS=$("$PYTHON" - "$CONFIG" <<'EOF'
import sys
from smcode.config import load_config, resolve_path
from smcode.semantic.model import model_tag
cfg = load_config(sys.argv[1])
print(resolve_path(cfg, "results"), resolve_path(cfg, "indexes"), resolve_path(cfg, "embeddings"), resolve_path(cfg, "runs"),
      model_tag(cfg.get("semantic", {}).get("base_model", "microsoft/unixcoder-base")))
EOF
) || die "не удалось прочитать пути из конфига $CONFIG (см. ошибку выше)"
read -r RESULTS_DIR INDEXES_DIR EMB_DIR RUNS_DIR BASE_TAG <<< "$PATHS"
[[ -n "$RESULTS_DIR" && -n "$RUNS_DIR" ]] || die "пустые пути из конфига: '$PATHS'"
echo "results: $RESULTS_DIR"
echo "runs: $RUNS_DIR"
echo "свободно на диске: $(df -h "$ROOT" | awk 'NR==2{print $4}')"
LOG="$RESULTS_DIR/run_all.log"
if [[ "$DRY" == "0" ]]; then
  mkdir -p "$RESULTS_DIR"
  exec > >(tee -a "$LOG") 2>&1
  echo "лог: $LOG ($(date))"
fi

# ----------------------------------------------------------------------------- CPU-часть
say "00 clone"
if [[ "$SKIP_CLONE" == "1" ]]; then
  echo "(пропуск: --skip-clone)"
else
  run env PYTHON="$PYTHON" bash scripts/00_clone.sh --config "$CONFIG" --jobs "$JOBS" $FORCE
fi

say "01 extract"
run "$PYTHON" scripts/01_extract.py --config "$CONFIG" --workers "$WORKERS" $FORCE

say "02 dedup"
run "$PYTHON" scripts/02_dedup_split.py --config "$CONFIG" --workers "$WORKERS" $FORCE

say "03 queries"
run "$PYTHON" scripts/03_build_queries.py --config "$CONFIG" $FORCE
if [[ "$SKIP_LLM" == "0" ]]; then
  say "03 queries: LLM (paraphrase/translate, vllm, cfg.transforms.llm_model)"
  run "$PYTHON" scripts/03_build_queries.py --config "$CONFIG" --llm-only $FORCE
fi

say "04 indexes: exact, winnowing, minhash (+ окна protected)"
run "$PYTHON" scripts/04_build_indexes.py --config "$CONFIG" --methods exact,winnowing,minhash --with-windows --workers "$WORKERS" $FORCE

# ----------------------------------------------------------------------------- GPU-часть
CFG_RUN="$CONFIG"
TRAINED=0
TRAIN_ARGS=()
if [[ -n "$EPOCHS" ]]; then TRAIN_ARGS+=(--epochs "$EPOCHS"); fi
if [[ -n "$BATCH_SIZE" ]]; then TRAIN_ARGS+=(--batch_size "$BATCH_SIZE"); fi
if [[ -n "$GRAD_ACCUM" ]]; then TRAIN_ARGS+=(--grad_accum "$GRAD_ACCUM"); fi
if [[ "$SKIP_GPU" == "0" ]]; then
  if [[ -n "$CHECKPOINT" ]]; then
    test -d "$CHECKPOINT" || die "чекпойнт не найден: $CHECKPOINT"
    TRAINED=1
  elif [[ "$SKIP_TRAIN" == "0" ]]; then
    say "06 embed: zero-shot (cfg.semantic.base_model) — базовая линия и кэш by_model/$BASE_TAG (латентность — у дообученной модели той же архитектуры)"
    run "$PYTHON" scripts/06_embed.py --config "$CONFIG" --no-latency $FORCE
    say "05 train: контрастивное дообучение энкодера → $RUNS_DIR/$RUN_NAME/best"
    run "$PYTHON" scripts/05_train_encoder.py --config "$CONFIG" --name "$RUN_NAME" ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"} $FORCE
    CHECKPOINT="$RUNS_DIR/$RUN_NAME/best"
    if [[ "$DRY" == "0" ]]; then
      test -d "$CHECKPOINT" || die "чекпойнт не создан: $CHECKPOINT (см. лог шага 05)"
    fi
    TRAINED=1
  fi
  if [[ "$TRAINED" == "1" ]]; then
    CFG_RUN="$RESULTS_DIR/run_all_config.yaml"
    say "производный конфиг с semantic.checkpoint=$CHECKPOINT → $CFG_RUN"
    if [[ "$DRY" == "0" ]]; then
      "$PYTHON" - "$CONFIG" "$CFG_RUN" "$CHECKPOINT" <<'EOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
cfg.setdefault("semantic", {})["checkpoint"] = sys.argv[3]
yaml.safe_dump(cfg, open(sys.argv[2], "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
EOF
    fi
  fi

  say "06 embed: эмбеддинги функций и запросов + латентность энкодера (модель из $CFG_RUN)"
  run "$PYTHON" scripts/06_embed.py --config "$CFG_RUN" $FORCE

  say "04 indexes: semantic, hybrid (комбинатор обучается на public_train при первой сборке; окна берутся из protected_windows.npz)"
  run "$PYTHON" scripts/04_build_indexes.py --config "$CFG_RUN" --methods semantic,hybrid --with-windows --workers "$WORKERS" $FORCE
  if [[ "$DRY" == "0" ]]; then
    for m in semantic hybrid; do
      test -f "$INDEXES_DIR/$m/meta.json" || die "индекс $m не построен (см. лог шага 04)"
    done
  fi

  if [[ "$SKIP_PRIVACY" == "0" ]]; then
    say "08 privacy: защиты × атаки A1/A2$( [[ "$WITH_A3" == "1" ]] && echo "/A3" ) (индекс — как у M3: функции + окна; полезность M3 и M4)"
    A3=()
    if [[ "$WITH_A3" == "1" ]]; then A3=(--a3); fi
    run "$PYTHON" scripts/08_privacy.py --config "$CFG_RUN" ${A3[@]+"${A3[@]}"} $FORCE
  fi
fi

# ----------------------------------------------------------------------------- оценка и отчёт
say "07 eval: все методы с готовым индексом, --bench 20 (латентность батча 1)"
run "$PYTHON" scripts/07_eval.py --config "$CFG_RUN" --bench 20 $FORCE

if [[ "$ABLATIONS" == "1" ]]; then
  say "абляция: M1 без фильтра общности → results/scores/winnowing_nofilter/"
  NF_INDEXES="${INDEXES_DIR}_nofilter"
  run "$PYTHON" scripts/04_build_indexes.py --config "$CFG_RUN" --methods winnowing --with-windows --workers "$WORKERS" \
      --set "paths.indexes=$NF_INDEXES" --set fingerprint.common_df=1000000000 --set fingerprint.common_file_frac=1.0 $FORCE
  run "$PYTHON" scripts/07_eval.py --config "$CFG_RUN" --methods winnowing --variant winnowing_nofilter --index-dir "$NF_INDEXES/winnowing" \
      --set fingerprint.common_df=1000000000 --set fingerprint.common_file_frac=1.0 --bench 20 $FORCE
  say "абляция: M1 на лексических токенах (классический Moss; T1–T4/combo его ломают) → results/scores/winnowing_lexical/"
  LX_INDEXES="${INDEXES_DIR}_lexical"
  run "$PYTHON" scripts/04_build_indexes.py --config "$CFG_RUN" --methods winnowing --with-windows --workers "$WORKERS" \
      --set "paths.indexes=$LX_INDEXES" --set fingerprint.token_mode=lexical $FORCE
  run "$PYTHON" scripts/07_eval.py --config "$CFG_RUN" --methods winnowing --variant winnowing_lexical --index-dir "$LX_INDEXES/winnowing" \
      --set fingerprint.token_mode=lexical --bench 20 $FORCE
fi

if [[ "$ABLATIONS" == "1" && "$SKIP_GPU" == "0" ]]; then
  if [[ -f "$INDEXES_DIR/hybrid/meta.json" || "$DRY" == "1" ]]; then
    say "абляция: hybrid с правилом двух порогов → results/scores/hybrid_rule/"
    run "$PYTHON" scripts/07_eval.py --config "$CFG_RUN" --methods hybrid --variant hybrid_rule --set semantic.hybrid_rule=two_threshold --bench 20 $FORCE
  fi
  if [[ "$TRAINED" == "1" ]]; then
    ZS_INDEXES="${INDEXES_DIR}_zeroshot"
    ZS_EMB="$EMB_DIR/by_model/$BASE_TAG"
    if [[ -f "$ZS_EMB/protected.npz" && -f "$ZS_EMB/queries/protected.npz" || "$DRY" == "1" ]]; then
      say "абляция: zero-shot энкодер (индекс $ZS_INDEXES/semantic, эмбеддинги $ZS_EMB) → results/scores/semantic_zeroshot/"
      run "$PYTHON" scripts/04_build_indexes.py --config "$CFG_RUN" --methods semantic --with-windows \
          --set semantic.checkpoint=null --set "paths.indexes=$ZS_INDEXES" --set "paths.embeddings=$ZS_EMB" $FORCE
      run "$PYTHON" scripts/07_eval.py --config "$CFG_RUN" --methods semantic --variant semantic_zeroshot --index-dir "$ZS_INDEXES/semantic" \
          --set semantic.checkpoint=null --set "paths.embeddings=$ZS_EMB" --bench 20 $FORCE
    else
      echo "(zero-shot эмбеддинги в $ZS_EMB не найдены — абляция semantic_zeroshot пропущена; запустите 06 без --checkpoint до обучения)"
    fi
  fi
  if [[ "$ABLATE_EVASION" == "1" && "$TRAINED" == "1" && "$SKIP_TRAIN" == "0" ]]; then
    NE_NAME="${RUN_NAME}_noevasion"
    NE_CKPT="$RUNS_DIR/$NE_NAME/best"
    say "абляция «неизвестный класс»: дообучение без insert_deadcode/reorder_stmts → $NE_CKPT, вариант semantic_noevasion"
    run "$PYTHON" scripts/05_train_encoder.py --config "$CONFIG" --name "$NE_NAME" --exclude_transforms insert_deadcode,reorder_stmts \
        ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"} $FORCE
    NE_TAG=$("$PYTHON" -c "import sys; from smcode.semantic.model import model_tag; print(model_tag(sys.argv[1]))" "$NE_CKPT")
    NE_EMB="$EMB_DIR/by_model/$NE_TAG"
    NE_INDEXES="${INDEXES_DIR}_noevasion"
    # 06 с чужим чекпойнтом делает его эмбеддинги активными (основные остаются в кэше by_model) — после абляции активные восстанавливаются
    run "$PYTHON" scripts/06_embed.py --config "$CONFIG" --checkpoint "$NE_CKPT" --no-latency --splits protected protected_windows $FORCE
    run "$PYTHON" scripts/04_build_indexes.py --config "$CFG_RUN" --methods semantic --with-windows \
        --set "semantic.checkpoint=$NE_CKPT" --set "paths.indexes=$NE_INDEXES" --set "paths.embeddings=$NE_EMB" $FORCE
    run "$PYTHON" scripts/07_eval.py --config "$CFG_RUN" --methods semantic --variant semantic_noevasion --index-dir "$NE_INDEXES/semantic" \
        --set "semantic.checkpoint=$NE_CKPT" --set "paths.embeddings=$NE_EMB" --bench 20 $FORCE
    say "восстановление активных эмбеддингов основной модели (из кэша by_model, без кодирования)"
    run "$PYTHON" scripts/06_embed.py --config "$CFG_RUN" --no-latency
  fi
fi

say "09 report: summary.json, таблицы T1–T7, рисунки F1–F7"
run "$PYTHON" scripts/09_report.py --config "$CFG_RUN" $FORCE

say "готово"
echo "results: $RESULTS_DIR (summary.json, tables/, scores/, privacy.json, latency_encoder.json); рисунки — cfg.paths.figures"
