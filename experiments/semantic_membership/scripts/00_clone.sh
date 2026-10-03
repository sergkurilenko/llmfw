#!/usr/bin/env bash
# Шаг 00: клонирование репозиториев (DESIGN.md §2.2).
#
# Список "<split> <owner>/<name>" печатает python -m smcode.data.list_repos (он же сохраняет
# распределение в data/functions/repo_splits.json; если seed/доли/repos.yaml изменились с момента
# сохранения, он завершается ошибкой — нужен --force, который пересчитывает распределение и
# переклонирует всё). Клоны: data/raw/<split>/<owner>__<name>, shallow (--depth 1, без
# сабмодулей), 3 попытки, параллельно (xargs -P), идемпотентно (клон с маркером .cloned
# пропускается). После клона .git удаляется.
# Лимит размера: «полезный» размер репозитория — файлы вне extract.skip_dirs и <= 1 МБ, т.е. то,
# что читает извлечение (python -m smcode.data.extract --measure); если он > MAX_MB или сырой
# размер дерева > MAX_RAW_MB, клон удаляется и попадает в data/raw/skipped.txt. Ошибки клонирования
# — data/raw/failed.txt.
#
# Использование:
#   scripts/00_clone.sh --config configs/default.yaml [--jobs 4] [--max-mb 300] [--max-raw-mb 2000] [--force] [--dry-run]
# Переменные окружения: PYTHON (интерпретатор venv), GIT_CLONE_TIMEOUT (сек, по умолчанию 1800),
# SMCODE_MAX_REPO_MB, SMCODE_MAX_RAW_MB.
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="configs/default.yaml"
JOBS=4
MAX_MB=${SMCODE_MAX_REPO_MB:-300}
MAX_RAW_MB=${SMCODE_MAX_RAW_MB:-2000}
FORCE=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --max-mb) MAX_MB="$2"; shift 2 ;;
    --max-raw-mb) MAX_RAW_MB="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

PYTHON=${PYTHON:-python}
RAW_DIR=$("$PYTHON" -c "import sys; from smcode.config import load_config, resolve_path; print(resolve_path(load_config(sys.argv[1]), 'raw_repos'))" "$CONFIG")
mkdir -p "$RAW_DIR"
LIST_FILE="$RAW_DIR/repo_list.txt"
LIST_ARGS="--print-splits"
if [[ "$FORCE" == "1" ]]; then LIST_ARGS="$LIST_ARGS --force"; fi
# shellcheck disable=SC2086
"$PYTHON" -m smcode.data.list_repos --config "$CONFIG" $LIST_ARGS > "$LIST_FILE"
echo "repos: $(wc -l < "$LIST_FILE"), dest: $RAW_DIR, jobs: $JOBS, max_mb: $MAX_MB (effective), max_raw_mb: $MAX_RAW_MB"

if [[ "$DRY_RUN" == "1" ]]; then
  cat "$LIST_FILE"
  exit 0
fi

export RAW_DIR MAX_MB MAX_RAW_MB FORCE CONFIG PYTHON
export GIT_TERMINAL_PROMPT=0
export GIT_CLONE_TIMEOUT=${GIT_CLONE_TIMEOUT:-1800}

clone_one() {
  local split="$1" repo="$2"
  local owner="${repo%%/*}" name="${repo#*/}"
  local dest="$RAW_DIR/$split/${owner}__${name}"
  local url="https://github.com/${repo}.git"
  if [[ -f "$dest/.cloned" && "$FORCE" != "1" ]]; then
    echo "[skip] $split $repo (already cloned)"
    return 0
  fi
  if grep -qx "$split $repo" "$RAW_DIR/skipped.txt" 2>/dev/null && [[ "$FORCE" != "1" ]]; then
    echo "[skip] $split $repo (in skipped.txt)"
    return 0
  fi
  rm -rf "$dest"
  mkdir -p "$(dirname "$dest")"
  local attempt
  for attempt in 1 2 3; do
    if timeout "$GIT_CLONE_TIMEOUT" git clone --quiet --depth 1 --no-tags --single-branch \
         --recurse-submodules=no "$url" "$dest" 2>"$dest.err"; then
      rm -f "$dest.err"
      break
    fi
    echo "[retry $attempt] $split $repo: $(tail -n 1 "$dest.err" 2>/dev/null)"
    rm -rf "$dest"
    if [[ "$attempt" == "3" ]]; then
      echo "[fail] $split $repo"
      echo "$split $repo clone_failed" >> "$RAW_DIR/failed.txt"
      return 0
    fi
    sleep $((attempt * 10))
  done
  rm -rf "$dest/.git"   # история не нужна; экономим место и ускоряем обход
  # размер: raw — всё дерево; eff — файлы вне skip_dirs и <= 1 МБ (то, что читает извлечение)
  local sizes raw_mb eff_mb
  if sizes=$("$PYTHON" -m smcode.data.extract --config "$CONFIG" --measure "$dest" 2>/dev/null); then
    raw_mb=${sizes%% *}
    eff_mb=${sizes##* }
  else
    raw_mb=$(du -sm "$dest" | cut -f1)
    eff_mb=$raw_mb
  fi
  if [[ "$eff_mb" -gt "$MAX_MB" || "$raw_mb" -gt "$MAX_RAW_MB" ]]; then
    echo "[skip-size] $split $repo (effective ${eff_mb} MB > ${MAX_MB} MB or raw ${raw_mb} MB > ${MAX_RAW_MB} MB)"
    rm -rf "$dest"
    echo "$split $repo" >> "$RAW_DIR/skipped.txt"
    return 0
  fi
  touch "$dest/.cloned"
  echo "[ok] $split $repo (raw ${raw_mb} MB, effective ${eff_mb} MB)"
}
export -f clone_one

# xargs передаёт строку "split repo" как два аргумента функции
xargs -P "$JOBS" -L 1 bash -c 'clone_one "$0" "$1"' < "$LIST_FILE"

echo "done. cloned: $(find "$RAW_DIR" -maxdepth 3 -name .cloned | wc -l); skipped: $(wc -l < "$RAW_DIR/skipped.txt" 2>/dev/null || echo 0); failed: $(wc -l < "$RAW_DIR/failed.txt" 2>/dev/null || echo 0)"
