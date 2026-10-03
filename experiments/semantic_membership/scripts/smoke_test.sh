#!/usr/bin/env bash
# Smoke-тест стенда (DESIGN.md §11): синтетические мини-репозитории → 01 → 02 → 03 → 04 → 07 → 09 на CPU,
# без сети и без torch; проверяет, что таблицы T1–T5 и рисунки созданы.
#
#   scripts/smoke_test.sh [--python PATH] [--workdir DIR] [--keep] [--gpu] [--seed N]
#
# Переменные окружения: PYTHON (интерпретатор; по умолчанию venv/bin/python, если есть, иначе python), SMCODE_INDEX_KEY
# (если не задан, на время теста ставится ключ "smoke-test-key"), TMPDIR (куда класть рабочий каталог).
# --keep     не удалять рабочий каталог (путь печатается в конце).
# --workdir  использовать указанный каталог (подразумевает --keep).
# --gpu      дополнительно прогнать GPU-часть на тех же данных: 05 (несколько шагов), 06, 04 semantic,hybrid,
#            07 для всех методов, 08 (A1 ridge/torch, без A3), 09. Нужны torch/transformers и доступ к HF-модели
#            cfg.semantic.base_model (скачивается при первом запуске). На CPU-машине не проверялось.
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x venv/bin/python ]]; then PYTHON=venv/bin/python; else PYTHON=python; fi
fi
KEEP=0
GPU=0
WORK=""
SEED=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON="$2"; shift 2 ;;
    --workdir) WORK="$2"; KEEP=1; shift 2 ;;
    --keep) KEEP=1; shift ;;
    --gpu) GPU=1; shift ;;
    --seed) SEED="$2"; shift 2 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$WORK" ]]; then
  WORK=$(mktemp -d "${TMPDIR:-/tmp}/smcode_smoke.XXXXXX")
fi
mkdir -p "$WORK"
export SMCODE_INDEX_KEY=${SMCODE_INDEX_KEY:-smoke-test-key}
export SMCODE_PROJECTION_KEY=${SMCODE_PROJECTION_KEY:-$SMCODE_INDEX_KEY}

cleanup() {
  if [[ "$KEEP" == "1" ]]; then
    echo "workdir kept: $WORK"
  else
    rm -rf "$WORK"
  fi
}
trap cleanup EXIT

step() { echo; echo "=== $*"; }
fail() { echo "SMOKE FAILED: $*" >&2; exit 1; }

step "python: $("$PYTHON" -c 'import sys; print(sys.version.split()[0], sys.executable)')"
"$PYTHON" -c "import numpy, scipy, sklearn, yaml, tqdm, matplotlib, tree_sitter_language_pack" \
  || fail "нет CPU-зависимостей (pip install -r requirements.txt)"

step "synthetic repos -> $WORK"
SEED_ARGS=()
if [[ -n "$SEED" ]]; then SEED_ARGS=(--seed "$SEED"); fi
CFG=$("$PYTHON" tests/synthetic_repos.py --out "$WORK" ${SEED_ARGS[@]+"${SEED_ARGS[@]}"})
echo "config: $CFG"
test -d "$WORK/raw/protected" || fail "репозитории не созданы"

step "01_extract"
"$PYTHON" scripts/01_extract.py --config "$CFG" --workers 1
for s in protected hard_neg public_train public_calib public_test; do
  test -s "$WORK/functions/$s.jsonl" || fail "нет functions/$s.jsonl"
done
test -s "$WORK/functions/protected_windows.jsonl" || fail "нет protected_windows.jsonl"

step "02_dedup_split"
"$PYTHON" scripts/02_dedup_split.py --config "$CFG" --workers 1
test -s "$WORK/functions/stats.json" || fail "нет functions/stats.json"

step "03_build_queries"
"$PYTHON" scripts/03_build_queries.py --config "$CFG"
for s in protected hard_neg public_calib public_test protected_windows public_calib_windows public_test_windows; do
  test -s "$WORK/queries/$s.jsonl" || fail "нет queries/$s.jsonl"
done

step "04_build_indexes (exact, winnowing, minhash)"
"$PYTHON" scripts/04_build_indexes.py --config "$CFG" --methods exact,winnowing,minhash --with-windows --workers 1
for m in exact winnowing minhash; do
  test -s "$WORK/indexes/$m/meta.json" || fail "нет indexes/$m/meta.json"
done
test -s "$WORK/results/index_stats.json" || fail "нет results/index_stats.json"

step "07_eval"
"$PYTHON" scripts/07_eval.py --config "$CFG" --methods exact,winnowing,minhash --bench 5
for m in exact winnowing minhash; do
  for s in protected hard_neg public_calib public_test protected_windows public_calib_windows public_test_windows; do
    test -s "$WORK/results/scores/$m/$s.jsonl" || fail "нет results/scores/$m/$s.jsonl"
  done
done

step "04/07: вариант M1 на лексических токенах (winnowing_lexical)"
"$PYTHON" scripts/04_build_indexes.py --config "$CFG" --methods winnowing --with-windows --workers 1 \
  --set "paths.indexes=$WORK/indexes_lexical" --set fingerprint.token_mode=lexical
"$PYTHON" scripts/07_eval.py --config "$CFG" --methods winnowing --variant winnowing_lexical --index-dir "$WORK/indexes_lexical/winnowing" \
  --set fingerprint.token_mode=lexical --bench 5
test -s "$WORK/results/scores/winnowing_lexical/protected.jsonl" || fail "нет results/scores/winnowing_lexical/protected.jsonl"

step "09_report"
"$PYTHON" scripts/09_report.py --config "$CFG"
test -s "$WORK/results/summary.json" || fail "нет results/summary.json"
for t in T1_data_stats T2_tpr_by_transform T3_fpr_guarantee T4_ablation T5_latency_memory; do
  test -s "$WORK/results/tables/$t.md" || fail "нет results/tables/$t.md"
done
for f in F1_tpr_by_transform F2_tpr_vs_length F3_roc_hardneg F4_fpr_claimed_vs_empirical F7_latency_vs_index; do
  test -s "$WORK/figures/$f.png" || fail "нет figures/$f.png"
done

step "sanity: summary.json"
"$PYTHON" - "$WORK/results/summary.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
ms = s["methods"]
assert {"exact", "winnowing", "minhash"} <= set(ms), sorted(ms)
ak = s["alpha_keys"][0]
ident = ms["exact"]["tpr"][ak]["conformal"]["by_transform"]["identity"]["value"]
assert ident == 1.0, f"exact identity TPR = {ident}"
fpr = ms["exact"]["fpr"][ak]["conformal"]["public_test"]["value"]
assert fpr <= 0.1, f"exact FPR public_test = {fpr}"
assert "winnowing_lexical" in ms, sorted(ms)
w, wl = ms["winnowing"]["tpr"][ak]["conformal"]["by_transform"], ms["winnowing_lexical"]["tpr"][ak]["conformal"]["by_transform"]
assert w["rename_ids"]["value"] >= wl["rename_ids"]["value"], (w["rename_ids"], wl["rename_ids"])  # abstract-токены инвариантны к переименованию
win = ms["winnowing"]["tpr_windows"][ak]
assert win["calibration"] == "public_calib_windows" and "public_test_windows" in win["fpr"], win.get("calibration")
assert ms["winnowing"]["thresholds"][ak]["mondrian"] is not None and ms["winnowing"]["fpr"][ak]["conformal"]["public_test"]["test_above_alpha"]
print("ok: exact identity TPR = 1.0, FPR public_test =", fpr, "; winnowing TPR =",
      round(ms["winnowing"]["tpr"][ak]["conformal"]["overall"]["value"], 3), "; lexical rename_ids TPR =", round(wl["rename_ids"]["value"], 3),
      "; windows FPR public_test_windows =", win["fpr"]["public_test_windows"]["value"])
EOF

if [[ "$GPU" == "1" ]]; then
  step "GPU part (optional): 05 -> 06 -> 04 semantic,hybrid -> 07 -> 08 -> 09"
  "$PYTHON" -c "import torch, transformers" || fail "нет torch/transformers (pip install -r requirements-gpu.txt)"
  "$PYTHON" scripts/05_train_encoder.py --config "$CFG" --name smoke --max_steps 5 --batch_size 4 --max_anchors 40 --val_pairs 16 --num_workers 0 --force
  # дообученный чекпойнт должен быть в конфиге шагов 04/07 (они берут модель из cfg.semantic.checkpoint)
  CFG_FT="$WORK/smoke_ft.yaml"
  "$PYTHON" - "$CFG" "$CFG_FT" "$WORK/runs/smoke/best" <<'EOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
cfg.setdefault("semantic", {})["checkpoint"] = sys.argv[3]
yaml.safe_dump(cfg, open(sys.argv[2], "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
EOF
  "$PYTHON" scripts/06_embed.py --config "$CFG_FT" --batch-size 8 --latency-threads 1
  "$PYTHON" scripts/04_build_indexes.py --config "$CFG_FT" --methods semantic,hybrid --with-windows --workers 1
  "$PYTHON" scripts/07_eval.py --config "$CFG_FT" --bench 5 --force
  "$PYTHON" scripts/07_eval.py --config "$CFG_FT" --methods hybrid --variant hybrid_rule --set semantic.hybrid_rule=two_threshold --bench 5
  "$PYTHON" scripts/08_privacy.py --config "$CFG_FT" --leaked-pairs 0 20 --force
  "$PYTHON" scripts/09_report.py --config "$CFG_FT" --force
  for t in T4_ablation T6_privacy; do test -s "$WORK/results/tables/$t.md" || fail "нет results/tables/$t.md"; done
  for f in F5_hybrid_ablation F6_privacy_utility; do test -s "$WORK/figures/$f.png" || fail "нет figures/$f.png"; done
fi

echo
echo "SMOKE OK ($WORK)"
