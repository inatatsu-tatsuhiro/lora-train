#!/usr/bin/env bash
# Training entrypoint for the SDXL LoRA pilot on SaladCloud / Kelpie.
# Training arguments mirror the Mac run (~/lora_train/train_mira.sh) exactly,
# with two cloud-only additions: --save_state and automatic --resume.
#
# FIX (2026-11-18, Noir failure): the previous version hardcoded Mira's names
# (10_mira / mira-dataset-r2.zip) in the unpack step and in the preconditions.
# Noir's dataset is 10_noir / noir-dataset-r2.zip, so the script silently skipped
# the unpack, downloaded the 6.6 GB base model, then died on
# `test -d "$DATA/10_mira"` with exit 1 and no error message (set -e).
# Kelpie job 3da1cd65 (noir-pilot-001) failed exactly that way at 02:08 JST,
# then never got reassigned for ~7h (~$3.17 wasted).
# This version detects whichever *-dataset-r2.zip / 10_* folder Kelpie synced,
# so Mira, Noir and future characters all use the same image, and it fails
# LOUDLY (explicit ERROR line) if no dataset folder is found.
set -euo pipefail

MODELS=/work/input/models
DATA=/work/input/dataset
OUT=/work/output
MODEL=$MODELS/animagine-xl-4.0-opt.safetensors
MODEL_URL=https://huggingface.co/cagliostrolab/animagine-xl-4.0/resolve/main/animagine-xl-4.0-opt.safetensors
mkdir -p "$MODELS" "$DATA" "$OUT"

# --- dataset: Kelpie sync delivers <name>-dataset-r2.zip; unpack it if needed ---
# Generic: unzip whichever *-dataset-r2.zip was synced, then use whichever
# 10_* folder it contains (10_mira, 10_noir, ...). No character name is hardcoded.
if ! ls -d "$DATA"/10_*/ >/dev/null 2>&1; then
  for zip in "$DATA"/*-dataset-r2.zip; do
    [ -f "$zip" ] || continue
    echo "[entry] unpacking dataset zip: $zip"
    unzip -q -o "$zip" -d "$DATA"
  done
fi
# Directories only (trailing slash); exactly one dataset folder is expected —
# sd-scripts trains on every 10_* folder under --train_data_dir, so more than
# one would silently mix characters (Claude review 2026-11-18).
DATASET_DIRS=$(ls -d "$DATA"/10_*/ 2>/dev/null || true)
DATASET_COUNT=$(printf '%s' "$DATASET_DIRS" | grep -c . || true)
if [ "$DATASET_COUNT" -eq 0 ]; then
  echo "[entry] ERROR: no 10_* dataset folder found in $DATA (contents: $(ls "$DATA" | tr '\n' ' '))" >&2
  exit 1
fi
if [ "$DATASET_COUNT" -gt 1 ]; then
  echo "[entry] ERROR: multiple 10_* dataset folders found, refusing to mix characters: $(printf '%s' "$DATASET_DIRS" | tr '\n' ' ')" >&2
  exit 1
fi
DATASET_DIR=${DATASET_DIRS%/}
echo "[entry] dataset folder: $DATASET_DIR"
# Output name: the job's OUTPUT_NAME wins; otherwise derive it from the
# dataset folder (10_noir -> noir_lora) so artifacts are never mislabeled
# as another character's (Claude review 2026-11-18).
NAME=${OUTPUT_NAME:-$(basename "$DATASET_DIR" | sed 's/^10_//')_lora}
echo "[entry] output name: $NAME"

# --- base model: prefer the R2-synced copy, otherwise download from HuggingFace ---
if [ ! -f "$MODEL" ]; then
  echo "[entry] downloading base model from HuggingFace"
  curl -L --retry 5 --retry-delay 10 -o "$MODEL.tmp" "$MODEL_URL"
  mv "$MODEL.tmp" "$MODEL"
fi

# --- preconditions (fail fast and LOUD, so Kelpie retries on a fresh node) ---
if [ ! -f "$MODEL" ]; then
  echo "[entry] ERROR: base model missing: $MODEL" >&2
  exit 1
fi
if [ ! -d "$DATASET_DIR" ]; then
  echo "[entry] ERROR: dataset folder missing: $DATASET_DIR" >&2
  exit 1
fi
ls "$DATASET_DIR"/*.txt >/dev/null
echo "[entry] model: $(du -h "$MODEL" | cut -f1), dataset files: $(ls "$DATASET_DIR" | wc -l)"

# --- resume: if a previous run left a state folder, continue from the newest ---
RESUME_ARGS=()
LATEST_STATE=$(ls -d "$OUT"/*-state 2>/dev/null | sort | tail -n 1 || true)
if [ -n "${LATEST_STATE:-}" ]; then
  RESUME_ARGS=(--resume "$LATEST_STATE")
  echo "[entry] resume from: $LATEST_STATE"
fi

cd /app/sd-scripts
python sdxl_train_network.py \
  --pretrained_model_name_or_path="$MODEL" \
  --train_data_dir="$DATA" \
  --output_dir="$OUT" \
  --output_name="$NAME" \
  --network_module=networks.lora \
  --network_dim=32 --network_alpha=16 \
  --resolution=1024,1024 --enable_bucket --min_bucket_reso=512 --max_bucket_reso=1024 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 --lr_scheduler=cosine --optimizer_type=adamw \
  --max_train_epochs=12 --save_every_n_epochs=3 \
  --mixed_precision=bf16 --cache_latents --no_half_vae \
  --seed=1234 --gradient_checkpointing --sdpa \
  --save_model_as=safetensors \
  --save_state \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee -a "$OUT/train.log"
