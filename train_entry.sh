#!/usr/bin/env bash
# Training entrypoint for the Mira SDXL LoRA pilot on SaladCloud / Kelpie.
# Training arguments mirror the Mac run (~/lora_train/train_mira.sh) exactly,
# with two cloud-only additions: --save_state and automatic --resume.
set -euo pipefail

MODELS=/work/input/models
DATA=/work/input/dataset
OUT=/work/output
NAME=${OUTPUT_NAME:-mira_lora}
MODEL=$MODELS/animagine-xl-4.0-opt.safetensors
MODEL_URL=https://huggingface.co/cagliostrolab/animagine-xl-4.0/resolve/main/animagine-xl-4.0-opt.safetensors
mkdir -p "$MODELS" "$DATA" "$OUT"

# --- dataset: Kelpie sync delivers mira-dataset-r2.zip; unpack it if needed ---
if [ ! -d "$DATA/10_mira" ]; then
  if [ -f "$DATA/mira-dataset-r2.zip" ]; then
    echo "[entry] unpacking dataset zip"
    unzip -q -o "$DATA/mira-dataset-r2.zip" -d "$DATA"
  fi
fi

# --- base model: prefer the R2-synced copy, otherwise download from HuggingFace ---
if [ ! -f "$MODEL" ]; then
  echo "[entry] downloading base model from HuggingFace"
  curl -L --retry 5 --retry-delay 10 -o "$MODEL.tmp" "$MODEL_URL"
  mv "$MODEL.tmp" "$MODEL"
fi

# --- preconditions (fail fast so Kelpie retries on a fresh node) ---
test -f "$MODEL"
test -d "$DATA/10_mira"
ls "$DATA/10_mira"/*.txt >/dev/null
echo "[entry] model: $(du -h "$MODEL" | cut -f1), dataset files: $(ls "$DATA/10_mira" | wc -l)"

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
