#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status.
set -euo pipefail
# Enable debugging output
#set -x

datetime="$(date '+%Y%m%d_%H%M%S')"
result_dir="results/${datetime}_SIDA-13B_eval"
mkdir -p "${result_dir}"
#data_root="/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only"
data_root="/home/infres/ziyliu-24/data/FakeParts2DataMockBin"
#data_root="/home/infres/ziyliu-24/data/FakeParts2DataMock"
#model_path="/home/infres/ziyliu-24/.cache/huggingface/hub/models--saberzl--SIDA-13B/snapshots/d62e8c3698687389318330fb5b38dad5f32308e0"
model_path="./ck/SIDA-13B"
conv="llava_v1" # "llava_v1", "llava_llama_2"

source /home/infres/ziyliu-24/miniconda3/etc/profile.d/conda.sh
conda activate sida311

deepspeed --master_port 29501 SIDAEval.py \
  --version "$model_path" \
  --dataset_dir "$data_root" \
  --log_base_dir "$result_dir" \
  --exp_name "SIDA-13B_eval" \
  --conv_type ${conv}