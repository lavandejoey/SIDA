


datetime="$(date '+%Y%m%d_%H%M%S')"
result_dir="results/${datetime}_SIDA-13B_eval"
mkdir -p "${result_dir}"

#data_root="/home/themis/workspace_firas/ff/FakeParts_V2_frames"

data_root="/home/themis/workspace_firas/ff/FakePartsMock"

model_path="./ck/SIDA-13B"
conv="llava_v1" # "llava_v1", "llava_llama_2"


deepspeed --master_port 29501 SIDAEval.py \
  --version "$model_path" \
  --dataset_dir "$data_root" \
  --log_base_dir "$result_dir" \
  --exp_name "SIDA-13B_eval" \
  --conv_type ${conv} \
  --val_batch_size 1
