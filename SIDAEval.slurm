#!/bin/bash
#SBATCH --job-name=SIDAEval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=A100,L40S,A40
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# -------- shell hygiene --------
set -euo pipefail
umask 077
mkdir -p logs

# -------- print job header --------
echo "================= SLURM JOB START ================="
echo "Job:    $SLURM_JOB_NAME  (ID: $SLURM_JOB_ID)"
echo "Node:   ${SLURMD_NODENAME:-$(hostname)}"
echo "GPUs:   ${SLURM_GPUS_ON_NODE:-unknown}  (${SLURM_JOB_GPUS:-not-set})"
echo "Start:  $(date)"
echo "==================================================="

# -------- conda / modules --------
source /home/infres/ziyliu-24/miniconda3/etc/profile.d/conda.sh
conda activate fakevlm310
module load cuda/12.4.1 ||  module load cuda/12.1

# -------- reproducibility & logging --------
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

# -------- CUDA / NCCL sanity --------
# Expose only the GPUs SLURM granted this job step
export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
# Reasonable defaults; flip IB if your fabric supports it
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=0
# If your cluster has no InfiniBand, uncomment:
# export NCCL_IB_DISABLE=1
# Optional perf knobs (tune as needed)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MALLOC_TRIM_THRESHOLD_=134217728   # 128MB
NP=${SLURM_GPUS_ON_NODE:-2}

# -------- paths / args (edit me) --------
PROJECT_ROOT="/home/infres/ziyliu-24/FakeParts2/Detectors/SIDA"
cd $PROJECT_ROOT || exit 1
#saberzl/SIDA-7B
#export MODEL_PATH=
#saberzl/SIDA-13B
export MODEL_PATH="/home/infres/ziyliu-24/.cache/huggingface/hub/models--saberzl--SIDA-13B/snapshots/d62e8c3698687389318330fb5b38dad5f32308e0"
export DATA_ROOT=$PROJECT_ROOT"/data"

# -------- quick hardware + torch sanity --------
nvidia-smi -L || true
srun --gres=gpu:"${NP}" \
python - <<'PY'
import os, torch
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.device_count():", torch.cuda.device_count())
assert torch.cuda.device_count() >= 1, "No CUDA devices visible to this job."
import transformers
print(f"transformers.__version__: {transformers.__version__}")
try:
    import flash_attn
    print("flash_attn.__version__:", flash_attn.__version__)
except ImportError:
    print("flash_attn not installed")
PY

# -------- run the actual job --------
#  deepspeed --include localhost: --master_port=24999 ${PROJECT_ROOT}/test.py \
#    --num_classes=2 \
#    --version=${MODEL_PATH} \
#    --dataset_dir=${DATA_BASE_TEST} \
#    --vision_pretrained="${PROJECT_ROOT}/ck/sam_vit_h_4b8939.pth" \
#    --test_dataset=${DATA_BASE_TEST} \
#    --test_only
srun --gres=gpu:"${NP}" \
  deepspeed --include localhost:0 --master_port=24999 test.py \
    --version ${MODEL_PATH} \
    --vision_pretrained ${PROJECT_ROOT}/ck/sam_vit_h_4b8939.pth \
    --dataset_dir ${DATA_ROOT} \
    --binary \
    --real_glob ${DATA_ROOT}"/0_real/**/*.jpg" \
    --fake_glob ${DATA_ROOT}"/1_fake/**/*.jpg" \
    --num_classes 2 \
    --test_only

EXIT_CODE=$?

echo "================== SLURM JOB END =================="
echo "End:   $(date)"
echo "Exit:  ${EXIT_CODE}"
echo "==================================================="
exit "${EXIT_CODE}"