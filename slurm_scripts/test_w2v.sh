#!/bin/sh
#SBATCH --job-name=synthesize
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --account=MST113025
#SBATCH -o ./slurm_logs/test/test-%A_%a.out
#SBATCH --ntasks-per-node=1
#SBATCH --array=1-7
module purge
module load miniconda3
conda activate task_vector
export HF_DATASETS_CACHE=/tmp
export TRANSFORMERS_CACHE=/tmp
export HF_DATASETS_CACHE=/tmp
export HUGGINGFACE_HUB_CACHE=/tmp
export TRANSFORMERS_CACHE=/tmp

target_domain="cooking"
# model=("openai/w2v-large" "outputs/w2v_slurp_"$target_domain"_large" "outputs/w2v_slurp_"$target_domain"_synthetic_large" "outputs/w2v_slurp_"$target_domain"_anti_large" "outputs/w2v_slurp_"$target_domain"_anti_synthetic_large" "outputs/w2v_slurp_"$target_domain"_anti_mixed_large/")
model=()
model+=("facebook/wav2vec2-conformer-rope-large-960h-ft")
model+=("outputs/w2v_slurp_"$target_domain"_large")
model+=("outputs/w2v_slurp_"$target_domain"_synthetic_large")
model+=("outputs/w2v_slurp_"$target_domain"_anti_large")
model+=("outputs/w2v_slurp_"$target_domain"_anti_synthetic_large")
model+=("outputs/w2v_slurp_"$target_domain"_anti_mixed_large/")
model+=("outputs/w2v_slurp_"$target_domain"_anti_mixed_large_continue/")

echo ${model[$SLURM_ARRAY_TASK_ID-1]};
sleep $(((SLURM_ARRAY_TASK_ID-1)*60));

python test_w2v.py --model_path "${model[$SLURM_ARRAY_TASK_ID-1]}" --domain "$target_domain";