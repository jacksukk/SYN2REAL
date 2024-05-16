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

target_domain="music"
# model=("openai/whisper-large" "outputs/whisper_slurp_dpwd__"$target_domain"_speech_t5_small" "outputs/whisper_slurp_dpwd__"$target_domain"_synthetic_speech_t5_small" "outputs/whisper_slurp_dpwd__"$target_domain"_anti_speech_t5_small" "outputs/whisper_slurp_dpwd__"$target_domain"_anti_synthetic_speech_t5_small" "outputs/whisper_slurp_dpwd__"$target_domain"_anti_mixed_speech_t5_small/")
model=()
model+=("openai/whisper-small")
model+=("outputs/whisper_slurp_dpwd__"$target_domain"_small")
model+=("outputs/whisper_slurp_dpwd__"$target_domain"_synthetic_speech_t5_small")
model+=("outputs/whisper_slurp_dpwd__"$target_domain"_anti_small")
model+=("outputs/whisper_slurp_dpwd__"$target_domain"_anti_synthetic_speech_t5_small")
model+=("outputs/whisper_slurp_dpwd__"$target_domain"_anti_mixed_speech_t5_small/")
model+=("outputs/whisper_slurp_dpwd__"$target_domain"_anti_mixed_speech_t5_small_continue/")

echo ${model[$SLURM_ARRAY_TASK_ID-1]};
# sleep $(((SLURM_ARRAY_TASK_ID-1)*60));

python test.py --model_path "${model[$SLURM_ARRAY_TASK_ID-1]}" --domain "$target_domain";