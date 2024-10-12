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


target_domain="music"
model=()
model+=("openai/whisper-small")
model+=("outputs/whisper_slurp_"$target_domain"_small")
model+=("outputs/whisper_slurp_"$target_domain"_synthetic_small")
model+=("outputs/whisper_slurp_"$target_domain"_anti_small")
model+=("outputs/whisper_slurp_"$target_domain"_anti_synthetic_small")
model+=("outputs/whisper_slurp_"$target_domain"_anti_mixed_small/")
model+=("outputs/whisper_slurp_"$target_domain"_anti_mixed_small_continue/")

echo ${model[$SLURM_ARRAY_TASK_ID-1]};

python test.py --model_path "${model[$SLURM_ARRAY_TASK_ID-1]}" --domain "$target_domain";