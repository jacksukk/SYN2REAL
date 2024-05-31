#!/bin/sh
#SBATCH --job-name=synthesize
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --account=MST113025
#SBATCH -o ./slurm_logs/test/test-%A_%a.out
#SBATCH --ntasks-per-node=1
#SBATCH --array=1-8
module purge
module load miniconda3
conda activate task_vector
export HF_DATASETS_CACHE=/tmp
export TRANSFORMERS_CACHE=/tmp
export HF_DATASETS_CACHE=/tmp
export HUGGINGFACE_HUB_CACHE=/tmp
export TRANSFORMERS_CACHE=/tmp

domain=('cooking' 'audio' 'transport' 'news' 'music' 'lists' 'weather' 'calendar' 'qa' 'general' 'datetime' 'recommendation' 'play' 'iot' 'social' 'takeaway' 'email' 'alarm')
# temp_domain=('cooking' 'audio' 'transport' 'news' 'music' 'lists' 'weather' 'calendar' 'qa' 'general' 'datetime' 'recommendation' 'play' 'iot' 'social' 'takeaway' 'email' 'alarm')
temp_domain=('music')
nums=(1 3 5 7 9 11 13 15)
# model=("openai/whisper-large" "outputs/whisper_slurp_"$target_domain"_large" "outputs/whisper_slurp_"$target_domain"_synthetic_large" "outputs/whisper_slurp_"$target_domain"_anti_large" "outputs/whisper_slurp_"$target_domain"_anti_synthetic_large" "outputs/whisper_slurp_"$target_domain"_anti_mixed_large/")

echo ${temp_domain[$SLURM_ARRAY_TASK_ID-1]};
sleep $(((SLURM_ARRAY_TASK_ID-1)*60));

python test.py --model_syn_anti outputs/whisper_slurp_${temp_domain[$SLURM_ARRAY_TASK_ID-1]}_anti${nums[$SLURM_ARRAY_TASK_ID-1]}_synthetic_small/ --model_anti outputs/whisper_slurp_${temp_domain[$SLURM_ARRAY_TASK_ID-1]}_anti${nums[$SLURM_ARRAY_TASK_ID-1]}_small/ --model_target_syn "jacksukk/whisper_slurp_music_anti_mixed_small_continue" --domain ${temp_domain[$SLURM_ARRAY_TASK_ID-1]} --weight 0.2