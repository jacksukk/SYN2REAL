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


domain=('cooking' 'audio' 'transport' 'news' 'music' 'lists' 'weather' 'calendar' 'qa' 'general' 'datetime' 'recommendation' 'play' 'iot' 'social' 'takeaway' 'email' 'alarm')
# temp_domain=('cooking' 'audio' 'transport' 'news' 'music' 'lists' 'weather' 'calendar' 'qa' 'general' 'datetime' 'recommendation' 'play' 'iot' 'social' 'takeaway' 'email' 'alarm')
temp_domain=('music')
echo ${temp_domain[$SLURM_ARRAY_TASK_ID-1]};
sleep $(((SLURM_ARRAY_TASK_ID-1)*60));

python test.py --model_syn_anti outputs/whisper_slurp_${temp_domain[$SLURM_ARRAY_TASK_ID-1]}_anti_synthetic_small/ --model_anti outputs/whisper_slurp_${temp_domain[$SLURM_ARRAY_TASK_ID-1]}_anti_small/ --model_target_syn outputs/whisper_slurp_${temp_domain[$SLURM_ARRAY_TASK_ID-1]}_anti_mixed_small_continue/ --domain ${temp_domain[$SLURM_ARRAY_TASK_ID-1]} --weight 0.2;
