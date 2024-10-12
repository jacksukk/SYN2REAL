#!/bin/sh
#SBATCH --job-name=synthesize
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --account=MST113025
#SBATCH -o ./slurm_logs/test/test-%A_%a.out
#SBATCH --ntasks-per-node=1
#SBATCH --array=1-18
module purge
module load miniconda3
conda activate task_vector


domain=('cooking' 'audio' 'transport' 'news' 'music' 'lists' 'weather' 'calendar' 'qa' 'general' 'datetime' 'recommendation' 'play' 'iot' 'social' 'takeaway' 'email' 'alarm');
mkdir -p data/synthetic/${domain[$SLURM_ARRAY_TASK_ID-1]};
python generate.py --domains ${domain[$SLURM_ARRAY_TASK_ID-1]};