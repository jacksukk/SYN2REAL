#!/bin/sh
#SBATCH --job-name=synthesize
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --account=MST113025
#SBATCH -o ./slurm_logs/slurm-%A_%a.out
#SBATCH --ntasks-per-node=1
#SBATCH --array=1-4
#SBATCH --partition=gp4d

module purge
module load miniconda3
conda activate task_vector
export HF_DATASETS_CACHE=/tmp
export TRANSFORMERS_CACHE=/tmp
export HF_DATASETS_CACHE=/tmp
export HUGGINGFACE_HUB_CACHE=/tmp
export TRANSFORMERS_CACHE=/tmp
domain=('cooking' 'audio' 'transport' 'news' 'music' 'lists' 'weather' 'calendar' 'qa' 'general' 'datetime' 'recommendation' 'play' 'iot' 'social' 'takeaway' 'email' 'alarm')
temp_domain=('cooking' 'weather' 'social' 'music')
final_train_domain=()
syn=()
for target_domain in "${temp_domain[@]}"; do
    
    train_domain=""

    for d in "${domain[@]}"; do
        if [ $d == "$target_domain" ]; then
            continue
        fi
        train_domain+="$d;"
    done

    # echo $train_domain;
    # final_train_domain=("$train_domain" "$target_domain" "$train_domain" "$target_domain");
    final_train_domain+=("$train_domain");

    syn+=("True");
    echo ${final_train_domain[$SLURM_ARRAY_TASK_ID-1]};
done
# echo ${final_train_domain[2]};
# python train.py --domains "${final_train_domain[$SLURM_ARRAY_TASK_ID-1]}" --syn "${syn[$SLURM_ARRAY_TASK_ID-1]}" --mix True --model_path openai/whisper-tiny --configs configs/whisper_tiny.yaml;
# python train_origin.py;
python train_w2v_from_checkpoint.py --domains "${final_train_domain[$SLURM_ARRAY_TASK_ID-1]}" --syn "${syn[$SLURM_ARRAY_TASK_ID-1]}" --mix True --model_path "outputs/w2v_slurp_"${temp_domain[$SLURM_ARRAY_TASK_ID-1]}"_anti_mixed_large/checkpoint-7000";
