#!/bin/sh
#SBATCH --job-name=synthesize
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --account=MST113025
#SBATCH -o ./slurm_logs/slurm-%A_%a.out
#SBATCH --ntasks-per-node=1
#SBATCH --array=1-16

module purge
module load miniconda3
conda activate task_vector
# export HF_DATASETS_CACHE=/tmp
# export TRANSFORMERS_CACHE=/tmp
# export HF_DATASETS_CACHE=/tmp
# export HUGGINGFACE_HUB_CACHE=/tmp
# export TRANSFORMERS_CACHE=/tmp

domain=('cooking' 'audio' 'transport' 'news' 'music' 'lists' 'weather' 'calendar' 'qa' 'general' 'datetime' 'recommendation' 'play' 'iot' 'social' 'takeaway' 'email' 'alarm')
multi_target_domain=("music" "music" "music" "music" "music" "music" "music" "music")
nums=(1 1 3 3 5 5 7 7 9 9 11 11 13 13 15 15)
final_train_domain=()
syn=()
for target_domain in "${multi_target_domain[@]}"; do
    train_domain=""
    for d in "${domain[@]}"; do
        if [ $d == "$target_domain" ]; then
            continue
        fi
        train_domain+="$d;"
    done

    # echo $train_domain;
    # echo $target_domain;
    final_train_domain+=("$train_domain" "$train_domain");

    # final_train_domain=("$target_domain" "$target_domain" "$target_domain");

    syn+=( "False" "True");
done
# syn=("True" "True");

echo ${final_train_domain[0]};
# torchrun --nproc_per_node 2 train.py;
# sleep $(((SLURM_ARRAY_TASK_ID-1)*120));
# if [ $SLURM_ARRAY_TASK_ID == 7 ]; then
#     echo $SLURM_ARRAY_TASK_ID;
python train.py --domains "${final_train_domain[$SLURM_ARRAY_TASK_ID-1]}" --syn "${syn[$SLURM_ARRAY_TASK_ID-1]}" --model_path openai/whisper-small --configs configs/whisper_small.yaml --numbers ${nums[$SLURM_ARRAY_TASK_ID-1]};
# fi
# python train.py --domains "${final_train_domain[$SLURM_ARRAY_TASK_ID-1]}" --syn "${syn[$SLURM_ARRAY_TASK_ID-1]}" --model_path openai/whisper-tiny --configs configs/whisper_tiny.yaml;
# python train_origin.py;


#  if [ $SLURM_ARRAY_TASK_ID == 15 ] || [ $SLURM_ARRAY_TASK_ID == 16 ] || [ $SLURM_ARRAY_TASK_ID == 17 ] || [ $SLURM_ARRAY_TASK_ID == 18 ] || [ $SLURM_ARRAY_TASK_ID == 19 ] ; then

# python train.py --domains "${final_train_domain[$SLURM_ARRAY_TASK_ID-1]}" --syn "${syn[$SLURM_ARRAY_TASK_ID-1]}" --model_path openai/whisper-base --configs configs/whisper_base.yaml;
# fi