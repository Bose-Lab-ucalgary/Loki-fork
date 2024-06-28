#!/bin/bash
#SBATCH --job-name=calc_emb_clip
#SBATCH --output=slurm_out/slurm-%x-%j.out
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100GB

for model in omiclip; do
    for test_case in SC_BC ; do
        echo "Calculating embeddings for $model on $test_case"
        python calculate_text_embeddings.py \
          --model $model \
          --output_path embeddings_$model \
          --test_case $test_case \
          --clip_context \
          --use_gpu
    done
done

