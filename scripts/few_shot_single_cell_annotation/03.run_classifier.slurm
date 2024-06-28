#!/bin/bash
#SBATCH --job-name=classifier_bayes
#SBATCH --output=slurm_out/slurm-%x-%j.out
#SBATCH --time=02:00:00
#SBATCH --partition=bigmemq
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100GB


label_type="cell_type"

for modality in text; do
    for model in omiclip; do
        for test_case in SC_BC; do
            echo "Run classifier for $model on $test_case using $modality modality"
            python supervised_classifier_bayes.py \
                --model $model \
                --test_case $test_case \
                --modality $modality \
                --label_col $label_type \
                --random_state 2024 \
                --output_dir results
        done
    done
done
