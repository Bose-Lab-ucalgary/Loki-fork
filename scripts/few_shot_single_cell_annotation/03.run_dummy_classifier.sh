#!/bin/bash
#SBATCH --job-name=classifier_svm
#SBATCH --output=slurm_out/slurm-%x-%j.out
#SBATCH --time=02:00:00
#SBATCH --partition=defq
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100GB

source ~/.bashrc
conda activate dalle2-pytorch
which python

# define label_col for each test case in the following dictionary
declare -A label_col_dict
label_col_dict["CRC-VAL-HE-7K"]="tissue_type"
label_col_dict["NCT-CRC-HE-100K"]="tissue_type"
label_col_dict["DigestPath"]="tumor_type"
label_col_dict["PanNuke"]="tumor_type"
label_col_dict["WSSS4LUAD"]="tissue_type"
label_col_dict["Breast"]="cell_type"
label_col_dict["HF"]="cell_type"
label_col_dict["KidneyCancer"]="cell_type"
label_col_dict["HCAHeart"]="cell_type"
label_col_dict["SC_BC"]="cell_type"
label_col_dict["SC_ColonCancer"]="cell_type"
label_col_dict["SC_KidneyCancer"]="cell_type"
label_col_dict["SC_RCC"]="cell_type"
label_col_dict["SC_LungCancer"]="cell_type"
label_col_dict["SC_BCC"]="cell_type"
label_col_dict["SC_gbm"]="cell_type"
label_col_dict["SC_ProstateCancer_recelltype"]="cell_type"


model='dummy'
for test_case in SC_ProstateCancer_recelltype; do
    label_type=${label_col_dict[$test_case]}
    python classifier_dummy.py  --model $model --test_case $test_case --label_col $label_type --output_dir dummy_out
done

