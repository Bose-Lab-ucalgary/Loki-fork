#!/bin/bash

# Description: This script prepares the data for the test case. It extracts the gene expression data and the cell type labels from the raw data and saves them in a format that can be used by the subsequent classifier model. The script also filters the genes based on the top_n_genes parameter and the exclude_genes and exclude_genes_pattern flags. The gene expression data is saved in a numpy array format and the cell type labels are saved in a csv file. The script takes the following parameters:
# test_case: The test case for which the data is being prepared. The test case is used to determine the data source and the column names for the gene expression data and the cell type labels.
# data_path: The path to the directory containing the raw data files.
# gene_symbol_col: The name of the column containing the gene symbols in the raw data files. If the gene_symbol_col is not specified, the gene names will be extracted from the first column of the raw data files.
# top_n_genes: The number of top genes to select based on the variance of the gene expression data.
# cell_type_col: The name of the column containing the cell type labels in the raw data files.
# exclude_genes: A flag indicating whether to exclude the genes that are provided in the housekeeping_genes.csv file from the gene expression data.
# exclude_genes_pattern: A flag indicating whether to exclude genes with certain patterns (mainly RNA genes) from the gene expression data.

# expected input files:
# 1. raw data files in h5ad format saved in the given {data_path} directory and named as {test_case}.h5ad

# output files:
# 1. numpy array containing the top-ranking gene names saved as {test_case}_top{top_n_genes}_names.npy in the test_case directory
# 2. csv file containing the cell type labels saved as {test_case}_labels.csv in the test_case directory


# define label_col for each test case in the following dictionary
declare -A ad_label_col_dict
ad_label_col_dict["SC_BC"]="celltype_major"

top_n_genes=100
data_path=data_source

for test_case in SC_BC; do
    label_col=${ad_label_col_dict[$test_case]}
    
    # if ad_gene_name_col_dict is not defined for the test case, then the gene names will be extracted from the first column (None)
    gene_name_col=${ad_gene_name_col_dict[$test_case]}
    python prepare_data.py \
        --test_case $test_case \
        --data_path $data_path \
        --gene_symbol_col $gene_name_col \
        --top_n_genes $top_n_genes \
        --cell_type_col $label_col \
        --exclude_genes \
        --exclude_genes_pattern
done
