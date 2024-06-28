# This script extract gene expression data and the annotations from the h5ad data file

import os
import numpy as np
import pandas as pd
import scanpy as sc
import warnings
from tqdm import tqdm
from scipy.sparse import csr_matrix, issparse
warnings.simplefilter(action='ignore')

EXCLUDE_PATTERN = ['ENSG', '\.', '-', '_']

def get_gene_expression_from_adata(ad, top_n_genes=100, gene_symbol_col=None, exclude_genes=None, exclude_genes_pattern=False):
    """
    get the gene expression data for the top_n_genes highly expressed genes in each cell
    The genes in the adata are not sorted by expression level.

    exclude_genes: list of genes (name) to exclude
    exclude_genes_pattern: exclude RNA genes from the list
    """

    if gene_symbol_col is not None:
        ad.var.index = ad.var[gene_symbol_col]

    if exclude_genes is not None:
        ad = ad[:, ~ad.var.index.isin(exclude_genes)]

    if exclude_genes_pattern:
        # exclude RNA genes containing the signatures in a list
        ad = ad[:, ~ad.var.index.str.contains('|'.join(EXCLUDE_PATTERN))]

    # Get the indices of the top-ranking values in each row (each cell)
    top_n_genes = min(top_n_genes, ad.var.shape[0])
    if issparse(ad.X):
        arr = ad.X.toarray()
    else:
        arr = ad.X
    
    # the top n indices are not sorted, sort them
    #top_n_indices = np.argpartition(arr, -top_n_genes, axis=1)[:, -top_n_genes:]

    # sort the top n indices in each row sorted in descending order
    top_n_indices = np.argsort(-arr, axis=1)[:, :top_n_genes]

    # Create a mask where only the top ranking values in each row are True
    mask = np.zeros_like(arr, dtype=bool)
    np.put_along_axis(mask, top_n_indices, True, axis=1)

    # Use this mask to set all other values to numpy.nan
    arr[~mask] = 0

    # create an annData object using the gene expression data
    arr_csr = csr_matrix(arr)

    # gabage collection
    del arr
    del mask

    arr_csr.eliminate_zeros()
    ad_top_genes = ad.copy()
    ad_top_genes.X = arr_csr

    del ad

    names = ad_top_genes.var.index.to_numpy()
    top_n_genes_array = names[top_n_indices]

    return ad_top_genes, top_n_genes_array

def get_cell_type_from_adata(ad, cell_type_col):
    """
    get the cell type data from the adata
    """

    cell_types = ad.obs[cell_type_col].to_numpy()
    return cell_types

def read_genes(exclude_genes_file):
    """
    read the genes to exclude from the file
    """
    import pandas as pd
    exclude_genes = pd.read_csv(exclude_genes_file)['genesymbol'].values

    return list(exclude_genes)


import argparse
parser = argparse.ArgumentParser(description='Prepare data for the model')
parser.add_argument('--test_case', type=str, required=True, help='Test case name')
parser.add_argument('--data_path', type=str, default="data_source", help='Path to the data source folder')
parser.add_argument('--top_n_genes', type=int, default=100, help='Number of top genes to consider')
parser.add_argument('--gene_symbol_col', type=str, default=None, help='Column name for gene symbol in adata.obs')
parser.add_argument('--cell_type_col', type=str, default='celltype_major', help='Column name for cell type')
parser.add_argument('--exclude_genes', action='store_true', help='Exclude genes from the list')
parser.add_argument('--exclude_genes_pattern', action='store_true', help='Exclude RNA genes from the list and other patterns')
args = parser.parse_args()

print(args)

data_path = args.data_path
test_case = args.test_case
top_n_genes = args.top_n_genes
gene_symbol_col = args.gene_symbol_col
cell_type_col = args.cell_type_col

if args.exclude_genes:
    exclude_genes_file = os.path.join(data_path, f"housekeeping_genes.csv")
    exclude_genes = read_genes(exclude_genes_file)
else:
    exclude_genes = None

print(f"Processing {test_case}")

os.makedirs(test_case, exist_ok=True)

ad = sc.read_h5ad(f"{data_path}/{test_case}.h5ad")
ad, text_features = get_gene_expression_from_adata(
        ad, 
        top_n_genes=top_n_genes,
        gene_symbol_col=gene_symbol_col,
        exclude_genes=exclude_genes,
        exclude_genes_pattern=args.exclude_genes_pattern
    )

ad.write(f"{test_case}/{test_case}_top{top_n_genes}.h5ad")
np.save(f"{test_case}/{test_case}_top{top_n_genes}_names.npy", text_features)

annotations = get_cell_type_from_adata(ad, cell_type_col)
anno_df = pd.DataFrame(annotations, columns=['cell_type'])

anno_df.to_csv(f"{test_case}/{test_case}_labels.csv", index=False)
