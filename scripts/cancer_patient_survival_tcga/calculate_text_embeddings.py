def main(
    model="omiclip",
    test_case='brca',
    context_length=76,
    exclude_genes=True,
    exclude_rna_genes=True
):
    import os

    import numpy as np
    import scanpy as sc
    import torch
    from open_clip import create_model_from_pretrained, get_tokenizer
    from scripts.encode import (PRETRAINED_NETS, get_text_embeddings_large,
                                model_paths)

    # The tokenizer script does not work for openaiclip model
    # The encode_text expect the contetx length to be 77 always
    if model == 'openaiclip':
        context_length = 77

    test_case = test_case.lower()
    name = f'TCGA-{test_case.upper()}'
    output_path = os.path.join(name, f"text_embeddings_{model}")

    ad_path = os.path.join('data_source', name, f'tcga_{test_case}.h5ad')
    ad = sc.read_h5ad(ad_path)

    if exclude_genes:
        excluded_genes_file = os.path.join(
            'data_source', 'housekeeping_genes.csv'
        )
        excluded_genes = read_genes(excluded_genes_file)
    else:
        excluded_genes = None
    ad_top_genes, top_n_genes_array = get_gene_expression_from_adata(
        ad,
        top_n_genes=50,
        exclude_genes=excluded_genes,
        exclude_rna_genes=exclude_rna_genes
    )

    ad_top_genes.write_h5ad(
        os.path.join('data_source', name, f'tcga_{test_case}_top_genes.h5ad')
    )
    np.save(
        os.path.join('data_source', name, f'tcga_{test_case}_top_genes.npy'),
        top_n_genes_array
    )

    sep = ' '
    labels = np.apply_along_axis(lambda x: sep.join(x), 1, top_n_genes_array)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(DEVICE)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Current working directory: {os.getcwd()}")

    model_path = model_paths[model]
    PRETRAINED_NET = PRETRAINED_NETS[model]

    os.makedirs(output_path, exist_ok=True)
    batch_size = 100 if DEVICE == "cpu" else 1000

    tokenizer = get_tokenizer(PRETRAINED_NET)
    model_pretrained, preprocess = create_model_from_pretrained(
        PRETRAINED_NET, device=DEVICE, pretrained=model_path
    )

    text_embeddings = get_text_embeddings_large(
        labels,
        model_pretrained,
        tokenizer,
        batch_size=batch_size,
        context_length=context_length,
        DEVICE=DEVICE
    )
    #    torch.save(
    #        text_embeddings,
    #        os.path.join(output_path, f"{model}_{test_case}_text_embeddings.pt")
    #    )

    assert len(text_embeddings) == ad_top_genes.obs.shape[
        0
    ], "Number of embeddings should be equal to the number of patients in the dataset"
    for i, emb in enumerate(text_embeddings):
        patient_id = ad_top_genes.obs.index[i]
        torch.save(
            emb,
            os.path.join(
                output_path,
                f"{model}_{test_case.lower()}_{patient_id}_text_embeddings.pt"
            )
        )


def get_gene_expression_from_adata(
    ad,
    top_n_genes=50,
    gene_symbol_col=None,
    exclude_genes=None,
    exclude_rna_genes=False
):
    """
    get the gene expression data for the top_n_genes highly expressed genes in each cell
    The genes in the adata are not sorted by expression level.

    exclude_genes: list of genes (name) to exclude
    exclude_rna_genes: exclude RNA genes from the list
    """

    import numpy as np
    from scipy.sparse import csr_matrix, issparse

    RNA_GENES_SIGNATURE = ['ENSG', '\.', '-', '_']

    if gene_symbol_col is not None:
        ad.var.index = ad.var[gene_symbol_col]

    if exclude_genes is not None:
        ad = ad[:, ~ad.var.index.isin(exclude_genes)]

    if exclude_rna_genes:
        # exclude RNA genes containing the signatures in a list
        #ad = ad[:, ~ad.var.index.str.startswith(tuple(RNA_GENES_SIGNATURE))]
        ad = ad[:, ~ad.var.index.str.contains('|'.join(RNA_GENES_SIGNATURE))]

    # Get the indices of the top 50 values in each row (each cell)
    top_n_genes = min(top_n_genes, ad.var.shape[0])
    if issparse(ad.X):
        arr = ad.X.toarray()
    else:
        arr = ad.X

    # sort the top n indices in each row sorted in descending order
    top_n_indices = np.argsort(-arr, axis=1)[:, :top_n_genes]

    # Create a mask where only the top 50 values in each row are True
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


def read_text_file(file_path, sep=' '):
    import numpy as np

    text_array = np.load(file_path, allow_pickle=True)
    cell_text_vector = np.apply_along_axis(lambda x: sep.join(x), 1, text_array)
    return cell_text_vector


def read_genes(exclude_genes_file):
    """
    read the genes to exclude from the file
    """
    import pandas as pd
    exclude_genes = pd.read_csv(exclude_genes_file)['genesymbol'].values

    return list(exclude_genes)


if __name__ == "__main__":

    import fire
    fire.Fire(main)
