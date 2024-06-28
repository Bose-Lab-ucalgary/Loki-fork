if __name__ == "__main__":
    # expected directory structure
    """
    data_source
    ├── bio_data 
    │   └── blca_tcga_pan_can_atlas_2018
    ├── diagnostic_slides -> contains all .svs files
    ├── patches 
    │   └── patches -> contains all .h5 files from CLAM
    └── TCGA-BLCA
       ├── tcga_blca.h5ad
       ├── tcga_blca_top_genes.h5ad
       ├── tcga_blca_top_genes.npy
       └── tcga_blca_wsi_paths.json -> json file of patient_id:[wsi_path_1, wsi_path_2, ..] pairs
    """

    import os

    from data import prepare_dataset

    root_dir = os.getcwd()
    data_dir = os.path.join(root_dir, "data_source")
    work_dir = root_dir

    if not os.path.exists(os.path.join(work_dir, "dataset_csv")):
        os.makedirs(os.path.join(work_dir, "dataset_csv"))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=str, default='LGG')

    args = parser.parse_args()
    cases = args.cases.split(',')
    for case in cases:
        print(f"Processing {case}")
        case = case.lower()

        prepare_dataset(
            case,
            data_dir=data_dir,
            output_dir=os.path.join(work_dir, "dataset_csv")
        )
