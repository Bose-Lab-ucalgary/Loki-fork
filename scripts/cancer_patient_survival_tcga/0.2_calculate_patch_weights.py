def calc_attention(case_id, slide_ids, data_dir, normalize=False, T=1):
    import os

    import torch
    from torch.nn import functional as F

    path_features = []
    for slide_id in slide_ids:
        wsi_path = os.path.join(
            data_dir, 'pt_files', 'image_{}.pt'.format(slide_id.rstrip('.svs'))
        )
        wsi_bag = torch.load(wsi_path)
        path_features.append(wsi_bag)
    path_features = torch.cat(path_features, dim=0)  # torch.Size([n, 768])

    text_embedding_path = os.path.join(
        data_dir, 'pt_files', 'text_{}.pt'.format(case_id)
    )
    omic_features = torch.load(text_embedding_path)  # torch.Size([768])

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    csim = cos(path_features, omic_features.unsqueeze(0))

    if normalize:
        dnorm = torch.sqrt(
            torch.tensor([omic_features.shape[0]], dtype=torch.float)
        )  # sqrt(768)
        csim = csim / dnorm
    csim = csim.unsqueeze(0)

    # apply temperature scaling
    csim = csim / T

    # calculate weighted embedding for each case (apply dropout for the weights)
    weights = torch.nn.functional.softmax(csim, dim=-1)  # torch.Size([1, n])

    weights_path = os.path.join(
        data_dir, 'pt_files', 'weights_{}.pt'.format(case_id)
    )
    torch.save(weights, weights_path)

    weighted_path_features = torch.mm(
        weights, path_features
    )  # torch.Size([1, 768])

    # normalize the weighted embedding
    weighted_path_features = F.normalize(weighted_path_features, p=2, dim=1)

    weighted_path_path = os.path.join(
        data_dir, 'pt_files', 'weighted_image_{}.pt'.format(case_id)
    )
    torch.save(weighted_path_features, weighted_path_path)


def main(
    test_case="BRCA",
    work_dir="/condo/wanglab/tmhpxz9/wc-coca/WSI_tasks/Survival_100perc_patches",
    normalize=False,
    T=1
):
    import os

    import pandas as pd
    from tqdm import tqdm

    os.chdir(work_dir)

    metadata_dir = os.path.join(work_dir, "dataset_csv")

    if True:
        name = f'TCGA_{test_case.upper()}'
        data_dir = os.path.join(work_dir, 'input', name)

        metadata_file = os.path.join(
            metadata_dir, f"tcga_{test_case.lower()}_all_clean.csv"
        )
        df = pd.read_csv(metadata_file)

        # one case may match multiple slides
        # each row corresponds to one slide
        case_slides = df.groupby('case_id')['slide_id'].apply(list
                                                             ).reset_index()
        print(f"Calculating attention for {test_case}...")
        print(f"Total cases: {len(case_slides)}")
        for i in tqdm(range(len(case_slides))):
            case_id = case_slides.loc[i, 'case_id']
            slide_ids = case_slides.loc[i, 'slide_id']
            calc_attention(
                case_id, slide_ids, data_dir, normalize=normalize, T=T
            )


if __name__ == '__main__':
    import fire
    fire.Fire(main)
