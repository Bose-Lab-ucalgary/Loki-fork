import os
import random

import numpy as np
import torch

from data import create_splits
from dataset_survival import Generic_MIL_Survival_Dataset


def seed_torch(seed=7, device=torch.device('cuda')):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_case(case='brca', seed=20, mode='coattn', n_bins=4, ignore=[]):
    cancer_type = case.upper()
    seed_torch(seed)
    dataset = Generic_MIL_Survival_Dataset(
        csv_path='./%s/%s_all_clean.csv' %
        ('dataset_csv', f'tcga_{cancer_type.lower()}'),
        mode=mode,
        apply_sig=False,
        data_dir=f'input/TCGA_{cancer_type.upper()}',
        shuffle=False,
        seed=seed,
        print_info=False,
        patient_strat=False,
        n_bins=n_bins,
        label_col='survival_months',
        ignore=ignore
    )

    create_splits(
        dataset.slide_data.case_id,
        dataset.slide_data.censorship,
        f"tcga_{cancer_type.lower()}",
        random_state=seed
    )


if __name__ == '__main__':

    run_case('lgg', seed=2024)
