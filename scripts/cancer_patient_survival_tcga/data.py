def read_clinical(df_p, df_s):
    """
    Read clinical data from patient and sample files
    """
    import numpy as np
    import pandas as pd

    age = df_p['AGE']
    site = df_s['TISSUE_SOURCE_SITE_CODE']
    is_female = df_p.SEX.str.contains('Female').astype(float)
    survival_months = df_p.OS_MONTHS
    survival_months_dss = df_p.DSS_MONTHS
    survival_months_pfi = df_p.PFS_MONTHS

    # if the patient is alive, censorship = 1; elif deceased, censorship = 0; else nan
    censorship = []
    for os_status in df_p.OS_STATUS:
        if os_status == '0:LIVING':
            censorship.append(1)
        elif os_status == '1:DECEASED':
            censorship.append(0)
        else:
            censorship.append(np.nan)

    censorship_dss = []
    for dss_status in df_p.DSS_STATUS:
        if dss_status == '0:ALIVE OR DEAD TUMOR FREE':
            censorship_dss.append(1)
        elif dss_status == '1:DEAD WITH TUMOR':
            censorship_dss.append(0)
        else:
            censorship_dss.append(np.nan)

    censorship_pfi = []
    for pfs_status in df_p.PFS_STATUS:
        if pfs_status == '0:CENSORED':
            censorship_pfi.append(1)
        elif pfs_status == '1:PROGRESSION':
            censorship_pfi.append(0)
        else:
            censorship_pfi.append(np.nan)

    #censorship = df_p.OS_STATUS.map(lambda x: 1 if x == '0:LIVING' else 0)
    #censorship_dss = df_p.DSS_STATUS.map(lambda x: 1 if x == '0:ALIVE OR DEAD TUMOR FREE' else 0)
    #censorship_pfi = df_p.PFS_STATUS.map(lambda x: 1 if x == '0:CENSORED' else 0)
    oncotree_code = df_s.ONCOTREE_CODE

    df = pd.DataFrame()

    df['age'] = age
    df['site'] = site
    df['is_female'] = is_female
    df['survival_months'] = survival_months
    df['survival_months_dss'] = survival_months_dss
    df['survival_months_pfi'] = survival_months_pfi
    df['censorship'] = censorship
    df['censorship_dss'] = censorship_dss
    df['censorship_pfi'] = censorship_pfi
    df['oncotree_code'] = oncotree_code

    return df


def complete_slide_path(slide_paths, patient_list):
    for patient_id in patient_list:
        if patient_id not in slide_paths:
            slide_paths[patient_id] = []
    return slide_paths


def exclude_patient_from_clinical(df):
    df = df.dropna(subset=['survival_months', 'censorship'])
    return df


def get_samples(
    cancer_type, data_dir=None, exclude_patients=[], exclude_slides=[]
):
    """
    Keep only the samples that have both WSI data and RNA-seq data
    """
    import json
    import os
    from pathlib import PurePath

    import pandas as pd

    name = f'TCGA-{cancer_type.upper()}'
    all_bio_data_dir = os.path.join(data_dir, "bio_data")
    type_bio_data_dir = os.path.join(
        all_bio_data_dir, f"{cancer_type}_tcga_pan_can_atlas_2018"
    )

    with open(
        os.path.join(data_dir, name, f'tcga_{cancer_type}_wsi_paths.json'), 'r'
    ) as f:
        slide_paths = json.load(f)

    patient_file = os.path.join(type_bio_data_dir, "data_clinical_patient.txt")
    sample_file = os.path.join(type_bio_data_dir, "data_clinical_sample.txt")

    df_patient = pd.read_csv(patient_file, sep="\t", skiprows=4, index_col=0)
    df_sample = pd.read_csv(sample_file, sep="\t", skiprows=4, index_col=0)

    df_clinical = read_clinical(df_patient, df_sample)

    slide_paths = complete_slide_path(
        slide_paths, df_patient.index.map(lambda x: x + '-01')
    )

    keep_patient_list = []
    keep_slide_list = []
    for patient_id in df_patient.index:
        if patient_id in exclude_patients:
            print("Excluding patient", patient_id)
            continue
        wsi_list = slide_paths.get(patient_id + '-01')

        for w in wsi_list:
            if PurePath(w).name in exclude_slides:
                print("Excluding slide", w)
                wsi_list.remove(w)

        if len(wsi_list) == 0:
            print("No WSI data for patient", patient_id)
            continue
        else:
            for w in wsi_list:
                w = PurePath(w).name
                keep_slide_list.append(w)
                keep_patient_list.append(patient_id)

    df = df_clinical.loc[keep_patient_list]
    df = df.reset_index().rename(columns={'PATIENT_ID': 'case_id'})
    df['slide_id'] = keep_slide_list

    # drop samples with missing survival information
    df = exclude_patient_from_clinical(df)

    return df


def prepare_dataset(
    cancer_type,
    data_dir=None,
    model='omiclip',
    exclude_patients_fpath=None,
    exclude_slides_fpath=None,
    output_dir=None
):

    if exclude_patients_fpath is not None:
        try:
            with open(exclude_patients_fpath, 'r') as f:
                exclude_patients = f.read().splitlines()
        except:
            exclude_patients = []

    if exclude_slides_fpath is not None:
        try:
            with open(exclude_slides_fpath, 'r') as f:
                exclude_slides = f.read().splitlines()
        except:
            exclude_slides = []

    df_c = get_samples(
        cancer_type,
        data_dir=data_dir,
        exclude_patients=exclude_patients,
        exclude_slides=exclude_slides
    )
    df_c.to_csv(f"{output_dir}/tcga_{cancer_type}_all_clean.csv", index=None)


def create_splits(series, censorship, name, splits=5, random_state=42):
    """
    Create stratified splits for cross validation

    Parameters
    ----------
    series : pandas.Series
        Series to split
    censorship : pandas.Series
        Series indicating censorship
    name : str
        Name of the dataset
    splits : int
        Number of splits to create
    random_state : int
        Random seed for reproducibility
    """

    import os

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold

    if random_state is not None:
        shuffle = True
    else:
        shuffle = False

    assert len(series) == len(
        censorship
    ), "series and censorship must have the same length"

    stratified_group_kfold = StratifiedGroupKFold(
        n_splits=splits, shuffle=shuffle, random_state=random_state
    )

    for i, (train_index, test_index) in enumerate(
        stratified_group_kfold.split(series, censorship, groups=series)
    ):
        train_case_ids = series.iloc[train_index].unique()
        test_case_ids = series.iloc[test_index].unique()

        # Pad the shorter list with None values
        length_diff = len(train_case_ids) - len(test_case_ids)
        if length_diff > 0:
            test_case_ids = np.append(test_case_ids, [None] * length_diff)
        elif length_diff < 0:
            train_case_ids = np.append(train_case_ids, [None] * (-length_diff))

        # Create a DataFrame for the split
        split_df = pd.DataFrame({'train': train_case_ids, 'val': test_case_ids})

        out_dir = os.path.join("splits", f"{splits}foldcv", name)

        os.makedirs(out_dir, exist_ok=True)
        # Save the DataFrame to a CSV file
        split_df.to_csv(os.path.join(out_dir, f'splits_{i}.csv'), index=True)
