import os

import numpy as np
import pandas as pd
import torch

from dataset_survival import Generic_MIL_Survival_Dataset
from training import kwargs_default, train

"""
The following functions `seed_torch`, `hazard2grade`, and `getPValue_Binary` are from/adapted from the MCAT repository (https://github.com/mahmoodlab/MCAT) 
with minor modifications. We thank the authors for their original work.
"""
def seed_torch(seed=7, device=torch.device('cuda')):
    import random

    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def hazard2grade(hazard, p):
    """
    The smaller the grade (strat), the smaller the hazard (risk)
    """
    for i in range(len(p)):
        if hazard < p[i]:
            return i
    return len(p)


def getPValue_Binary(
    df: pd.DataFrame = None,
    risk_percentiles=[50, 50],
    save_path=None,
    **plot_kwargs
):
    from lifelines.statistics import logrank_test
    results_df = df.copy()

    p = np.percentile(results_df['risk'], risk_percentiles)
    results_df.insert(
        0, 'strat', [hazard2grade(risk, p) for risk in results_df['risk']]
    )
    T_low, T_high = results_df['survival'][
        results_df['strat'] == 0
    ], results_df['survival'][results_df['strat'] == len(risk_percentiles)]
    E_low, E_high = 1 - results_df['censorship'][
        results_df['strat'] == 0], 1 - results_df['censorship'][
            results_df['strat'] == len(risk_percentiles)]

    surv_groups = {'low risk': T_low, 'high risk': T_high}

    event_groups = {'low risk': E_low, 'high risk': E_high}

    plot_survival(surv_groups, event_groups, save_path=save_path, **plot_kwargs)

    low_vs_high = logrank_test(
        durations_A=T_low,
        durations_B=T_high,
        event_observed_A=E_low,
        event_observed_B=E_high
    ).p_value
    return np.array([low_vs_high])


def plot_survival(
    surv_groups,
    event_groups,
    save_path=None,
    loc=slice(0, 150),
    ci_show=False,
    ci_force_lines=True
):
    """
    groups: a dictionary of survival times. k -> group name; v -> list/array of survival times
    """
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter

    censor_style = {'ms': 10, 'marker': '+'}

    kmf = KaplanMeierFitter()

    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = plt.subplot()

    for k, v in surv_groups.items():
        c = event_groups[k]
        kmf.fit(v, c, label=k)
        kmf.plot_survival_function(
            ax=ax,
            loc=loc,
            ci_show=ci_show,
            ci_force_lines=ci_force_lines,
            show_censors=True,
            censor_styles=censor_style
        )

    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.001, 0.2))

    ax.tick_params(axis='both', which='major', labelsize=12)
    #plt.legend(fontsize=32, prop=font_manager.FontProperties(family='sans', style='normal', size=32))
    plt.xlabel('Time')
    plt.ylabel('Surviving ratio')

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()


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

    #create_splits(dataset.slide_data.case_id, dataset.slide_data.censorship, f"tcga_{cancer_type.lower()}", random_state=seed)

    results_dir = 'results_all_patches'
    which_splits = '5foldcv'
    split_dir = f"tcga_{cancer_type.lower()}"

    ### Creates results_dir Directory.
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    ### Appends to the results_dir path: 1) which splits were used for training (e.g. - 5foldcv), and then 2) the parameter code and 3) experiment code
    results_dir = os.path.join(
        results_dir, which_splits, f"tcga_{cancer_type.lower()}_cosine_s{seed}"
    )
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    split_dir = os.path.join('./splits', which_splits, split_dir)

    dfs_val = []

    for cur in range(5):
        train_dataset, val_dataset = dataset.return_splits(
            from_id=False, csv_path='{}/splits_{}.csv'.format(split_dir, cur)
        )

        print(
            'training: {}, validation: {}'.format(
                len(train_dataset), len(val_dataset)
            )
        )
        datasets = (train_dataset, val_dataset)

        kwargs_default.update(
            {
                'fusion': 'concat',
                'lr': 0.002,
                'results_dir': results_dir,
                'alpha_surv': 0.1
            }
        )

        (results_train_dict, results_val_dict
        ), (train_cindex, val_cindex) = train(datasets, cur, **kwargs_default)

        df_train = pd.DataFrame(results_train_dict).T
        df_train['censorship'] = df_train['censorship'].astype(int)
        df_train['survival'] = df_train['survival'].astype(float)
        df_train.to_csv(
            os.path.join(results_dir, f"result_training_fold{cur}.csv")
        )

        df_val = pd.DataFrame(results_val_dict).T
        df_val['censorship'] = df_val['censorship'].astype(int)
        df_val['survival'] = df_val['survival'].astype(float)
        df_val.to_csv(os.path.join(results_dir, f"result_val_fold{cur}.csv"))

        dfs_val.append(df_val)

    dfs_val_conc = pd.concat(dfs_val)
    getPValue_Binary(
        dfs_val_conc, [50, 50],
        loc=None,
        save_path=os.path.join(results_dir, 'survival_plot_50v50.png')
    )
    getPValue_Binary(
        dfs_val_conc, [30, 70],
        loc=None,
        save_path=os.path.join(results_dir, 'survival_plot_30v70.png')
    )


if __name__ == '__main__':
    import fire
    fire.Fire(run_case)
