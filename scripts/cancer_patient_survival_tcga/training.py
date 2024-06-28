"""
This code is adapted from the MCAT repository (https://github.com/mahmoodlab/MCAT) 
with minor modifications. We thank the authors for their original work.
"""

import os
from argparse import Namespace

import numpy as np
import torch
from sksurv.metrics import concordance_index_censored

from loss_func import CoxSurvLoss, CrossEntropySurvLoss, NLLSurvLoss
from survival_model import cos_Surv
from utils import get_optim, get_split_loader, l1_reg_all, print_network


def train_loop_survival_coattn(
    epoch,
    model,
    loader,
    optimizer,
    n_classes,
    writer=None,
    loss_fn=None,
    reg_fn=None,
    lambda_reg=0.,
    gc=16
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (data_WSI, data_omic, label, event_time,
                    c) in enumerate(loader):
        data_WSI = data_WSI.to(device)
        data_omic = data_omic.to(device)
        label = label.type(torch.LongTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)

        hazards, S, Y_hat, A = model(x_path=data_WSI, x_omic=data_omic)
        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        if (batch_idx + 1) % 100 == 0:
            print(
                'batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}'
                .format(
                    batch_idx, loss_value + loss_reg, label.item(),
                    float(event_time), float(risk)
                )
            )
        loss = loss / gc + loss_reg
        loss.backward()

        if (batch_idx + 1) % gc == 0:
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)
    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool),
        all_event_times,
        all_risk_scores,
        tied_tol=1e-08
    )[0]
    print(
        'Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'
        .format(epoch, train_loss_surv, train_loss, c_index)
    )


def validate_survival_coattn(
    cur,
    epoch,
    model,
    loader,
    n_classes,
    early_stopping=None,
    monitor_cindex=None,
    writer=None,
    loss_fn=None,
    reg_fn=None,
    lambda_reg=0.,
    results_dir=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (data_WSI, data_omic, label, event_time,
                    c) in enumerate(loader):
        data_WSI = data_WSI.to(device)
        data_omic = data_omic.to(device)

        label = label.type(torch.LongTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)

        with torch.no_grad():
            hazards, S, Y_hat, A = model(
                x_path=data_WSI, x_omic=data_omic
            )  # return hazards, S, Y_hat, A_raw, results_dict

        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c, alpha=0)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.cpu().numpy()
        all_event_times[batch_idx] = event_time

        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg

    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool),
        all_event_times,
        all_risk_scores,
        tied_tol=1e-08
    )[0]

    if early_stopping:
        assert results_dir
        early_stopping(
            epoch,
            val_loss_surv,
            model,
            ckpt_name=os.path.join(
                results_dir, "s_{}_minloss_checkpoint.pt".format(cur)
            )
        )

        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


kwargs_default = {
    'task_type': 'survival',
    'mode': 'coattn',
    'alpha_surv': 0.0,
    'bag_loss': 'nll_surv',
    'reg_type': 'omic',
    'lambda_reg': 1e-5,
    'gc': 32,
    'n_classes': 4,
    'fusion': None,
    'opt': 'adam',
    'batch_size': 1,
    'dropout': 0.25,
    'lr': 2e-4,
    'reg': 1e-5,
    'testing': False,
    'weighted_sample': True,
    'early_stopping': False,
    'max_epochs': 20,
    'results_dir': None,
}


def train(datasets: tuple, cur: int, **kwargs):
    """
        train for a single fold
    """

    # show the training parameters
    print(kwargs)

    args = Namespace(**kwargs)

    writer = None
    print('\nTraining Fold {}!'.format(cur))

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split = datasets
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    print('\nInit loss function...', end=' ')
    if args.task_type == 'survival':
        if args.bag_loss == 'ce_surv':
            loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
        elif args.bag_loss == 'nll_surv':
            loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
        elif args.bag_loss == 'cox_surv':
            loss_fn = CoxSurvLoss()
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    if args.reg_type == 'omic':
        reg_fn = l1_reg_all
    else:
        reg_fn = None

    print('Done!')

    print('\nInit Model...', end=' ')
    args.fusion = None if args.fusion == 'None' else args.fusion

    model_dict = {
        'fusion': args.fusion,
        'n_classes': args.n_classes,
        'dropout': args.dropout
    }
    model = cos_Surv(**model_dict)

    # for param in model.coattn.parameters():
    #     param.requires_grad = False

    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda'))
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(
        train_split,
        training=True,
        testing=args.testing,
        weighted=args.weighted_sample,
        mode=args.mode,
        batch_size=args.batch_size
    )
    val_loader = get_split_loader(
        val_split,
        testing=args.testing,
        mode=args.mode,
        batch_size=args.batch_size
    )
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(
            warmup=0, patience=10, stop_epoch=20, verbose=True
        )
    else:
        early_stopping = None

    print('\nSetup Validation C-Index Monitor...', end=' ')
    monitor_cindex = Monitor_CIndex()
    print('Done!')

    for epoch in range(args.max_epochs):
        if args.task_type == 'survival':
            if args.mode == 'coattn':
                train_loop_survival_coattn(
                    epoch, model, train_loader, optimizer, args.n_classes,
                    writer, loss_fn, reg_fn, args.lambda_reg, args.gc
                )
                stop = validate_survival_coattn(
                    cur, epoch, model, val_loader, args.n_classes,
                    early_stopping, monitor_cindex, writer, loss_fn, reg_fn,
                    args.lambda_reg, args.results_dir
                )
            else:
                raise NotImplementedError

    torch.save(
        model.state_dict(),
        os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))
    )
    model.load_state_dict(
        torch.load(
            os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))
        )
    )
    results_val_dict, val_cindex = summary_survival(
        model, val_loader, args.n_classes
    )
    results_train_dict, train_cindex = summary_survival(
        model, train_loader, args.n_classes
    )

    print('Train c-Index: {:.4f}'.format(train_cindex))
    print('Val c-Index: {:.4f}'.format(val_cindex))
    return (results_train_dict, results_val_dict), (train_cindex, val_cindex)


def summary_survival(model, loader, n_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    test_loss = 0.

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data_WSI, data_omic, label, event_time,
                    c) in enumerate(loader):
        data_WSI = data_WSI.to(device)
        data_omic = data_omic.to(device)
        label = label.to(device)

        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            hazards, survival, Y_hat, Adict = model(
                x_path=data_WSI, x_omic=data_omic
            )

        risk = -torch.sum(survival, dim=1).cpu().numpy()
        risk = risk[0]
        event_time = event_time[0]
        c = c[0]
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c
        all_event_times[batch_idx] = event_time
        patient_results.update(
            {
                slide_id:
                    {
                        'slide_id': np.array(slide_id),
                        'risk': risk,
                        'disc_label': label.item(),
                        'survival': event_time,
                        'censorship': c
                    }
            }
        )

    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool),
        all_event_times,
        all_risk_scores,
        tied_tol=1e-08
    )[0]
    return patient_results, c_index


def get_attention(model, p_id, dataset):
    device = torch.device('cuda')
    try:
        p_train_idx = np.argwhere(dataset.slide_data.case_id.values == p_id)[0,
                                                                             0]
    except:
        return

    data_WSI, data_omic, label, event_time, c = dataset[p_train_idx]
    data_WSI = data_WSI.to(device)
    data_omic = data_omic.to(device)

    with torch.no_grad():
        hazards, survival, Y_hat, Adict = model(
            x_path=data_WSI, x_omic=data_omic
        )

    attention = Adict['coattn'].flatten().cpu().numpy()

    return attention


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


class Monitor_CIndex:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.best_score = None

    def __call__(self, val_cindex, model, ckpt_name:str='checkpoint.pt'):

        score = val_cindex

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        else:
            pass

    def save_checkpoint(self, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), ckpt_name)
