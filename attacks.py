import argparse 
import os, random
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from datasets import concatenate_datasets, load_dataset

import scipy.stats as stats
from sklearn import metrics
import matplotlib.pyplot as plt

from metrics import MetricsLogger, evaluate

# we assume the adversary has black-box access to the model's outputs on a given datapoint, 
# but including the full output distribution and loss values with respect to a target label.

# we also assume the adversary has access to some data of a similar distribution to 
# what the model was trained on; knows the model's architecture and some details about
# the training regime, so that they can train 'shadow models' of the same architecture
# and with similar learning rate, num_epochs and so on.

def loss_threshold(target_model, 
                   in_data, 
                   out_data, 
                   metric='rescaled logit', 
                   num_samples=None, # number of samples of each of members/nonmembers to compare
                   batch_size=64,
                   plot=True):
    """given a trained model instance, and some examples of member and non-member data for that model,
    runs the Yeom et al. loss threshold technique and plot the distributions of member and non-member
    loss/confidence/logit scores"""

    if num_samples is None:
        # if number of examples is not given,
        # just take as many as possible from the data:
        num_samples = min([len(in_data), len(out_data)])//2
    
    member_indices = np.random.choice(len(in_data), num_samples, replace=False)
    nonmember_indices = np.random.choice(len(out_data), num_samples, replace=False)

    member_dataloader = DataLoader(in_data.select(member_indices), 
                    batch_size=batch_size, 
                    collate_fn=target_model.collate_fn, 
                    shuffle=False)
    nonmember_dataloader = DataLoader(out_data.select(nonmember_indices), 
                    batch_size=batch_size, 
                    collate_fn=target_model.collate_fn,
                    shuffle=False)

    assert metric in ['loss', 'confidence', 'rescaled logit'], "Attack must be based on ['loss', 'confidence', 'rescaled logit']"
    target_model.eval()
    mia_loss_fn =  torch.nn.CrossEntropyLoss(reduction='none') 

    
    member_cls_preds, nonmember_cls_preds, member_data, nonmember_data = [], [], [], []

    # first for members, then non-members:
    for _cls_preds, _data, _loader in zip([member_cls_preds, nonmember_cls_preds], 
                                          [member_data, nonmember_data], 
                                          [member_dataloader, nonmember_dataloader]):

        # loop over the chosen data samples:
        for _b, _batch in enumerate(tqdm(_loader)):
            x, targets = _batch
            with torch.no_grad():

                # record the model's output (whatever metric we use)
                pred_logits = target_model(x)
                pred_ints = torch.max(pred_logits, dim=-1).indices
                if metric == 'loss':
                    # raw loss
                    _loss = mia_loss_fn(pred_logits, targets)
                    _data.append(-_loss)
                elif metric == 'confidence':
                    # exp of neg loss
                    _loss = mia_loss_fn(pred_logits, targets)
                    _conf = torch.exp(-_loss)
                    _data.append(_conf)
                elif metric == 'rescaled logit':
                    # something like softmax??
                    
                    _preds = pred_logits - torch.max(pred_logits, dim=-1, keepdims=True).values
                    _preds = torch.exp(_preds)
                    _preds = _preds / torch.sum(_preds, dim=-1,keepdims=True)
                    _bs = _preds.size(0)

                    _y_true = _preds[np.arange(_bs), targets[:_bs]]
                    _preds[np.arange(_bs), targets[:_bs]] = 0
                    _y_wrong = torch.sum(_preds, dim=-1)
            
                    _rescaled_logit = torch.log(_y_true + 1e-45) - torch.log(_y_wrong + 1e-45)

                    _data.append(_rescaled_logit)
                    
                _cls_preds.append(pred_ints)
                
    member_data = torch.cat(member_data, dim=0).cpu().numpy()
    nonmember_data = torch.cat(nonmember_data, dim=0).cpu().numpy()
    member_cls_preds  = torch.cat(member_cls_preds, dim=0).cpu().numpy()
    assert len(member_data) == len(nonmember_data) == len(member_cls_preds) == num_samples

    # build dataset of member and nonmember datapoints
    X = np.concatenate([member_data, nonmember_data], axis=0)
    y = [1] * num_samples + [0] * num_samples

    fpr, tpr, thresholds = metrics.roc_curve(y, X)
    auc = metrics.roc_auc_score(y, X)
    bal_acc = np.max(1-(fpr+(1-tpr))/2)
    tpr_gfpr = tpr[np.where(fpr<.001)[0][-1]]
    print(f"MIA Results: (based on metric: '{metric}')")
    print(f"{'AUC':>25} = {auc:.4f}")
    print(f"{'(Max) Balanced Accuracy':>25} = {bal_acc:.4f}")
    print(f"{'TPR@FPR=0.01':>25} = {tpr_gfpr:.4f}")

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

        ax1.plot(fpr, fpr, c=[0.8]*3, linestyle=':', label="Chance level, AUC=0.5")
        ax1.plot(fpr, tpr, label=f"AUC={auc:.4f}")
        ax1.set_xlabel('False Positive Rate'); ax1.set_ylabel('True Positive Rate')
        # ax1.set_xscale("log"); ax1.set_yscale("log") # why???
        ax1.title.set_text(f"ROC curve from Yeom et al.")
        ax1.legend(loc='upper left')

        ax2.hist(X[:num_samples], bins='auto', alpha=0.5, label='member')
        ax2.hist(X[num_samples:], bins='auto', alpha=0.5, label='non-member')
        ax2.title.set_text(f"Distribution of {metric} values")
        ax2.legend(loc='upper right')
    
        plt.tight_layout()
        plt.show()
    return X, y, member_cls_preds, bal_acc, auc



def online_lira_on_datapoint(target_model,
                             target_datapoint,
                             distribution_data,
                             shadow_train_size=4000,
                             # model_out_data,
                             num_shadow=8, # number of shadow models to train
                             ### hyperparameters for shadow training:
                             batch_size=64, 
                             lr=1e-4,
                             num_epochs=5,
                             metric='rescaled logit',  # score to use for comparing non/member distributions
                             plot=True):
    """runs the online LiRA attack to classify a specific datapoint
    as a member or non-member of the target model's training data
    by training N shadow models, half with and half without that datapoint
    using some data sampled from the distribution of the target model"""
    
    shadow_train_data = distribution_data

    # print(f'Running online LiRA against training datapoint: {target_datapoint}')
    in_scores, out_scores = [], []
    
    # train shadow models:
    for s in range(num_shadow):
        
        # the first half of our shadow models include the target datapoint in training;
        # the second half do not include it:
        in_model = s < num_shadow // 2
        
        
        seed_all(s)
        model_label = "in" if in_model else "out"
        print(f'  {s}/{num_shadow} Training shadow model S{s}_{model_label}:')
        
        shadow_model = target_model.__class__(num_classes=target_model.num_classes, pretrained=target_model.pretrained)

        if in_model:
            # print(f'  (containing the target data point)')
            _shadow_inds = np.random.choice(len(shadow_train_data), 
                                            shadow_train_size-1, 
                                            replace=False)
            
            _shadow_dataset = concatenate_datasets([shadow_train_data.select(_shadow_inds),
                                                    # add the target point as well:
                                                   target_datapoint])
        else:
            # print(f'  (NOT containing the target data point)')
            _shadow_inds = np.random.choice(len(shadow_train_data), 
                                            shadow_train_size, 
                                            replace=False)
            _shadow_dataset = shadow_train_data.select(_shadow_inds)
        shadow_train_loader = DataLoader(_shadow_dataset,
                                         batch_size=batch_size,
                                         collate_fn=shadow_model.collate_fn,
                                         shuffle=True)

        train_model(s, shadow_model, shadow_train_loader, None, lr, num_epochs)

        mia_loader = DataLoader(target_datapoint,
                    batch_size=1,
                    collate_fn=target_model.collate_fn,
                    shuffle=True)
        [_dist_score] = get_score(shadow_model, [mia_loader], metric=metric)

        if in_model:
            in_scores.append(_dist_score)
        else:
            out_scores.append(_dist_score)
    assert len(in_scores) == len(out_scores) == num_shadow // 2
    in_scores = np.concatenate(in_scores, axis=0)
    out_scores = np.concatenate(out_scores, axis=0)

    # 2. Fitting Gaussians to IN/OUT distribution and compute the likelihood ratio
    mu_in, std_in = np.mean(in_scores), np.std(in_scores)
    mu_out, std_out = np.mean(out_scores), np.std(out_scores)

    _miatest_loader = DataLoader(target_datapoint,
                    batch_size=1,
                    collate_fn=target_model.collate_fn,
                    shuffle=True)
    [score] = get_score(target_model, [_miatest_loader], metric=metric)

    lk_in = -stats.norm.logpdf(score, mu_in, std_in+1e-30)[0]
    lk_out = -stats.norm.logpdf(score, mu_out, std_out+1e-30)[0]
    lira = (lk_in / (lk_in + lk_out))

    print(f'{lk_in=}')
    print(f'{lk_out=}')
    print(f'LiRA membership probability: {lira:.1%}')

    return lira
                
def offline_lira(target_model,
                 distribution_data,
                 member_test_data,
                 nonmember_test_data,
                 num_sample_pairs=400,
                 shadow_train_size=4000,
                 num_shadow=16,
                 ### hyperparameters for shadow training:
                 batch_size=64, 
                 lr=1e-4,
                 num_epochs=5,
                 metric='rescaled logit',  # score to use for comparing non/member distributions
                 use_global_var=False,  # global variance instead of default per-example variance
                 use_cached_shadows=False, # load saved model checkpoints if already trained
                 plot=True):

    print(f'Running offline LiRA, validating on {num_sample_pairs} member datapoints and {num_sample_pairs} non-member datapoints')

    assert num_shadow % 2 == 0, "requires an even number of shadow models to train on in/out data"
    num_samples = num_sample_pairs*2
    
    # where to save shadow model checkpoints:
    SAVE_ROOT = 'save/models/lira_shadow/'
    if not os.path.exists(SAVE_ROOT):
        print(f'Creating directory: {SAVE_ROOT}')
        os.makedirs(SAVE_ROOT)    

    model_dir = os.path.join(SAVE_ROOT, "offline")
    if os.path.exists(model_dir):
        for _pt in os.listdir(model_dir):
            os.remove(os.path.join(model_dir, _pt))
    else:
        os.makedirs(model_dir)

    member_indices = np.random.choice(len(member_test_data), num_sample_pairs, replace=False)
    nonmember_indices = np.random.choice(len(nonmember_test_data), num_sample_pairs, replace=False)

    member_dataloader = DataLoader(member_test_data.select(member_indices),
                batch_size=batch_size,
                collate_fn=target_model.collate_fn,
                shuffle=True)
    nonmember_dataloader = DataLoader(nonmember_test_data.select(nonmember_indices),
                    batch_size=batch_size,
                    collate_fn=target_model.collate_fn,
                    shuffle=True)

    shadow_train_data = distribution_data
    
    # 1. Randomly Sample N subsets of D DISJOINT from D_train to train a N shadow models
    if not use_cached_shadows:
        for s in range(num_shadow):
            print(f'Training shadow model {s}/{num_shadow}:')
            seed_all(s)

            # print("="*20)

            shadow_model = target_model.__class__(num_classes=target_model.num_classes, pretrained=target_model.pretrained)

            _inds = np.random.choice(len(shadow_train_data), shadow_train_size, replace=False)
            _shadow_data = shadow_train_data.select(_inds)
            shadow_train_loader = DataLoader(_shadow_data,
                            batch_size=batch_size,
                            collate_fn=shadow_model.collate_fn,
                            shuffle=True)

            train_model(s, shadow_model, shadow_train_loader, None, lr=lr, num_epochs=num_epochs)

            model_path = os.path.join(model_dir, f"model_{s}.pt")
            print(f'Saving shadow model checkpoint to {model_path}')
            torch.save(shadow_model.state_dict(), model_path)
            # np.save(os.path.join(model_path, f"index_{s}.npy"), _inds)
    else:
        assert len(os.listdir(model_dir)) == num_shadow

    # 2. Having trained N shadow models offline:
    #   for each example (x,y): compute loss(f_i,x) from each shadow model f_i
    #   construct OUT Gaussian distribution from N loss values
    #   compute the probability of (x, y) is a non-member
    shadow_scores = []

    for s in range(num_shadow):
        shadow_model = target_model.__class__(num_classes=target_model.num_classes, pretrained=target_model.pretrained)
        shadow_model.load_state_dict(torch.load(os.path.join(model_dir, f'model_{s}.pt')))

        [_in_score, _out_score] = get_score(shadow_model, [member_dataloader, nonmember_dataloader], metric=metric)

        shadow_scores.append(np.concatenate([_in_score, _out_score], axis=0))

    mu_out = np.mean(shadow_scores, axis=0)
    if use_global_var:
        sigma_out = np.repeat([np.std(shadow_scores)], num_samples)
    else:
        sigma_out = np.std(shadow_scores, axis=0)

    [miatest_member, miatest_nonmember] = get_score(target_model, 
                                          [member_dataloader, nonmember_dataloader], 
                                          metric=metric)
    miatest_both = np.concatenate([miatest_member, miatest_nonmember], axis=0)

    lk_out = stats.norm.logpdf(miatest_both, mu_out, sigma_out+1e-30)
    X = lk_out  # one-sided OUT probability (likelihood)
    y = [0] * num_sample_pairs + [1] * num_sample_pairs
    

    # 3. Thresholding and evaluate the attacks with TPR@FPR=0.01
    fpr, tpr, thresholds = metrics.roc_curve(y, X)
    auc = metrics.auc(fpr, tpr)
    bal_acc = np.max(1 - (fpr + (1-tpr))/2)
    print(f"ACC={bal_acc:.4f}, AUC={auc:.4f}, TPR@FPR=0.01={tpr[np.where(fpr<0.01)[0][-1]]}")

    if plot:
        plt.plot(fpr, fpr, '-', label="Chance level, AUC=0.5")
        plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")

        plt.xlabel('FPR'); plt.ylabel('TPR'); 
        # plt.xscale("log"); plt.yscale("log") # why???
        plt.title(f"Offline LiRA ROC curve"); plt.legend(loc='upper left'); plt.tight_layout()
        # plt.savefig(os.path.join(SAVE_ROOT, f'lira_{save_name}_{num_sample_pairs}.png'))

        plt.show()
    return X, y
    


#### helper functions for lira:

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train_model(idx, model, train_loader, val_loader, lr, num_epochs):

    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    logger = MetricsLogger(steps_per_epoch=len(train_loader))

    for e in range(num_epochs):

        pbar = tqdm(train_loader, ncols=100)
        for b, batch in enumerate(pbar):
            x, targets = batch

            # pass data through model:
            opt.zero_grad()
            pred_logits = model(x)
            loss = loss_fn(pred_logits, targets)

            # update weights:
            loss.backward()
            opt.step()

            # compute train accuracy:
            pred_ints = pred_logits.max(axis=1).indices
            acc = (targets == pred_ints).float().mean()

            # record loss/accuracy and update progress bar:
            logger.log(e, b, {'train_loss': loss.item(), 'train_acc': acc.item()})
            pbar.set_description(f"E:{e}/{num_epochs} | batch_loss:{loss:<6.3f} batch_acc:{acc:<6.1%}") 

        # evaluate after each epoch:
        if val_loader is not None:
            # don't validate, not necessary for shadow models
            val_loss, val_acc = evaluate(model, val_loader, loss_fn, set_eval=False, ncols=100) 
            logger.log(e, b, {'val_loss': val_loss, 'val_acc': val_acc})
    
def get_score(model, dataloaders, metric='loss'):
    assert metric in ['loss', 'prob', 'rescaled logit'], "Attack must be based on ['loss', 'prob', 'rescaled logit']"

    # model.eval()
    loss_fn =  torch.nn.CrossEntropyLoss(reduction='none')

    data_lst = []
    for _loader in dataloaders:
        _data = []
        for _b, _batch in enumerate(_loader):
            x, targets = _batch
            with torch.no_grad():
                pred_logits = model(x)
                _loss = loss_fn(pred_logits, targets)
                _prob = torch.exp(-_loss)

                _preds = pred_logits - torch.max(pred_logits, dim=-1, keepdims=True).values
                _preds = torch.exp(_preds)
                _preds = _preds / torch.sum(_preds, dim=-1,keepdims=True)

                _bs = _preds.size(0)
                _y_true = _preds[np.arange(_bs), targets[:_bs]]
                _preds[np.arange(_bs), targets[:_bs]] = 0
                _y_wrong = torch.sum(_preds, dim=-1)

                _rescaled_logits = torch.log(_y_true + 1e-45) - torch.log(_y_wrong + 1e-45)

                if metric == 'loss':
                    _data.append(_loss)
                elif metric == 'prob':
                    _data.append(_prob)
                elif metric == 'rescaled logit':
                    _data.append(_rescaled_logits)
        data_lst.append(_data)

    data_lst = [torch.cat(_data, dim=0).cpu().numpy() for _data in data_lst]
    return data_lst

def sweep(tpr, fpr, thresholds, n_pos, n_neg):
    """
        tpr, fpr, thresholds: 1darray
    """
    assert all([isinstance(_arr, np.ndarray) for _arr in [tpr, fpr, thresholds]])
    assert len(tpr) == len(fpr) == len(thresholds)

    tp = tpr * n_pos; fn = n_pos - tp
    tn = (1 - fpr) * n_neg; fp = n_neg - tn
    acc = (tp + tn) / (n_pos + n_neg)

    pre = tp / (tp + fp); rec = tp / (tp + fn)
    f1 = (2 * pre * rec) / (pre + rec)

    best_thresh = thresholds[np.argmax(acc)]
    return
