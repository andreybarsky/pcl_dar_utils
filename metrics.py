import torch
import numpy as np
import pandas as pd

from matplotlib import colormaps
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from tqdm import tqdm
from math import ceil

class MetricsLogger():
    """helper class for training and validation metrics"""
    def __init__(self, steps_per_epoch: int=None):
        self.steps_per_epoch = steps_per_epoch

        self.metrics_dict = {}

        self.recorded_metrics = set()
        self.recorded_steps = set()

    def log(self, 
            epoch:int, 
            step:int, # relative to epoch,
            metrics: dict):
        """accepts the current epoch and training step (within that epoch)
        and a dict of metrics (expecting train_loss, train_acc, val_loss, val_acc, etc.)
        and records to the logger for later plotting"""
        
            # train_loss:float=None, 
            # train_acc: float=None, 
            # val_loss:  float=None, 
            # val_acc:   float=None):

        if self.steps_per_epoch is not None:
            global_step = (epoch * self.steps_per_epoch) + step
        else:
            global_step = epoch ## TODO: find a workaround?

        for key, val in metrics.items():
            if key not in self.metrics_dict:
                self.metrics_dict[key] = {}
                self.recorded_metrics.add(key)
            self.metrics_dict[key][global_step] = val

        self.recorded_steps.add(global_step)


    def __getattr__(self, key):
        ### retrieve dict values as attributes
        if key in self.metrics_dict:
            return self.metrics_dict[key]
        else:
            raise ValueError
    
    def plot(self, smoothing=True, size=100):
        # smoothing is currently broken - I will just switch to pandas to make it easier

        total_steps = max(self.recorded_steps)
        if self.steps_per_epoch is not None:
            total_epochs = ceil(total_steps / self.steps_per_epoch)
        else:
            total_epochs = 1

        # assume that we have training loss and accuracy:
        
        if 'train_loss' in self.recorded_metrics and 'train_acc' in self.recorded_metrics:
            
            train_loss = pd.Series(self.train_loss)
            train_acc = pd.Series(self.train_acc)
            
            if smoothing:
                # exponential moving average:
                
                train_loss = train_loss.ewm(alpha=0.2).mean()
                train_acc = train_acc.ewm(alpha=0.2).mean()

        train_x = self.train_loss.keys()
        assert list(train_x) == sorted(train_x)
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(size//11, 6))
        
        # show epoch boundaries:
        if total_epochs > 1:
            for e in range(1,total_epochs):
                for ax in (ax1, ax2):
                    ax.axvline((self.steps_per_epoch*e), c=[0.8]*3, linestyle=':')

        # consistent colormaps for train and val:
        train_color = colormaps['Oranges'](0.5)
        # val_color = colormaps['Blues'](0.7)
        val_color = train_color
        
        # plot loss:
        
        if 'train_loss' in self.recorded_metrics:
            ax1.plot(train_x, train_loss, c=train_color, label='Train')

        if 'val_loss' in self.recorded_metrics:
            vloss_x = self.val_loss.keys()
            vloss_y = self.val_loss.values() 
            ax1.plot(vloss_x, vloss_y, c=val_color, linestyle='--', marker='o', label='Valid')
        ax1.set_title('Loss')
        ax1.legend()

        # plot accuracy:
        if 'train_acc' in self.recorded_metrics:
            ax2.plot(train_x, train_acc, c=train_color, label='Train')
        
        if 'val_acc' in self.recorded_metrics:
            vacc_x = self.val_acc.keys()
            vacc_y = self.val_acc.values() 
            ax2.plot(vacc_x, vacc_y, c=val_color, linestyle='--', marker='o', label='Valid')
        ax2.set_title('Accuracy')
        # format as percentage:
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax2.yaxis.tick_right()
        ax2.legend()

        plt.tight_layout()
        plt.show()


    def plot_fl(self, clients, smoothing=True, size=100):
        """display training curves for server and client machines in a FL setting.
        assumes this method is being called from the centralized logger, and accepts
        the loggers of all involved client models."""

        # get logger objects from client list:
        client_loggers = [cl.metrics for cl in clients]
        
        epoch_boundaries = list(self.val_loss.keys())
        if epoch_boundaries[0] != 0: # don't assume that initial validation was done
            epoch_boundaries = [0] + epoch_boundaries
        global_epoch_size = epoch_boundaries[1]
        
        total_epochs = len(epoch_boundaries) - 1

        # place grey lines at epoch boundaries:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(size//11, 6))
        for ep_step in epoch_boundaries:
            for ax in (ax1, ax2):
                ax.axvline((ep_step), c=[0.8]*3, linestyle=':')

        # setup colours for plots:
        num_clients = len(client_loggers)
        client_shades = np.linspace(0.2, 0.5, num_clients)
        train_cmap = colormaps['Oranges']
        train_colors = [train_cmap(shade) for shade in client_shades]
    
        # val_cmap = colormaps['Blues']
        val_cmap = colormaps['Oranges']
        val_colors = [val_cmap(shade) for shade in client_shades]
        
        # plot training loss for each client:
        for cid, clog in enumerate(client_loggers):

            train_loss = pd.Series(clog.train_loss)
            train_acc = pd.Series(clog.train_acc)

            # train_x = clog.train_loss.keys()
            # we need to correct the x-axis for each client
            # to account for gaps in the local train step (compared to the global)
            local_epoch_size = clog.steps_per_epoch
            train_x = []
            gap_train_loss, gap_train_acc = [], []
            for e in range(total_epochs):
                epoch_x = list(range(global_epoch_size*e, global_epoch_size*e + local_epoch_size))
                epoch_loss = list(train_loss)[e*local_epoch_size : (e+1)*local_epoch_size]
                epoch_acc = list(train_acc)[e*local_epoch_size : (e+1)*local_epoch_size]
                if smoothing: # here we smooth every epoch separately, to avoid smearing effects
                    # exponential moving average:
                    epoch_loss = list(pd.Series(epoch_loss).ewm(alpha=0.05).mean())
                    epoch_acc = list(pd.Series(epoch_acc).ewm(alpha=0.05).mean())
                
                train_x.extend(epoch_x)
                gap_train_loss.extend(epoch_loss)
                gap_train_acc.extend(epoch_acc)
                
                # fill with blank gaps for the rest of the global epoch:
                for lst in [train_x, gap_train_loss, gap_train_acc]:
                    lst.extend([None]*(global_epoch_size - local_epoch_size))
            train_loss = gap_train_loss
            train_acc = gap_train_acc
            ###
                  
            tloss_label = 'Local train loss' if cid == num_clients-1 else ''
            tacc_label = 'Local train acc' if cid == num_clients-1 else ''
            ax1.plot(train_x, train_loss, c=train_colors[cid], label=tloss_label)
            ax2.plot(train_x, train_acc, c=train_colors[cid], label=tacc_label)

            # val_x = clog.val_loss.keys()
            val_x = [(e+1)*global_epoch_size for e in range(total_epochs)]
            
            val_loss = clog.val_loss.values() 
            val_acc = clog.val_acc.values() 
            vloss_label = 'Local val loss' if cid == num_clients-1 else ''
            vacc_label = 'Local val acc' if cid == num_clients-1 else ''
            ax1.plot(val_x, val_loss, c=val_colors[cid], marker='o', linestyle='--', label=vloss_label)
            ax2.plot(val_x, val_acc, c=val_colors[cid], marker='o', linestyle='--', label=vacc_label)

        # now add global validation on top:
        global_x = epoch_boundaries
        global_loss = self.val_loss.values()
        global_acc = self.val_acc.values()
     
        ax1.plot(global_x, global_loss, c=colormaps['Blues'](0.7), marker='o', linestyle='--', label=f'Global val loss')
        ax2.plot(global_x, global_acc, c=colormaps['Blues'](0.7), marker='o', linestyle='--', label=f'Global val acc')

        ax1.set_title('Loss')
        ax1.legend()
        ax2.set_title('Accuracy')
        # format as percentage:
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax2.yaxis.tick_right()
        ax2.legend()
        plt.show()

def evaluate_batch(model: torch.nn.Module,
                   val_loader: torch.utils.data.DataLoader,
                   loss_fn: torch.nn.modules.loss._Loss):
    """evaluate a model's predictions on one batch of the validation set"""
    with torch.no_grad():
        try:
            val_x, val_y = next(val_loader.iterator)
        except:# StopIteration:
            # if we've hit the end of the validation data,
            # just loop back around again
            val_loader.iterator = iter(val_loader)
            val_x, val_y = next(val_loader.iterator)
        val_out = model(val_x)
    val_loss = loss_fn(val_out, val_y).item()
    val_acc = (val_out.argmax(axis=1) == val_y).float().mean().item()
    return val_loss, val_acc


def evaluate(model: torch.nn.Module,
             val_loader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.modules.loss._Loss,
             set_eval=False, **kwargs):
    """evaluate a model's predictions on the entirety of the validation set"""
    losses, accs = [], []
    val_loader.iterator = iter(val_loader)

    if set_eval:
        # put model into eval mode so it doesn't update the batch norm stats:
        model.eval()
        # off by default; has some odd interactions in the federated setting
    
    pbar = tqdm(val_loader, **kwargs)
    for b, vbatch in enumerate(pbar):

        vbatch_x, vbatch_y = vbatch
        vbatch_out = model(vbatch_x)
        losses.append(loss_fn(vbatch_out, vbatch_y).item())
        accs.append((vbatch_out.argmax(axis=1) == vbatch_y).float().mean().item())

        if b != len(pbar)-1:
            pbar.set_description(f"Val | val_batch_loss:{losses[-1]:<6.3f} val_batch_acc:{accs[-1]:<6.1%}")

        else:
            val_loss = np.mean(losses)
            val_acc = np.mean(accs)
            pbar.set_description(f"Val | val_loss:{val_loss:<6.3f} val_acc:{val_acc:<6.1%}")

    if set_eval:
        # put model back into train mode:
        model.train()
    return val_loss, val_acc

def tensor_bytes(tensor):
    """calculate the size of a torch tensor
    for logging the total communication cost in the federated setting"""
    return tensor.element_size() * tensor.nelement()

def format_bytes(size: int, metric=True, decimal_places=1):
    """formats number of bytes in human-readable SI units. (MB, GB, etc.)
    or in binary units if metric=False (MiB, GiB, etc.)"""
    if metric:
        power = 1000
        names = {0 : 'B', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB', 5: 'PB'}
    else: # binary units; technically "discouraged"
        power = 1024
        names = {0 : 'B', 1: 'KiB', 2: 'MiB', 3: 'GiB', 4: 'TiB', 5: 'PiB'}        
    n = 0
    while size > power:
        size /= power
        n += 1
    return f'{size:.{decimal_places}f} {names[n]}'