import numpy as np
import torch
import os
import datetime


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, epoch, save_path):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, save_path)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, save_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, save_path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, os.path.join(save_path, "checkpoint.pth.tar"))
        self.val_loss_min = val_loss


class Loss(object):
    def __init__(self, lossList=[]):
        self.lossList = lossList

    def update(self, **kwargs):
        if self.lossList == []:
            for key, value in kwargs.items():
                setattr(self, f"{key}_this_epoch", [])
                setattr(self, f"{key}_history", [])
                self.lossList.append(key)
        for key, value in kwargs.items():
            assert key in self.lossList
            getattr(self, f"{key}_this_epoch").append(value.item())

    def update_epoch(self):
        avg = self.get_average_epoch()
        for n in self.lossList:
            getattr(self, f"{n}_history").append(avg[n])
        self.new_epoch()

    def new_epoch(self):
        for n in self.lossList:
            setattr(self, f"{n}_this_epoch", [])

    def get_average_epoch(self):
        avg = {}
        for n in self.lossList:
            avg[n] = np.average(getattr(self, f"{n}_this_epoch"))
        return avg


class Metric(object):
    def __init__(self, metricList=["l1_left", "l1_right"]):
        self.metricList = metricList
        for n in metricList:
            setattr(self, f"{n}_this_epoch", [])
            setattr(self, f"{n}_history", [])

    def update(self, input):
        for key in input:
            assert key in self.metricList
            getattr(self, f"{key}_this_epoch").append(input[key].item())

    def update_epoch(self):
        avg = self.get_average_epoch()
        for n in self.metricList:
            getattr(self, f"{n}_history").append(avg[n])
        self.new_epoch()

    def new_epoch(self):
        for n in self.metricList:
            setattr(self, f"{n}_this_epoch", [])

    def get_average_epoch(self):
        avg = {}
        for n in self.metricList:
            avg[n] = np.average(getattr(self, f"{n}_this_epoch"))
        return avg


class TimeRecorder(object):
    def __init__(self, total_epoch, iter_per_epoch):
        self.total_epoch = total_epoch
        self.iter_per_epoch = iter_per_epoch
        self.start_train_time = datetime.datetime.now()
        self.start_epoch_time = datetime.datetime.now()
        self.t_last = datetime.datetime.now()

    def get_iter_time(self, epoch, iter):
        dt = (datetime.datetime.now() - self.t_last).__str__()
        self.t_last = datetime.datetime.now()
        remain_time = self.cal_remain_time(epoch, iter, self.total_epoch, self.iter_per_epoch)
        end_time = (datetime.datetime.now() + datetime.timedelta(seconds=remain_time)).strftime("%Y-%m-%d %H:%S:%M")
        remain_time = datetime.timedelta(seconds=remain_time).__str__()
        return dt, remain_time, end_time

    def cal_remain_time(self, epoch, iter, total_epoch, iter_per_epoch):
        t_used = (datetime.datetime.now() - self.start_train_time).total_seconds()
        time_per_iter = t_used / (epoch * iter_per_epoch + iter + 1)
        remain_iter = total_epoch * iter_per_epoch - (epoch * iter_per_epoch + iter + 1)
        remain_time_second = time_per_iter * remain_iter
        return remain_time_second


if __name__ == '__main__':
    l = Loss(['l1' ,'lpips'])
    l.update(l1=torch.tensor([1]), lpips=torch.tensor([2]), total=torch.tensor([3]))
    l.update(l1=torch.tensor([4]), lpips=torch.tensor([5]), total=torch.tensor([6]))
    l.update_epoch()