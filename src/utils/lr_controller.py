import numpy as np
from keras import backend as K
import warnings


class ReduceLROnPlateau():
    def __init__(self, model, curmonitor=np.Inf, factor=0.1, patience=10, mode='min',
                 min_delta=1e-4, cooldown=0, min_lr=0, verbose=1,
                 **kwargs):

        self.curmonitor = curmonitor
        if factor > 1.0:
            raise ValueError('ReduceLROnPlateau does not support a factor > 1.0.')

        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.model = model
        self.verbose = verbose
        self.monitor_op = None
        self._reset()

    def _reset(self):
        if self.mode == 'min':
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def update_monitor(self, curmonitor):
        self.curmonitor = curmonitor

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, curmonitor):
        curlr = K.get_value(self.model.optimizer.lr)
        self.curmonitor = curmonitor
        if self.curmonitor is None:
            warnings.warn('errro input of monitor', RuntimeWarning)
        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(self.curmonitor, self.best):
                self.best = self.curmonitor
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = float(K.get_value(self.model.optimizer.lr))
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        K.set_value(self.model.optimizer.lr, new_lr)
                        if self.verbose > 0:
                            print('\nEpoch %05d: ReduceLROnPlateau reducing '
                                  'learning rate to %s.' % (epoch + 1, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0
        return curlr

    def in_cooldown(self):
        return self.cooldown_counter > 0