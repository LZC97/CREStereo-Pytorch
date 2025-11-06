import math
import torch

class LinearWarmupConstLinearDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warm_up: float = 0.02,
        const_range: float = 0.6,
        min_lr_rate: float = 0.05,
        base_lr: float = 1e-4,
        total_epoch: int = -1,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: the base optimizer
            warm_up: the linear warm up ratio in total epochs
            const_range: the const lr area in total epochs
            min_lr_rate: the ratio wrt base_lr of min lr
            base_lr: the learning rate of const range
            total_epoch: the total epochs to train
            last_epoch: the last epoch
        """

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        self.optimizer = optimizer

        self.warm_up = warm_up
        self.const_range = const_range
        self.min_lr_rate = min_lr_rate
        self.base_lr = base_lr
        self.total_epoch = total_epoch

        if self.total_epoch <= 0:
            raise ValueError(f"total_epoch must be a positive integer, got {self.total_epoch}")

        super().__init__(optimizer, last_epoch)

    def get_lr(self, epoch):
        # Ensure epoch is at least 1 for proper calculation
        if epoch < 1:
            epoch = 1

        warm_up_epochs = self.total_epoch * self.warm_up
        const_range_epochs = self.total_epoch * self.const_range

        if epoch <= warm_up_epochs:
            # Linear warmup: from min_lr_rate * base_lr to base_lr
            if warm_up_epochs > 0:
                lr = (1 - self.min_lr_rate) * self.base_lr / warm_up_epochs * epoch + self.min_lr_rate * self.base_lr
            else:
                lr = self.base_lr
        elif epoch <= const_range_epochs:
            # Constant phase
            lr = self.base_lr
        else:
            # Linear decay: from base_lr to min_lr_rate * base_lr
            lr = (self.min_lr_rate - 1) * self.base_lr / (
                (1 - self.const_range) * self.total_epoch
            ) * epoch + (1 - self.min_lr_rate * self.const_range) / (1 - self.const_range) * self.base_lr
        return [lr for _ in self.optimizer.param_groups]

    def step(self, epoch=None):
        if epoch is None and self.last_epoch < 0:
            epoch = 0
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))

        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:
            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr(epoch))):
                param_group, lr = data
                param_group["lr"] = lr
                self.print_lr(self.verbose, i, lr, epoch)


# Ref: https://github.com/pytorch/vision/pull/6555
class ConsinAnnealingWarmupRestartsWithDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int,
        T_mult: float = 1.0,
        eta_min: float = 0.0001,
        T_warmup: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: the base optimizer
            T_0: the base number of iterations for a cycle
            T_mult: the scaling factor for how much a cycle lasts
            eta_min: the minimum learning rate
            T_warmup: the number of linear warmup iterations
            gamma: the exponential decay factor for the maximum learning rate
            last_epoch: the last epoch
        """

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        self.optimizer = optimizer

        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_warmup = T_warmup
        self.gamma = gamma

        # iterations in current cycle
        self.T_cur = 0
        self._last_lr = 0
        self.N_cycle = 0
        self._C_cycle = 0

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs

        elif self.T_cur < self.T_warmup:
            return [(base_lr - self.eta_min) * self.T_cur / self.T_warmup + self.eta_min for base_lr in self.base_lrs]

        else:
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos(math.pi * (self.T_cur - self.T_warmup) / (self.T_i - self.T_warmup)))
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.N_cycle = self.N_cycle + 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    # reset current epoch and compute current cycle
                    self.T_cur = epoch % self.T_0
                    self.N_cycle = epoch // self.T_0
                else:
                    # compute through how many cycles we have exponentiated the cycle size
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.N_cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (self.T_mult - 1) - self.T_warmup
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:
            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group["lr"] = lr
                self.print_lr(self.verbose, i, lr, epoch)

        self.base_lrs = [group["initial_lr"] * (self.gamma**self.N_cycle) for group in self.optimizer.param_groups]
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
