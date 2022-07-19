import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0
            module.track_running_stats = False
        if isinstance(module, nn.SyncBatchNorm):
            module.need_sync = False

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum
            module.track_running_stats = True
        if isinstance(module, nn.SyncBatchNorm):
            module.need_sync = True

    model.apply(_enable)
