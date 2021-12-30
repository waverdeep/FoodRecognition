import torch.nn as nn


def set_criterion(name, params=None):
    if name == 'MSELoss':
        return nn.MSELoss()
    elif name == 'L1Loss':
        return nn.L1Loss()
    elif name == 'NLLLoss':
        return nn.NLLLoss()
    elif name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()

