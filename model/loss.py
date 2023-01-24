# AI/ML Framework
import torch.nn.functional as F
import torch.nn as nn


def cross_entropy_loss(output, target):
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(output, target)
    return loss