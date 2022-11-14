# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 10/27/2022
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.linalg import norm as torch_norm
from torch import distributed



class ArcFace(nn.Module):
    def __init__(self, cin, cout, s=32, m=0.5):
        super().__init__()
        self.m = m
        self.s = s
        self.cout = cout
        self.fc = nn.Linear(cin, cout, bias=False)

    def forward(self, x, label=None):
        if label is None:
            w_L2 = torch_norm(self.fc.weight.detach(), dim=1, keepdim=True)
            x_L2 = torch_norm(x, dim=1, keepdim=True)
            logit = F.linear(x / x_L2, self.fc.weight / w_L2)
        else:
            one_hot = F.one_hot(label, num_classes=self.cout)
            w_L2 = torch_norm(self.fc.weight.detach(), dim=1, keepdim=True)
            x_L2 = torch_norm(x, dim=1, keepdim=True)
            cos = F.linear(x / x_L2, self.fc.weight / w_L2)
            theta_yi = torch.acos(cos * one_hot)
            logit = torch.cos(theta_yi + self.m) * one_hot + cos * (1 - one_hot)
            logit = logit * self.s

        return logit


class DistCrossEntropyFunc(torch.autograd.Function):
    """
    CrossEntropy loss is calculated in parallel, allreduce denominator into single gpu and calculate softmax.
    Implemented of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, label: torch.Tensor):
        """ """
        batch_size = logits.size(0)
        # for numerical stability
        max_logits, _ = torch.max(logits, dim=1, keepdim=True)
        # local to global
        # distributed.all_reduce(max_logits, distributed.ReduceOp.MAX)
        logits.sub_(max_logits)
        logits.exp_()
        sum_logits_exp = torch.sum(logits, dim=1, keepdim=True)
        # local to global
        # distributed.all_reduce(sum_logits_exp, distributed.ReduceOp.SUM)
        logits.div_(sum_logits_exp)
        index = torch.where(label != -1)[0]
        # loss
        loss = torch.zeros(batch_size, 1, device=logits.device)
        loss[index] = logits[index].gather(1, label[index])
        distributed.all_reduce(loss, distributed.ReduceOp.SUM)
        ctx.save_for_backward(index, logits, label)
        return loss.clamp_min_(1e-30).log_().mean() * (-1)

    @staticmethod
    def backward(ctx, loss_gradient):
        """
        Args:
            loss_grad (torch.Tensor): gradient backward by last layer
        Returns:
            gradients for each input in forward function
            `None` gradients for one-hot label
        """
        (
            index,
            logits,
            label,
        ) = ctx.saved_tensors
        batch_size = logits.size(0)
        one_hot = torch.zeros(
            size=[index.size(0), logits.size(1)], device=logits.device
        )
        one_hot.scatter_(1, label[index], 1)
        logits[index] -= one_hot
        logits.div_(batch_size)
        return logits * loss_gradient.item(), None


class DistCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(DistCrossEntropy, self).__init__()

    def forward(self, logit_part, label_part):
        return DistCrossEntropyFunc.apply(logit_part, label_part)


if __name__ == '__main__':
    loss = ArcFace(128, 3)

    x = torch.rand(1, 128)
    y = torch.rand(1)

    l = loss(x, y)
