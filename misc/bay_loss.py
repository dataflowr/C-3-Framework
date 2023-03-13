import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss
from torch.nn.modules import Module
from post_prob import Post_Prob
from bay_loss_trainer import parse_args

import numpy as np

class Bay_Loss(_Loss):
    def __init__(self, use_background, device):
        super(Bay_Loss, self).__init__()
        self.device = device
        self.use_bg = use_background

    def forward(self, prob_list, target_list, pre_density):
        loss = 0
        for idx, prob in enumerate(prob_list):  # iterative through each sample
            if prob is None:  # image contains no annotation points
                pre_count = torch.sum(pre_density[idx])
                target = torch.zeros((1,), dtype=torch.float32, device=self.device)
            else:
                N = len(prob)
                if self.use_bg:
                    target = torch.zeros((N,), dtype=torch.float32, device=self.device)
                    target[:-1] = target_list[idx]
                else:
                    target = target_list[idx]
                pre_count = torch.sum(pre_density[idx].view((1, -1)) * prob, dim=1)  # flatten into vector

            loss += torch.sum(torch.abs(target - pre_count))
        loss = loss / len(prob_list)

        return loss

if __name__ == "__main__":
    args=parse_args()
    data = torch.zeros(1, 1, 1, 1)
    data += 0.001
    target = torch.zeros(1, 1, 1, 1)
    data = Variable(data, requires_grad=True)
    target = Variable(target)
    device=torch.device("cpu")
    post_prob=Post_Prob(args.sigma,
                        args.crop_size,
                        args.downsample_ratio,
                        args.background_ratio,
                        args.use_background,
                        device)
    prob_list = post_prob(points, st_sizes)
    model = Bay_Loss(True,device)
    loss = model(post_prob,data, target)
    loss.backward()
    print(loss)
    print(data.grad)