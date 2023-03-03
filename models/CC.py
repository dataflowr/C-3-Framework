import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import imp


class CrowdCounter(nn.Module):
    def __init__(self, gpus, model_name):
        super(CrowdCounter, self).__init__()

        net = getattr(
            imp.load_source("network_src", "models/SCC_Model/" + model_name + ".py"),
            model_name,
        )

        self.CCN = net()
        if len(gpus) > 1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
            self.CCN = self.CCN.cuda()
        self.loss_mse_fn = nn.MSELoss().cuda()

    @property
    def loss(self):
        return self.loss_mse

    def forward(self, img, gt_map):
        density_map = self.CCN(img)
        self.loss_mse = self.build_loss(density_map.squeeze(), gt_map.squeeze())
        return density_map

    def build_loss(self, density_map, gt_data):
        loss_mse = self.loss_mse_fn(density_map, gt_data)
        return loss_mse

    def test_forward(self, img):
        density_map = self.CCN(img)
        return density_map
