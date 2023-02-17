import os
import numpy as np
import torch
import imp

from config import cfg

# ------------prepare enviroment------------
seed = cfg.SEED
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

gpus = cfg.GPU_ID
if len(gpus) == 1:
    torch.cuda.set_device(gpus[0])

torch.backends.cudnn.benchmark = True


# ------------prepare data loader------------
data_mode = cfg.DATASET

dataset_import_path = "datasets/" + data_mode
loading_data = getattr(imp.load_source("loader", dataset_import_path + "/loading_data.py"), "loading_data")
cfg_data = getattr(imp.load_source("settings", dataset_import_path + "/setting.py"), "cfg_data")

# ------------Prepare Trainer------------
net = cfg.NET
if net in [
    "MCNN",
    "AlexNet",
    "VGG",
    "VGG_DECODER",
    "Res50",
    "Res101",
    "CSRNet",
    "Res101_SFCN",
]:
    from trainer import Trainer
elif net in ["SANet"]:
    from trainer_for_M2TCC import Trainer  # double losses but single output
elif net in ["CMTL"]:
    from trainer_for_CMTL import Trainer  # double losses and double outputs

# ------------Start Training------------
pwd = os.path.split(os.path.realpath(__file__))[0]
cc_trainer = Trainer(loading_data, cfg_data, pwd)
cc_trainer.forward()
