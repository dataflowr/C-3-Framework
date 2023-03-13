import os
from easydict import EasyDict as edict
import time
import torch

# For easy copy-paste in a colab environment
__C = edict()
cfg = __C
__C.SEED = 3035
__C.DATASET = 'SHHB'
if __C.DATASET == 'UCF50':
	from datasets.UCF50.setting import cfg_data
	__C.VAL_INDEX = cfg_data.VAL_INDEX 
if __C.DATASET == 'GCC':
	from datasets.GCC.setting import cfg_data
	__C.VAL_MODE = cfg_data.VAL_MODE 
__C.NET = 'Res101_SFCN'
__C.PRE_GCC = False
__C.PRE_GCC_MODEL = 'path to model'
__C.RESUME = False
__C.RESUME_PATH = './exp/04-25_09-19_SHHB_VGG_1e-05/latest_state.pth'
__C.GPU_ID = [0,1]
__C.LR = 1e-5
__C.LR_DECAY = 0.995
__C.LR_DECAY_START = -1
__C.NUM_EPOCH_LR_DECAY = 1
__C.MAX_EPOCH = 200
__C.LAMBDA_1 = 1e-4
__C.PRINT_FREQ = 10
now = time.strftime("%m-%d_%H-%M", time.localtime())
__C.EXP_NAME = now \
			 + '_' + __C.DATASET \
             + '_' + __C.NET \
             + '_' + str(__C.LR)
if __C.DATASET == 'UCF50':
	__C.EXP_NAME += '_' + str(__C.VAL_INDEX)	
if __C.DATASET == 'GCC':
	__C.EXP_NAME += '_' + __C.VAL_MODE	
__C.EXP_PATH = './exp'
__C.VAL_DENSE_START = 50
__C.VAL_FREQ = 10
__C.VISIBLE_NUM_IMGS = 1