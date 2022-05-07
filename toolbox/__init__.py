from .metrics import averageMeter, runningScore
from .log import get_logger
from .loss import MscCrossEntropyLoss
from .utils import ClassWeight, save_ckpt, load_ckpt, class_to_RGB, adjust_lr
from .ranger.ranger import Ranger
from .ranger.ranger913A import RangerVA
from .ranger.rangerqh import RangerQH
import torch.nn as nn

def get_dataset(cfg):
    assert cfg['dataset'] in ['nyuv2', 'sunrgbd', 'sunrgbd224']

    if cfg['dataset'] == 'nyuv2':
        from .datasets.nyuv2 import NYUv2
        return NYUv2(cfg, mode='train'), NYUv2(cfg, mode='test')
    if cfg['dataset'] == 'sunrgbd':
        from .datasets.sunrgbd import SUNRGBD
        return SUNRGBD(cfg, mode='train'), SUNRGBD(cfg, mode='test')
    if cfg['dataset'] == 'sunrgbd224':
        from .datasets.sunrgbd224 import SUNRGBD224
        return SUNRGBD224(cfg, mode='train'), SUNRGBD224(cfg, mode='test')


def get_model(cfg):
    if cfg['model_name'] == 'rednet':
        from .models.rednet import RedNet
        return RedNet(num_classes=cfg['n_classes'], pretrained=True)

    if cfg['model_name'] == 'bbsnet':
        from .models.BBSnetmodel.BBSnet import BBSNet
        return BBSNet(n_class=cfg['n_classes'])





