# @Time    : 2023/8/26 13:02
# @Author  : zhangchenming

from .models.acvnet.trainer import Trainer as ACVNetTrainer
from .models.gwcnet.trainer import Trainer as GwcNetTrainer
from .models.igev_pp.trainer import Trainer as IGEVPPTrainer
from .models.msnet.trainer import Trainer as MSNetTrainer
from .models.lightstereo.trainer import Trainer as LightStereoTrainer
from .models.banet_2d.trainer import Trainer as BANetTrainer
from .models.lightgev.trainer import Trainer as LightGEVTrainer

__all__ = {
    'MSNet2D': MSNetTrainer,
    'IGEV_PP': IGEVPPTrainer,
    'GwcNet': GwcNetTrainer,
    'ACVNet': ACVNetTrainer,
    'LightStereo': LightStereoTrainer,
    'BANet': BANetTrainer,
    'LightGEV': LightGEVTrainer,
    'LightGEVModify': LightGEVTrainer,
    'LightGEVModify_2': LightGEVTrainer,
    'LightGEVModify_3': LightGEVTrainer,
    'LightGEVModify_4': LightGEVTrainer,
}


def build_trainer(args, cfgs, local_rank, global_rank, logger, tb_writer):
    trainer = __all__[cfgs.MODEL.NAME](args, cfgs, local_rank, global_rank, logger, tb_writer)
    return trainer