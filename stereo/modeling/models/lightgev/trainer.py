from stereo.modeling.trainer_template import TrainerTemplate
from .lightgev import LightGEV
from .lightgev_modify import LightGEVModify
from .lightgev_modify_gate_refiner import LightGEVModify_2
from .lightgev_modify_3 import LightGEVModify_3
from .lightgev_modify_4 import LightGEVModify_4

__all__ = {
    'LightGEV': LightGEV,
    'LightGEVModify': LightGEVModify,
    'LightGEVModify_2': LightGEVModify_2,
    'LightGEVModify_3': LightGEVModify_3,
    'LightGEVModify_4': LightGEVModify_4,
}

class Trainer(TrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer):
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)
        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)