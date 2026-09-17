# @Time    : 2023/8/28 22:18
# @Author  : zhangchenming
import sys
import os
import argparse
import datetime
import tqdm
import shutil
import torch
import torch.distributed as dist

from easydict import EasyDict
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, './')
from stereo.utils import common_utils
from stereo.modeling import build_trainer
from cfgs.data_basic import DATA_PATH_DICT

import warnings
# 同时忽略 FutureWarning 和 UserWarning （包括PyTorch产生的）
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    # mode
    parser.add_argument('--dist_mode', action='store_true', default=False, help='torchrun ddp multi gpu')
    parser.add_argument('--cfg_file', type=str, default=None, required=True, help='specify the config for training')
    parser.add_argument('--fix_random_seed', action='store_true', default=False,
                        help='--seed 값으로 random/numpy/torch 시드를 고정한다')
    parser.add_argument('--seed', type=int, default=0,
                        help='난수 시드 (--fix_random_seed 와 함께 사용). '
                             'cross-domain 성능은 run 마다 크게 흔들리므로(측정: KITTI EPE '
                             '동일 구조 두 run 이 1.2 차이) 여러 seed 로 돌려 평균±범위로 보고할 것.')

    # save path
    parser.add_argument('--save_root_dir', type=str, default='./train_outputs', help='save root dir for this experiment')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--cover_old_exp', action='store_true', default=False)

    # dataloader
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='data loader pin memory')

    args = parser.parse_args()
    yaml_config = common_utils.config_loader(args.cfg_file)
    cfgs = EasyDict(yaml_config)

    dataset_names = [x.DATASET for x in cfgs.DATA_CONFIG.DATA_INFOS]
    unique_dataset_names = list(set(dataset_names))
    if len(unique_dataset_names) == 1:
        exp_dataset_dir = unique_dataset_names[0]
    else:
        exp_dataset_dir = 'MultiDataset'
    args.exp_group_path = os.path.join(exp_dataset_dir, cfgs.MODEL.NAME)
    args.tag = os.path.basename(args.cfg_file)[:-5]

    for each in cfgs.DATA_CONFIG.DATA_INFOS:
        dataset_name = each.DATASET
        if dataset_name == 'KittiDataset':
            dataset_name = 'KittiDataset15' if 'kitti15' in each.DATA_SPLIT.EVALUATING else 'KittiDataset12'
        each.DATA_PATH = DATA_PATH_DICT[dataset_name]
        assert os.path.exists(each.DATA_PATH), '[Errno 2] No such file or directory: {}, You must modify the data root path in cfgs/databasic.py to the path of your own dataset.'.format(each.DATA_PATH)

    args.run_mode = 'train'
    return args, cfgs


def main():
    args, cfgs = parse_config()
    if args.dist_mode:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        group_rank = int(os.environ["GROUP_RANK"])
    else:
        local_rank = 0
        global_rank = 0
        group_rank = 0

    # env
    torch.cuda.set_device(local_rank)
    if args.fix_random_seed:
        # dist 모드에서는 rank 마다 다른 시드를 줘야 augmentation 이 겹치지 않는다
        seed = args.seed if not args.dist_mode else args.seed + dist.get_rank()
        common_utils.set_random_seed(seed=seed)
        args.resolved_seed = seed

    # savedir
    args.output_dir = str(os.path.join(args.save_root_dir, args.exp_group_path, args.tag, args.extra_tag))
    cfgs.EVALUATOR.REGION_CSV_DIR = os.path.join(args.output_dir, 'region_metrics')
    if os.path.exists(args.output_dir) and args.cover_old_exp and global_rank == 0:
        shutil.rmtree(args.output_dir)
    if args.dist_mode:
        dist.barrier()
    if os.path.exists(args.output_dir) and args.extra_tag != 'debug' and cfgs.MODEL.CKPT == -1:
        raise Exception('There is already an exp with this name')

    args.ckpt_dir = os.path.join(args.output_dir, 'ckpt')
    if not os.path.exists(args.ckpt_dir) and local_rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
    if args.dist_mode:
        dist.barrier()

    # logger
    log_file = os.path.join(args.output_dir, 'train_{}_{}.log'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), group_rank))
    logger = common_utils.create_logger(log_file, rank=local_rank)
    tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard')) if global_rank == 0 else None
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    common_utils.log_configs(cfgs, logger=logger)

    # trainer
    model_trainer = build_trainer(args, cfgs, local_rank, global_rank, logger, tb_writer)

    tbar = tqdm.trange(model_trainer.last_epoch + 1, model_trainer.total_epochs,
                       desc='epochs', dynamic_ncols=True, disable=(local_rank != 0),
                       bar_format='{l_bar}{bar}{r_bar}\n')
    # train loop
    for current_epoch in tbar:
        model_trainer.train(current_epoch, tbar)
        model_trainer.save_ckpt(current_epoch)
        # 마지막 에폭은 EVAL_INTERVAL과 상관없이 항상 평가 (save_ckpt와 동일한 규칙)
        if (current_epoch % cfgs.TRAINER.EVAL_INTERVAL == 0
                or current_epoch == model_trainer.total_epochs - 1):
            model_trainer.evaluate(current_epoch)

    # 학습 종료 후 최종 geo_gate_alpha 기록 (LightGEV 계열 모델만 해당)
    model_trainer.log_geo_gate_alpha('final')


if __name__ == '__main__':
    main()