# @Time    : 2024/1/20 03:13
# @Author  : zhangchenming
import os
import copy
import time
import glob
import math
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from functools import partial
from stereo.datasets import build_dataloader
from stereo.utils import common_utils
from stereo.utils.common_utils import color_map_tensorboard, write_tensorboard
from stereo.utils.warmup import LinearWarmup
from stereo.utils.clip_grad import ClipGrad
from stereo.utils.lamb import Lamb
from stereo.evaluation.metric_per_image import epe_metric, d1_metric, threshold_metric
from stereo.optim.schedulers.warmup_cosine import WarmupCosineScheduler

from stereo.utils.data_probe import probe_batch


class TrainerTemplate:
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer, model):
        self.args = args
        self.cfgs = cfgs
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.logger = logger
        self.tb_writer = tb_writer

        self.model = self.build_model(model)

        if self.args.run_mode in ['train', 'eval']:
            self.eval_set, self.eval_loader, self.eval_sampler = self.build_eval_loader()

        if self.args.run_mode == 'test':
            self.test_set, self.test_loader, self.test_sampler = self.build_test_loader()

        if self.args.run_mode == 'train':
            self.train_set, self.train_loader, self.train_sampler = self.build_train_loader()

            if 'MAX_ITER' in cfgs.OPTIMIZATION and cfgs.OPTIMIZATION.MAX_ITER > 0:
                self.max_iter = cfgs.OPTIMIZATION.MAX_ITER
                self.total_epochs = math.ceil(cfgs.OPTIMIZATION.MAX_ITER / len(self.train_loader))
            else:
                self.total_epochs = cfgs.OPTIMIZATION.NUM_EPOCHS
                self.max_iter = self.total_epochs * len(self.train_loader)

            self.last_epoch = -1

            self.optimizer, self.scheduler = self.build_optimizer_and_scheduler()
            self.scaler = torch.cuda.amp.GradScaler(enabled=cfgs.OPTIMIZATION.AMP)

            if self.cfgs.MODEL.CKPT > -1:
                self.resume_ckpt()

            self.warmup_scheduler = self.build_warmup()
            self.clip_gard = self.build_clip_grad()

            # train EPE / train loss 를 val 과 같은 기준으로 재기 위한 로더.
            # TRAINER.TRAIN_EVAL_BATCHES: 0 이면 비활성, 음수(기본)면 val 과 같은 배치 수.
            self.train_eval_loader = None
            self.train_eval_batches = self.cfgs.TRAINER.get('TRAIN_EVAL_BATCHES', -1)
            if self.train_eval_batches != 0:
                self.train_eval_set, self.train_eval_loader, _ = self.build_train_eval_loader()
                if self.train_eval_loader is not None and self.train_eval_batches < 0:
                    self.train_eval_batches = len(self.eval_loader)

            # get_loss 가 학습 전용 출력을 요구해서 실패하는 모델이 있으면 한 번만 경고하고 끈다
            self.eval_loss_enabled = self.cfgs.TRAINER.get('EVAL_LOSS', True)
            self.eval_loss_skipped = 0

            # LightGEV 계열 모델의 learnable geometry gate scalar 초기값 기록
            self.log_geo_gate_alpha('init')

    def build_train_loader(self):
        train_set, train_loader, train_sampler = build_dataloader(
            data_cfg=self.cfgs.DATA_CONFIG,
            batch_size=self.cfgs.OPTIMIZATION.BATCH_SIZE_PER_GPU,
            is_dist=self.args.dist_mode,
            workers=self.args.workers,
            pin_memory=self.args.pin_memory,
            mode='training')
        self.logger.info('Total samples for train dataset: %d' % (len(train_set)))
        return train_set, train_loader, train_sampler

    def build_eval_loader(self):
        eval_set, eval_loader, eval_sampler = build_dataloader(
            data_cfg=self.cfgs.DATA_CONFIG,
            batch_size=self.cfgs.EVALUATOR.BATCH_SIZE_PER_GPU,
            is_dist=self.args.dist_mode,
            workers=self.args.workers,
            pin_memory=self.args.pin_memory,
            mode='evaluating')
        self.logger.info('Total samples for eval dataset: %d' % (len(eval_set)))
        return eval_set, eval_loader, eval_sampler
    
    def build_test_loader(self):
        test_set, test_loader, test_sampler = build_dataloader(
            data_cfg=self.cfgs.DATA_CONFIG,
            batch_size=self.cfgs.EVALUATOR.BATCH_SIZE_PER_GPU,
            is_dist=self.args.dist_mode,
            workers=self.args.workers,
            pin_memory=self.args.pin_memory,
            mode='testing')
        self.logger.info('Total samples for test dataset: %d' % (len(test_set)))
        return test_set, test_loader, test_sampler

    def build_train_eval_loader(self):
        """training split 을 evaluating 전처리로 읽는 로더.

        train EPE 를 val EPE 와 같은 기준으로 재기 위한 것이다. self.train_loader 는
        RandomCrop 증강이 붙고 셔플되므로 val 과 직접 비교할 수 없다. 여기서는
        DATA_SPLIT.EVALUATING 자리에 TRAINING 목록을 넣고 mode='evaluating' 으로 만들어
        전처리를 val 과 완전히 동일하게 맞춘다. SequentialSampler 라 앞에서 N 배치를
        자르면 매 epoch 같은 이미지가 나오므로 곡선이 epoch 간 비교 가능하다.
        """
        data_cfg = copy.deepcopy(self.cfgs.DATA_CONFIG)
        kept = []
        for data_info in data_cfg.DATA_INFOS:
            if 'TRAINING' in data_info.DATA_SPLIT:
                data_info.DATA_SPLIT['EVALUATING'] = data_info.DATA_SPLIT['TRAINING']
                kept.append(data_info)
        if not kept:
            self.logger.info('No TRAINING split found; train-eval pass disabled.')
            return None, None, None
        data_cfg.DATA_INFOS = kept

        train_eval_set, train_eval_loader, train_eval_sampler = build_dataloader(
            data_cfg=data_cfg,
            batch_size=self.cfgs.EVALUATOR.BATCH_SIZE_PER_GPU,
            is_dist=self.args.dist_mode,
            workers=self.args.workers,
            pin_memory=self.args.pin_memory,
            mode='evaluating')
        self.logger.info('Total samples for train-eval dataset: %d' % (len(train_eval_set)))
        return train_eval_set, train_eval_loader, train_eval_sampler

    def build_model(self, model):
        if self.cfgs.OPTIMIZATION.get('FREEZE_BN', False):
            model = common_utils.freeze_bn(model)
            self.logger.info('Freeze the batch normalization layers')

        if self.cfgs.OPTIMIZATION.SYNC_BN and self.args.dist_mode:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.logger.info('Convert batch norm to sync batch norm')
        model = model.to(self.local_rank)

        if self.args.dist_mode:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[self.local_rank], output_device=self.local_rank,
                find_unused_parameters=self.cfgs.MODEL.FIND_UNUSED_PARAMETERS)

        # load pretrained model
        if self.cfgs.MODEL.PRETRAINED_MODEL:
            self.logger.info('Loading parameters from checkpoint %s' % self.cfgs.MODEL.PRETRAINED_MODEL)
            if not os.path.isfile(self.cfgs.MODEL.PRETRAINED_MODEL):
                raise FileNotFoundError
            common_utils.load_params_from_file(
                model, self.cfgs.MODEL.PRETRAINED_MODEL, device='cuda:%d' % self.local_rank,
                dist_mode=self.args.dist_mode, logger=self.logger, strict=False)
        return model

    def build_optimizer_and_scheduler(self):
        # optimizer 생성
        if self.cfgs.OPTIMIZATION.OPTIMIZER.NAME == 'Lamb':
            optimizer_cls = Lamb
        else:
            optimizer_cls = getattr(torch.optim, self.cfgs.OPTIMIZATION.OPTIMIZER.NAME)

        valid_arg = common_utils.get_valid_args(
            optimizer_cls, 
            self.cfgs.OPTIMIZATION.OPTIMIZER, 
            ['name']
        )
        optimizer = optimizer_cls(
            params=[p for p in self.model.parameters() if p.requires_grad], 
            **valid_arg
        )

        # ------------------------------
        # 🔥 Lambda 스케줄러 직접 구현
        # ------------------------------
        if self.cfgs.OPTIMIZATION.SCHEDULER.NAME == "LambdaLR":
            lr_decay = self.cfgs.OPTIMIZATION.SCHEDULER.get("LR_DECAY", [])

            def lr_step(step):
                scale = 1.0
                # lr_decay = [s1, v1, s2, v2, ...]
                for s, v in zip(lr_decay[::2], lr_decay[1::2]):
                    if step > s:
                        scale = v
                return scale

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, 
                lr_lambda=lr_step
            )
            return optimizer, scheduler

        # ------------------------------
        # 기본 스케줄러 (기존 코드)
        # ------------------------------
        self.cfgs.OPTIMIZATION.SCHEDULER.TOTAL_STEPS = self.max_iter

        try:
            scheduler_cls = getattr(torch.optim.lr_scheduler, self.cfgs.OPTIMIZATION.SCHEDULER.NAME)
        except:
            if self.cfgs.OPTIMIZATION.SCHEDULER.NAME == 'WarmupCosine':
                scheduler_cls = WarmupCosineScheduler

        valid_arg = common_utils.get_valid_args(
            scheduler_cls, 
            self.cfgs.OPTIMIZATION.SCHEDULER,
            ['name', 'on_epoch']
        )
        scheduler = scheduler_cls(optimizer, **valid_arg)

        return optimizer, scheduler


    def resume_ckpt(self):
        self.logger.info('Resume from ckpt:%d' % self.cfgs.MODEL.CKPT)
        ckpt_path = str(os.path.join(self.args.ckpt_dir, 'checkpoint_epoch_%d.pth' % self.cfgs.MODEL.CKPT))
        checkpoint = torch.load(ckpt_path, map_location='cuda:%d' % self.local_rank)
        self.last_epoch = checkpoint['epoch']
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scaler.load_state_dict(checkpoint['scaler_state'])
        if self.args.dist_mode:
            self.model.module.load_state_dict(checkpoint['model_state'])
        else:
            self.model.load_state_dict(checkpoint['model_state'])

    def build_warmup(self):
        last_step = (self.last_epoch + 1) * len(self.train_loader) - 1
        if 'WARMUP' in self.cfgs.OPTIMIZATION.SCHEDULER:
            warmup_steps = self.cfgs.OPTIMIZATION.SCHEDULER.WARMUP.get('WARM_STEPS', 1)
            warmup_scheduler = LinearWarmup(
                self.optimizer,
                warmup_period=warmup_steps,
                last_step=last_step)
        else:
            warmup_scheduler = LinearWarmup(
                self.optimizer,
                warmup_period=1,
                last_step=last_step)

        return warmup_scheduler

    def build_clip_grad(self):
        clip_gard = None
        if 'CLIP_GRAD' in self.cfgs.OPTIMIZATION:
            clip_type = self.cfgs.OPTIMIZATION.CLIP_GRAD.get('TYPE', None)
            clip_value = self.cfgs.OPTIMIZATION.CLIP_GRAD.get('CLIP_VALUE', 0.1)
            max_norm = self.cfgs.OPTIMIZATION.CLIP_GRAD.get('MAX_NORM', 35)
            norm_type = self.cfgs.OPTIMIZATION.CLIP_GRAD.get('NORM_TYPE', 2)
            clip_gard = ClipGrad(clip_type, clip_value, max_norm, norm_type)
        return clip_gard

    def get_geo_gate_alpha(self):
        """LightGEV 모델의 learnable geometry gate scalar 값들을 {name: float}로 반환."""
        model = self.model.module if self.args.dist_mode else self.model
        alphas = {}
        for name, param in model.named_parameters():
            if name.split('.')[-1] == 'geo_gate_alpha':
                alphas[name] = param.detach().float().mean().item()
        return alphas

    def log_geo_gate_alpha(self, stage, current_epoch=None, tb_step=None):
        """geo_gate_alpha 값을 파싱 가능한 형태로 로그(및 tensorboard)에 기록.

        stage: 'init' | 'epoch' | 'final'
        기록 형식: "GeoGateAlpha epoch 3: geo_gate_alpha=0.987654"
        """
        alphas = self.get_geo_gate_alpha()
        if not alphas:
            return alphas

        tag = 'epoch %d' % current_epoch if stage == 'epoch' else stage
        body = ', '.join('{}={:.6f}'.format(k, v) for k, v in alphas.items())
        self.logger.info('GeoGateAlpha {}: {}'.format(tag, body))

        if self.global_rank == 0 and self.tb_writer is not None and tb_step is not None:
            write_tensorboard(self.tb_writer,
                              {'scalar/train/{}'.format(k): v for k, v in alphas.items()},
                              tb_step)
        return alphas

    def train(self, current_epoch, tbar):
        self.model.train()
        if self.cfgs.OPTIMIZATION.get('FREEZE_BN', False):
            self.model = common_utils.freeze_bn(self.model)
        if self.args.dist_mode:
            self.train_sampler.set_epoch(current_epoch)
        self.train_one_epoch(current_epoch=current_epoch, tbar=tbar)
        self.log_geo_gate_alpha('epoch', current_epoch=current_epoch, tb_step=current_epoch)
        if self.args.dist_mode:
            dist.barrier()
        if self.cfgs.OPTIMIZATION.SCHEDULER.ON_EPOCH:
            self.scheduler.step()
            self.warmup_scheduler.lrs = [group['lr'] for group in self.optimizer.param_groups]

    def evaluate(self, current_epoch):
        self.model.eval()
        self.eval_one_epoch(current_epoch=current_epoch)
        if self.train_eval_loader is not None:
            self.train_eval_one_epoch(current_epoch=current_epoch)
        if self.args.dist_mode:
            dist.barrier()

    def save_ckpt(self, current_epoch):
        if (current_epoch % self.cfgs.TRAINER.CKPT_SAVE_INTERVAL == 0 or current_epoch == self.total_epochs - 1) and self.global_rank == 0:
            ckpt_list = glob.glob(os.path.join(self.args.ckpt_dir, 'checkpoint_epoch_*.pth'))
            ckpt_list.sort(key=os.path.getmtime)
            if len(ckpt_list) >= self.cfgs.TRAINER.MAX_CKPT_SAVE_NUM:
                for cur_file_idx in range(0, len(ckpt_list) - self.cfgs.TRAINER.MAX_CKPT_SAVE_NUM + 1):
                    os.remove(ckpt_list[cur_file_idx])
            ckpt_name = os.path.join(self.args.ckpt_dir, 'checkpoint_epoch_%d.pth' % current_epoch)
            common_utils.save_checkpoint(self.model, self.optimizer, self.scheduler, self.scaler,
                                         self.args.dist_mode, current_epoch, filename=ckpt_name)
        if self.args.dist_mode:
            dist.barrier()

    def train_one_epoch(self, current_epoch, tbar):
        start_epoch = self.last_epoch + 1
        logger_iter_interval = self.cfgs.TRAINER.LOGGER_ITER_INTERVAL
        total_loss = 0.0
        loss_func = self.model.module.get_loss if self.args.dist_mode else self.model.get_loss

        train_loader_iter = iter(self.train_loader)
        for i in range(0, len(self.train_loader)):
            total_iter = current_epoch * len(self.train_loader) + i
            if total_iter >= self.max_iter:
                break

            self.optimizer.zero_grad()
            lr = self.optimizer.param_groups[0]['lr']

            start_timer = time.time()
            data = next(train_loader_iter)

            for k, v in data.items():
                data[k] = v.to(self.local_rank) if torch.is_tensor(v) else v
            data_timer = time.time()

            with torch.cuda.amp.autocast(enabled=self.cfgs.OPTIMIZATION.AMP):
                model_pred = self.model(data)
                infer_timer = time.time()
                loss, tb_info = loss_func(model_pred, data)

            # 不要在autocast下调用, calls backward() on scaled loss to create scaled gradients.
            self.scaler.scale(loss).backward()
            # 做梯度剪裁的时候需要先unscale, unscales the gradients of optimizer's assigned params in-place
            self.scaler.unscale_(self.optimizer)
            # 梯度剪裁
            if self.clip_gard is not None:
                self.clip_gard(self.model)
            # optimizer's gradients are already unscaled, so scaler.step does not unscale them
            self.scaler.step(self.optimizer)
            # Updates the scale for next iteration.
            self.scaler.update()
            # torch.cuda.empty_cache()

            # warmup_scheduler period>1 和 batch_scheduler 不要同时使用
            with self.warmup_scheduler.dampening():
                if not self.cfgs.OPTIMIZATION.SCHEDULER.ON_EPOCH:
                    self.scheduler.step()

            total_loss += loss.item()

            trained_time_past_all = tbar.format_dict['elapsed']
            single_iter_second = trained_time_past_all / (total_iter + 1 - start_epoch * len(self.train_loader))
            remaining_second_all = single_iter_second * (self.total_epochs * len(self.train_loader) - total_iter - 1)
            if total_iter % logger_iter_interval == 0:
                message = ('Training Epoch:{:>2d}/{} Iter:{:>4d}/{} '
                           'Loss:{:#.6g}({:#.6g}) LR:{:.4e} '
                           'DataTime:{:.2f} InferTime:{:.2f}ms'
                           'Time cost: {}/{}'
                           ).format(current_epoch, self.total_epochs, i, len(self.train_loader),
                                    loss.item(), total_loss / (i + 1), lr,
                                    data_timer - start_timer, (infer_timer - data_timer) * 1000,
                                    tbar.format_interval(trained_time_past_all),
                                    tbar.format_interval(remaining_second_all))
                self.logger.info(message)

            if self.cfgs.TRAINER.TRAIN_VISUALIZATION:
                pred = model_pred['disp_pred']
                gt = data['disp']

                # pred가 GT 해상도와 다를 때만 업샘플
                if pred.shape[-2:] != gt.shape[-2:]:
                    pred = F.interpolate(pred, size=gt.shape[-2:], mode='bilinear', align_corners=False)

                # numpy 변환 및 색상 맵 생성
                tb_info['image/train/disp'] = color_map_tensorboard(gt[0], pred.squeeze(1)[0])
                tb_info['image/train/image'] = torch.cat([data['left'][0], data['right'][0]], dim=1) / 256
                # tb_info['image/train/disp'] = color_map_tensorboard(data['disp'][0], model_pred['disp_pred'].squeeze(1)[0])

            tb_info.update({'scalar/train/lr': lr})
            if total_iter % logger_iter_interval == 0 and self.local_rank == 0 and self.tb_writer is not None:
                write_tensorboard(self.tb_writer, tb_info, total_iter)

    @torch.no_grad()
    def compute_eval_loss(self, data):
        """평가 중 loss 계산.

        get_loss 는 보통 학습 전용 보조 출력(disp_4 등)을 요구한다. 최상위 모듈의
        training 플래그만 켜서 그 출력을 만들게 하고, 하위 모듈(BatchNorm/Dropout)은
        eval 상태로 둔다 -- .train() 과 달리 플래그가 재귀 전파되지 않으므로 running
        stat 은 갱신되지 않고 예측도 바뀌지 않는다 (LightStereo/LightGEV/LightGEVModify
        에서 disp_pred 차이 0 확인).

        주의: forward 가 self.training 으로 '보조 출력 추가' 이상의 분기를 하는 모델
        (예: IGEV_PP 의 test_mode = not self.training) 에서는 예측 자체가 달라진다.
        그런 모델은 config 에 TRAINER.EVAL_LOSS: false 로 끌 것.

        유효 픽셀이 하나도 없는 샘플(예: SceneFlow val 에 GT 가 전부 max_disp 이상인
        프레임이 1장 있다) 에서는 get_loss 가 빈 텐서의 평균을 내서 NaN 을 만든다.
        metric 함수들은 num_valid_pixels > 0 로 방어하지만 loss 는 그렇지 않으므로,
        여기서 비유한 값을 걸러 epoch 평균이 오염되지 않게 한다.

        Returns:
            (loss_sum, count) 또는 계산 불가/비유한 시 (None, 0)
        """
        if not self.eval_loss_enabled:
            return None, 0

        core = self.model.module if self.args.dist_mode else self.model
        was_training = core.training
        core.training = True
        try:
            with torch.cuda.amp.autocast(enabled=self.cfgs.OPTIMIZATION.AMP):
                model_pred = core(data)
                loss, _ = core.get_loss(model_pred, data)
            if not torch.isfinite(loss):
                self.eval_loss_skipped += 1
                return None, 0
            return float(loss.item()), 1
        except Exception as e:  # noqa: BLE001 - 모델마다 요구 키가 달라 광범위하게 잡는다
            self.eval_loss_enabled = False
            self.logger.warning(
                'Eval loss disabled for this run: get_loss failed ({}: {}). '
                'Set TRAINER.EVAL_LOSS: false to silence this.'.format(type(e).__name__, e))
            return None, 0
        finally:
            core.training = was_training

    def train_eval_one_epoch(self, current_epoch):
        """train split 을 val 과 동일한 전처리/모드로 평가한다 (train_metrics 로 기록)."""
        self.eval_one_epoch(current_epoch=current_epoch,
                            loader=self.train_eval_loader,
                            max_batches=self.train_eval_batches,
                            tag='train')

    def eval_one_epoch(self, current_epoch, loader=None, max_batches=None, tag='val'):
        """tag='val' 이면 "Epoch N metrics: {...}", 'train' 이면 "Epoch N train_metrics: {...}" 로 기록."""
        loader = self.eval_loader if loader is None else loader
        num_batches = len(loader) if not max_batches or max_batches <= 0 else min(max_batches, len(loader))

        metric_func_dict = {
            'epe': epe_metric,
            'd1_all': d1_metric,
            'thres_1': partial(threshold_metric, threshold=1),
            'thres_2': partial(threshold_metric, threshold=2),
            'thres_3': partial(threshold_metric, threshold=3),
        }

        evaluator_cfgs = self.cfgs.EVALUATOR
        local_rank = self.local_rank

        epoch_metrics = {}
        for k in evaluator_cfgs.METRIC:
            epoch_metrics[k] = {'indexes': [], 'values': []}

        loss_sum, loss_count = 0.0, 0
        self.eval_loss_skipped = 0
        for i, data in enumerate(loader):
            if i >= num_batches:
                break
            for k, v in data.items():
                data[k] = v.to(local_rank) if torch.is_tensor(v) else v

            with torch.cuda.amp.autocast(enabled=self.cfgs.OPTIMIZATION.AMP):
                infer_start = time.time()
                model_pred = self.model(data)
                infer_time = time.time() - infer_start

            disp_pred = model_pred['disp_pred']
            disp_gt = data["disp"]

            # pred가 GT 해상도와 다를 때만 업샘플
            if disp_pred.shape[-2:] != disp_gt.shape[-2:]:
                disp_pred = F.interpolate(disp_pred, size=disp_gt.shape[-2:], mode='bilinear', align_corners=False)

            mask = (disp_gt < evaluator_cfgs.MAX_DISP) & (disp_gt > 0)
            if 'occ_mask' in data and evaluator_cfgs.get('APPLY_OCC_MASK', False):
                mask = mask & ~data['occ_mask'].to(torch.bool)

            for m in evaluator_cfgs.METRIC:
                if m not in metric_func_dict:
                    raise ValueError("Unknown metric: {}".format(m))
                metric_func = metric_func_dict[m]
                res = metric_func(disp_pred.squeeze(1), disp_gt, mask)
                epoch_metrics[m]['indexes'].extend(data['index'].tolist())
                epoch_metrics[m]['values'].extend(res.tolist())

            batch_loss, n = self.compute_eval_loss(data)
            if n:
                loss_sum += batch_loss
                loss_count += n

            if i % self.cfgs.TRAINER.LOGGER_ITER_INTERVAL == 0:
                message = ('{} Epoch:{:>2d} Iter:{:>4d}/{} InferTime: {:.2f}ms'
                           ).format('Evaluating' if tag == 'val' else 'Train-eval',
                                    current_epoch, i, num_batches, infer_time * 1000)
                self.logger.info(message)

                if self.cfgs.TRAINER.EVAL_VISUALIZATION and self.tb_writer is not None:
                    tb_info = {
                        'image/eval/image': torch.cat([data['left'][0], data['right'][0]], dim=1) / 256,
                        'image/eval/disp': color_map_tensorboard(disp_gt[0], disp_pred.squeeze(1)[0])
                    }
                    write_tensorboard(self.tb_writer, tb_info, current_epoch * num_batches + i)

        # gather from all gpus
        if self.args.dist_mode:
            dist.barrier()
            self.logger.info("Start reduce metrics.")
            for k in epoch_metrics.keys():
                indexes = torch.tensor(epoch_metrics[k]["indexes"]).to(local_rank)
                values = torch.tensor(epoch_metrics[k]["values"]).to(local_rank)
                gathered_indexes = [torch.zeros_like(indexes) for _ in range(dist.get_world_size())]
                gathered_values = [torch.zeros_like(values) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_indexes, indexes)
                dist.all_gather(gathered_values, values)
                unique_dict = {}
                for key, value in zip(torch.cat(gathered_indexes, dim=0).tolist(),
                                      torch.cat(gathered_values, dim=0).tolist()):
                    if key not in unique_dict:
                        unique_dict[key] = value
                epoch_metrics[k]["indexes"] = list(unique_dict.keys())
                epoch_metrics[k]["values"] = list(unique_dict.values())

        results = {}
        for k in epoch_metrics.keys():
            results[k] = torch.tensor(epoch_metrics[k]["values"]).mean()

        # loss 는 배치 평균이라 인덱스 기반 dedup 이 아니라 합/개수로 모은다
        if loss_count:
            if self.args.dist_mode:
                packed = torch.tensor([loss_sum, float(loss_count)], device=local_rank)
                dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                loss_sum, loss_count = packed[0].item(), packed[1].item()
            results['loss'] = torch.tensor(loss_sum / loss_count)
            if self.eval_loss_skipped:
                self.logger.info(
                    '[%s] loss: %d/%d batches skipped (no valid GT pixel -> non-finite loss)'
                    % (tag, self.eval_loss_skipped, self.eval_loss_skipped + loss_count))

        if local_rank == 0 and self.tb_writer is not None:
            tb_info = {}
            for k, v in results.items():
                tb_info[f'scalar/{tag}/{k}'] = v.item()

            write_tensorboard(self.tb_writer, tb_info, current_epoch)

        label = 'metrics' if tag == 'val' else '{}_metrics'.format(tag)
        self.logger.info(f"Epoch {current_epoch} {label}: {results}")