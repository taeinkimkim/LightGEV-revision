# =================================================================================
# LightGEVModify_4 : GEV_ITERS ablation (full, iter 2 / 4 / 6)  -  kitti 평가
#
# 주의: cross-domain 수치는 run 간 편차가 크다. iter 간 차이가 편차보다 작으면
#       단일 run 으로 순위를 매기지 말 것.
# =================================================================================

CKPT_ROOT=train_outputs/SceneFlowDataset/LightGEVModify_4
# EXTRA_TAG=pretrain_trainBS12_valBS12_LR0.0004_modify_4_gate_normT_biasF_same_loss
EXTRA_TAG=pretrain_trainBS12_valBS12_LR0.0004_modify_4_gate_normT_biasF
OUT_ROOT=/media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/kitti_eval/lightgev_modify_4
EPOCH=89

evaluate () {   # $1 = variant
    python tools/eval_single.py \
      --model_name lightgev_modify_4_pretrained_sceneflow_$1_bs12_lr0004_normT_biasF_valid_GT_revised \
      --cfg cfgs/lightgev_modify_4/lightgev_modify_4_eval_kitti_$1.yaml \
      --ckpt ${CKPT_ROOT}/lightgev_modify_4_train_sceneflow_$1/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
      --output_root ${OUT_ROOT} \
      --device cuda:0
}

# evaluate full_iter_2
# evaluate full_iter_2_r_075
# evaluate full_iter_4_r_075
evaluate full_iter_6_r_075
# evaluate full_iter_8_r_075
# evaluate full_iter_10_r_075
# evaluate full_iter_6_r_075_dil_2
# evaluate full_iter_6_r_075_dil_1
# evaluate refiner_only_iter_6_r_075
