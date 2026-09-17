### Zero-shot Evaluation
# ------------------------------------------------------------ [KITTI Dataset] ------------------------------------------------------
# LightStereo
python tools/eval_single.py \
  --model_name lightstereo_pretrained_sceneflow_bs12_lr0004_datafixed_valid_GT_revised \
  --cfg cfgs/lightstereo/lightstereo_m_eval_kitti.yaml \
  --ckpt train_outputs/SceneFlowDataset/LightStereo/lightstereo_m_train_sceneflow/pretrain_trainBS12_valBS12_LR0.0004_datafixed/ckpt/checkpoint_epoch_89.pth \
  --output_root /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/kitti_eval/lightstereo/lightstereo \
  --device cuda:0 

# LightGEV
# python tools/eval_single.py \
#   --model_name lightgev_pretrained_sceneflow_aggregation_only_bs4_lr0004_fixed \
#   --cfg cfgs/lightgev/lightgev_eval_kitti_aggregation_only.yaml \
#   --ckpt train_outputs/SceneFlowDataset/LightGEV/lightgev_train_sceneflow_aggregation_only/pretrain_trainBS4_valBS4_LR0.0004_EP30_fixed/ckpt/checkpoint_epoch_29.pth \
#   --output_root /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/kitti_eval/lightgev \
#   --device cuda:0 

# python tools/eval_single.py \
#   --model_name lightgev_pretrained_sceneflow_full_iter2_bs4_lr0004_fixed \
#   --cfg cfgs/lightgev/lightgev_eval_kitti_full_iter_2.yaml \
#   --ckpt train_outputs/SceneFlowDataset/LightGEV/lightgev_train_sceneflow_full_iter_2/pretrain_trainBS4_valBS4_LR0.0004_EP30_fixed/ckpt/checkpoint_epoch_29.pth \
#   --output_root /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/kitti_eval/lightgev \
#   --device cuda:0 

##  Comparison
# LightStereo vs LightGEV
# python tools/panel_compare.py \
#   --input_root /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/kitti_eval \
#   --output /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/kitti_eval/_compare/ \
#   --models lightstereo_pretrained_sceneflow_bs12_lr0004 lightgev_pretrained_sceneflow_iter4_bs12_lr0004_fixed