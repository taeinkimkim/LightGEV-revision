### Zero-shot Evaluation
# ------------------------------------------------------------ [ETH3D Dataset] ------------------------------------------------------
LightStereo
python tools/eval_single.py \
  --model_name lightstereo_pretrained_sceneflow_bs12_lr0004_datafixed_valid_GT_revised \
  --cfg cfgs/lightstereo/lightstereo_m_eval_eth3d.yaml \
  --ckpt train_outputs/SceneFlowDataset/LightStereo/lightstereo_m_train_sceneflow/pretrain_trainBS12_valBS12_LR0.0004_datafixed/ckpt/checkpoint_epoch_89.pth \
  --output_root /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/eth3d_eval/lightstereo/lightstereo \
  --device cuda:0 

# # LightGEV
# python tools/eval_single.py \
#   --model_name lightgev_pretrained_sceneflow_full_iter_4_bs12_lr0004_fixed \
#   --cfg cfgs/lightgev/lightgev_eval_eth3d_full_iter_4.yaml \
#   --ckpt train_outputs/SceneFlowDataset/LightGEV/lightgev_train_sceneflow_full_iter_4/pretrain_trainBS12_valBS12_LR0.0004_fixed/ckpt/checkpoint_epoch_89.pth \
#   --output_root /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/eth3d_eval/lightgev \
#   --device cuda:0 

##  Comparison
# LightStereo vs LightGEV
# python tools/panel_compare.py \
#   --input_root outputs/eth3d_eval \
#   --output /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/eth3d_eval/_compare/ \
#   --models lightstereo_pretrained_sceneflow lightgev_pretrained_sceneflow_final2