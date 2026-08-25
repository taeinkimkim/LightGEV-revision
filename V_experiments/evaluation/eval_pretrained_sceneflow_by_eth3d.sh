### Zero-shot Evaluation
# ------------------------------------------------------------ [ETH3D Dataset] ------------------------------------------------------
# LightStereo
python tools/eval_single.py \
  --model_name lightstereo_pretrained_sceneflow \
  --cfg cfgs/lightstereo/taein/lightstereo_m_eval_eth3d.yaml \
  --ckpt train_outputs/SceneFlowDataset/LightStereo/lightstereo_m_train_sceneflow/pretrain/ckpt/checkpoint_epoch_23.pth \
  --output_root outputs/eth3d_eval \
  --device cuda:0 

# LightGEV
python tools/eval_single.py \
  --model_name lightgev_pretrained_sceneflow_final2 \
  --cfg cfgs/lightgev/lightgev_eval_eth3d.yaml \
  --ckpt train_outputs/SceneFlowDataset/LightGEV/lightgev_train_sceneflow_full_iter_2/final_2/ckpt/checkpoint_epoch_26.pth \
  --output_root outputs/eth3d_eval \
  --device cuda:0 

##  Comparison
# LightStereo vs LightGEV
python tools/panel_compare.py \
  --input_root outputs/eth3d_eval \
  --output outputs/eth3d_eval/_compare/ \
  --models lightstereo_pretrained_sceneflow lightgev_pretrained_sceneflow_final2