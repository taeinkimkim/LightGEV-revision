### Zero-shot Evaluation
# ------------------------------------------------------------ [SceneFlow Dataset] ----------------------------------------------------
# LightStereo
python tools/eval_single.py \
  --model_name lightstereo_pretrained_sceneflow \
  --cfg cfgs/lightstereo/taein/lightstereo_m_eval_sceneflow.yaml \
  --ckpt train_outputs/SceneFlowDataset/LightStereo/lightstereo_m_train_sceneflow/pretrain/ckpt/checkpoint_epoch_23.pth \
  --output_root outputs/sceneflow_eval \
  --device cuda:0 \
  --max_disp 192 \
  --tau_px 10.0 \
  --no_npz \
  --no_vis