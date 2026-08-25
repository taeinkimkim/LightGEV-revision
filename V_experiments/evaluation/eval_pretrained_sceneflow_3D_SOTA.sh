### Zero-shot Evaluation
# ------------------------------------------------------------ [SceneFlow Dataset] ----------------------------------------------------
# IGEV++
python tools/eval_single.py \
  --model_name igev_pp_pretrained_sceneflow \
  --cfg cfgs/igev_pp/igev_pp_eval_sceneflow.yaml \
  --ckpt ckpts/igev_pp/sceneflow.pth \
  --output_root outputs/sceneflow_eval \
  --device cuda:0 \
  --no_npz \
  --no_vis

# ACVNet
python tools/eval_single.py \
  --model_name acvnet_pretrained_sceneflow \
  --cfg cfgs/acvnet/acvnet_eval_sceneflow.yaml \
  --ckpt ckpts/acvnet/sceneflow.ckpt \
  --output_root outputs/sceneflow_eval \
  --device cuda:0 \
  --no_npz \
  --no_vis

# GwcNet
python tools/eval_single.py \
  --model_name gwcnet_pretrained_sceneflow \
  --cfg cfgs/gwcnet/taein/gwcnet_eval_sceneflow.yaml \
  --ckpt ckpts/gwcnet/checkpoint_000014.ckpt \
  --output_root outputs/sceneflow_eval \
  --device cuda:0 \
  --no_npz \
  --no_vis







# ------------------------------------------------------------ [KITTI Dataset] ------------------------------------------------------
# IGEV++
python tools/eval_single.py \
  --model_name igev_pp_pretrained_sceneflow \
  --cfg cfgs/igev_pp/igev_pp_eval_kitti.yaml \
  --ckpt ckpts/igev_pp/sceneflow.pth \
  --output_root outputs/kitti_eval \
  --device cuda:0 \
  --max_disp 192 \
  --tau_px 10.0

# ACVNet
python tools/eval_single.py \
  --model_name acvnet_pretrained_sceneflow \
  --cfg cfgs/acvnet/acvnet_eval_kitti.yaml \
  --ckpt ckpts/acvnet/sceneflow.ckpt \
  --output_root outputs/kitti_eval \
  --device cuda:0 \
  --max_disp 192 \
  --tau_px 10.0

# GwcNet
python tools/eval_single.py \
  --model_name gwcnet_pretrained_sceneflow \
  --cfg cfgs/gwcnet/taein/gwcnet_eval_kitti.yaml \
  --ckpt ckpts/gwcnet/checkpoint_000014.ckpt \
  --output_root outputs/kitti_eval \
  --device cuda:0 \
  --max_disp 192 \
  --tau_px 10.0






# ------------------------------------------------------------ [Middlebury Dataset] ------------------------------------------------------
# IGEV++
python tools/eval_single.py \
  --model_name igev_pp_pretrained_sceneflow \
  --cfg cfgs/igev_pp/igev_pp_eval_middlebury.yaml \
  --ckpt ckpts/igev_pp/sceneflow.pth \
  --output_root outputs/middlebury_eval \
  --device cuda:0 \
  --max_disp 64 \
  --tau_px 10.0

# ACVNet
python tools/eval_single.py \
  --model_name acvnet_pretrained_sceneflow \
  --cfg cfgs/acvnet/acvnet_eval_middlebury.yaml \
  --ckpt ckpts/acvnet/sceneflow.ckpt \
  --output_root outputs/middlebury_eval \
  --device cuda:0 \
  --max_disp 64 \
  --tau_px 10.0

# GwcNet
python tools/eval_single.py \
  --model_name gwcnet_pretrained_sceneflow \
  --cfg cfgs/gwcnet/taein/gwcnet_eval_middlebury.yaml \
  --ckpt ckpts/gwcnet/checkpoint_000014.ckpt \
  --output_root outputs/middlebury_eval \
  --device cuda:0 \
  --max_disp 64 \
  --tau_px 10.0







# ------------------------------------------------------------ [ETH3D Dataset] ------------------------------------------------------
# IGEV++
python tools/eval_single.py \
  --model_name igev_pp_pretrained_sceneflow \
  --cfg cfgs/igev_pp/igev_pp_eval_eth3d.yaml \
  --ckpt ckpts/igev_pp/sceneflow.pth \
  --output_root outputs/eth3d_eval \
  --device cuda:0 \
  --max_disp 64 \
  --tau_px 10.0

# ACVNet
python tools/eval_single.py \
  --model_name acvnet_pretrained_sceneflow \
  --cfg cfgs/acvnet/acvnet_eval_eth3d.yaml \
  --ckpt ckpts/acvnet/sceneflow.ckpt \
  --output_root outputs/eth3d_eval \
  --device cuda:0 \
  --max_disp 64 \
  --tau_px 10.0

# GwcNet
python tools/eval_single.py \
  --model_name gwcnet_pretrained_sceneflow \
  --cfg cfgs/gwcnet/taein/gwcnet_eval_eth3d.yaml \
  --ckpt ckpts/gwcnet/checkpoint_000014.ckpt \
  --output_root outputs/eth3d_eval \
  --device cuda:0 \
  --max_disp 64 \
  --tau_px 10.0