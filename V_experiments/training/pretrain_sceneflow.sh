export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDNN_BENCHMARK=True
export TORCHAMP_USE_FP16_REDUCE=True
export PYTHONUNBUFFERED=1

# LightStereo
python tools/train.py \
    --cfg_file=cfgs/lightstereo/lightstereo_m_train_sceneflow.yaml \
    --fix_random_seed \
    --save_root_dir=train_outputs --extra_tag='pretrain_trainBS12_valBS12_LR0.0004_datafixed' \
    --cover_old_exp --workers 8 --pin_memory