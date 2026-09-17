export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDNN_BENCHMARK=True
export TORCHAMP_USE_FP16_REDUCE=True
export PYTHONUNBUFFERED=1


# python tools/train.py \
#     --cfg_file=cfgs/lightgev/lightgev_train_sceneflow_no_gev.yaml \
#     --fix_random_seed \
#     --save_root_dir=train_outputs --extra_tag='ablation' \
#     --cover_old_exp --workers 8 --pin_memory

# python tools/train.py \
#     --cfg_file=cfgs/lightgev/lightgev_train_sceneflow_aggregation_only.yaml \
#     --fix_random_seed \
#     --save_root_dir=train_outputs --extra_tag='ablation' \
#     --cover_old_exp --workers 8 --pin_memory

# python tools/train.py \
#     --cfg_file=cfgs/lightgev/lightgev_train_sceneflow_refiner_only.yaml \
#     --fix_random_seed \
#     --save_root_dir=train_outputs --extra_tag='ablation' \
#     --cover_old_exp --workers 8 --pin_memory

# python tools/train.py \
#     --cfg_file=cfgs/lightgev/lightgev_train_sceneflow_full_iter_0.yaml \
#     --fix_random_seed \
#     --save_root_dir=train_outputs --extra_tag='ablation' \
#     --cover_old_exp --workers 8 --pin_memory

# python tools/train.py \
#     --cfg_file=cfgs/lightgev/lightgev_train_sceneflow_full_iter_2.yaml \
#     --fix_random_seed \
#     --save_root_dir=train_outputs --extra_tag='ablation' \
#     --cover_old_exp --workers 8 --pin_memory

# python tools/train.py \
#     --cfg_file=cfgs/lightgev/lightgev_train_sceneflow_full_iter_4.yaml \
#     --fix_random_seed \
#     --save_root_dir=train_outputs --extra_tag='ablation' \
#     --cover_old_exp --workers 8 --pin_memory

# python tools/train.py \
#     --cfg_file=cfgs/lightgev/lightgev_train_sceneflow_full_iter_6.yaml \
#     --fix_random_seed \
#     --save_root_dir=train_outputs --extra_tag='ablation' \
#     --cover_old_exp --workers 8 --pin_memory

# python tools/train.py \
#     --cfg_file=cfgs/lightgev/lightgev_train_sceneflow_full_ch_8.yaml \
#     --fix_random_seed \
#     --save_root_dir=train_outputs --extra_tag='ablation' \
#     --cover_old_exp --workers 8 --pin_memory

# python tools/train.py \
#     --cfg_file=cfgs/lightgev/lightgev_train_sceneflow_full_ch_32.yaml \
#     --fix_random_seed \
#     --save_root_dir=train_outputs --extra_tag='ablation' \
#     --cover_old_exp --workers 8 --pin_memory

# python tools/train.py \
#     --cfg_file=cfgs/lightgev/lightgev_train_sceneflow_full_dil_1.yaml \
#     --fix_random_seed \
#     --save_root_dir=train_outputs --extra_tag='ablation' \
#     --cover_old_exp --workers 8 --pin_memory

# python tools/train.py \
#     --cfg_file=cfgs/lightgev/lightgev_train_sceneflow_full_dil_2.yaml \
#     --fix_random_seed \
#     --save_root_dir=train_outputs --extra_tag='ablation' \
#     --cover_old_exp --workers 8 --pin_memory

# python tools/train.py \
#     --cfg_file=cfgs/lightgev/lightgev_train_sceneflow_aggregation_only.yaml \
#     --fix_random_seed \
#     --save_root_dir=train_outputs --extra_tag='pretrain_trainBS4_valBS4_LR0.0004_EP30_fixed' \
#     --cover_old_exp --workers 8 --pin_memory

python tools/train.py \
    --cfg_file=cfgs/lightgev/lightgev_train_sceneflow_full_iter_4.yaml \
    --fix_random_seed \
    --save_root_dir=train_outputs --extra_tag='pretrain_trainBS4_valBS4_LR0.0004_EP30_fixed' \
    --cover_old_exp --workers 8 --pin_memory