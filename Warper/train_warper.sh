export PATH=/usr/local/cuda/bin:~/.local/bin:$PATH
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
save_dir=/home/wcc/RodinHD/data/save_warping_module256
src_root=/home/wcc/RodinHD/data/save_triplane_and_mlp256
src_data=/home/wcc/RodinHD/data/portrait3dclip_data
tgt_data=/home/wcc/RodinHD/data/portrait3dexpclip_data
tgt_root=/home/wcc/RodinHD/data/save_triplane256
latent_root=/home/wcc/RodinHD/data/emo_feature
ms_feature_root=/home/wcc/RodinHD/data/emo_feature/ms_latent
fitting_obj_list=/home/wcc/RodinHD/data/portrait3dclip_data/fitting_obj_list.txt
ckpt_dir=latest
mpirun -np 4 python main.py \
    ${fitting_obj_list} \
    ${save_dir} \
    --fp16 \
    --src_root ${src_root} \
    --src_data ${src_data} \
    --tgt_data ${tgt_data} \
    --tgt_root ${tgt_root} \
    --latent_root ${latent_root} \
    --ms_feature_root ${ms_feature_root} \
    --dataset portrait3d \
    --workspace ${save_dir} \
    --bound 1 --scale 1 --dt_gamma 0 \
    --start_idx 0 --end_idx 55 \
    --batch_size 30 \
    --triplane_channels 8 --resolution0 128 --resolution1 128 \
    --ckpt ${ckpt_dir} \
    --num_epochs 2000 --lr0 2e-5 --eval_freq 10 \
    --num_rays 32768
