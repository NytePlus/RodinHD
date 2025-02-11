export PATH=/usr/local/cuda/bin:~/.local/bin:$PATH
save_dir=/home/wcc/RodinHD/data/save_warping_module
src_root=/home/wcc/RodinHD/data/save_triplane_and_mlp2
src_data=/home/wcc/RodinHD/data/portrait3d_data
tgt_data=/home/wcc/RodinHD/data/portrait3dexp_data
latent_root=/home/wcc/RodinHD/data/feature2/latent
fitting_obj_list=/home/wcc/RodinHD/data/portrait3d_data/fitting_obj_list.txt
ckpt_dir=latest
mpirun -np 4 python main.py \
    ${fitting_obj_list} \
    ${save_dir} \
    --fp16 \
    --src_root ${src_root} \
    --src_data ${src_data} \
    --tgt_data ${tgt_data} \
    --latent_root ${latent_root} \
    --dataset portrait3d \
    --workspace ${save_dir} \
    --bound 1 --scale 0.8 --dt_gamma 0 \
    --start_idx 0 --end_idx 2 \
    --triplane_channels 32 \
    --ckpt ${ckpt_dir} \
    --num_epochs 100 --lr0 2e-3 --eval_freq 10 \
    --num_rays 32768
