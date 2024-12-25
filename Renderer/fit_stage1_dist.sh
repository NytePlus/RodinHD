export PATH=/usr/local/cuda/bin:~/.local/bin:$PATH
export OMP_NUM_THREADS=4
export MAX_JOBS=2
save_dir=/home/wcc/RodinHD/data/save_triplane_and_mlp2
data_root=/home/wcc/RodinHD/data/portrait3d_data
fitting_obj_list=/home/wcc/RodinHD/data/portrait3d_data/fitting_obj_list.txt
ckpt_dir=latest
mpirun -np 6 python main.py \
    ${fitting_obj_list} \
    ${save_dir} \
    --workspace ${save_dir} \
    -O \
    --dataset portrait3d\
    --start_idx 0 --end_idx 64 \
    --bound 1.0 --scale 0.8 --dt_gamma 0 \
    --triplane_channels 32 \
    --ckpt ${ckpt_dir} \
    --data_root ${data_root} \
    --out_loop_eps 30 --iters 5000 --lr0 2e-3 --lr1 2e-4 --eval_freq 10 \
    --l1_reg_weight 1e-4 \
    --tv_weight 1e-2 \
    --dist_weight 0 \
    --iwc_weight 0.1 \
    --num_rays 16384 --max_ray_batch 16384 \
    --max_steps 1024
