export PATH=/usr/local/cuda/bin:~/.local/bin:$PATH
export OMP_NUM_THREADS=4
export MAX_JOBS=2
save_dir=/home/wcc/RodinHD/data/save_triplane_and_mlp_test
data_root=/home/wcc/RodinHD/data/raw_data_300
fitting_obj_list=/home/wcc/RodinHD/data/fitting_obj_list_300.txt
ckpt_dir=latest
mpirun -np 4 python main.py \
    ${fitting_obj_list} \
    ${save_dir} \
    --workspace ${save_dir} \
    -O \
    --start_idx 0 --end_idx 64 \
    --bound 1.0 --scale 0.4 --dt_gamma 0 \
    --triplane_channels 32 \
    --ckpt ${ckpt_dir} \
    --data_root ${data_root} \
    --out_loop_eps 30 --iters 5000 --lr0 2e-2 --lr1 2e-4 --eval_freq 1 \
    --l1_reg_weight 1e-4 \
    --tv_weight 1e-2 \
    --dist_weight 0 \
    --iwc_weight 0.1 \
    --num_rays -1 --max_ray_batch 2048 \
    --max_steps 512 --downscale 64 \
    --ray_shuffle \
    --no_grid
