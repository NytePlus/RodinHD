export PATH=/usr/local/cuda/bin:~/.local/bin:$PATH
export OMP_NUM_THREADS=4
export MAX_JOBS=2
fitting_obj_list=/home/wcc/RodinHD/data/portrait3dexp_data/fitting_obj_list.txt
save_dir=/home/wcc/RodinHD/data/save_triplane
ckpt=/home/wcc/RodinHD/data/save_triplane/checkpoints/ngp_ep0017.pth.pth
data_root=/home/wcc/RodinHD/data/portrait3dexp_data
mpirun -np 8 python main.py \
    ${fitting_obj_list} \
    ${save_dir} \
    --workspace ${save_dir} \
    -finetune \
    -O --start_idx 0 --end_idx 192 \
    --bound 1.0 --scale 0.8 --dt_gamma 0 \
    --triplane_channels 32 \
    --data_root ${data_root} \
    --ckpt ${ckpt} \
    --out_loop_eps 1 \
    --iters 3000  --lr1 0 --lr0 2e-3 --eval_freq 10 \
    --num_rays 16384 --max_ray_batch 8192 \
    --l1_reg_weight 0 \
    --tv_weight 0 \
    --dist_weight 0 \
    --iwc_weight 0
