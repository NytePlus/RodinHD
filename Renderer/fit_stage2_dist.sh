export PATH=/usr/local/cuda/bin:~/.local/bin:$PATH
export OMP_NUM_THREADS=4
export MAX_JOBS=2
export CUDA_VISIBLE_DEVICES=2,3,4,5
fitting_obj_list=/home/wcc/RodinHD/data/portrait3dexpclip_data/fitting_obj_list.txt
save_dir=/home/wcc/RodinHD/data/save_triplane256c0
ckpt=/home/wcc/RodinHD/data/save_triplane_and_mlp256/checkpoints/ngp_ep0017.pth.pth
data_root=/home/wcc/RodinHD/data/portrait3dexpclip_data
mpirun -np 4 python main.py \
    ${fitting_obj_list} \
    ${save_dir} \
    --workspace ${save_dir} \
    -O --start_idx 0 --end_idx 1 \
    -finetune \
    --bound 1.0 --scale 1 --dt_gamma 0 \
    --downscale 4 --resolution0 128 --resolution1 128 \
    --triplane_channels 8 \
    --data_root ${data_root} \
    --ckpt ${ckpt} \
    --iters 20000  --lr1 0 --lr0 2e-2 --eval_freq 30 \
    --num_rays 24576 \
    --l1_reg_weight 0 \
    --tv_weight 0 \
    --dist_weight 0 \
    --iwc_weight 0
