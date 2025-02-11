export PATH=/usr/local/cuda/bin:~/.local/bin:$PATH
export OMP_NUM_THREADS=4
export MAX_JOBS=2
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
fitting_obj_list=/home/wcc/RodinHD/data/portrait3dexpclip_data/fitting_obj_list.txt
save_dir=/home/wcc/RodinHD/data/save_triplane5
ckpt=/home/wcc/RodinHD/data/save_triplane_and_mlp5/checkpoints/ngp_ep0040.pth.pth
data_root=/home/wcc/RodinHD/data/portrait3dexpclip_data
mpirun -np 6 python main.py \
    ${fitting_obj_list} \
    ${save_dir} \
    --workspace ${save_dir} \
    -O --start_idx 585 --end_idx 891 \
    -finetune \
    --bound 1.0 --scale 1 --dt_gamma 0 \
    --triplane_channels 32 \
    --data_root ${data_root} \
    --ckpt ${ckpt} \
    --iters 20000  --lr1 0 --lr0 2e-3 --eval_freq 30 \
    --num_rays 8192 --max_ray_batch 8192 \
    --l1_reg_weight 0 \
    --tv_weight 0 \
    --dist_weight 0 \
    --iwc_weight 0
