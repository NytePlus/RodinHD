export PATH=/usr/local/cuda/bin:~/.local/bin:$PATH
export OMP_NUM_THREADS=4
export MAX_JOBS=2
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
save_dir=/home/wcc/RodinHD/data/save_expression_mlp3
data_root=/home/wcc/RodinHD/data/metahuman_data
feat_root=/home/wcc/RodinHD/data/metahuman_exp/latent
fitting_obj_list=/home/wcc/RodinHD/data/metahuman_data/fitting_obj_list.txt
ckpt_dir=latest
mpirun -np 6 python main.py \
    ${fitting_obj_list} \
    ${save_dir} \
    --workspace ${save_dir} \
    -O \
    --start_idx 0 --end_idx 126 \
    --bound 1.0 --scale 1 --dt_gamma 0 \
    --downscale 2 --resolution0 256 --resolution1 256 --triplane_channels 8 --expression_channels 24\
    --ckpt ${ckpt_dir} \
    --data_root ${data_root} \
    --out_loop_eps 30 --iters 3400 --lr0 2e-2 --lr1 2e-3 --eval_freq 20 \
    --l1_reg_weight 1e-4 \
    --tv_weight 1e-2 \
    --dist_weight 0 \
    --iwc_weight 0.1 \
    --num_rays 1280
