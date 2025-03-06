export PATH=/usr/local/cuda/bin:~/.local/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
fitting_obj_list=/home/wcc/RodinHD/data/metahuman_data/fitting_obj_list.txt
save_dir=/home/wcc/RodinHD/data/save_triplane_and_mlp4/4plus
ckpt=/home/wcc/RodinHD/data/save_triplane_and_mlp4/checkpoints/ngp_ep0053.pth.pth
data_root=/home/wcc/RodinHD/data/metahuman_data
python main.py \
    ${fitting_obj_list} \
    ${save_dir} \
    --dataset metahuman \
    --workspace ${save_dir} \
    -O --start_idx 0 --end_idx 1 \
    --bound 1.0 --scale 1 --dt_gamma 0 \
    --downscale 1 --resolution0 512 --resolution1 512 \
    --triplane_channels 32 \
    --data_root ${data_root} \
    --ckpt ${ckpt} \
    --iters 2000  --lr1 0 --lr0 2e-2 --eval_freq 30 \
    --num_rays 8192 \
    --l1_reg_weight 0 \
    --tv_weight 0 \
    --dist_weight 0 \
    --iwc_weight 0
