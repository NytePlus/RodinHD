export PATH=/usr/local/cuda/bin:~/.local/bin:$PATH
save_dir=/home/wcc/RodinHD/data/save_triplane_and_mlp_300_test
data_root=/home/wcc/RodinHD/data/raw_data_300
fitting_obj_list=/home/wcc/RodinHD/data/fitting_obj_list_300.txt
ckpt_dir=latest
python main.py \
    ${fitting_obj_list} \
    ${save_dir} \
    --workspace ${save_dir} \
    -O \
    --start_idx 0 --end_idx 1 \
    --bound 1.0 --scale 0.4 --dt_gamma 0 \
    --triplane_channels 32 \
    --ckpt ${ckpt_dir} \
    --data_root ${data_root} \
    --out_loop_eps 30 --iters 5000 --lr0 2e-2 --lr1 2e-3 --eval_freq 1 \
    --l1_reg_weight 1e-4 \
    --tv_weight 1e-2 \
    --dist_weight 0 \
    --iwc_weight 0.1 \
    --num_rays -1 \
    --max_steps 1024 \
    --ray_shuffle \
    --max_ray_batch 8192 \
    --update_extra_interval 1 \
    --downscale 2
