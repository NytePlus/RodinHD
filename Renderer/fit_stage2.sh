export PATH=/usr/local/cuda/bin:~/.local/bin:$PATH
fitting_obj_list=/home/wcc/RodinHD/data/metahuman_data/fitting_obj_list.txt
save_dir=/home/wcc/RodinHD/data/save_triplane2
ckpt=/home/wcc/RodinHD/data/save_triplane_and_mlp3/checkpoints/ngp_ep0017.pth.pth
data_root=/home/wcc/RodinHD/data/metahuman_data
python main.py \
    ${fitting_obj_list} \
    ${save_dir} \
    --dataset metahuman \
    --workspace ${save_dir} \
    -O --start_idx 0 --end_idx 1 \
    --bound 1.0 --scale 2.5 --dt_gamma 0 \
    --triplane_channels 32 \
    --data_root ${data_root} \
    --ckpt ${ckpt} \
    --iters 30000  --lr1 0 --eval_freq 10 \
    --num_rays 32768
