export PATH=/usr/local/cuda/bin:~/.local/bin:$PATH
export CUDA_VISIBLE_DEVICES=7
save_dir=/data1/wcc/RodinHD/data/save_warping_module256/validation/snapshot_1d
fitting_obj_list=/home/wcc/RodinHD/data/portrait3expclip_data/fitting_obj_list.txt
ckpt=/home/wcc/RodinHD/data/save_triplane_and_mlp256/checkpoints/ngp_ep0017.pth.pth
data_root=/home/wcc/RodinHD/data/portrait3expclip_data
python main.py \
    ${fitting_obj_list} \
    ${save_dir} \
    --workspace ${save_dir} \
    -O \
    --start_idx 0 --end_idx 4 \
    --bound 1.0 --scale 1 --dt_gamma 0 \
    --downscale 4 --resolution0 128 --resolution1 128 \
    --triplane_channels 8 \
    --ckpt ${ckpt} \
    --data_root ${data_root} \
    --test \
    --eval_video
