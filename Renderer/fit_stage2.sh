export PATH=/usr/local/cuda/bin:~/.local/bin:$PATH
fitting_obj_list=/home/wcc/RodinHD/data/fitting_obj_list.txt
save_dir=/home/wcc/RodinHD/data/save_triplane
ckpt=/home/wcc/RodinHD/data/save_triplane_and_mlp/checkpoints/ngp_ep0016.pth.pth
data_root=/home/wcc/RodinHD/data/raw_data
rm -rf ${save_dir}
python main.py \
    ${fitting_obj_list} \
    ${save_dir} \
    --workspace ${save_dir} \
    -O --start_idx 64 --end_idx 7485 \
    --bound 1.0 --scale 0.4 --dt_gamma 0 \
    --triplane_channels 32 \
    --data_root ${data_root} \
    --ckpt ${ckpt} \
    --iters 30000  --lr1 0 --lr0 2e-4 --eval_freq 10 \
    --num_rays 6144 \
    --num_steps 512 --upsample_steps 512
