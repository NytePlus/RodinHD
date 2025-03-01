export PATH=/usr/local/cuda/bin:~/.local/bin:$PATH
save_dir=/home/wcc/RodinHD/data/save_triplane_and_mlp3/validation
data_root=/home/wcc/RodinHD/data/portrait3d_data
fitting_obj_list=/home/wcc/RodinHD/data/portrait3d_data/fitting_obj_list.txt
ckpt_dir=/home/wcc/RodinHD/data/save_triplane_and_mlp3/checkpoints/ngp_ep0017.pth.pth
mpirun -np 6 python main.py \
    ${fitting_obj_list} \
    ${save_dir} \
    --workspace ${save_dir} \
    -O \
    --start_idx 0 --end_idx 1 \
    --bound 1.0 --scale 0.8 --dt_gamma 0 \
    --triplane_channels 32 \
    --ckpt ${ckpt_dir} \
    --data_root ${data_root} \
    --test \
    --eval_video
