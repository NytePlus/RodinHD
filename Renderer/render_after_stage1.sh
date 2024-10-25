export PATH=/usr/local/cuda/bin:~/.local/bin:$PATH
save_dir=/home/wcc/RodinHD/data/save_triplane_and_mlp
data_root=/home/wcc/RodinHD/data/raw_data
fitting_obj_list=/home/wcc/RodinHD/data/fitting_obj_list.txt
ckpt_dir=latest
python main.py \
    ${fitting_obj_list} \
    ${save_dir} \
    --workspace ${save_dir} \
    -O \
    --start_idx 0 --end_idx 2 \
    --bound 1.0 --scale 0.4 --dt_gamma 0 \
    --triplane_channels 32 \
    --ckpt ${ckpt_dir} \
    --data_root ${data_root} \
    --test \
    --eval_video \
    --grid_size 512
