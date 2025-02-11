export PATH=/usr/local/cuda/bin:~/.local/bin:$PATH
save_dir=/home/wcc/RodinHD/data/save_warping_module2
src_root=/home/wcc/RodinHD/data/save_triplane_and_mlp3
src_data=/home/wcc/RodinHD/data/portrait3d_data
tgt_root=/home/wcc/RodinHD/data/save_triplane2
tgt_data=/home/wcc/RodinHD/data/portrait3dexp_data
fitting_obj_list=/home/wcc/RodinHD/data/portrait3d_data/fitting_obj_list.txt
ckpt_dir=/home/wcc/RodinHD/data/save_wraping_module2
python main.py \
    ${fitting_obj_list} \
    --src_root ${src_root} --src_data ${src_data}\
    --tgt_root ${tgt_root} --tgt_data ${tgt_data}\
    --workspace ${save_dir} \
    --start_idx 0 --end_idx 2 \
    --triplane_channels 32 \
    --ckpt ${ckpt_dir} \
    --num_epochs 100 --lr0 2e-3 --eval_freq 10
