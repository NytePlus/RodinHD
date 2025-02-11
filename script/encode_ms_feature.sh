root_path=/home/wcc/RodinHD/data/portrait3dexp_data
output_dir=/home/wcc/RodinHD/data/feature2
txt_file=/home/wcc/RodinHD/data/portrait3dexp_data/fitting_obj_list.txt
vae_dir=/home/wcc/RodinHD/script/vae_model

CUDA_VISIBLE_DEVICES=0 python encode_multiscale_feature.py \
  --root ${root_path} --output_dir ${output_dir} --txt_file ${txt_file} --vae_dir ${vae_dir}\
  --start_idx 0 --end_idx 64