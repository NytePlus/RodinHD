root_path=/home/wcc/RodinHD/data/portrait3d_data
output_dir=/home/wcc/RodinHD/data/feature
txt_file=/home/wcc/RodinHD/data/portrait3d_data/fitting_obj_list.txt
vae_dir=/home/wcc/RodinHD/script/models--stabilityai--stable-diffusion-xl-base-1.0
python encode_multiscale_feature.py \
  --root {root_path} --output_dir {output_dir} --txt_file {txt_file} --vae_dir {vae_dir}\
  --start_idx 0 --end_idx 64