data_dir=/home/wcc/RodinHD/data/portrait3d_data
output_path=/home/wcc/RodinHD/data/save_diffusion
base_diffusion_ckpt=/home/wcc/RodinHD/data/save_diffusion/checkpoints/model020000.pt
render_ckpt=/home/wcc/RodinHD/data/save_triplane_and_mlp2/checkpoints/ngp_ep0017.pth.pth
txt_file=/home/wcc/RodinHD/data/portrait3d_data/fitting_obj_list.txt
render_cam_path=/home/wcc/RodinHD/data/portrait3d_data
latent_root=/home/wcc/RodinHD/data/feature/latent
ms_feature_root=/home/wcc/RodinHD/data/feature/ms_latent
num_samples=10

MODEL_FLAGS="--learn_sigma True --uncond_p 0. --image_size 128  --diffusion_steps 1000 --sample_respacing 10 --predict_xstart False --model_path $base_diffusion_ckpt "
TRAIN_FLAGS="--lr 1e-5 --batch_size 1 --schedule_sampler loss-second-moment  --use_tv False " 
DIFFUSION_FLAGS="--noise_schedule cosine_light"
SAMPLE_FLAGS="--num_samples $num_samples --sample_c 1.3 --exp_name $output_path --eta 1"
DATASET_FLAGS="--data_dir $data_dir --mode triplane --start_idx 0 --end_idx $num_samples --txt_file $txt_file --latent_root $latent_root --ms_feature_root $ms_feature_root"
python ../base_sample.py $MODEL_FLAGS  $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS  $DATASET_FLAGS 

cd ..
cd Renderer
python main.py $txt_file  $output_path/LR  --workspace $output_path/render_lr --data_root $render_cam_path -O --start_idx 0 --end_idx $num_samples --bound 1.0 --scale 0.8 --dt_gamma 0 --ckpt $render_ckpt  --iters 15000 --lr1 0 --test
