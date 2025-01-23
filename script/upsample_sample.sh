low_res_triplane_dir=/path/to/low_res_triplane_dir
output_path=/home/wcc/RodinHD/data/save_updiffusion
sr_diffusion_ckpt=/path/to/sr_diffusion_ckpt
render_ckpt=/home/wcc/RodinHD/data/save_triplane_and_mlp2/checkpoints/ngp_ep0017.pth.pth
txt_file=/home/wcc/RodinHD/data/save_triplane_and_mlp2/fitting_obj_list.txt
render_cam_path=/home/wcc/RodinHD/data/portrait3d_data
latent_root=/home/wcc/RodinHD/data/feature/latent
num_samples=10

MODEL_FLAGS="--learn_sigma False --model_path $sr_diffusion_ckpt --n_feats 64 --ch_mult 1 2 4"
TRAIN_FLAGS="--lr 1e-5 --batch_size 1  --use_tv False --predict_xstart True --diffusion_steps 100 --super_res 128 --predict_type xstart" 
DIFFUSION_FLAGS="--noise_schedule sigmoid --image_size 512"
SAMPLE_FLAGS="--num_samples 10 --sample_c 1 --exp_name $output_path --sample_respacing 10"
DATASET_FLAGS="--data_dir $low_res_triplane_dir --mode triplane --start_idx 0 --end_idx $num_samples --txt_file $txt_file --latent_root $latent_root"
python ../upsample_sample.py $MODEL_FLAGS  $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS  $DATASET_FLAGS --eta $eta

cd ..
cd Renderer
python main.py $txt_file  $output_path/HR  --workspace $output_path/render_hr --data_root $render_cam_path -O --start_idx 0 --end_idx $num_samples --bound 1.0 --scale 0.6 --dt_gamma 0 --ckpt $render_ckpt  --iters 15000 --lr1 0 --test --inference_real
    