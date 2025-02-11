output_path=/home/wcc/RodinHD/data/save_updiffusion_2
txt_file=/home/wcc/RodinHD/data/save_triplane_and_mlp2/fitting_obj_list.txt
hr_triplane_path=/home/wcc/RodinHD/data/save_triplane_and_mlp2
image_root=/home/wcc/RodinHD/data/portrait3d_data
render_ckpt=/home/wcc/RodinHD/data/save_triplane_and_mlp2/checkpoints/ngp_ep0017.pth.pth
latent_root=/home/wcc/RodinHD/data/feature/latent
num_samples=10000

MODEL_FLAGS="--learn_sigma False --uncond_p 0 --image_size 512 --super_res 128 --finetune_decoder True --diffusion_steps 100 --predict_xstart True --n_feats 64 --ch_mult 1 2 4 --local_rank 0 --predict_type xstart --decoder_type vae_latent" 
TRAIN_FLAGS="--lr 1e-5 --batch_size 1 --schedule_sampler uniform --use_tv False --exp_name $output_path --log_interval 100 --save_interval 5000 --render_weight 1 --render_lpips_weight 0.5 --patch_size 64 --use_tv False --dtype 16 --use_renderer $render_ckpt --use_checkpoint True --use_vgg True "
DIFFUSION_FLAGS="--noise_schedule sigmoid  "
SAMPLE_FLAGS="--num_samples 10 --sample_c 1.0"
DATASET_FLAGS="--data_dir $hr_triplane_path --mode triplane --start_idx 0 --end_idx $num_samples --txt_file $txt_file --latent_root $latent_root "
DEEPSPEED_FLAGS="--deepspeed_config pretrained_diffusion/configs/upsample_deepspeed.json"

deepspeed upsample_train.py $MODEL_FLAGS  $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS  $DATASET_FLAGS $DEEPSPEED_FLAGS 