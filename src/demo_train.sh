#!/usr/bin/python3.6

# ------------------------------- traning arguments -------------------------------
# --gpu_id: the gpu device you want to use in current task
# --gpu_memory_fraction: upper bound of gpu memory fraction that can be used
# --mixed_precision_training: whether use mixed precision training or not
# --data_dir: the root path of training data folder
# --train_input_folder: the input images of network (raw SIM image or WF iamge)
# --train_gt_folder: the gt images of training (GT-SIM image)
# --validate_input_folder: the input images of network (raw SIM image or WF iamge) for validation
# --validate_gt_folder: the gt images for validation (GT-SIM image)
# --model_name: 'DFCAN' or 'DFGAN'
# --save_weights_dir: root directory where models weights will be saved in
# --save_weights_name: folder name where model weights will be saved in
# --patch_height: the height of input image patches
# --patch_width: the width of input image patches
# --input_channels: 1 for WF input and 9 for SIM reconstruction
# --scale_factor: 2 for linear SIM and 3 for nonlinear SIM
# --iterations: total training iterations
# --batch_size: batch size for training
# --start_lr: initial learning rate of training, typically set as 10-4
# --lr_decay_factor: learning rate decay factor, typically set as 0.5

# --d_start_lr: initial learning rate for the discriminator (only for DFGAN model)
# --g_start_lr: initial learning rate for the generator (only for DFGAN model)
# --train_discriminator_times: times of training discriminator in each iteration (only for DFGAN model)
# --train_generator_times: times of training generator in each iteration (only for DFGAN model)

# ------------------------------- train a DFCAN model -------------------------------
# input_channels: 1 for SISR model, 9 for SIM reconstruction model
python train_DFCAN.py --gpu_id '0' --gpu_memory_fraction 0.3 --mixed_precision_training 1 \
                      --data_dir "../dataset/train/F-actin" \
                      --save_weights_dir "../trained_models" \
                      --patch_height 128 --patch_width 128 --input_channels 9 \
                      --scale_factor 2 --iterations 200000 --sample_interval 1000 \
                      --validate_interval 2000 --validate_num 500 --batch_size 4 \
                      --start_lr 1e-4 --lr_decay_factor 0.5 --load_weights 0 \
                      --model_name "DFCAN" --optimizer_name "adam"

# ------------------------------- train a DFGAN model -------------------------------
# input_channels: 1 for SISR model, 9 for SIM reconstruction model
python train_DFGAN.py --gpu_id '0' --gpu_memory_fraction 0.3 --mixed_precision_training 1 \
                      --data_dir "../dataset/train/F-actin" \
                      --save_weights_dir "../trained_models" \
                      --patch_height 128 --patch_width 128 --input_channels 9 \
                      --scale_factor 2 --iterations 200000 --sample_interval 500 \
                      --validate_interval 1000 --validate_num 500 --batch_size 2 \
                      --d_start_lr 2e-5 --g_start_lr 1e-4 --lr_decay_factor 0.5 \
                      --train_discriminator_times 1 --train_generator_times 3 \
                      --load_weights 0 --model_name "DFGAN" --optimizer_name "adam"
