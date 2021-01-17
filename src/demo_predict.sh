#!/usr/bin/python3.6

# ------------------------------- arguments for 2D model -------------------------------
# --gpu_id: the gpu device you want to use in current task
# --gpu_memory_fraction: upper bound of gpu memory fraction that can be used
# --data_dir: the root path of test data folder
# --folder_test: the raw SIM image or WF iamge folder
# --model_name: select 'DFCAN' or 'DFGAN' to perform SR image prediction
# --model_weights: the pre-trained model file to be loaded
# --input_height: the height of input images
# --input_width: the width of input images
# --scale_factor: 2 for linear SIM and 3 for nonlinear SIM

# ------------------------------- predict with DFCAN SISR model  -------------------------------
python predict.py --gpu_id '0' --gpu_memory_fraction 0.3 \
                  --data_dir "../dataset/test/F-actin" \
                  --folder_test "input_wide_field_images" \
                  --model_name "DFCAN" \
                  --model_weights "../trained_models/DFCAN-SISR_F-actin/weights.best" \
                  --input_height 502 --input_width 502 --scale_factor 2

# ------------------------------- predict with DFCAN SIM reconstruction model  -------------------------------
python predict.py --gpu_id '0' --gpu_memory_fraction 0.3 \
                  --data_dir "../dataset/test/F-actin" \
                  --folder_test "input_raw_sim_images" \
                  --model_name "DFCAN" \
                  --model_weights "../trained_models/DFCAN-SIM_F-actin/weights.best" \
                  --input_height 502 --input_width 502 --scale_factor 2

# ------------------------------- predict with DFGAN SISR model  -------------------------------
python predict.py --gpu_id '0' --gpu_memory_fraction 0.3 \
                  --data_dir "../dataset/test/F-actin" \
                  --folder_test "input_wide_field_images" \
                  --model_name "DFGAN" \
                  --model_weights "../trained_models/DFGAN-SISR_F-actin/weights.best" \
                  --input_height 502 --input_width 502 --scale_factor 2

# ------------------------------- predict with DFGAN SIM reconstruction model  -------------------------------
python predict.py --gpu_id '0' --gpu_memory_fraction 0.3 \
                  --data_dir "../dataset/test/F-actin" \
                  --folder_test "input_raw_sim_images" \
                  --model_name "DFGAN" \
                  --model_weights "../trained_models/DFGAN-SIM_F-actin/weights.best" \
                  --input_height 502 --input_width 502 --scale_factor 2

