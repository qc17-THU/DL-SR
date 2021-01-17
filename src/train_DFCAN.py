import argparse
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
import datetime
from keras.callbacks import TensorBoard
import glob
import os
import tensorflow as tf
from models import *
from utils.lr_controller import ReduceLROnPlateau
from utils.data_loader import data_loader, data_loader_multi_channel
from utils.utils import img_comp
from utils.loss import loss_mse_ssim

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=int, default=1)
parser.add_argument("--gpu_memory_fraction", type=float, default=0.3)
parser.add_argument("--mixed_precision_training", type=int, default=1)
parser.add_argument("--data_dir", type=str, default="../dataset/train/F-actin")
parser.add_argument("--save_weights_dir", type=str, default="../trained_models")
parser.add_argument("--model_name", type=str, default="DFCAN")
parser.add_argument("--patch_height", type=int, default=128)
parser.add_argument("--patch_width", type=int, default=128)
parser.add_argument("--input_channels", type=int, default=9)
parser.add_argument("--scale_factor", type=int, default=2)
parser.add_argument("--norm_flag", type=int, default=1)
parser.add_argument("--iterations", type=int, default=1000000)
parser.add_argument("--sample_interval", type=int, default=1000)
parser.add_argument("--validate_interval", type=int, default=2000)
parser.add_argument("--validate_num", type=int, default=500)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--start_lr", type=float, default=1e-4)
parser.add_argument("--lr_decay_factor", type=float, default=0.5)
parser.add_argument("--load_weights", type=int, default=0)
parser.add_argument("--optimizer_name", type=str, default="adam")

args = parser.parse_args()
gpu_id = str(args.gpu_id)
gpu_memory_fraction = args.gpu_memory_fraction
mixed_precision_training = str(args.mixed_precision_training)
data_dir = args.data_dir
save_weights_dir = args.save_weights_dir
validate_interval = args.validate_interval
batch_size = args.batch_size
start_lr = args.start_lr
lr_decay_factor = args.lr_decay_factor
patch_height = args.patch_height
patch_width = args.patch_width
input_channels = args.input_channels
scale_factor = args.scale_factor
norm_flag = args.norm_flag
validate_num = args.validate_num
iterations = args.iterations
load_weights = args.load_weights
optimizer_name = args.optimizer_name
model_name = args.model_name
sample_interval = args.sample_interval

os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = mixed_precision_training
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

data_name = data_dir.split('/')[-1]
if input_channels == 1:
    save_weights_name = model_name + '-SISR_' + data_name
    cur_data_loader = data_loader
    train_images_path = data_dir + '/training_wf/'
    validate_images_path = data_dir + '/validate_wf/'
else:
    save_weights_name = model_name + '-SIM_' + data_name
    cur_data_loader = data_loader_multi_channel
    train_images_path = data_dir + '/training/'
    validate_images_path = data_dir + '/validate/'
save_weights_path = save_weights_dir + '/' + save_weights_name + '/'
train_gt_path = data_dir + '/training_gt/'
validate_gt_path = data_dir + '/validate_gt/'
sample_path = save_weights_path + 'sampled_img/'

if not os.path.exists(save_weights_path):
    os.mkdir(save_weights_path)
if not os.path.exists(sample_path):
    os.mkdir(sample_path)

# --------------------------------------------------------------------------------
#                           select models and optimizer
# --------------------------------------------------------------------------------
modelFns = {'DFCAN': DFCAN16.DFCAN}
modelFN = modelFns[model_name]
optimizer_g = optimizers.adam(lr=start_lr, beta_1=0.9, beta_2=0.999)

# --------------------------------------------------------------------------------
#                              define combined model
# --------------------------------------------------------------------------------
g = modelFN((patch_height, patch_width, input_channels))
g.compile(loss=loss_mse_ssim, optimizer=optimizer_g)
lr_controller = ReduceLROnPlateau(model=g, factor=lr_decay_factor, patience=10, mode='min', min_delta=1e-4,
                                  cooldown=0, min_lr=start_lr * 0.1, verbose=1)

# --------------------------------------------------------------------------------
#                                 about Tensorboard
# --------------------------------------------------------------------------------
log_path = save_weights_path + 'graph'
if not os.path.exists(log_path):
    os.mkdir(log_path)
callback = TensorBoard(log_path)
callback.set_model(g)
train_names = 'training_loss'
val_names = ['val_MSE', 'val_SSIM', 'val_PSNR', 'val_NRMSE']


def write_log(callback, names, logs, batch_no):
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = logs
    summary_value.tag = names
    callback.writer.add_summary(summary, batch_no)
    callback.writer.flush()


# --------------------------------------------------------------------------------
#                             Sample and validate
# --------------------------------------------------------------------------------
def Validate(iter, sample=0):
    validate_path = glob.glob(validate_images_path + '*')
    validate_path.sort()
    if sample == 1:
        r, c = 3, 3
        mses, nrmses, psnrs, ssims = [], [], [], []
        img_show, gt_show, output_show = [], [], []
        validate_path = np.random.choice(validate_path, size=r)
        for path in validate_path:
            [img, gt] = cur_data_loader([path], validate_images_path, validate_gt_path, patch_height,
                                        patch_width, 1, norm_flag=norm_flag, scale=scale_factor)
            output = np.squeeze(g.predict(img))
            mses, nrmses, psnrs, ssims = img_comp(gt, output, mses, nrmses, psnrs, ssims)
            img_show.append(np.squeeze(np.mean(img, 3)))
            gt_show.append(np.squeeze(gt))
            output_show.append(output)
            # show some examples
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            axs[row, 1].set_title('MSE=%.4f, SSIM=%.4f, PSNR=%.4f' % (mses[row], ssims[row], psnrs[row]))
            for col, image in enumerate([img_show, output_show, gt_show]):
                axs[row, col].imshow(np.squeeze(image[row]))
                axs[row, col].axis('off')
            cnt += 1
        fig.savefig(sample_path + '%d.png' % iter)
        plt.close()
    else:
        if validate_num < validate_path.__len__():
            validate_path = validate_path[0:validate_num]
        mses, nrmses, psnrs, ssims = [], [], [], []
        for path in validate_path:
            [img, gt] = cur_data_loader([path], validate_images_path, validate_gt_path, patch_height,
                                        patch_width, 1, norm_flag=norm_flag, scale=scale_factor)
            output = np.squeeze(g.predict(img))
            mses, nrmses, psnrs, ssims = img_comp(gt, output, mses, nrmses, psnrs, ssims)

        # if best, save weights.best
        g.save_weights(save_weights_path + 'weights.latest')
        if min(validate_nrmse) > np.mean(nrmses):
            g.save_weights(save_weights_path + 'weights.best')

        validate_nrmse.append(np.mean(nrmses))
        curlr = lr_controller.on_epoch_end(iter, np.mean(nrmses))
        write_log(callback, val_names[0], np.mean(mses), iter)
        write_log(callback, val_names[1], np.mean(ssims), iter)
        write_log(callback, val_names[2], np.mean(psnrs), iter)
        write_log(callback, val_names[3], np.mean(nrmses), iter)
        write_log(callback, 'lr', curlr, iter)


# --------------------------------------------------------------------------------
#                             if exist, load weights
# --------------------------------------------------------------------------------
if load_weights:
    if os.path.exists(save_weights_path + 'weights.best'):
        g.save_weights(save_weights_path + 'weights.best')
        print('Loading weights successfully: ' + save_weights_path + 'weights.best')
    elif os.path.exists(save_weights_path + 'weights.latest'):
        g.save_weights(save_weights_path + 'weights.latest')
        print('Loading weights successfully: ' + save_weights_path + 'weights.latest')


# --------------------------------------------------------------------------------
#                                    training
# --------------------------------------------------------------------------------
start_time = datetime.datetime.now()
loss_record = []
validate_nrmse = [np.Inf]
lr_controller.on_train_begin()
images_path = glob.glob(train_images_path + '/*')
for it in range(iterations):
    # ------------------------------------
    #         train generator
    # ------------------------------------
    input_g, gt_g = cur_data_loader(images_path, train_images_path, train_gt_path, patch_height, patch_width,
                                    batch_size, norm_flag=norm_flag, scale=scale_factor)
    loss_generator = g.train_on_batch(input_g, gt_g)
    loss_record.append(loss_generator)

    elapsed_time = datetime.datetime.now() - start_time
    print("%d epoch: time: %s, g_loss = %s" % (it + 1, elapsed_time, loss_generator))

    if (it + 1) % sample_interval == 0:
        images_path = glob.glob(train_images_path + '/*')
        Validate(it + 1, sample=1)

    if (it + 1) % validate_interval == 0:
        Validate(it + 1, sample=0)
        write_log(callback, train_names, np.mean(loss_record), it + 1)
        loss_record = []
