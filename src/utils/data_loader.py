import numpy as np
import glob
import cv2 as cv
import imageio
from .utils import prctile_norm


def data_loader(images_path, data_path, gt_path, height, width, batch_size, norm_flag=1, resize_flag=0, scale=2, bn=0):
    batch_images_path = np.random.choice(images_path, size=batch_size)
    image_batch = []
    gt_batch = []
    for path in batch_images_path:
        img = imageio.imread(path).astype(np.float)
        if resize_flag == 1:
            img = cv.resize(img, (height*scale, width*scale))
        path_gt = path.replace(data_path, gt_path)
        gt = imageio.imread(path_gt).astype(np.float)
        if norm_flag:
            img = prctile_norm(img)
            gt = prctile_norm(gt)
        else:
            img = img / 65535
            gt = gt / 65535
        image_batch.append(img)
        gt_batch.append(gt)

    image_batch = np.array(image_batch)
    gt_batch = np.array(gt_batch)
    if bn:
        image_batch = (image_batch - 0.5) / 0.5
        gt_batch = (gt_batch - 0.5) / 0.5
    if resize_flag == 1:
        image_batch = image_batch.reshape((batch_size, width*scale, height*scale, 1))
    else:
        image_batch = image_batch.reshape((batch_size, width, height, 1))
    gt_batch = gt_batch.reshape((batch_size, width*scale, height*scale, 1))

    return image_batch, gt_batch


def data_loader_multi_channel(images_path, data_path, gt_path, height, width, batch_size, norm_flag=1, resize_flag=0, scale=2, wf=0):
    batch_images_path = np.random.choice(images_path, size=batch_size)
    image_batch = []
    gt_batch = []
    for path in batch_images_path:
        img_path = glob.glob(path + '/*.tif')
        img_path.sort()
        cur_img = []
        for cur in img_path:
            img = imageio.imread(cur).astype(np.float)
            if resize_flag == 1:
                img = cv.resize(img, (height * scale, width * scale))
            cur_img.append(img)

        path_gt = path.replace(data_path, gt_path) + '.tif'
        cur_gt = imageio.imread(path_gt).astype(np.float)

        if norm_flag:
            cur_img = prctile_norm(np.array(cur_img))
            cur_gt = prctile_norm(cur_gt)
        else:
            cur_img = np.array(cur_img) / 65535
            cur_gt = cur_gt / 65535
        image_batch.append(cur_img)
        gt_batch.append(cur_gt)

    image_batch = np.array(image_batch)
    gt_batch = np.array(gt_batch)

    image_batch = np.transpose(image_batch, (0, 2, 3, 1))
    gt_batch = gt_batch.reshape((batch_size, width*scale, height*scale, 1))

    if wf == 1:
        image_batch = np.mean(image_batch, 3)
        for b in range(batch_size):
            image_batch[b, :, :] = prctile_norm(image_batch[b, :, :])
        image_batch = image_batch[:, :, :, np.newaxis]

    return image_batch, gt_batch
