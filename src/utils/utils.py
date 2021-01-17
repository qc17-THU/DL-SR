import numpy as np
from skimage.measure import compare_mse, compare_nrmse, compare_psnr, compare_ssim


def prctile_norm(x, min_prc=0, max_prc=100):
    y = (x-np.percentile(x, min_prc))/(np.percentile(x, max_prc)-np.percentile(x, min_prc)+1e-7)
    y[y > 1] = 1
    y[y < 0] = 0
    return y


def img_comp(gt, pr, mses=None, nrmses=None, psnrs=None, ssims=None):
    if ssims is None:
        ssims = []
    if psnrs is None:
        psnrs = []
    if nrmses is None:
        nrmses = []
    if mses is None:
        mses = []
    gt, pr = np.squeeze(gt), np.squeeze(pr)
    gt = gt.astype(np.float32)
    if gt.ndim == 2:
        n = 1
        gt = np.reshape(gt, (1, gt.shape[0], gt.shape[1]))
        pr = np.reshape(pr, (1, pr.shape[0], pr.shape[1]))
    else:
        n = np.size(gt, 0)

    for i in range(n):
        mses.append(compare_mse(prctile_norm(np.squeeze(gt[i])), prctile_norm(np.squeeze(pr[i]))))
        nrmses.append(compare_nrmse(prctile_norm(np.squeeze(gt[i])), prctile_norm(np.squeeze(pr[i]))))
        psnrs.append(compare_psnr(prctile_norm(np.squeeze(gt[i])), prctile_norm(np.squeeze(pr[i])), 1))
        ssims.append(compare_ssim(prctile_norm(np.squeeze(gt[i])), prctile_norm(np.squeeze(pr[i]))))
    return mses, nrmses, psnrs, ssims


def diffxy(img, order=3):
    for _ in range(order):
        img = prctile_norm(img)
        d = np.zeros_like(img)
        dx = (img[1:-1, 0:-2] + img[1:-1, 2:]) / 2
        dy = (img[0:-2, 1:-1] + img[2:, 1:-1]) / 2
        d[1:-1, 1:-1] = img[1:-1, 1:-1] - (dx + dy) / 2
        d[d < 0] = 0
        img = d
    return img


def rm_outliers(img, order=3, thresh=0.2):
    img_diff = diffxy(img, order)
    mask = img_diff > thresh
    img_rm_outliers = img
    img_mean = np.zeros_like(img)
    for i in [-1, 1]:
        for a in range(0, 2):
            img_mean = img_mean + np.roll(img, i, axis=a)
    img_mean = img_mean / 4
    img_rm_outliers[mask] = img_mean[mask]
    img_rm_outliers = prctile_norm(img_rm_outliers)
    return img_rm_outliers


