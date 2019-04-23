import timer


import numpy as np
import cv2

from skimage.filters import sobel
import skimage.morphology as morphology
from skimage.restoration import inpaint


def do(img, avg_window_size, select_window_size, bright_average):
    timer.start()

    kernel = np.ones((avg_window_size, avg_window_size), np.float32) / (avg_window_size ** 2)
    img_filtered = cv2.filter2D(img.astype(np.float32), -1, kernel)

    img_diff = cv2.divide(img.astype(np.float32), img_filtered.astype(np.float32))
    img_diff_max = np.max(img_diff)
    img_diff = img_diff / img_diff_max
    img_diff[img_diff > 1] = 1

    edge = sobel(img_diff)
    edge_top1 = np.percentile(edge, 95)
    edge[edge > edge_top1] = 1
    edge[edge <= edge_top1] = 0
    edge_dilated = morphology.binary_dilation(edge, selem=morphology.disk(1))
    img_noise_map = edge_dilated

    img_denoised = img
    idx = np.where(img_noise_map == 1)
    num_idx = len(idx[0])
    img_pad = cv2.copyMakeBorder(img, select_window_size, select_window_size, select_window_size, select_window_size, cv2.BORDER_REPLICATE)

    for i in range(num_idx):
        r = idx[0][i] + select_window_size
        c = idx[1][i] + select_window_size
        window = img_pad[r - select_window_size : r + select_window_size + 1, c - select_window_size : c + select_window_size + 1]
        img_denoised[idx[0][i], idx[1][i]] = np.median(window)

    img_corrected = img_denoised.astype(dtype = 'double') / bright_average.astype(dtype = 'double')
    img_corrected[img_corrected > 1] = 1

    timer.end()

    return img_corrected
