import run_spike_gpu
import run_spike_cpu

import cv2
import skimage.io as io
import numpy as np


cpu = False
#cpu = True

img = cv2.imread('tomo_test_pattern0000.tiff', -1)
avg_window_size = 5
select_window_size = 5
bright_average = io.imread('bright_average.tiff')

if cpu:
    run_spike_cpu.do(img, avg_window_size, select_window_size, bright_average)
else:
    run_spike_gpu.do(img, avg_window_size, select_window_size, bright_average)
