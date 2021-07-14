import numpy as np


def psnr(video, reference):
    sum_mse = 0
    for frame_no, frame in video:
        sum_mse += np.sum((frame - reference[frame_no]) ** 2) / frame.size
    avg_mse = sum_mse / (frame_no + 1)
    return -10 * np.log10(avg_mse)
