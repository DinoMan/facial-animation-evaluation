import numpy as np

def ccc(gt, pred):
    mean_pred = np.mean(pred)
    mean_gt = np.mean(gt)

    std_pred = np.std(pred)
    std_gt = np.std(gt)

    pcc = np.corrcoef(gt, pred)[0, 1]
    return 2.0 * pcc * std_pred * std_gt / (std_pred ** 2 + std_gt ** 2 + (mean_pred - mean_gt) ** 2)


def sagr(gt, pred):
    return np.mean(np.sign(gt) == np.sign(pred))
