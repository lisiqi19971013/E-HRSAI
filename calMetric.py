import glob
import numpy as np
import os
import cv2
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio


def calpsnr(gt, pred):
    return peak_signal_noise_ratio(gt, pred)


def calssim(gt, pred):
    return structural_similarity(gt, pred, channel_axis=2, gaussian_weights=True)


path = "./output_main"

file = '/home/lisiqi/data/DeOccDepthEstimation(DODE)/test.txt'

with open(file, 'r') as f:
    lines = f.readlines()

psnr = []
ssim = []

psnr_pf = []
ssim_pf = []

psnr_lb = []
ssim_lb = []

psnr_zl = []
ssim_zl = []

imgList = glob.glob(os.path.join(path, '*_gt.png'))
assert len(imgList) == len(lines)

for i, f in enumerate(imgList):
    id = os.path.split(f)[-1]
    id = int(id.split('_')[0])

    op = cv2.imread(f.replace('gt', 'output'))
    gt = cv2.imread(f)
    p = calpsnr(gt, op)
    s = calssim(gt, op)

    psnr.append(p)
    ssim.append(s)

    if 'pf' in lines[id]:
        psnr_pf.append(p)
        ssim_pf.append(s)
    elif 'zl' in lines[id]:
        psnr_zl.append(p)
        ssim_zl.append(s)
    else:
        psnr_lb.append(p)
        ssim_lb.append(s)

    print(i, p, s)

psnr_pf1 = sum(psnr_pf) / len(psnr_pf)
psnr_lb1 = sum(psnr_lb) / len(psnr_lb)
psnr_zl1 = sum(psnr_zl) / len(psnr_zl)
psnr1 = sum(psnr) / len(psnr)
ssim_pf1 = sum(ssim_pf) / len(ssim_pf)
ssim_lb1 = sum(ssim_lb) / len(ssim_lb)
ssim_zl1 = sum(ssim_zl) / len(ssim_zl)
ssim1 = sum(ssim) / len(ssim)


line = f'Dense occlusion: PSNR {psnr_pf1}, SSIM {ssim_pf1}\n' \
       f'Extreamly dense occlusion: PSNR {(psnr_lb1+psnr_zl1)/2}, SSIM {(ssim_lb1+ssim_zl1)/2}\n' \
       f'Total PNSR: {psnr1}, SSIM: {ssim1}'
print(line)

with open(os.path.join(path, 'res.txt'), 'w') as f:
    f.writelines(line)