import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
sys.path.append('..')
from utils.dataset import EOccSR_dataset
from model.EDeOccSR import EDeOccSR
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import warnings
warnings.filterwarnings("ignore")
import cv2


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def saveImg(img, path):
    img[img > 1] = 1
    img[img < 0] = 0
    img = np.array(img[0].cpu().permute(1, 2, 0) * 255)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


if __name__ == '__main__':
    import numpy as np
    import torch
    from torch import nn

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='/lisiqi/EOccSR/')
    parser.add_argument('--arg', type=bool, default=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--ckpt', type=str, default='./checkpoint_main.pth')
    parser.add_argument('--opFolder', type=str, default='./output_main')
    # /home/lisiqi/code/E-OccSR-New/log_sr/2023-03-16/bs4_rate4_(our_dataset_split_by_occlusions)/checkpoint_max_psnr.pth

    args = parser.parse_args()

    device = args.device
    rate = 4
    ckpt_path = args.ckpt

    run_dir = os.path.split(ckpt_path)[0]
    print('rundir:', os.path.abspath(run_dir))

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True

    testFolder = EOccSR_dataset(args=args, train=False)
    testLoader = DataLoaderX(testFolder, batch_size=1, shuffle=False, pin_memory=False, num_workers=1)

    test_total_iter = len(testLoader)
    print(test_total_iter)

    model = EDeOccSR(input_img_channel=17*3, input_event_channel=36, dim=48, num_blocks=[1,2,2,3], num_refinement_blocks=2,
                 heads=[1,2,4,8], ffn_expansion_factor=2.66, LayerNorm_type='WithBias', bias=False, img_channel=3, channel_att=True, rate=rate)

    a = torch.load(ckpt_path)
    model.load_state_dict(a['state_dict'])
    model = model.to(device)

    opFolder = args.opFolder
    os.makedirs(opFolder, exist_ok=True)

    with torch.no_grad():
        model.eval()
        for i, (event_vox, imgs, mask, gt_x4, event01, event21) in enumerate(testLoader):
            event_vox = event_vox.cuda()
            imgs = imgs.cuda().float()
            mask = mask.cuda().float()
            event01 = event01.cuda().float()
            event21 = event21.cuda().float()

            gt = gt_x4.cuda().float()

            output = model(imgs, mask, event_vox, event01, event21)

            saveImg(output, os.path.join(opFolder, '%d_output.png'%i))
            saveImg(gt, os.path.join(opFolder, '%d_gt.png'%i))
            print(i)
