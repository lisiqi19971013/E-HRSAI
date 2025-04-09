import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '4'
import numpy as np
import torch
import torch.nn as nn
from kornia.filters.canny import canny, Canny
from kornia.filters.sobel import sobel
import torch.nn.functional as F
from kornia.filters.gaussian import gaussian_blur2d
from kornia.filters.kernels import get_canny_nms_kernel
from kornia.filters.sobel import spatial_gradient
from kornia.color import rgb_to_grayscale
import math


class CmpLoss(nn.Module):
    def __init__(self):
        super(CmpLoss, self).__init__()

        self.l1loss = nn.L1Loss()

    def forward(self, output, neg, pos, mask):
        mask = mask - mask.min()
        mask = mask / mask.max()

        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0

        # mask = nn.Sigmoid(mask)
        mask = mask.bool()
        # L_pos = self.l1loss(output, pos)
        # L_neg = self.l1loss(output[~mask], neg[~mask])

        L_pos = self.l1loss(output, pos)
        L_neg = self.l1loss(output*(~mask), neg*(~mask))
        return L_pos/L_neg


class TextureLoss(nn.Module):
    def __init__(self, low_threshold=0.05, high_threshold=0.1):
        super(TextureLoss, self).__init__()
        self.l1loss = nn.L1Loss()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.kernel_size = (5, 5)
        self.sigma = (1, 1)
        self.eps = 1e-6

    def calEdge(self, input):
        device: torch.device = input.device
        dtype: torch.dtype = input.dtype

        # To Grayscale
        if input.shape[1] == 3:
            input = rgb_to_grayscale(input)

        # Gaussian filter
        blurred: torch.Tensor = gaussian_blur2d(input, self.kernel_size, self.sigma)

        # Compute the gradients
        gradients: torch.Tensor = spatial_gradient(blurred, normalized=False)

        # Unpack the edges
        gx: torch.Tensor = gradients[:, :, 0]
        gy: torch.Tensor = gradients[:, :, 1]

        # Compute gradient magnitude and angle
        magnitude: torch.Tensor = torch.sqrt(gx * gx + gy * gy + self.eps)
        angle: torch.Tensor = torch.atan2(gy, gx)

        # Radians to Degrees
        angle = 180.0 * angle / math.pi

        # Round angle to the nearest 45 degree
        angle = torch.round(angle / 45) * 45

        # Non-maximal suppression
        nms_kernels: torch.Tensor = get_canny_nms_kernel(device, dtype)
        nms_magnitude: torch.Tensor = F.conv2d(magnitude, nms_kernels, padding=nms_kernels.shape[-1] // 2)

        # Get the indices for both directions
        positive_idx: torch.Tensor = (angle / 45) % 8
        positive_idx = positive_idx.long()

        negative_idx: torch.Tensor = ((angle / 45) + 4) % 8
        negative_idx = negative_idx.long()

        # Apply the non-maximum suppresion to the different directions
        channel_select_filtered_positive: torch.Tensor = torch.gather(nms_magnitude, 1, positive_idx)
        channel_select_filtered_negative: torch.Tensor = torch.gather(nms_magnitude, 1, negative_idx)

        channel_select_filtered: torch.Tensor = torch.stack([channel_select_filtered_positive, channel_select_filtered_negative], 1)

        is_max: torch.Tensor = channel_select_filtered.min(dim=1)[0] > 0.0

        magnitude = magnitude * is_max

        # Threshold
        edges: torch.Tensor = F.threshold(magnitude, self.high_threshold, 0.0)

        edges = F.threshold(-edges, -self.high_threshold, -1)
        edges = -edges
        # low: torch.Tensor = magnitude > self.low_threshold
        # high: torch.Tensor = magnitude > self.high_threshold
        #
        # edges = low * 0.5 + high * 0.5
        # edges = edges.to(dtype)

        return edges

    def forward(self, op, gt):
        edge_op = self.calEdge(op)
        edge_gt = self.calEdge(gt)

        # edge_op = sobel(op)
        # edge_op = edge_op / edge_op.max()
        # edge_op[edge_op>=20/255] = 1
        # edge_op[edge_op<20/255] = 0
        #
        # edge_gt = sobel(gt)
        # edge_gt = edge_gt / edge_gt.max()
        # edge_gt[edge_gt>=20/255] = 1
        # edge_gt[edge_gt<20/255] = 0

        return self.l1loss(edge_op, edge_gt)


if __name__ == '__main__':
    import numpy as np
    from PIL import Image
    from torchvision import transforms

    # img = transforms.ToTensor()(Image.open('/home2/lisiqi/data/DeOccDepthEstimation(DODE)/0043/pf_0.5_2.0/gt_kinect_x2_cvt.png'))
    img = transforms.ToTensor()(Image.open('/home/lisiqi/data/EF-SAI-Dataset/dense/1/outdoor/test/gt/0004.png'))
    img = torch.unsqueeze(img, dim=0)
    l = TextureLoss(high_threshold=0.05)

    os.makedirs('./check', exist_ok=True)

    mag = l.calEdge(img)
    transforms.ToPILImage()(mag[0,0]).save('./check/edge.png')
    transforms.ToPILImage()(img[0]).save('./check/img.png')

    # class test(nn.Module):
    #     def __init__(self):
    #         super(test, self).__init__()
    #         self.conv = nn.Conv2d(3,3,1)
    #
    #     def forward(self, x):
    #         return self.conv(x)
    #
    # model = test().cuda()
    #
    # x = torch.zeros([1, 3, 256, 256]).cuda()
    # gt = torch.zeros([1, 3, 256, 256]).cuda()
    #
    # output = model(x)
    #
    # l = TextureLoss()
    # loss = l(output, gt)
    # loss.backward()


# if __name__ == '__main__':
#     import sys
#     sys.path.append('..')
#     from utils.dataset import dataset_DODE
#     from model.model_sai import EFDeOccNet
#     from torch.utils.data import DataLoader
#     from prefetch_generator import BackgroundGenerator
#
#     from torchvision import transforms
#
#
#     class DataLoaderX(DataLoader):
#         def __iter__(self):
#             return BackgroundGenerator(super().__iter__())
#
#
#     import torch
#     from torch import nn
#     from lpips import lpips
#
#     device = 'cuda'
#     random_seed = 1996
#     batch_size = 1
#
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.fastest = True
#
#     testFolder = dataset_DODE(train=False, arg=True)
#     testLoader = DataLoaderX(testFolder, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)
#
#     test_total_iter = len(testLoader)
#     print(test_total_iter)
#
#     model = EFDeOccNet(input_img_channel=17*3, input_event_channel=36, dim=48, num_blocks=[1,2,2,3], num_refinement_blocks=2,
#                  heads=[1,2,4,8], ffn_expansion_factor=2.66, LayerNorm_type='WithBias', bias=False, img_channel=3, channel_att=True)
#
#     # a = torch.load('/home/lisiqi/code/E-OccSR-New/log_sai/2023-02-19/bs8_(our_dataset_add_cmpLoss_from_scratch)/checkpoint.pth.tar')
#     # model.load_state_dict(a['state_dict'])
#
#     if torch.cuda.device_count() > 1:
#         model = nn.DataParallel(model)
#     model = model.to(device)
#
#     # loss_l1 = nn.L1Loss()
#     # loss_lpips = lpips.LPIPS(net='vgg', spatial=False).cuda()
#     # loss_cmp = CmpLoss()
#
#     os.makedirs('./check', exist_ok=True)
#     with torch.no_grad():
#         model.eval()
#         for i, (event_vox, img, mask, gt_img, _, _, _, _) in enumerate(testLoader):
#             event_vox = event_vox[:,1,...].cuda()
#             img = img[:,1,...].cuda().float()
#             mask = mask[:,1,...].cuda().float()
#             gt_img = gt_img.cuda().float()
#
#             output, _, mask_out = model(img, mask, event_vox)
#
#
#             # Lpips = torch.sum(loss_lpips.forward(output, gt_img, normalize=True)) / batch_size
#             # L1Loss = loss_l1(output, gt_img)
#             # LossCmp = loss_cmp(output, img[:, 24:27, ...], gt_img, mask_out[:, 24:27, ...])
#
#             # mask = mask_out[:, 24:27, ...]
#             # mask = mask - mask.min()
#             # mask = mask / mask.max()
#             #
#             # mask[mask >= 0.5] = 1
#             # mask[mask < 0.5] = 0
#             #
#             # mask = mask.bool()
#             #
#             # transforms.ToPILImage()(output[0]).save(f'./check/{i}_img.png')
#             # transforms.ToPILImage()(mask[0].float()).save(f'./check/{i}_mask.png')
#             # transforms.ToPILImage()((output*mask)[0]).save(f'./check/{i}_pos_op.png')
#             # transforms.ToPILImage()((gt_img*mask)[0]).save(f'./check/{i}_pos_gt.png')
#             # transforms.ToPILImage()((output*(~mask))[0]).save(f'./check/{i}_neg_op.png')
#             # transforms.ToPILImage()((img[:, 24:27, ...]*(~mask))[0]).save(f'./check/{i}_neg_gt.png')
#
#             break
#
#
#
