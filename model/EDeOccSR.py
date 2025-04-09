from model.model_sai import EFDeOccNet
from model.basic_encoder import TransformerBlock, Upsample
from torch import nn
import torch
from model.model_pcda import Alignment
import torch.nn.functional as F
from model.FlowEstimation import FlowNet, backwarp_2d


class AdaptiveFusion(nn.Module):
    def __init__(self, channel):
        super(AdaptiveFusion, self).__init__()
        self.global_conv = nn.Sequential(nn.Conv2d(4, 4, 7, 1, 3), nn.LeakyReLU(0.1), nn.Conv2d(4, 1, 1))
        self.conv1 = nn.Conv2d(2*channel, channel//4, 1)
        self.del_conv1 = nn.Conv2d(channel//4, channel//4, kernel_size=1, stride=1, padding=0)
        self.del_conv2 = nn.Conv2d(channel//4, channel//4, kernel_size=3, stride=1, padding=1)
        self.del_conv3 = nn.Conv2d(channel//4, channel//4, kernel_size=5, stride=1, padding=2)
        self.del_conv4 = nn.Conv2d(channel//4, channel//4, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(channel, channel, 1)
        self.lr_relu = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fea1, fea2):
        f1_1 = fea1.mean(dim=1, keepdim=True)
        f1_2 = fea1.max(dim=1, keepdim=True)[0]
        f2_1 = fea2.mean(dim=1, keepdim=True)
        f2_2 = fea2.max(dim=1, keepdim=True)[0]
        f = torch.cat([f1_1, f1_2, f2_1, f2_2], dim=1)
        global_att = self.global_conv(f)

        fea = self.lr_relu(self.conv1(torch.cat([fea1, fea2], dim=1)))
        fea = torch.cat([self.del_conv1(fea), self.del_conv2(fea), self.del_conv3(fea), self.del_conv4(fea)], dim=1)
        spatial_att = self.conv2(self.lr_relu(fea))

        attn = self.sigmoid(spatial_att + global_att)

        return fea1 * attn + fea2 * (1-attn)


class EDeOccSR(nn.Module):
    def __init__(self, input_img_channel=17, input_event_channel=36, dim=48, num_blocks=[4,6,6,8], num_refinement_blocks=4,
                 heads=[1,2,4,8], ffn_expansion_factor=2.66, LayerNorm_type='WithBias', bias=False, img_channel=1, channel_att=True, rate=2,
                 flow=True, defConv=True, adapFuse=True):
        super(EDeOccSR, self).__init__()
        assert rate == 1 or rate == 2 or rate == 4, "SR rate must be 1 or 2 or 4."
        self.ef_sai_net = EFDeOccNet(input_img_channel, input_event_channel, dim, num_blocks, num_refinement_blocks,
                                     heads, ffn_expansion_factor, LayerNorm_type, bias, img_channel, channel_att)

        self.defConv = defConv
        self.flow = flow
        self.adapFuse = adapFuse

        if rate != 1:
            if flow:
                self.flow_net = FlowNet()
            if defConv:
                self.alignment = Alignment(int(dim * 2 * 2))
            if adapFuse:
                self.adaptive_fusion = AdaptiveFusion(int(dim * 2 * 2))

        if rate == 2:
            k = int(dim * 2 ** 2) * 3
            self.up2 = nn.Sequential(nn.Conv2d(k, k, kernel_size=3, stride=1, padding=1, bias=False), nn.PixelShuffle(2))
            k = k//4
            self.refinement2 = nn.Sequential(*[TransformerBlock(dim=k, num_heads=4, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(2)])
            self.output2 = nn.Conv2d(k, img_channel, kernel_size=3, stride=1, padding=1, bias=bias)
        if rate == 4:
            k = int(dim * 2 ** 2) * 3
            self.up2 = nn.Sequential(nn.Conv2d(k, k, kernel_size=3, stride=1, padding=1, bias=False), nn.PixelShuffle(2))
            k = k//4
            self.refinement2 = nn.Sequential(*[TransformerBlock(dim=k, num_heads=4, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(2)])
            self.output2 = nn.Conv2d(k, img_channel, kernel_size=3, stride=1, padding=1, bias=bias)

            self.up4 = nn.Sequential(nn.Conv2d(k, k, kernel_size=3, stride=1, padding=1, bias=False), nn.PixelShuffle(2))
            k = k//4
            self.refinement4 = nn.Sequential(*[TransformerBlock(dim=k, num_heads=4, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(2)])
            self.output4 = nn.Conv2d(k, img_channel, kernel_size=3, stride=1, padding=1, bias=bias)

        self.rate = rate

    def forward(self, img, mask, event_vox, e01, e21):
        bs = event_vox.shape[0]
        if self.rate == 1:
            event_vox = event_vox[:, 1, ...]
            img = img[:, 1, ...]
            mask = mask[:, 1, ...]
        else:
            event_vox = event_vox.view(bs*3, event_vox.shape[2], event_vox.shape[3], event_vox.shape[4])
            img = img.view(bs*3, img.shape[2], img.shape[3], img.shape[4])
            mask = mask.view(bs*3, mask.shape[2], mask.shape[3], mask.shape[4])

        deocc, output_fea, mask_out = self.ef_sai_net(img, mask, event_vox)
        if self.rate == 1:
            return deocc
        else:
            deocc = deocc.view(bs, 3, 3, deocc.shape[2], deocc.shape[3])
            img_0 = deocc[:, 0, ...]
            img_1 = deocc[:, 1, ...]
            img_2 = deocc[:, 2, ...]

            output_fea1 = output_fea.view(bs, 3, -1, output_fea.shape[2], output_fea.shape[3]).clone()
            fea_0 = output_fea1[:, 0, ...]
            fea_1 = output_fea1[:, 1, ...]
            fea_2 = output_fea1[:, 2, ...]

            if self.defConv:
                fea_0_dc_aligned, fea_2_dc_aligned = self.alignment(output_fea)
            else:
                fea_0_dc_aligned, fea_2_dc_aligned = fea_0, fea_2

            if self.flow:
                flow = self.flow_net(img_0, img_1, img_2, e01, e21)
                warped_fea, _ = backwarp_2d(torch.cat([fea_0, fea_2]), y_displacement=flow[:, 0, ...], x_displacement=flow[:, 1, ...])
                (fea_0_of_aligned, fea_2_of_aligned) = torch.chunk(warped_fea, chunks=2)
            else:
                fea_0_of_aligned, fea_2_of_aligned = fea_0, fea_2

            if self.adapFuse:
                fea_0_aligned = self.adaptive_fusion(fea_0_dc_aligned, fea_0_of_aligned)
                fea_2_aligned = self.adaptive_fusion(fea_2_dc_aligned, fea_2_of_aligned)
            else:
                fea_0_aligned = fea_0_dc_aligned + fea_0_of_aligned
                fea_2_aligned = fea_2_dc_aligned + fea_2_of_aligned

            fea_aligned = torch.cat([fea_1, fea_0_aligned, fea_2_aligned], dim=1)

            fea_up2 = self.up2(fea_aligned)
            fea_up2 = self.refinement2(fea_up2)
            output_x2 = self.output2(fea_up2)
            if self.rate == 2:
                return output_x2
            else:
                fea_up4 = self.up4(fea_up2)
                fea_up4 = self.refinement4(fea_up4)
                output_x4 = self.output4(fea_up4)
                return output_x4


if __name__ == '__main__':
    import os
    import numpy as np
    import torch
    from torchvision import transforms
    from PIL import Image
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # img = torch.zeros([2, 3, 51, 256, 256]).cuda()
    # mask = torch.zeros([2, 3, 17, 256, 256]).cuda()
    # event_vox = torch.zeros([2, 3, 36, 256, 256]).cuda()
    # e01 = torch.zeros([2, 5, 256, 256]).cuda()
    # e21 = torch.zeros([2, 5, 256, 256]).cuda()
    # with torch.no_grad():
    #     o = m(img, mask, event_vox, e01, e21)

    folder = []
    with open('/home/lisiqi/data/DeOccDepthEstimation(DODE)/total.txt', 'r') as f:
        for line in f.readlines():
            p1 = line.strip('\n')
            folder.append(p1)

    m = EDeOccSR(input_img_channel=17*3, input_event_channel=36, dim=48, num_blocks=[1,2,2,3], num_refinement_blocks=2,
                 heads=[1,2,4,8], ffn_expansion_factor=2.66, LayerNorm_type='WithBias', bias=False, img_channel=3, channel_att=True, rate=4).cuda()
    a = torch.load('/home/lisiqi/code/DeOccTest/log_ours/2022-07-16/bs8_EFDeOccNet/checkpoint_max_psnr.pth')
    b = torch.load('/home/lisiqi/code/E-OccSR-New/log_flow/2023-02-13/bs32_(flow_estimation)/checkpoint.pth.tar')

    m.ef_sai_net.load_state_dict(a['state_dict'])
    m.flow_net.load_state_dict(b['state_dict'])

    for idx in range(len(folder)):
        event_vox = torch.unsqueeze(torch.from_numpy(np.load(os.path.join(folder[idx], 'data', 'event.npy'))[:,:,2:-2,2:-2]), dim=0).cuda()
        imgs = torch.unsqueeze(torch.from_numpy(np.load(os.path.join(folder[idx], 'data', 'frame.npy'))[:,:,2:-2,2:-2]), dim=0).cuda()
        mask = torch.unsqueeze(torch.from_numpy(np.load(os.path.join(folder[idx], 'data', 'mask.npy'))[:,:,2:-2,2:-2]), dim=0).float().cuda()
        gt = torch.unsqueeze(transforms.ToTensor()(Image.open(os.path.join(folder[idx], 'gt_kinect_cvt.png')))[:,2:-2,2:-2], dim=0).cuda()
        data = np.load(os.path.join(folder[idx], 'data', 'flow.npy'))[:, 2:-2, 2:-2]
        data = torch.from_numpy(data)
        event01 = torch.unsqueeze(data[9:9+5, ...], dim=0).cuda()
        event21 = torch.unsqueeze(data[9+5:, ...], dim=0).cuda()

        output_x4 = m(imgs, mask, event_vox, event01, event21)
        gt = torch.zeros([1,3,1024,1024]).cuda()

        # with torch.no_grad():
        #     output_x2 = m(imgs, mask, event_vox, event01, event21)

            # img_0, img_2, img_1, fea_0, fea_2, fea_1 = m(imgs, mask, event_vox, event01, event21)
            # transforms.ToPILImage()(img_0[0,...]).save(f'/home/lisiqi/data/DeOccDepthEstimation(DODE)/check/{idx}_0.png')
            # transforms.ToPILImage()(img_1[0,...]).save(f'/home/lisiqi/data/DeOccDepthEstimation(DODE)/check/{idx}_1.png')
            # transforms.ToPILImage()(img_2[0,...]).save(f'/home/lisiqi/data/DeOccDepthEstimation(DODE)/check/{idx}_2.png')

            # os.makedirs()

        break
        # print(idx)


