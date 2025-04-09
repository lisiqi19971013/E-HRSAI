import sys
sys.path.append('..')
import torch.nn as nn
import torch
from ops.dcn import (ModulatedDeformConvPack, modulated_deform_conv)
from distutils.version import LooseVersion
import torchvision


class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            print(f'Offset abs mean is {offset_absmean}, larger than 50.')
        # return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
        #                                  self.dilation, self.groups, self.deformable_groups)
        if LooseVersion(torchvision.__version__) >= LooseVersion('0.9.0'):
            return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                                 self.dilation, mask)
        else:
            return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
                                         self.dilation, self.groups, self.deformable_groups)


class PCDAlignment(nn.Module):
    def __init__(self, num_feat=64, deformable_groups=8):
        super(PCDAlignment, self).__init__()

        # Pyramid has three levels:
        # L3: level 3, 1/4 spatial size
        # L2: level 2, 1/2 spatial size
        # L1: level 1, original spatial size
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()

        # Pyramids
        for i in range(3, 0, -1):
            level = f'l{i}'
            self.offset_conv1[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
            if i == 3:
                self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            else:
                self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
                self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.dcn_pack[level] = DCNv2Pack(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)

            if i < 3:
                self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)

        # Cascading dcn
        self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.cas_offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.cas_dcnpack = DCNv2Pack(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat_l, ref_feat_l):
        """Align neighboring frame features to the reference frame features.

        Args:
            nbr_feat_l (list[Tensor]): Neighboring feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).
            ref_feat_l (list[Tensor]): Reference feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).

        Returns:
            Tensor: Aligned features.
        """
        # Pyramids
        upsampled_offset, upsampled_feat = None, None
        for i in range(3, 0, -1):
            level = f'l{i}'
            offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
            offset = self.lrelu(self.offset_conv1[level](offset))
            if i == 3:
                offset = self.lrelu(self.offset_conv2[level](offset))
            else:
                offset = self.lrelu(self.offset_conv2[level](torch.cat([offset, upsampled_offset], dim=1)))
                offset = self.lrelu(self.offset_conv3[level](offset))

            feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset)
            if i < 3:
                feat = self.feat_conv[level](torch.cat([feat, upsampled_feat], dim=1))
            if i > 1:
                feat = self.lrelu(feat)

            if i > 1:  # upsample offset and features
                # x2: when we upsample the offset, we should also enlarge
                # the magnitude.
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)

        # Cascading
        offset = torch.cat([feat, ref_feat_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset))))
        feat = self.lrelu(self.cas_dcnpack(feat, offset))
        return feat


class Alignment(nn.Module):
    def __init__(self, dim, deformable_groups=8):
        super(Alignment, self).__init__()
        self.align = PCDAlignment(dim, deformable_groups)
        self.conv1 = nn.Sequential(nn.Conv2d(dim, dim, 3, 2, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True), nn.Conv2d(dim, dim, 3, 1, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(dim, dim, 3, 2, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True), nn.Conv2d(dim, dim, 3, 1, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True))

    def forward(self, fea):
        # feature  (bs * 3) * dim * H * W
        _, c, h ,w = fea.shape
        fea1 = self.conv1(fea)
        fea2 = self.conv2(fea1)

        fea = fea.view(-1, 3, c, h, w)
        fea1 = fea1.view(-1, 3, c, int(h/2), int(w/2))
        fea2 = fea2.view(-1, 3, c, int(h/4), int(w/4))

        ref_fea    = [fea[:, 1, ...].clone(), fea1[:, 1, ...].clone(), fea2[:, 1, ...].clone()]
        nbr_feat_l = [fea[:, 0, ...].clone(), fea1[:, 0, ...].clone(), fea2[:, 0, ...].clone()]
        nbr_feat_r = [fea[:, 2, ...].clone(), fea1[:, 2, ...].clone(), fea2[:, 2, ...].clone()]
        # aligned_feat = [self.align(nbr_feat_l, ref_fea), fea[:, 1, ...].clone(), self.align(nbr_feat_r, ref_fea)]
        # aligned_feat = torch.stack(aligned_feat, dim=1)
        # return aligned_feat.view(-1, 3*c, h, w)
        return self.align(nbr_feat_l, ref_fea), self.align(nbr_feat_r, ref_fea)


if __name__ == '__main__':
    m = Alignment(192).cuda()
    x = torch.zeros([6, 192, 256, 256]).cuda()
    o = m(x)
    # f1 = torch.zeros([2, 64, 256, 256])
    # f2 = torch.zeros([2, 64, 128, 128])
    # f3 = torch.zeros([2, 64, 64, 64])
    #
    # x1 = [f1, f2, f3]
    # x2 = [f1, f2, f3]
    # m = PCDAlignment()
    # out = m(x1, x2)