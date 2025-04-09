import torch
import torch.nn.functional as F
from torch import nn


class down_light(nn.Module):
    def __init__(self, inChannels, outChannels, norm='BN'):
        super(down_light, self).__init__()
        self.relu1 = nn.LeakyReLU(0.1, True)
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=4, stride=2, padding=1)
        self.norm = norm
        if norm:
            self.bn = nn.BatchNorm2d(outChannels)

    def forward(self, x):
        x = self.conv1(x)
        if self.norm:
            x = self.bn(x)
        x = self.relu1(x)
        return x


class up_light(nn.Module):
    def __init__(self, inChannels, outChannels, norm=False):
        super(up_light, self).__init__()
        bias = False if norm else True
        self.conv1 = nn.Conv2d(inChannels, outChannels, 3, stride=1, padding=1, bias=bias)
        self.norm = norm
        if norm == 'BN':
            self.bn1 = nn.BatchNorm2d(outChannels)
        elif norm == 'IN':
            self.bn1 = nn.InstanceNorm2d(outChannels, track_running_stats=True)
        elif norm == False:
            print('No Normalization.')
        else:
            raise ValueError("Choose BN or IN or False.")

    def forward(self, x, skpCn):
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = self.conv1(torch.cat((x, skpCn), 1))
        if self.norm:
            x = F.leaky_relu(self.bn1(x), negative_slope=0.1)
        else:
            x = F.leaky_relu(x, negative_slope=0.1)
        return x


class UNetLight(nn.Module):
    def __init__(self, inChannels, outChannels, layers=[128,256,256,512,512], norm='BN'):
        super(UNetLight, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(inChannels, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.1, True))
        self.down1 = down_light(64, layers[0], norm)
        self.down2 = down_light(layers[0], layers[1], norm)
        self.down3 = down_light(layers[1], layers[2], norm)
        self.down4 = down_light(layers[2], layers[3], norm)
        self.down5 = down_light(layers[3], layers[4], norm)
        self.up1 = up_light(layers[4]+layers[3], layers[3], norm)
        self.up2 = up_light(layers[3]+layers[2], layers[2], norm)
        self.up3 = up_light(layers[2]+layers[1], layers[1], norm)
        self.up4 = up_light(layers[1]+layers[0], layers[0], norm)
        self.up5 = up_light(layers[0]+64, 64, norm)
        self.conv3 = nn.Conv2d(64, outChannels, 1)

    def forward(self, x):
        s1 = self.conv1(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        x = self.down5(s5)

        x = self.up1(x, s5)
        x = self.up2(x, s4)
        x = self.up3(x, s3)
        x = self.up4(x, s2)
        x = self.up5(x, s1)
        x = self.conv3(x)
        return x


class MaskPredNet(nn.Module):
    def __init__(self, img_channel=1):
        super(MaskPredNet, self).__init__()
        self.pred = UNetLight(img_channel+1, 1)
        self.img_channel = img_channel

    def forward(self, img, mask_init):
        bs, C, H ,W = img.shape
        # print(img.shape, mask_init.shape)
        img = img.reshape([-1, self.img_channel, H, W])
        mask_init = mask_init.reshape([-1, 1, H, W])
        # print(img.shape, mask_init.shape)
        x = torch.cat([img, mask_init], dim=1)
        mask = F.sigmoid(self.pred(x))
        mask = torch.repeat_interleave(mask, self.img_channel, dim=1)
        mask = mask.reshape([bs, -1, H, W])
        return mask


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        self.multi_channel = True
        self.return_mask = True

        super(PartialConv2d, self).__init__(*args, **kwargs)

        self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            if self.weight_maskUpdater.type() != input.type():
                self.weight_maskUpdater = self.weight_maskUpdater.to(input)
            mask = mask_in
            self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                        padding=self.padding, dilation=self.dilation, groups=1)

            # for mixed precision training, change 1e-8 to 1e-6
            self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)

            self.update_mask = torch.clamp(self.update_mask, 0, 1)
            self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output