'''
frame+event -> mask
frame partial conv, concat event -> restormer (encoder decoder之间加channel attention)
multi-frame -> restormer encoder -> feature用deformerable convolution做alignment -> channel attention -> decoder （考虑中间做对齐还是最后做对齐）
'''
import torch
from torch import nn
from model.basic_encoder import BasicAE
from model.subnet import MaskPredNet, PartialConv2d
from model.FlowEstimation import FlowNet


class EFDeOccNet(nn.Module):
    def __init__(self, input_img_channel=17, input_event_channel=36, dim=48, num_blocks=[4,6,6,8], num_refinement_blocks=4,
                 heads=[1,2,4,8], ffn_expansion_factor=2.66, LayerNorm_type='WithBias', bias=False, img_channel=1, channel_att=True):
        super().__init__()
        self.mask_pred = MaskPredNet(img_channel)
        self.img_embed = PartialConv2d(input_img_channel, dim, 3, padding=1)
        self.event_embed = nn.Conv2d(input_event_channel, dim, kernel_size=3, padding=1)
        self.deocc = BasicAE((dim+dim), num_blocks, num_refinement_blocks, heads, ffn_expansion_factor, LayerNorm_type, bias, img_channel, channel_att)

    def forward(self, img, mask_init, event):
        # if len(img.shape) == 5:
        #     event = event[:, 1, ...]
        #     img = img[:, 1, ...]
        #     mask_init = mask_init[:, 1, ...]

        # bs, N, C, H, W = img.shape
        # img = img.reshape(bs*N, -1, H, W)
        # mask_init = mask_init.reshape(bs*N, -1, H, W)
        # event = event.reshape(bs*N, -1, H, W)

        mask = self.mask_pred(img, mask_init)

        img_fea, _ = self.img_embed(img, mask)
        event_fea = self.event_embed(event)
        fea = torch.cat([img_fea, event_fea], dim=1)
        output, output_fea = self.deocc(fea)
        return output, output_fea, mask

        # output = output.reshape(bs, N, -1, H, W)
        # I0 = output[:, 0, ...]
        # I1 = output[:, 1, ...]
        # I2 = output[:, 2, ...]
        # I01, I21, flow = self.flow_net(I0, I1, I2, e_01, e_21)
        # return I1, I01, I21


if __name__ == '__main__':
    m = EFDeOccNet(input_img_channel=51, img_channel=3)
    img = torch.zeros([2, 3, 51, 256, 256])
    mask_init = torch.zeros([2, 3, 17, 256, 256])
    event = torch.zeros([2, 3, 36, 256, 256])
    e_01 = torch.zeros([2, 5, 256, 256])
    e_21 = torch.zeros([2, 5, 256, 256])

    output = m(img, mask_init, event, e_01, e_21)

