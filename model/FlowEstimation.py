import torch
from model.subnet import UNetLight
import torch.nn as nn


def create_meshgrid(width, height, is_cuda):
    x, y = torch.meshgrid([torch.arange(0, width), torch.arange(0, height)])
    x, y = (x.transpose(0, 1).float(), y.transpose(0, 1).float())
    if is_cuda:
        x = x.cuda()
        y = y.cuda()
    return x, y


def compute_source_coordinates(y_displacement, x_displacement):
    width, height = y_displacement.size(-1), y_displacement.size(-2)
    x_target, y_target = create_meshgrid(width, height, y_displacement.is_cuda)
    x_source = x_target + x_displacement.squeeze(1)
    y_source = y_target + y_displacement.squeeze(1)
    out_of_boundary_mask = ((x_source.detach() < 0) | (x_source.detach() >= width) | (y_source.detach() < 0) | (y_source.detach() >= height))
    return y_source, x_source, out_of_boundary_mask


def backwarp_2d(source, y_displacement, x_displacement):
    width, height = source.size(-1), source.size(-2)
    y_source, x_source, out_of_boundary_mask = compute_source_coordinates(y_displacement, x_displacement)
    x_source = (2.0 / float(width - 1)) * x_source - 1
    y_source = (2.0 / float(height - 1)) * y_source - 1
    x_source = x_source.masked_fill(out_of_boundary_mask, 0)
    y_source = y_source.masked_fill(out_of_boundary_mask, 0)
    grid_source = torch.stack([x_source, y_source], -1)
    target = torch.nn.functional.grid_sample(source, grid_source, align_corners=True)
    out_of_boundary_mask = out_of_boundary_mask.unsqueeze(1)
    target.masked_fill_(out_of_boundary_mask.expand_as(target), 0)
    return target, out_of_boundary_mask


class FlowNet(nn.Module):
    def __init__(self, img_channel=3):
        super(FlowNet, self).__init__()
        self.net = UNetLight(5+2*img_channel, 2)

    def forward(self, I0, I1, I2, e_01, e_21):
        input1 = torch.cat([I0, I1, e_01], dim=1)
        input2 = torch.cat([I2, I1, e_21], dim=1)
        input = torch.cat([input1, input2])
        flow = self.net(input)
        return flow
        # warped, warped_invalid = backwarp_2d(torch.cat([I0, I2]), y_displacement=flow[:, 0, ...], x_displacement=flow[:, 1, ...])
        # (I01, I21) = torch.chunk(warped, chunks=2)
        # return I01, I21, flow


class Flow(nn.Module):
    def __init__(self):
        super(Flow, self).__init__()
        self.flow = nn.Parameter(torch.nn.init.normal(torch.full(size=(2, 2, 256, 256),fill_value=0.0),mean=0.0,std=0.0))

    def forward(self, I0, I2):
        warped, warped_invalid = backwarp_2d(torch.cat([I0, I2]), y_displacement=self.flow[:, 0, ...], x_displacement=self.flow[:, 1, ...])
        (I01, I21) = torch.chunk(warped, chunks=2)
        return I01, I21


if __name__ == '__main__':
    I0 = torch.zeros([1, 3, 256, 256])
    I1 = torch.zeros([1, 3, 256, 256])
    I2 = torch.zeros([1, 3, 256, 256])
    e_01 = torch.zeros([1, 5, 256, 256])
    e_21 = torch.zeros([1, 5, 256, 256])
    m = Flow()
    o = m(I0, I1, I2, e_01, e_21)
