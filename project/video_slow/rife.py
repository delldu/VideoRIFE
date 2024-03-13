"""Create model."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright 2022 Dell(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2022年 09月 28日 星期三 09:13:36 CST
# ***
# ************************************************************************************/
#
import os

# RIFE model v4.13.2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional

import todos
import pdb

# The following comes from mmcv/ops/point_sample.py
def grid_sample(im, grid):
    """Given an input and a flow-field grid, computes the output using input
    values and pixel locations from grid. 
    Args:
        im (torch.Tensor): Input feature map, shape (N, C, H, W)
        grid (torch.Tensor): Point coordinates, shape (N, Hg, Wg, 2)
    Returns:
        torch.Tensor: A tensor with sampled points, shape (N, C, Hg, Wg)
    """
    n, c, h, w = im.shape
    gn, gh, gw, _ = grid.shape
    assert n == gn

    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]

    # if align_corners:
    #     x = ((x + 1) / 2) * (w - 1)
    #     y = ((y + 1) / 2) * (h - 1)
    # else:
    #     x = ((x + 1) * w - 1) / 2
    #     y = ((y + 1) * h - 1) / 2
    x = ((x + 1) / 2) * (w - 1)
    y = ((y + 1) / 2) * (h - 1)

    x = x.view(n, -1)
    y = y.view(n, -1)

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1

    wa = ((x1 - x) * (y1 - y)).unsqueeze(1)
    wb = ((x1 - x) * (y - y0)).unsqueeze(1)
    wc = ((x - x0) * (y1 - y)).unsqueeze(1)
    wd = ((x - x0) * (y - y0)).unsqueeze(1)

    # Apply default for grid_sample function zero padding
    im_padded = F.pad(im, pad=[1, 1, 1, 1], mode='constant', value=0.0)
    padded_h = h + 2
    padded_w = w + 2
    # save points positions after padding
    x0, x1, y0, y1 = x0 + 1, x1 + 1, y0 + 1, y1 + 1

    # Clip coordinates to padded image size
    x0 = torch.where(x0 < 0, torch.tensor(0), x0)
    x0 = torch.where(x0 > padded_w - 1, torch.tensor(padded_w - 1), x0)
    x1 = torch.where(x1 < 0, torch.tensor(0), x1)
    x1 = torch.where(x1 > padded_w - 1, torch.tensor(padded_w - 1), x1)
    y0 = torch.where(y0 < 0, torch.tensor(0), y0)
    y0 = torch.where(y0 > padded_h - 1, torch.tensor(padded_h - 1), y0)
    y1 = torch.where(y1 < 0, torch.tensor(0), y1)
    y1 = torch.where(y1 > padded_h - 1, torch.tensor(padded_h - 1), y1)

    im_padded = im_padded.view(n, c, -1)

    x0_y0 = (x0 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x0_y1 = (x0 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y0 = (x1 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y1 = (x1 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)

    Ia = torch.gather(im_padded, 2, x0_y0)
    Ib = torch.gather(im_padded, 2, x0_y1)
    Ic = torch.gather(im_padded, 2, x1_y0)
    Id = torch.gather(im_padded, 2, x1_y1)

    return (Ia * wa + Ib * wb + Ic * wc + Id * wd).reshape(n, c, gh, gw)


def make_grid(B: int, H: int, W: int):
    hg = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
    vg = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([hg, vg], dim=1)
    return grid


def warp(x, flow, grid):
    B, C, H, W = x.size()
    flow = torch.cat(
        [flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), 
        flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], dim=1)
    g = (grid + flow).permute(0, 2, 3, 1)
    # onnx support
    # return F.grid_sample(input=x, grid=g, mode="bilinear", padding_mode="border", align_corners=True)
    return grid_sample(x, g)


def resize(x, scale: float):
    return F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=False)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True
        ),
        nn.LeakyReLU(0.2, True),
    )


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 32, 3, 2, 1)
        self.cnn1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.cnn2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(32, 8, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x0 = self.cnn0(x)
        x = self.relu(x0)
        x1 = self.cnn1(x)
        x = self.relu(x1)
        x2 = self.cnn2(x)
        x = self.relu(x2)
        x3 = self.cnn3(x)
        return x3


class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super().__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
        )
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * 6, 4, 2, 1), 
            nn.PixelShuffle(2),
        )


    def start(self, x, scale: float=8.0) -> List[torch.Tensor]:
        """flow is None"""
        x = resize(x, 1.0 / scale)

        feat = self.conv0(x)
        feat = self.convblock(feat)
        feat = self.lastconv(feat)
        feat = resize(feat, scale)
        flow = feat[:, 0:4] * scale
        mask = feat[:, 4:5]
        return flow, mask

    def forward(self, x, flow, scale: float=1.0) -> List[torch.Tensor]:
        x = resize(x, 1.0 / scale)
        """flow is not None"""
        flow = resize(flow, 1.0 / scale) / scale
        x = torch.cat((x, flow), dim=1)

        feat = self.conv0(x)
        feat = self.convblock(feat)
        feat = self.lastconv(feat)
        feat = resize(feat, scale)
        flow = feat[:, 0:4] * scale
        mask = feat[:, 4:5]
        return flow, mask


class IFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.MAX_H = 2048
        self.MAX_W = 4096
        self.MAX_TIMES = 32
        # Define max GPU memory -- 8G, 440ms

        self.block0 = IFBlock(7 + 16, c=192)
        self.block1 = IFBlock(8 + 4 + 16, c=128)
        self.block2 = IFBlock(8 + 4 + 16, c=96)
        self.block3 = IFBlock(8 + 4 + 16, c=64)
        self.encode = Head()

        self.load_weights()

        self.blocks = nn.ModuleList([self.block1, self.block2, self.block3])

    def load_weights(self, model_path="models/flownet_v4.13.2.pkl"):
        print(f"Loading weights from {model_path} ......")
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        sd = torch.load(checkpoint)
        new_sd = {}
        for k, v in sd.items():
            k = k.replace("module.", "")
            new_sd[k] = v
        self.load_state_dict(new_sd)


    def forward(self, x):
        B2, C2, H2, W2 = x.size()

        assert C2 == 6, "x channel must be 6"
        pad_h = self.MAX_TIMES - (H2 % self.MAX_TIMES)
        pad_w = self.MAX_TIMES - (W2 % self.MAX_TIMES)
        x = F.pad(x, (0, pad_w, 0, pad_h), 'reflect')
        B, C, H, W = x.size()

        I1 = x[:, 0:3, :, :]
        I2 = x[:, 3:6, :, :]

        scale_list: List[float] = [8.0, 4.0, 2.0, 1.0]

        # timestep:float=0.5
        timestep = torch.ones(B, 1, H, W).to(x.device) * 0.5

        F1 = self.encode(I1)
        F2 = self.encode(I2)

        xx = torch.cat((I1, I2, F1, F2, timestep), dim=1)
        flow, mask = self.block0.start(xx, scale=scale_list[0])

        W_I1 = I1
        W_I2 = I2
        grid = make_grid(B, H, W).to(x.device)
        for i, block in enumerate(self.blocks):  # self.block1, self.block2, self.block3
            W_F1 = warp(F1, flow[:, 0:2], grid)
            W_F2 = warp(F2, flow[:, 2:4], grid)
            xx = torch.cat((W_I1, W_I2, W_F1, W_F2, timestep, mask), dim=1)
            fd, mask = block(xx, flow, scale=scale_list[i + 1])
            flow = flow + fd

            W_I1 = warp(I1, flow[:, 0:2], grid)
            W_I2 = warp(I2, flow[:, 2:4], grid)

        mask = torch.sigmoid(mask)
        middle = W_I1 * mask + W_I2 * (1.0 - mask)

        return middle[:, :, 0:H2, 0:W2]
