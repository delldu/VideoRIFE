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
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import todos

import pdb

# The following comes from RIFE_HDv3.py/IFNet_HDv3.py, thanks authors !!!


def standard_flow_grid(flow):
    B, C, H, W = flow.shape
    hg = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
    vg = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([hg, vg], dim=1).to(flow.device)
    return grid


def warp(input, flow, backwarp_grid):
    _, _, H, W = input.shape
    flow = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], dim=1)

    g = (backwarp_grid + flow).permute(0, 2, 3, 1)  # ==> [1, 544, 960, 2]
    return F.grid_sample(input=input, grid=g, mode="bilinear", padding_mode="border", align_corners=True)


def resize(x, scale: float):
    return F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=False)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True
        ),
        nn.PReLU(out_planes),
    )


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock0 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock1 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock2 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock3 = nn.Sequential(conv(c, c), conv(c, c))
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, 4, 2, 1),
            nn.PReLU(c // 2),
            nn.ConvTranspose2d(c // 2, 4, 4, 2, 1),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, 4, 2, 1),
            nn.PReLU(c // 2),
            nn.ConvTranspose2d(c // 2, 1, 4, 2, 1),
        )

    def forward(self, x, flow, scale: float) -> List[torch.Tensor]:
        x = resize(x, 1.0 / scale)
        flow = resize(flow, 1.0 / scale) * 1.0 / scale
        feat = self.conv0(torch.cat((x, flow), dim=1))
        feat = self.convblock0(feat) + feat
        feat = self.convblock1(feat) + feat
        feat = self.convblock2(feat) + feat
        feat = self.convblock3(feat) + feat
        flow = self.conv1(feat)
        mask = self.conv2(feat)
        flow = resize(flow, scale) * scale
        mask = resize(mask, scale)
        return flow, mask


class IFNet(nn.Module):
    """Intermediate Flow Network -- RIFE(Real-Time Intermediate Flow Estimation)"""

    def __init__(self):
        super(IFNet, self).__init__()
        # Define max GPU/CPU memory -- 4G, 440ms
        self.MAX_H = 2048
        self.MAX_W = 4096
        self.MAX_TIMES = 16

        self.block0 = IFBlock(7 + 4, c=90)
        self.block1 = IFBlock(7 + 4, c=90)
        self.block2 = IFBlock(7 + 4, c=90)
        self.block_tea = IFBlock(10 + 4, c=90)
        self.blocks = nn.ModuleList([self.block0, self.block1, self.block2])

        self.load_weights()

    def load_weights(self, model_path="models/video_slow.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        todos.model.load(self, checkpoint)

        # self.load_state_dict(torch.load(checkpoint))

    def forward(self, x):
        B, C, H, W = x.shape
        img0 = x[:, 0:3]
        img1 = x[:, 3:6]

        warped_img0 = img0
        warped_img1 = img1
        flow = torch.zeros((B, 4, H, W)).to(x.device)
        mask = torch.zeros((B, 1, H, W)).to(x.device)
        grid = standard_flow_grid(flow[:, 0:2])

        scale_list = [4.0, 2.0, 1.0]
        for i, block in enumerate(self.blocks):
            t0 = torch.cat((warped_img0, warped_img1, mask), dim=1)
            t1 = torch.cat((warped_img1, warped_img0, -mask), dim=1)

            f0, m0 = block(t0, flow, scale_list[i])
            flow_1 = torch.cat((flow[:, 2:4], flow[:, 0:2]), dim=1)
            f1, m1 = block(t1, flow_1, scale_list[i])
            f1 = torch.cat((f1[:, 2:4], f1[:, 0:2]), dim=1)

            flow = flow + (f0 + f1) / 2.0
            mask = mask + (m0 + (-m1)) / 2.0

            warped_img0 = warp(img0, flow[:, 0:2], grid)
            warped_img1 = warp(img1, flow[:, 2:4], grid)

        mask = torch.sigmoid(mask)
        middle = warped_img0 * mask + warped_img0 * (1.0 - mask)

        return middle.clamp(0.0, 1.0)
