"""RIFE appliction class."""# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020, All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 12月 20日 星期日 08:42:09 CST
# ***
# ************************************************************************************/
#
import inspect
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import model_device, model_load, model_setenv, model_save

import pdb

import warnings
warnings.filterwarnings("ignore")

def abs_weight_file(funcname, filename):
    dir = os.path.dirname(inspect.getfile(funcname))
    return os.path.join(dir, 'weights/%s' % (filename))

class RIFE(nn.Module):
    """RIFE."""

    def __init__(self):
        """Init model."""
        super(RIFE, self).__init__()

        model_setenv()

        flow = FlowModel()
        context = ContextModel()
        fusion = FusionModel()

        model_load(flow, abs_weight_file(self.__init__, "Flow.pth"))
        model_load(context, abs_weight_file(self.__init__, "Context.pth"))
        model_load(fusion, abs_weight_file(self.__init__, "Fusion.pth"))

        self.device = model_device()
        self.flow = flow.to(self.device)
        self.context = context.to(self.device)
        self.fusion = fusion.to(self.device)

    def train(self):
        self.flow.train()
        self.context.train()
        self.fusion.train()

    def eval(self):
        self.flow.eval()
        self.context.eval()
        self.fusion.eval()

    def save(self):
        '''Save models. '''
        outputdir = "output"
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        model_save(self.flow, "{}/Flow.pth".format(outputdir))
        model_save(self.context, "{}/Context.pth".format(outputdir))
        model_save(self.fusion, "{}/Fusion.pth".format(outputdir))

    def forward(self, img1, img2):
        """Forward."""
        imgs = torch.cat((img1, img2), 1)
        imgs = imgs.to(self.device)

        flow, _ = self.flow(imgs)
        # pdb.set_trace()
        # (Pdb) imgs.size()
        # torch.Size([1, 6, 2176, 3840])
        # (Pdb) imgs.mean()
        # tensor(0.2947, device='cuda:0')

        # (Pdb) flow.size(), flow.mean(), flow.min(), flow.max()
        # (torch.Size([1, 2, 1088, 1920]), tensor(-0.3612, device='cuda:0'), 
        # tensor(-2.2808, device='cuda:0'), tensor(1.3152, device='cuda:0'))

        # (Pdb) self.predict(imgs, flow).size()
        # torch.Size([1, 3, 2176, 3840])

        return self.predict(imgs, flow)

    def predict(self, imgs, flow, training=True, flow_gt=None):
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]

        c0 = self.context(img0, flow)
        c1 = self.context(img1, -flow)

        # why flow must be multi * 2.0 ?
        flow = F.interpolate(flow, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        refine_output, warped_img0, warped_img1, warped_img0_gt, warped_img1_gt = \
            self.fusion(img0, img1, flow, c0, c1, flow_gt)

        res = torch.sigmoid(refine_output[:, :3]) * 2 - 1
        mask = torch.sigmoid(refine_output[:, 3:4])
        merged_img = warped_img0 * mask + warped_img1 * (1 - mask)
        pred = merged_img + res

        del c0, c1, flow, refine_output, warped_img0, warped_img1, res, mask, merged_img
        torch.cuda.empty_cache()

        return torch.clamp(pred, 0, 1)

    def slow(self, img1, img2, exp):
        scale = (2 ** exp)
        seq = [None] * (scale + 1)
        seq[0] = img1.cpu()
        seq[scale] = img2.cpu()
        skip = scale
        while skip > 0:
            for start in range(0, scale, skip):
                stop = start + skip
                with torch.no_grad():
                    mid = self.forward(seq[start], seq[stop])
                seq[(start + stop)//2] = mid.cpu()
            skip = skip // 2
        return seq

backwarp_tensor_grid = {}
def warp(image_tensor, flow_tensor):
    k = (str(flow_tensor.device), str(flow_tensor.size()))
    # pdb.set_trace()
    # (Pdb) image_tensor.size(), flow_tensor.size()
    # (torch.Size([1, 3, 1088, 1920]), torch.Size([1, 2, 1088, 1920]))
    # (Pdb) k
    # ('cuda:0', 'torch.Size([1, 2, 1088, 1920])')
    B = image_tensor.shape[0]
    H = image_tensor.shape[2]
    W = image_tensor.shape[3]

    assert image_tensor.shape[2] == flow_tensor.shape[2]
    assert image_tensor.shape[3] == flow_tensor.shape[3]

    if k not in backwarp_tensor_grid:
        tenHorizontal = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
        tenVertical = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
        backwarp_tensor_grid[k] = torch.cat([tenHorizontal, tenVertical], 1).to(flow_tensor.device)

    flow_tensor = torch.cat([flow_tensor[:, 0:1, :, :] / ((W - 1.0) / 2.0),
                         flow_tensor[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
    g = (backwarp_tensor_grid[k] + flow_tensor).permute(0, 2, 3, 1)
    # pdb.set_trace()
    # (Pdb) backwarp_tensor_grid[k].size()
    # torch.Size([1, 2, 1088, 1920]), range [-1.0, 1.0]    
    # (Pdb) flow_tensor.size()
    # torch.Size([1, 2, 1088, 1920])
    # (Pdb) g.size()
    # torch.Size([1, 1088, 1920, 2])

    return F.grid_sample(input=image_tensor, grid=torch.clamp(g, -1, 1), mode='bilinear', padding_mode='zeros', align_corners=True)

def flow_conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.PReLU(out_planes)
    )

def flow_conv_wo_act(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
    )


class FlowResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(FlowResBlock, self).__init__()
        if in_planes == out_planes and stride == 1:
            self.conv0 = nn.Identity()
        else:
            self.conv0 = nn.Conv2d(in_planes, out_planes,
                                   3, stride, 1, bias=False)
        self.conv1 = flow_conv(in_planes, out_planes, 5, stride, 2)
        self.conv2 = flow_conv_wo_act(out_planes, out_planes, 3, 1, 1)
        self.relu1 = nn.PReLU(1)
        self.relu2 = nn.PReLU(out_planes)
        self.fc1 = nn.Conv2d(out_planes, 16, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(16, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        y = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        w = x.mean(3, True).mean(2, True)
        w = self.relu1(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        x = self.relu2(x * w + y)
        return x


class FlowBlock(nn.Module):
    ''' Intermediate Flow Block'''

    def __init__(self, in_planes, scale=1, c=64):
        super(FlowBlock, self).__init__()
        self.scale = scale
        self.conv0 = flow_conv(in_planes, c, 5, 2, 2)
        self.res0 = FlowResBlock(c, c)
        self.res1 = FlowResBlock(c, c)
        self.res2 = FlowResBlock(c, c)
        self.res3 = FlowResBlock(c, c)
        self.res4 = FlowResBlock(c, c)
        self.res5 = FlowResBlock(c, c)
        self.conv1 = nn.Conv2d(c, 8, 3, 1, 1)
        self.up = nn.PixelShuffle(2)

    def forward(self, x):
        if self.scale != 1:
            x = F.interpolate(x, scale_factor=1. / self.scale, mode="bilinear",
                              align_corners=False)
        x = self.conv0(x)
        x = self.res0(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.conv1(x)
        flow = self.up(x)
        if self.scale != 1:
            flow = F.interpolate(flow, scale_factor=self.scale, mode="bilinear",
                                 align_corners=False)
        return flow


class FlowModel(nn.Module):
    ''' Intermediate Flow Model'''
    def __init__(self):
        super(FlowModel, self).__init__()
        self.block0 = FlowBlock(6, scale=8, c=192)
        self.block1 = FlowBlock(8, scale=4, c=128)
        self.block2 = FlowBlock(8, scale=2, c=96)
        self.block3 = FlowBlock(8, scale=1, c=48)

    def forward(self, x):
        img0 = x[:, :3]
        img1 = x[:, 3:]
        x = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        flow0 = self.block0(x)
        F1 = flow0
        warped_img0 = warp(img0, F1)
        warped_img1 = warp(img1 -F1)
        flow1 = self.block1(torch.cat((warped_img0, warped_img1, F1), 1))
        F2 = (flow0 + flow1)
        warped_img0 = warp(img0, F2)
        warped_img1 = warp(img1, -F2)
        flow2 = self.block2(torch.cat((warped_img0, warped_img1, F2), 1))
        F3 = (flow0 + flow1 + flow2)
        warped_img0 = warp(img0, F3)
        warped_img1 = warp(img1, -F3)
        flow3 = self.block3(torch.cat((warped_img0, warped_img1, F3), 1))
        F4 = (flow0 + flow1 + flow2 + flow3)
        return F4, [F1, F2, F3, F4]


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes,
                                 kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes)
    )

def conv_woact(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
    )

class ResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(ResBlock, self).__init__()
        if in_planes == out_planes and stride == 1:
            self.conv0 = nn.Identity()
        else:
            self.conv0 = nn.Conv2d(in_planes, out_planes,
                                   3, stride, 1, bias=False)
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv_woact(out_planes, out_planes, 3, 1, 1)
        self.relu1 = nn.PReLU(1)
        self.relu2 = nn.PReLU(out_planes)
        self.fc1 = nn.Conv2d(out_planes, 16, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(16, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        y = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        w = x.mean(3, True).mean(2, True)
        w = self.relu1(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        x = self.relu2(x * w + y)
        return x

c = 32
class ContextModel(nn.Module):
    def __init__(self):
        super(ContextModel, self).__init__()
        # conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv0 = conv(3, c, 3, 2, 1)
        # ResBlock(in_planes, out_planes, stride=2)
        self.conv1 = ResBlock(c, c)
        self.conv2 = ResBlock(c, 2*c)
        self.conv3 = ResBlock(2*c, 4*c)
        self.conv4 = ResBlock(4*c, 8*c)

    def forward(self, x, flow):
        # pdb.set_trace()
        # (torch.Size([1, 3, 2176, 3840]), torch.Size([1, 2, 1088, 1920]))
        x = self.conv0(x)
        x = self.conv1(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear",
            align_corners=False) * 0.5
        f1 = warp(x, flow)
        x = self.conv2(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear",
                             align_corners=False) * 0.5
        f2 = warp(x, flow)
        x = self.conv3(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear",
                             align_corners=False) * 0.5
        f3 = warp(x, flow)
        x = self.conv4(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear",
                             align_corners=False) * 0.5
        f4 = warp(x, flow)
        # pdb.set_trace()
        # (Pdb) f1.size(), f2.size(), f3.size(), f4.size()
        # (torch.Size([1, 32, 544, 960]), 
        # torch.Size([1, 64, 272, 480]), 
        # torch.Size([1, 128, 136, 240]), 
        # torch.Size([1, 256, 68, 120]))
        # (Pdb) flow.size()
        # torch.Size([1, 2, 68, 120])

        return [f1, f2, f3, f4]


class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.conv0 = conv(8, c, 3, 2, 1)
        self.down0 = ResBlock(c, 2*c)
        self.down1 = ResBlock(4*c, 4*c)
        self.down2 = ResBlock(8*c, 8*c)
        self.down3 = ResBlock(16*c, 16*c)
        self.up0 = deconv(32*c, 8*c)
        self.up1 = deconv(16*c, 4*c)
        self.up2 = deconv(8*c, 2*c)
        self.up3 = deconv(4*c, c)
        self.conv = nn.Conv2d(c, 16, 3, 1, 1)
        self.up4 = nn.PixelShuffle(2)

    def forward(self, img0, img1, flow, c0, c1, flow_gt):
        # (Pdb) img0.size()
        # torch.Size([1, 3, 2176, 3840])
        # (Pdb) img1.size()
        # torch.Size([1, 3, 2176, 3840])
        # (Pdb) flow.size()
        # torch.Size([1, 2, 2176, 3840])
        # (Pdb) flow_gt is None
        # True
        # len(c0), len(c1) --> (4, 4)
        # (Pdb) c0[0].size(), c0[1].size(), c0[2].size(), c0[3].size()
        # (torch.Size([1, 32, 544, 960]), 
        # torch.Size([1, 64, 272, 480]), 
        # torch.Size([1, 128, 136, 240]), 
        # torch.Size([1, 256, 68, 120]))

        warped_img0 = warp(img0, flow)
        warped_img1 = warp(img1, -flow)
        if flow_gt == None:
            warped_img0_gt, warped_img1_gt = None, None
        else:
            warped_img0_gt = warp(img0, flow_gt[:, :2])
            warped_img1_gt = warp(img1, flow_gt[:, 2:4])
        x = self.conv0(torch.cat((warped_img0, warped_img1, flow), 1))
        # (Pdb) xxx=torch.cat((warped_img0, warped_img1, flow), 1)
        # (Pdb) xxx.size()
        # torch.Size([1, 8, 2176, 3840])
        # (Pdb) self.conv0(xxx).size()
        # torch.Size([1, 32, 1088, 1920])

        s0 = self.down0(x)
        s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
        s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
        s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
        x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
        x = self.up1(torch.cat((x, s2), 1))
        x = self.up2(torch.cat((x, s1), 1))
        x = self.up3(torch.cat((x, s0), 1))
        x = self.up4(self.conv(x))
        # pdb.set_trace()
        # (Pdb) warped_img0_gt is None
        # True
        # (Pdb) warped_img0.size()
        # torch.Size([1, 3, 2176, 3840])
        # x.size()
        # torch.Size([1, 4, 2176, 3840])

        return x, warped_img0, warped_img1, warped_img0_gt, warped_img1_gt

