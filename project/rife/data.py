"""Data loader."""# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020, All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 12月 20日 星期日 08:42:09 CST
# ***
# ************************************************************************************/
#

import os
import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as T
import torchvision.utils as utils

train_dataset_rootdir = "dataset/train/"
test_dataset_rootdir = "dataset/test/"
VIDEO_SEQUENCE_LENGTH = 2  # for Video Slow

def multiple_scale(data, multiple=32):
    '''
    Scale image to a multiple.
    input data is tensor, with CxHxW format.
    '''

    C, H, W = data.shape
    Hnew = ((H - 1) // multiple + 1)*multiple
    Wnew = ((W - 1) // multiple + 1)*multiple
    if Hnew == H and Wnew == W:
        return data

    # Padding with zeros ...        
    temp = data.new_zeros(C, Hnew, Wnew)
    temp[:, 0:H, 0:W] = data

    return temp


def get_transform(train=True):
    """Transform images."""
    ts = []
    # if train:
    #     ts.append(T.RandomHorizontalFlip(0.5))

    ts.append(T.ToTensor())
    return T.Compose(ts)

class Video(data.Dataset):
    """Define Video Frames Class."""

    def __init__(self, seqlen=VIDEO_SEQUENCE_LENGTH, transforms=get_transform()):
        """Init dataset."""
        super(Video, self).__init__()
        self.seqlen = seqlen
        self.transforms = transforms
        self.root = ""
        self.images = []
        self.height = 0
        self.width = 0

    def reset(self, root):
        # print("Video Reset Root: ", root)
        self.root = root
        self.images = list(sorted(os.listdir(root)))

        # Suppose the first image size is video frame size
        if len(self.images) > 0:
            filename = os.path.join(self.root, self.images[0])
            img = self.transforms(Image.open(filename).convert("RGB"))
            C, H, W = img.size()
            self.height = H
            self.width = W

    def __getitem__(self, idx):
        """Load images."""
        n = len(self.images)
        filelist = []
        delta = (self.seqlen - 1)/2
        for k in range(-int(delta), int(delta + 0.5) + 1):
            if (idx + k < 0):
                filename = self.images[0]
            elif (idx + k >= n):
                filename = self.images[n - 1]
            else:
                filename = self.images[idx + k]
            filelist.append(os.path.join(self.root, filename))
        # print("filelist: ", filelist)
        sequence = []
        for filename in filelist:
            img = Image.open(filename).convert("RGB")
            img = self.transforms(img)
            img = multiple_scale(img)
            C, H, W = img.size()
            img = img.view(1, C, H, W)
            sequence.append(img)
        return torch.cat(sequence, dim=0)

    def __len__(self):
        """Return total numbers of images."""
        return len(self.images)

    def __repr__(self):
        """
        Return printable representation of the dataset object.
        """
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of samples: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms: '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transforms.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

def RIFEDatasetTest():
    """Test dataset ..."""

    ds = Video()
    ds.reset(train_dataset_rootdir)
    print(ds)
    # src, tgt = ds[0]
    # grid = utils.make_grid(torch.cat([src.unsqueeze(0), tgt.unsqueeze(0)], dim=0), nrow=2)
    # ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    # image = Image.fromarray(ndarr)
    # image.show()

if __name__ == '__main__':
    RIFEDatasetTest()
