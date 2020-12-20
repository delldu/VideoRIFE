"""Model predict."""# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020, All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 12月 20日 星期日 08:42:09 CST
# ***
# ************************************************************************************/
#
import argparse
import os

import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from rife.rife import RIFE
from rife.data import Video

if __name__ == "__main__":
    """Demo."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="demo", help="input image")
    parser.add_argument(
        '--output', type=str, default="output", help="output video folder")
    parser.add_argument("--exp", type=int, default=4,
                        help='Increase the frames by 2**N. Example exp=1 ==> 2x frames, exp=2, means 4x')
    args = parser.parse_args()

    # Create directory to store results
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    model = RIFE()
    model.eval()

    video = Video()
    video.reset(args.input)

    # totensor = transforms.ToTensor()
    toimage = transforms.ToPILImage()

    progress_bar = tqdm(total = len(video) - 1)

    count = 0
    for index in range(len(video) - 1):
        progress_bar.update(1)

        # videos format: TxCxHxW, Here T = 2
        frames = video[index]
        frame0 = frames[0:1]
        frame1 = frames[1:2]

        seqence = model.slow(frame0, frame1, args.exp)
        for i in range(2**args.exp):
            oldimg = seqence[i][:, :, 0:video.height, 0:video.width]
            toimage(oldimg.squeeze()).save("{}/{:06d}.png".format(args.output, count))
            count += 1
