# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020-2022(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 12月 28日 星期一 14:29:37 CST
# ***
# ************************************************************************************/
#

import pdb
import os
import time
from tqdm import tqdm

import torch
import todos
import video_slow

SO_B, SO_C, SO_H, SO_W = 1, 6, 512, 512


def compile():
    model, device = video_slow.get_tvm_model()

    todos.data.mkdir("output")
    if not os.path.exists("output/video_slow.so"):
        input = torch.randn(SO_B, SO_C, SO_H, SO_W)
        todos.tvmod.compile(model, device, input, "output/video_slow.so")
    todos.model.reset_device()


def predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    device = todos.model.get_device()
    tvm_model = todos.tvmod.load("output/video_slow.so", str(device))

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    mean_time = 0
    progress_bar = tqdm(total=len(image_filenames))

    I1 = I2 = None
    OUTPUT_COUNT = 0
    for i, filename in enumerate(image_filenames):
        progress_bar.update(1)

        input_tensor = todos.data.load_tensor(filename)
        input_tensor = todos.data.resize_tensor(input_tensor, SO_H, SO_W)

        if i == 0:
            I1 = input_tensor
            I2 = input_tensor
        else:
            I1 = I2
            I2 = input_tensor

        start_time = time.time()
        outputs = todos.model.forward(model, device, torch.cat([I1, I2], dim=1))
        mean_time += time.time() - start_time

        for output_tensor in outputs[1:]:  # skip first
            output_file = f"{output_dir}/{OUTPUT_COUNT + 1:06d}.png"
            todos.data.save_tensor([output_tensor], output_file)
            OUTPUT_COUNT = OUTPUT_COUNT + 1

    mean_time = mean_time / len(image_filenames)

    print(f"Mean spend {mean_time:0.4f} seconds")
    os.system("nvidia-smi | grep python")

    todos.model.reset_device()


if __name__ == "__main__":
    compile()
    predict("images/*.png", "output/so")
