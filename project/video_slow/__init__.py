"""Image/Video Slow Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, 2022(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
import torch

import redos
import todos
from . import rife

import pdb


def get_model():
    """Create model."""

    model_path = "models/video_slow.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = rife.IFNet()
    todos.model.load(model, checkpoint)
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    model = torch.jit.script(model)

    todos.data.mkdir("output")
    if not os.path.exists("output/video_slow.torch"):
        model.save("output/video_slow.torch")

    return model, device


def model_forward(model, device, input_tensor, multi_times=8):
    # zeropad for model
    H, W = input_tensor.size(2), input_tensor.size(3)
    if H % multi_times != 0 or W % multi_times != 0:
        input_tensor = todos.data.zeropad_tensor(input_tensor, times=multi_times)

    torch.cuda.synchronize()
    with torch.jit.optimized_execution(False):
        output_tensor = todos.model.forward(model, device, input_tensor)
    torch.cuda.synchronize()

    return output_tensor[:, :, 0:H, 0:W]


def model_forward_times(model, device, i1, i2, slow_times=1):
    inputs = [i1, i2]
    outputs = []
    for n in range(slow_times):
        outputs = []
        outputs.append(inputs[0].cpu())
        for i in range(len(inputs) - 1):
            images = torch.cat((inputs[i].cpu(), inputs[i + 1].cpu()), dim=1)
            middle = model_forward(model, device, images)
            outputs.append(middle.cpu())
            outputs.append(inputs[i + 1].cpu())
        inputs = outputs  # for next time
    return outputs


def image_predict(input_files, slow_times, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()

    # load files
    image_filenames = todos.data.load_files(input_files)

    # Start predict
    progress_bar = tqdm(total=len(image_filenames))

    I1 = I2 = None
    OUTPUT_COUNT = 0
    for i, filename in enumerate(image_filenames):
        progress_bar.update(1)

        input_tensor = todos.data.load_tensor(filename)
        if i == 0:
            I1 = input_tensor
            I2 = input_tensor
        else:
            I1 = I2
            I2 = input_tensor

        outputs = model_forward_times(model, device, I1, I2, slow_times)

        for output_tensor in outputs[1:]:  # skip first
            output_file = f"{output_dir}/{OUTPUT_COUNT + 1:06d}.png"
            todos.data.save_tensor([output_tensor], output_file)
            OUTPUT_COUNT = OUTPUT_COUNT + 1
    todos.model.reset_device()

def video_predict(input_file, slow_times, output_file):
    # load video
    video = redos.video.Reader(input_file)
    if video.n_frames < 1:
        print(f"Read video {input_file} error.")
        return False

    # Create directory to store result
    output_dir = output_file[0 : output_file.rfind(".")]
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()

    print(f"  Slow down {input_file}, save to {output_file} ...")
    progress_bar = tqdm(total=video.n_frames)

    global I1, I2, OUTPUT_COUNT
    I1 = I2 = None
    OUTPUT_COUNT = 0

    def slow_video_frame(no, data):
        global I1, I2, OUTPUT_COUNT
        progress_bar.update(1)

        input_tensor = todos.data.frame_totensor(data)

        # convert tensor from 1x4xHxW to 1x3xHxW
        input_tensor = input_tensor[:, 0:3, :, :]
        if no == 0:
            I1 = input_tensor
            I2 = input_tensor
        else:
            I1 = I2
            I2 = input_tensor

        outputs = model_forward_times(model, device, I1, I2, slow_times)

        for output_tensor in outputs[1:]:  # skip first
            output_file = f"{output_dir}/{OUTPUT_COUNT + 1:06d}.png"
            todos.data.save_tensor([output_tensor], output_file)
            OUTPUT_COUNT = OUTPUT_COUNT + 1

    video.forward(callback=slow_video_frame)

    redos.video.encode(output_dir, output_file)

    # delete temp files
    for i in range(OUTPUT_COUNT):
        temp_output_file = "{}/{:06d}.png".format(output_dir, i + 1)
        os.remove(temp_output_file)
    os.removedirs(output_dir)
    todos.model.reset_device()

    return True
