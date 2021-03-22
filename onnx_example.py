import torch
import torch.nn.functional as F
import pdb


@torch.jit.script
def GridSampleFunction(x, m):
    return F.grid_sample(x, m, align_corners=True)


class GridSample(torch.nn.Module):
    def __init__(self):
        super(GridSample,self).__init__()

    def forward(self, x, m):
        return GridSampleFunction(x, m)

x = torch.rand(1, 1, 10, 10)
m = torch.Tensor([[[1, 0, 0], [0, 1, 0]]])
m = F.affine_grid(m, x.size(), align_corners=True).type_as(x)

model = GridSample()

# traced_script_module = torch.jit.trace(model, (x, m))


@torch.jit.script
def loop(x, y):
    for i in range(int(y)):
        x = x + i
    return x

class LoopModel(torch.nn.Module):
    def forward(self, x, y):
        return loop(x, y)

model = LoopModel()
dummy_input = torch.ones(2, 3, dtype=torch.long)
loop_count = torch.tensor(5, dtype=torch.long)
torch.onnx.export(model, (dummy_input, loop_count), 'loop.onnx', verbose=True,
                  input_names=['input_data', 'loop_range'])

