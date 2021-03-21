import torch
from model import get_model, model_device
import pdb

def export_torch_model():
    """Export torch model."""

    weight_file = "models/VideoRIFE.pth"
    script_file = "output/VideoRIFE.pt"

    # 1. Load model
    print("Loading model ...")
    model = get_model(weight_file)
    model.eval()

    device = model_device()

    # 2. Model export
    print("Export model ...")
    input = torch.randn(2, 3, 512, 512).to(device)

    traced_script_module = torch.jit.trace(model, input)
    traced_script_module.save(script_file)

if __name__ == "__main__":
    """export torch model."""
    export_torch_model()