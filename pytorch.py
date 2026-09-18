"""Prepare perceptual-loss weights for native training.

python pytorch.py prepare
"""
import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def prepare():
    vgg = torch.hub.load_state_dict_from_url(
        "https://download.pytorch.org/models/vgg16-397923af.pth", map_location="cpu", check_hash=True)
    lpips = torch.hub.load_state_dict_from_url(
        "https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/vgg.pth",
        map_location="cpu", file_name="lpips-vgg-v0.1.pth")
    tensors = {name: value.contiguous() for name, value in vgg.items() if name.startswith("features.")}
    tensors.update({f"linear.{i}": lpips[f"lin{i}.model.1.weight"].flatten().contiguous() for i in range(5)})
    path = Path(__file__).resolve().parent / "data/.flowdit/lpips-vgg.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path, metadata={"source": "torchvision VGG16 / LPIPS v0.1, perceptual loss only"})
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    parser.parse_args()
    prepare()
