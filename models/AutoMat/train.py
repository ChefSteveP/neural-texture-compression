from types import SimpleNamespace

import torch
from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms


def transform_batch(batch):
    return {
        "color": [transforms.ToTensor()(img.convert("RGB")) for img in batch["color"]],
        "normal": [
            transforms.ToTensor()(img.convert("RGB")) for img in batch["normal"]
        ],
    }


def train(args):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    training_dataset = load_dataset("dream-textures/textures-color-normal-1k")["train"]
    training_dataset.set_transform(transform_batch)
    dataloader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    for batch in dataloader:
        print(batch["color"][0].shape)
        print(batch["normal"][0].shape)
        break


train(
    args=SimpleNamespace(
        batch_size=16,
    )
)
