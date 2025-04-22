from model import ConvAutoencoder

import piq
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset
from datasets import load_dataset
from PIL import Image
from piq import psnr, ssim, multi_scale_ssim

class TexturesDataset(Dataset):
    def __init__(self, split="train"):
        raw_dataset = load_dataset(
            "dream-textures/textures-color-normal-1k", split=split
        )
        self.valid_samples = []

        for i in range(len(raw_dataset)):
            if i == 465:
                continue

            sample = raw_dataset[i]

            color, normal = sample["color"], sample["normal"]
            if (
                isinstance(color, Image.Image)
                and color.mode == "RGB"
                and isinstance(normal, Image.Image)
                and normal.mode == "RGB"
            ):
                self.valid_samples.append(sample)
            else:
                print(
                    f"Skipping sample {i}: invalid mode. Color: {getattr(color, 'mode', None)}, Normal: {getattr(normal, 'mode', None)}"
                )

        print(
            f"Loaded {len(self.valid_samples)} valid RGB samples out of {len(raw_dataset)}."
        )

        self.transform = transforms.Compose(
            [transforms.Resize((512, 512)), transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        color = self.transform(sample["color"])
        normal = self.transform(sample["normal"])
        return torch.cat((color, normal), dim=0)
    
def get_weighted_loss(weights):
    def loss_fn(x, y):
        mse_loss = nn.functional.mse_loss(x, y)
        l1_loss = nn.functional.l1_loss(x, y)
        ssim_loss = 1 - piq.ssim(x, y, data_range=1.0)
        ms_ssim_loss = 1 - piq.multi_scale_ssim(x, y, data_range=1.0)
        return (weights[0] * mse_loss +
                weights[1] * l1_loss +
                weights[2] * ssim_loss +
                weights[3] * ms_ssim_loss)
    return loss_fn

def train_autoencoder(num_epochs = 50): 
    # load dataset
    dataset = TexturesDataset()

    # train / set size
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    # set batch size
    batch_size = 16

    # split train / test set
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    # dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # model hyperparameters
    learning_rate = 1e-3
    weights = (0.20, 0.75, 0.0, 0.05)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # model 
    model = ConvAutoencoder().to(device)
    criterion = get_weighted_loss(weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # loss 
    loss_history = {
        "total": [],
        "mse": [],
        "l1": [],
        "ssim": [],
        "ms_ssim": []
    }

    # training loop
    for epoch in range(num_epochs):
        total_loss = 0
        total_mse = 0
        total_l1 = 0
        total_ssim = 0
        total_ms_ssim = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)

            # Individual losses
            mse_loss = nn.functional.mse_loss(output, data)
            l1_loss = nn.functional.l1_loss(output, data)
            ssim_loss = 1 - piq.ssim(output, data, data_range=1.0)
            ms_ssim_loss = 1 - piq.multi_scale_ssim(output, data, data_range=1.0)

            # Weighted loss
            loss = criterion(output, data)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_l1 += l1_loss.item()
            total_ssim += ssim_loss.item()
            total_ms_ssim += ms_ssim_loss.item()

        # Store epoch-wise averages
        n = len(train_loader)
        loss_history["total"].append(total_loss / n)
        loss_history["mse"].append(total_mse / n)
        loss_history["l1"].append(total_l1 / n)
        loss_history["ssim"].append(total_ssim / n)
        loss_history["ms_ssim"].append(total_ms_ssim / n)

        print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss / n:.4f}, "
            f"MSE: {total_mse / n:.4f}, L1: {total_l1 / n:.4f}, "
            f"SSIM: {total_ssim / n:.4f}, MS-SSIM: {total_ms_ssim / n:.4f}")
        
    # return model and test_loader for evaluation
    return model, test_loader