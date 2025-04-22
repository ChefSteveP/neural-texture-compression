from pathlib import Path

import matplotlib.pyplot as plt
import piq
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from datasets import load_dataset
from model import ConvAutoencoder
from PIL import Image
from piq import multi_scale_ssim, psnr, ssim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from train import TexturesDataset


def download_file_with_progress(url, filename):
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 KB
    t = tqdm(
        total=total_size, unit="iB", unit_scale=True, desc="Downloading model weights"
    )

    with open(filename, "wb") as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

    if total_size != 0 and t.n != total_size:
        print("WARNING: Downloaded size does not match expected size.")


def evaluate(model, test_loader, device):
    model.eval()
    total_psnr = total_ssim = total_mse = total_ms_ssim = 0.0
    num_samples = 0

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)

            total_psnr += psnr(output, data).item()
            total_ssim += ssim(output, data, data_range=1.0).item()
            total_mse += nn.MSELoss()(output, data).item()
            total_ms_ssim += multi_scale_ssim(output, data, data_range=1.0).item()
            num_samples += 1

    print("\nEvaluation Results:")
    print(f"Average PSNR: {total_psnr / num_samples:.4f}")
    print(f"Average SSIM: {total_ssim / num_samples:.4f}")
    print(f"Average MSE: {total_mse / num_samples:.4f}")
    print(f"Average MS-SSIM: {total_ms_ssim / num_samples:.4f}")


def show_reconstructions(model, test_loader, device, n=5):
    model.eval()
    data = next(iter(test_loader))[:n].to(device)

    with torch.no_grad():
        reconstructed = model(data)

    data = data.cpu()
    reconstructed = reconstructed.cpu()
    fig, axs = plt.subplots(4, n, figsize=(n * 2, 6))
    for i in range(n):
        axs[0, i].imshow(data[i, :3].permute(1, 2, 0))
        axs[1, i].imshow(data[i, 3:].permute(1, 2, 0))
        axs[2, i].imshow(reconstructed[i, :3].permute(1, 2, 0))
        axs[3, i].imshow(reconstructed[i, 3:].permute(1, 2, 0))
        for ax in axs[:, i]:
            ax.axis("off")
    axs[0, 0].set_title("Original Color")
    axs[1, 0].set_title("Original Normal")
    axs[2, 0].set_title("Reconstructed Color")
    axs[3, 0].set_title("Reconstructed Normal")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Using device: {device}")

    # Dropbox model URL (direct link)
    dropbox_url = "https://www.dropbox.com/scl/fi/v5bfah5hzpn6b1h0wjabv/conv_autoencoder_weights.pt?rlkey=4qad4l1kk4qehu08m8gz160j6&st=st4f22bn&dl=1"
    local_model_path = Path("conv_autoencoder_weights.pt")

    if not local_model_path.exists():
        download_file_with_progress(dropbox_url, local_model_path)

    # Load model
    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(local_model_path, map_location=device))
    print("Loaded model from conv_autoencoder_weights.pt")

    # Load test data
    dataset = TexturesDataset()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Evaluate and visualize
    evaluate(model, test_loader, device)
    show_reconstructions(model, test_loader, device, n=5)
