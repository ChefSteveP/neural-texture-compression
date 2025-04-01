import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from model import VQVAE, GSSOFT

def shift(x):
    """Shift pixel values from [0, 1] to [-0.5, 0.5] for model input."""
    return x - 0.5

def evaluate_model(args):
    # Handle OpenMP error
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define model
    if args.model == "VQVAE":
        model = VQVAE(args.channels, args.latent_dim, args.num_embeddings, args.embedding_dim)
    elif args.model == "GSSOFT":
        model = GSSOFT(args.channels, args.latent_dim, args.num_embeddings, args.embedding_dim)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # Load checkpoint
    if args.checkpoint is None:
        model_name = f"{args.model}_C_{args.channels}_N_{args.latent_dim}_M_{args.num_embeddings}_D_{args.embedding_dim}"
        checkpoint_paths = list(Path(model_name).glob("model.ckpt-*.pt"))
        if not checkpoint_paths:
            raise FileNotFoundError(f"No checkpoints found in {model_name} directory")
        
        # Get the latest checkpoint
        checkpoint_path = max(checkpoint_paths, key=lambda p: int(p.stem.split('-')[1]))
    else:
        checkpoint_path = args.checkpoint
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    # Use weights_only=True to avoid security warnings
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])

    # Data transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(shift)
    ])

    # Load dataset
    dataset = datasets.CIFAR10("./CIFAR10", train=False, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                            num_workers=args.num_workers)

    # Metrics to track
    total_perplexity = 0
    total_samples = 0
    total_bpd = 0
    
    # Get a batch of images for visualization
    print("Loading test images...")
    try:
        test_images, test_labels = next(iter(dataloader))
        test_images = test_images[:args.num_examples]
        test_images = test_images.to(device)
        
        # Get class names for labels
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
        test_classes = [classes[label] for label in test_labels[:args.num_examples]]
    except Exception as e:
        print(f"Error loading test images: {e}")
        raise

    print("Running model inference...")
    try:
        with torch.no_grad():
            if args.model == "VQVAE":
                dist, vq_loss, perplexity = model(test_images)
                kl = args.latent_dim * 8 * 8 * np.log(args.num_embeddings)
            else:  # GSSOFT
                dist, kl, perplexity = model(test_images)
            
            # Calculate bits per dimension
            targets = (test_images + 0.5) * 255
            targets = targets.long()
            logp = dist.log_prob(targets).sum((1, 2, 3)).mean()
            N = 3 * 32 * 32  # CIFAR image size (3 channels, 32x32 pixels)
            elbo = (kl - logp) / N
            bpd = elbo / np.log(2)
            
            # Get reconstructions
            print("Generating reconstructions...")
            reconstructions = torch.argmax(dist.logits, dim=-1).float() / 255
            
            # Track metrics
            total_perplexity += perplexity.item()
            total_samples += test_images.size(0)
            total_bpd += bpd.item() * test_images.size(0)
    except Exception as e:
        print(f"Error during model inference: {e}")
        raise

    # Create visualization grid
    print("Creating visualization...")
    fig, axes = plt.subplots(args.num_examples, 2, figsize=(10, args.num_examples * 2.5))
    
    # Handle case with only one example
    if args.num_examples == 1:
        axes = [axes]
    
    for i in range(args.num_examples):
        # Display original
        orig_img = (test_images[i] + 0.5).cpu().permute(1, 2, 0).numpy()
        axes[i][0].imshow(np.clip(orig_img, 0, 1))
        axes[i][0].set_title(f"Original ({test_classes[i]})")
        axes[i][0].axis('off')
        
        # Display reconstruction
        recon_img = reconstructions[i].cpu().permute(1, 2, 0).numpy()
        axes[i][1].imshow(np.clip(recon_img, 0, 1))
        axes[i][1].set_title("Reconstruction")
        axes[i][1].axis('off')
    
    # Show metrics
    avg_perplexity = total_perplexity / total_samples
    avg_bpd = total_bpd / total_samples
    
    plt.suptitle(f"Model: {args.model}, Perplexity: {avg_perplexity:.2f}, BPD: {avg_bpd:.2f}")
    plt.tight_layout()
    
    # Save or display
    if args.output:
        output_path = args.output
        plt.savefig(output_path)
        print(f"Saved visualization to {output_path}")
    else:
        print("Displaying visualization (close the window to continue)...")
        plt.show()

    print(f"Evaluation Results:")
    print(f"Average Perplexity: {avg_perplexity:.4f}")
    print(f"Average Bits per Dimension (BPD): {avg_bpd:.4f}")
    
    # Additional detailed metrics (if needed)
    if args.detailed:
        # Run evaluation on the entire dataset
        total_perplexity = 0
        total_logp = 0
        total_kl = 0
        total_samples = 0
        
        print("Evaluating on full test set...")
        for images, _ in dataloader:
            images = images.to(device)
            batch_size = images.size(0)
            
            with torch.no_grad():
                if args.model == "VQVAE":
                    dist, vq_loss, perplexity = model(images)
                    batch_kl = args.latent_dim * 8 * 8 * np.log(args.num_embeddings)
                else:  # GSSOFT
                    dist, batch_kl, perplexity = model(images)
                
                targets = (images + 0.5) * 255
                targets = targets.long()
                logp = dist.log_prob(targets).sum((1, 2, 3)).mean()
                
                total_perplexity += perplexity.item()
                total_logp += logp.item() * batch_size
                total_kl += batch_kl * batch_size if isinstance(batch_kl, float) else batch_kl.item() * batch_size
                total_samples += batch_size
        
        # Calculate overall metrics
        avg_perplexity = total_perplexity / total_samples
        avg_logp = total_logp / total_samples
        avg_kl = total_kl / total_samples
        avg_elbo = (avg_kl - avg_logp) / N
        avg_bpd = avg_elbo / np.log(2)
        
        print(f"Full Dataset Evaluation Results:")
        print(f"Average Perplexity: {avg_perplexity:.4f}")
        print(f"Average Log Likelihood: {avg_logp:.4f}")
        print(f"Average KL: {avg_kl:.4f}")
        print(f"Average ELBO: {avg_elbo:.4f}")
        print(f"Average Bits per Dimension (BPD): {avg_bpd:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VQVAE or GSSOFT model")
    parser.add_argument("--model", choices=["VQVAE", "GSSOFT"], default="VQVAE",
                        help="Select model to evaluate (either VQVAE or GSSOFT)")
    parser.add_argument("--channels", type=int, default=256, 
                        help="Number of channels in conv layers")
    parser.add_argument("--latent-dim", type=int, default=8, 
                        help="Dimension of categorical latents")
    parser.add_argument("--num-embeddings", type=int, default=128, 
                        help="Number of codebook embeddings size")
    parser.add_argument("--embedding-dim", type=int, default=32, 
                        help="Dimension of codebook embeddings")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to checkpoint file (if not provided, latest checkpoint will be used)")
    parser.add_argument("--batch-size", type=int, default=32, 
                        help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=2, 
                        help="Number of dataloader workers")
    parser.add_argument("--num-examples", type=int, default=5, 
                        help="Number of examples to visualize")
    parser.add_argument("--output", type=str, default=None, 
                        help="Path to save visualization (if not provided, will display)")
    parser.add_argument("--detailed", action="store_true", 
                        help="Run detailed evaluation on the entire dataset")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    try:
        evaluate_model(args)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()