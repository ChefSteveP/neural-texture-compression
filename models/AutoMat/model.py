import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self):
        self.latent_dim = 128
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1),  # 512x512x6 -> 512x512x64
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # -> (64, 256, 256)
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # -> (128, 128, 128)
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # -> (256, 64, 64)
            
            nn.Flatten(),
            nn.Linear(256 * 64 * 64, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 64 * 64),
            nn.ReLU(),
            nn.Unflatten(1, (256, 64, 64)),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # -> (128, 128, 128)
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 256, 256)
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 6, kernel_size=4, stride=2, padding=1),  # -> (6, 512, 512)
            nn.Sigmoid()  # For pixel values in [0,1]
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out