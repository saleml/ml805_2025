import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset
transform = transforms.ToTensor()
batch_size = 128
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)


class SimpleUnet(nn.Module):
    """A simplified U-Net architecture for diffusion models."""

    def __init__(self):
        super().__init__()
        # Number of channels for each layer
        c = 64

        # TODO: Task 1 - Implement time embedding
        self.time_embed = nn.Sequential(
            # YOUR CODE HERE
        )

        # Encoder (downsampling)
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1), nn.GroupNorm(8, c), nn.SiLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(c, c * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, c * 2),
            nn.SiLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(c * 2, c * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, c * 2),
            nn.SiLU(),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c * 2, c * 4, 3, padding=1),
            nn.GroupNorm(8, c * 4),
            nn.SiLU(),
            nn.Conv2d(c * 4, c * 2, 3, padding=1),
        )

        # Decoder (upsampling)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(c * 2, c * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, c * 2),
            nn.SiLU(),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, 1, 3, padding=1),
        )

    def forward(self, x, t):
        # TODO: Task 2 - Implement forward pass
        # YOUR CODE HERE
        pass


class SimpleDiffusion:
    """A simplified diffusion model."""

    def __init__(self, n_steps=1000):
        self.n_steps = n_steps
        self.device = device

        self.beta = torch.linspace(1e-5, 0.02, n_steps).to(device)

        # TODO: Task 3 - Implement noise schedule
        # YOUR CODE HERE

        # Create model and optimizer
        self.model = SimpleUnet().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

    def noise_images(self, x, t):
        """Add noise to images at timestep t."""
        # TODO: Task 4 - Implement noising process
        # YOUR CODE HERE
        pass

    def train_step(self, images):
        """Single training step."""
        self.optimizer.zero_grad()
        t = torch.randint(0, self.n_steps, (images.shape[0],), device=self.device)
        noisy_images, noise = self.noise_images(images, t)
        noise_pred = self.model(noisy_images, t)
        loss = nn.MSELoss()(noise_pred, noise)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def sample(self, n_samples):
        """Sample new images."""
        # TODO: Task 5 - Implement sampling process
        # YOUR CODE HERE
        pass


def train_and_sample(n_epochs=1, test_mode=False):
    """Train the model and show samples."""
    diffusion = SimpleDiffusion(n_steps=500)

    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch+1}/{n_epochs}")

        total_batches = 2 if test_mode else len(train_loader)
        pbar = tqdm(total=total_batches, desc="Training")

        running_loss = 0
        for i, (images, _) in enumerate(train_loader):
            if test_mode and i >= 2:
                break

            images = images.to(device)
            loss = diffusion.train_step(images)
            running_loss = 0.9 * running_loss + 0.1 * loss

            pbar.update(1)
            pbar.set_description(f"Loss: {running_loss:.4f}")

        pbar.close()

        print("Generating samples...")
        samples = diffusion.sample(4)

        plt.figure(figsize=(8, 8))
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.imshow(samples[i, 0].cpu(), cmap="gray")
            plt.axis("off")
        plt.show()


# Run training
print(f"Training on device: {device}")
train_and_sample(n_epochs=20, test_mode=False)

# TODO: Task 6 - Implement your improvements here
# YOUR CODE HERE
