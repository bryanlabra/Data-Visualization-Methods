import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import gpytorch
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
import umap

# Data loading function
def get_mnist_dataloader(batch_size=256, root="data"):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts images to [0, 1] range by default
    ])
    mnist_train = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    return train_loader

# PPCA Model class
class PPCA(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(PPCA, self).__init__()
        self.mean = nn.Parameter(torch.zeros(input_dim))
        self.W = nn.Parameter(torch.randn(input_dim, latent_dim))
        self.sigma_sq = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x = x - self.mean
        cov = self.W @ self.W.T + self.sigma_sq * torch.eye(self.W.size(0), device=x.device)
        return cov

# GPLVM Model class
class ProjectedGPLVM(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, latent_dim, data_dim):
        super(ProjectedGPLVM, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # Projector layer to map latent space (latent_dim) to data space (data_dim)
        self.projector = torch.nn.Linear(latent_dim, data_dim)

    def forward(self, x):
        # Compute the GP output in the latent space
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_distribution = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
        # Directly project `x` to observed space without operating on `latent_distribution`
        projected_mean = self.projector(x)
        
        # Use a simple isotropic covariance for the projected data
        projected_covariance = torch.eye(projected_mean.size(1)) * 0.1  # Adjust scale if needed
        return gpytorch.distributions.MultivariateNormal(projected_mean, projected_covariance)

# Autoencoder Model class
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=2):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder network with convolutional and linear layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, latent_dim)  # Project to the 2D latent space
        )
        
        # Decoder network with transposed convolutions and linear layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14 -> 28x28
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, x):
        # Encode to latent space
        latent = self.encoder(x)
        # Decode back to original dimension
        reconstruction = self.decoder(latent)
        return latent, reconstruction

# Visualization for PPCA
def plot_ppca(projections, labels):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(projections[:, 0].detach().numpy(), projections[:, 1].detach().numpy(), c=labels, cmap="tab10", alpha=0.6)
    plt.colorbar(scatter, ticks=range(10), label="Digit Class")
    plt.xlabel("PPCA Component 1")
    plt.ylabel("PPCA Component 2")
    plt.title("PPCA on MNIST Dataset")
    plt.show()

# Visualization for GPLVM
def plot_gplvm(latent_x, labels):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_x[:, 0].detach().numpy(), latent_x[:, 1].detach().numpy(), c=labels, cmap="tab10", alpha=0.6)
    plt.colorbar(scatter, ticks=range(10), label="Digit Class")
    plt.xlabel("GPLVM Component 1")
    plt.ylabel("GPLVM Component 2")
    plt.title("GPLVM on MNIST Dataset")
    plt.show()

# t-SNE function for visualization
def plot_tsne(data, labels, n_components=2, perplexity=30, learning_rate=200):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate)
    projections = tsne.fit_transform(data)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(projections[:, 0], projections[:, 1], c=labels, cmap="tab10", alpha=0.6)
    plt.colorbar(scatter, ticks=range(10), label="Digit Class")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title("t-SNE on MNIST Dataset")
    plt.show()

# UMAP function for visualization
def plot_umap(data, labels, n_components=2, n_neighbors=15, min_dist=0.1):
    umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist)
    projections = umap_model.fit_transform(data)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(projections[:, 0], projections[:, 1], c=labels, cmap="tab10", alpha=0.6)
    plt.colorbar(scatter, ticks=range(10), label="Digit Class")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.title("UMAP on MNIST Dataset")
    plt.show()

# Isomap function for visualization
def plot_isomap(data, labels, n_components=2, n_neighbors=5):
    isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors)
    projections = isomap.fit_transform(data)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(projections[:, 0], projections[:, 1], c=labels, cmap="tab10", alpha=0.6)
    plt.colorbar(scatter, ticks=range(10), label="Digit Class")
    plt.xlabel("Isomap Component 1")
    plt.ylabel("Isomap Component 2")
    plt.title("Isomap on MNIST Dataset")
    plt.show()

# LLE function for visualization
def plot_lle(data, labels, n_components=2, n_neighbors=10):
    lle = LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors)
    projections = lle.fit_transform(data)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(projections[:, 0], projections[:, 1], c=labels, cmap="tab10", alpha=0.6)
    plt.colorbar(scatter, ticks=range(10), label="Digit Class")
    plt.xlabel("LLE Component 1")
    plt.ylabel("LLE Component 2")
    plt.title("LLE on MNIST Dataset")
    plt.show()

# Autoencoder visualization function
def plot_autoencoder(latent_space, labels):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_space[:, 0], latent_space[:, 1], c=labels, cmap="tab10", alpha=0.6)
    plt.colorbar(scatter, ticks=range(10), label="Digit Class")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("Autoencoder Latent Space on MNIST Dataset")
    plt.show()