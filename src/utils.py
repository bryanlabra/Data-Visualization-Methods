# utils.py
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import gpytorch
from sklearn.manifold import TSNE  # Import t-SNE from scikit-learn

# Data loading function
def get_mnist_dataloader(batch_size=256, root="data"):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
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