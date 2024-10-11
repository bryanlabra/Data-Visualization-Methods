# GPLVM.py
import torch
import gpytorch
from utils import get_mnist_dataloader, ProjectedGPLVM, plot_gplvm

def train_gplvm(train_loader, latent_dim=2, data_dim=784, num_epochs=10, learning_rate=0.1):
    # Load one batch for simplicity (for full training, use all batches)
    images, labels = next(iter(train_loader))
    images = images.view(images.size(0), -1)  # Flatten images to vectors
    
    # Initialize random latent variables (latent space)
    latent_x = torch.randn(images.size(0), latent_dim, requires_grad=True)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ProjectedGPLVM(latent_x, images, likelihood, latent_dim, data_dim)
    
    # Set up optimizer for latent variables and model parameters
    optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': latent_x}], lr=learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    # Training loop
    model.train()
    likelihood.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Generate model output
        output = model(latent_x)
        
        # Calculate loss and ensure it is a scalar by summing over batch
        loss = -mll(output, images).sum()  # Use .sum() or .mean() to get a scalar
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    return latent_x.detach(), labels

if __name__ == "__main__":
    # Parameters
    batch_size = 256
    latent_dim = 2
    data_dim = 784  # Match MNIST image size
    num_epochs = 10
    learning_rate = 0.1

    # Load Data
    train_loader = get_mnist_dataloader(batch_size=batch_size)

    # Train GPLVM and visualize results
    latent_x, labels = train_gplvm(train_loader, latent_dim=latent_dim, data_dim=data_dim, num_epochs=num_epochs, learning_rate=learning_rate)
    plot_gplvm(latent_x, labels)