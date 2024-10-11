# PPCA.py
import torch
import torch.optim as optim
from utils import get_mnist_dataloader, PPCA, plot_ppca

def train_ppca(train_loader, input_dim=784, latent_dim=2, num_epochs=10, learning_rate=0.01):
    # Initialize PPCA Model
    ppca = PPCA(input_dim, latent_dim)
    optimizer = optim.Adam(ppca.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for images, _ in train_loader:
            images = images.view(-1, input_dim)
            optimizer.zero_grad()
            
            # Compute negative log-likelihood as loss
            cov = ppca(images)
            log_likelihood = -0.5 * torch.logdet(cov) - 0.5 * torch.einsum("bi,ij,bj->b", images, torch.inverse(cov), images).mean()
            loss = -log_likelihood
            
            # Backpropagate and optimize
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    return ppca

if __name__ == "__main__":
    # Parameters
    batch_size = 256
    input_dim = 28 * 28  # MNIST images are 28x28 pixels
    latent_dim = 2
    num_epochs = 10
    learning_rate = 0.01

    # Load Data
    train_loader = get_mnist_dataloader(batch_size=batch_size)

    # Train and visualize PPCA
    ppca = train_ppca(train_loader, input_dim=input_dim, latent_dim=latent_dim, num_epochs=num_epochs, learning_rate=learning_rate)
    
    # Get projections for visualization
    images, labels = next(iter(train_loader))
    images = images.view(-1, input_dim)
    centered_data = images - ppca.mean
    projections = centered_data @ ppca.W
    plot_ppca(projections, labels)
