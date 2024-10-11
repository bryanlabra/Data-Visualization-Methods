# Autoencoder.py
import torch
from torch import nn, optim
from utils import get_mnist_dataloader, Autoencoder, plot_autoencoder

def train_autoencoder(model, dataloader, num_epochs=20, learning_rate=0.001):
    criterion = nn.MSELoss()  # Mean Squared Error for reconstruction loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for images, _ in dataloader:
            images = images.view(images.size(0), -1)  # Flatten images to vectors

            # Forward pass
            latent, reconstructed = model(images)
            loss = criterion(reconstructed, images)  # Calculate reconstruction loss

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")

def get_latent_space(model, dataloader):
    model.eval()
    images, labels = next(iter(dataloader))
    images = images.view(images.size(0), -1)  # Flatten images to vectors
    with torch.no_grad():
        latent, _ = model(images)  # Only use the encoder part
    return latent.numpy(), labels.numpy()

if __name__ == "__main__":
    # Parameters
    batch_size = 256
    input_dim = 784  # MNIST images are 28x28
    hidden_dim = 128
    latent_dim = 2
    num_epochs = 20
    learning_rate = 0.001

    # Load Data
    train_loader = get_mnist_dataloader(batch_size=batch_size)

    # Initialize the Autoencoder model
    model = Autoencoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)

    # Train the Autoencoder
    train_autoencoder(model, train_loader, num_epochs=num_epochs, learning_rate=learning_rate)

    # Get the 2D latent representation
    latent_space, labels = get_latent_space(model, train_loader)

    # Plot the latent space
    plot_autoencoder(latent_space, labels)