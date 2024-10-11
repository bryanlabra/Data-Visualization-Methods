# Autoencoder.py  #source ./.venv/bin/activate
import torch
from torch import nn, optim
from torchvision import transforms
from utils import get_mnist_dataloader, ConvAutoencoder, plot_autoencoder

# Initialize weights with Xavier initialization
def initialize_weights(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                layer.bias.data.fill_(0.0)

def train_autoencoder(model, dataloader, num_epochs=31, initial_lr=0.001):
    criterion = nn.BCELoss()  # Use Binary Cross-Entropy Loss for reconstruction
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    for epoch in range(num_epochs):
        total_loss = 0
        model.train()  # Set model to training mode
        for images, _ in dataloader:
            images = images.to(next(model.parameters()).device)
            
            # Forward pass
            latent, reconstructed = model(images)
            loss = criterion(reconstructed, images)  # Calculate reconstruction loss

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        # Update learning rate based on plateau in loss
        scheduler.step(total_loss / len(dataloader))
        
        # Print epoch loss and current learning rate
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

def get_latent_space(model, dataloader):
    model.eval()
    images, labels = next(iter(dataloader))
    images = images.to(next(model.parameters()).device)
    with torch.no_grad():
        latent, _ = model(images)  # Only use the encoder part
    return latent.cpu().numpy(), labels.numpy()

if __name__ == "__main__":
    # Parameters
    batch_size = 256
    latent_dim = 4  # Increased latent dimension for richer representation
    num_epochs = 30
    initial_lr = 0.001

    # Load Data
    train_loader = get_mnist_dataloader(batch_size=batch_size)

    # Initialize the Convolutional Autoencoder model
    model = ConvAutoencoder(latent_dim=latent_dim)
    initialize_weights(model)  # Apply Xavier initialization
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Train the Convolutional Autoencoder with AdamW and scheduler
    train_autoencoder(model, train_loader, num_epochs=num_epochs, initial_lr=initial_lr)

    # Get the higher-dimensional latent representation and reduce it to 2D
    latent_space, labels = get_latent_space(model, train_loader)

    # Optionally use PCA or t-SNE for 2D visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    latent_space_2d = pca.fit_transform(latent_space)

    # Plot the 2D latent space
    plot_autoencoder(latent_space_2d, labels)