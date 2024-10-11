# UMAP.py
import torch
from utils import get_mnist_dataloader, plot_umap

def prepare_data_for_umap(loader):
    images, labels = next(iter(loader))
    images = images.view(images.size(0), -1)  # Flatten images to vectors
    return images.numpy(), labels.numpy()

if __name__ == "__main__":
    # Parameters
    batch_size = 256

    # Load Data
    train_loader = get_mnist_dataloader(batch_size=batch_size)

    # Prepare data for UMAP
    images, labels = prepare_data_for_umap(train_loader)

    # Plot UMAP
    plot_umap(images, labels)