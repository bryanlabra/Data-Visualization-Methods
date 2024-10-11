# Benchmarking-Visualization-Methods
python version 3.9.6


### Visualization Methods

Probabilistic Principal Component Analysis (PPCA)

Gaussian Process Latent Variable Model (GPLVM)

t-distributed Stochastic Neighbor Embedding (t-SNE)

Uniform Manifold Approximation (UMAP)

## Visualizating the MNIST Dataset

### 1. PPCA

what parameters can be adjusted for better results?

### 2. GPLVM

what parameters can be adjusted for better performance?

### 3. t-SNE
what parameters can be adjusted for improved outcomes?

*Perplexity*: Try values between 5 and 50 to see if it improves separation. Higher perplexity considers more global relationships, which may reduce overlap.

*Learning Rate*: Adjusting the learning rate (e.g., between 50 and 500) might change how clusters are positioned.

### 4. UMAP

what parameters can be adjusted for improved outcomes?

*n_neighbors*: Try increasing or decreasing this to change the emphasis on local versus global structure.

*min_dist*: A smaller min_dist can create tighter clusters, while a larger value can spread points further apart.

### 5. Isomap

*Increase n_neighbors*: This may help in clustering more global relationships.
*Experiment with Data Subsets*: Sometimes, Isomap can perform better on smaller, more focused subsets of data.

### 6. LLE

### 7. Autoencoders

**Model Adjustments**:

*Increase Hidden Layers*: Adding more hidden layers with larger dimensions in the encoder and decoder might help the model capture more complex features.

*Add Convolutional Layers*: For image data like MNIST, convolutional autoencoders (using convolutional layers instead of fully connected layers) can capture spatial hierarchies and may perform better.

*Regularization Techniques*: Adding dropout or batch normalization layers can help improve generalization and potentially create better-separated clusters.

**Training Adjustments**:

*Increase Training Epochs*: Allowing the autoencoder to train for more epochs might improve the reconstruction quality and latent representation.

*Tune Learning Rate*: A lower learning rate could help the model converge to a better solution, albeit more slowly.

## Visualizating the PBMC dataset

### 1. Data Loading: 
Loads the PBMC 3K dataset using Scanpy’s built-in dataset loader.

### 2.	Preprocessing: 
Applies sc.pp.recipe_zheng17 to normalize, log-transform, and scale the data, preparing it for analysis.

### 3.	Dimensionality Reduction:
PCA: Computes principal components to reduce the dataset’s dimensionality.

Neighbors: Constructs a neighborhood graph, a prerequisite for clustering and UMAP/t-SNE.

### 4.	Clustering: 
Uses the Louvain algorithm to group cells into clusters, adding a louvain column to the data object to label each cell’s cluster.

### 5.	Additional Embeddings:
UMAP and t-SNE: Calculates UMAP and t-SNE embeddings for visualizing clusters in 2D.

### 6.	Visualization:
Plots PCA, UMAP, and t-SNE, coloring cells by Louvain clusters to visualize how cells are grouped in each projection.

