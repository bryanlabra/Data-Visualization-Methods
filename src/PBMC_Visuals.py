import scanpy as sc

# Load the dataset
alldata = sc.datasets.pbmc3k()

# Preprocessing
sc.pp.recipe_zheng17(alldata)  # Normalization and log transformation

# Dimensionality reduction
sc.tl.pca(alldata)  # PCA computation
sc.pp.neighbors(alldata, n_neighbors=10, n_pcs=40)  # Compute the neighborhood graph

# Run Louvain clustering
sc.tl.louvain(alldata)  # This will add 'louvain' to alldata.obs

# Compute UMAP and t-SNE for visualization
sc.tl.umap(alldata)
sc.tl.tsne(alldata)

# Plot PCA with Louvain clusters
sc.pl.pca(alldata, color='louvain')

# Plot UMAP with Louvain clusters
sc.pl.umap(alldata, color='louvain')

# Plot t-SNE with Louvain clusters
sc.pl.tsne(alldata, color='louvain')

#Force Atlas 2
sc.tl.draw_graph(alldata, layout='fa')  # ForceAtlas2 layout
sc.pl.draw_graph(alldata, color='louvain')

#Diffusion Map 
sc.tl.diffmap(alldata)
sc.pl.diffmap(alldata, color='louvain')

