import scanpy as sc
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import seaborn as sns
import scvi


import warnings, math
from typing import Dict, Optional, Iterable
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, silhouette_score, accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_distances
import scanpy as sc



def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compute_pca_embeddings(
    adata,
    n_components=50,
    latent_key="X_pca",
    normalize=True,
    log1p=True,
    scale=True,
    zero_center=True,
    svd_solver=None,
    random_state=0,
):
    """
    Compute PCA embeddings on AnnData and store them in adata.obsm[latent_key].

    Parameters:
    - adata: AnnData object (must contain data in adata.X or specified layer).
    - layer: Layer name to use for PCA (default: None → adata.X).
    - n_components: Number of principal components to compute.
    - latent_key: Key under adata.obsm to store PCA output.
    - normalize: If True, normalize per cell (scanpy.pp.normalize_total).
    - log1p: If True, apply log1p transformation.
    - scale: If True, scale data (zero-mean, unit variance).
    - zero_center: Passed to scanpy.pp.pca; if True, center variables (regular PCA), else applies SVD.
    - svd_solver: Optional solver choice for PCA (default: None → auto).
    - random_state: Random seed for reproducibility.

    Returns:
    - Updated AnnData object with PCA embeddings stored in adata.obsm[latent_key].
    """
    import scanpy as sc

    # Work on a copy to avoid modifying adata.X in place
    adata_copy = adata.copy()

    # Normalize, log-transform, and scale as requested
    if normalize:
        sc.pp.normalize_total(adata_copy)
    if log1p:
        sc.pp.log1p(adata_copy)
    if scale:
        sc.pp.scale(adata_copy)

    # Compute PCA using the specified data source (adata.X or the layer)
    sc.pp.pca(
        adata_copy,
        n_comps=n_components,
        zero_center=zero_center,
        svd_solver=svd_solver,
        random_state=random_state,
    )

    # Store the result in the original AnnData object's obsm
    adata.obsm[latent_key] = adata_copy.obsm["X_pca"][:, :n_components]

    return adata

def compute_scvi_embeddings(
    adata,
    batch_key="dataset_id",
    n_layers=2,
    n_latent=50,
    gene_likelihood="nb"
):
    """
    Compute and store scVI latent embeddings in adata.obsm["X_scvi"],
    assuming raw counts are in adata.X.

    Parameters:
    - adata: AnnData object with raw count data in adata.X.
    - batch_key: Key in adata.obs indicating batch labels (default: "dataset_id").
    - n_layers: Number of hidden layers in encoder/decoder.
    - n_latent: Dimensionality of the latent space.
    - gene_likelihood: Likelihood model to use ("nb", "zinb", etc.).

    Returns:
    - Updated AnnData object with latent embeddings in .obsm["X_scvi"].
    """
    import scvi

    scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key)
    model = scvi.model.SCVI(
        adata,
        n_layers=n_layers,
        n_latent=n_latent,
        gene_likelihood=gene_likelihood
    )
    model.train()
    adata.obsm["X_scvi"] = model.get_latent_representation()
    return adata

def compute_scgpt_embeddings(
    data_path: str,
    model_dir: str,
    gene_col: str = "feature_name",
    batch_size: int = 128,
    return_new_adata: bool = False,
):
    """
    Computes SCGPT embeddings for a given h5ad file.

    Parameters:
        data_path (str): Path to the input .h5ad file.
        model_dir (str): Path to the pretrained SCGPT model directory.
        gene_col (str): Column name for gene features.
        batch_size (int): Batch size to use for embedding.
        return_new_adata (bool): Whether to return a new AnnData object.

    Returns:
        AnnData: The embedded AnnData object (modified in-place if return_new_adata=False).
    """

    from scgpt.tasks import embed_data
    
    adata = sc.read_h5ad(data_path)
    adata = embed_data(
        adata_or_file=adata,
        model_dir=model_dir,
        gene_col=gene_col,
        batch_size=batch_size,
        return_new_adata=return_new_adata,
    )
    return adata

def compute_umap(adata, use_rep="custom", umap_key="custom", n_neighbors=15, min_dist=0.3, random_state=0):
    """
    Compute neighbors and UMAP for a given embedding and store in adata.obsm[umap_key].

    Parameters:
    - adata: AnnData object with embeddings in adata.obsm[use_rep].
    - use_rep: Key in adata.obsm to use as input for computing neighbors.
    - umap_key: Key in adata.obsm to store UMAP coordinates.
    - n_neighbors: Number of neighbors for the KNN graph (default: 15).
    - min_dist: Controls how tightly UMAP packs points together (default: 0.3).
    - random_state: Random seed for reproducibility.

    Returns:
    - Updated AnnData object with UMAP stored in adata.obsm[umap_key].
    """
    import scanpy as sc

    # Compute neighbors from the specified representation
    sc.pp.neighbors(adata, use_rep=f"X_{use_rep}", n_neighbors=n_neighbors)

    # Compute UMAP
    sc.tl.umap(adata, min_dist=min_dist, random_state=random_state)

    # Save UMAP to a custom key to avoid overwriting
    adata.obsm[f"X_umap_{umap_key}"] = adata.obsm["X_umap"]

    return adata

def compute_leiden(adata, use_rep="custom", leiden_key="custom", n_neighbors=15, resolution=1.0):
    """
    Compute neighbors and Leiden clustering for a given embedding and store in adata.obs["leiden_{leiden_key}"].

    Parameters:
    - adata: AnnData object with embeddings in adata.obsm[f"X_{use_rep}"].
    - use_rep: Key in adata.obsm to use for neighbors (without "X_" prefix).
    - leiden_key: Key suffix for adata.obs (e.g., "scvi" → "leiden_scvi").
    - n_neighbors: Number of neighbors for KNN graph (default: 15).
    - resolution: Resolution parameter for Leiden clustering (higher → more clusters).
    - random_state: Random seed for reproducibility.

    Returns:
    - Updated AnnData object with cluster labels in adata.obs[f"leiden_{leiden_key}"].
    """
    import scanpy as sc

    # Compute neighbors from specified embedding
    sc.pp.neighbors(adata, use_rep=f"X_{use_rep}", n_neighbors=n_neighbors)

    # Run Leiden clustering
    sc.tl.leiden(
        adata,
        key_added=f"leiden_{leiden_key}",
        resolution=resolution,
    )

    return adata

def compute_clustering_metrics(embedding, labels, clusters):
    """
    Compute silhouette score, ARI, and NMI for given embedding, labels, and clusters.
    """
    silhouette = silhouette_score(embedding, labels)
    ari = adjusted_rand_score(labels, clusters)
    nmi = normalized_mutual_info_score(labels, clusters)
    return silhouette, ari, nmi

def print_clustering_metrics(name, silhouette, ari, nmi):
    print(f"{name}: Silhouette={silhouette:.3f}, ARI={ari:.3f}, NMI={nmi:.3f}")

