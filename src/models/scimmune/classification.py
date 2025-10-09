# %% [markdown]
# ## Utils

# %%
def prepend_zeros(example, num_metadata_tokens=6):
    # Convert tensor to list for manipulation
    import torch

    expressions = example["expressions"].tolist()
    zeros = [0.0] * num_metadata_tokens
    expressions = zeros + expressions

    # Convert back to tensor if you're using HF with torch format
    example["expressions"] = torch.tensor(expressions, dtype=torch.float)
    return example

def move_cls_to_front(example, cls_token_id=60695):
    genes = example["genes"].tolist()
    
    import torch

    if cls_token_id in genes:
        genes.remove(cls_token_id)
        genes = [cls_token_id] + genes

    example["genes"] = torch.tensor(genes, dtype=torch.long)
    return example

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
    save_path: str = None
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
    import scanpy as sc
    adata = sc.read_h5ad(data_path)
    adata = embed_data(
        adata_or_file=adata,
        model_dir=model_dir,
        gene_col=gene_col,
        batch_size=batch_size,
        return_new_adata=return_new_adata,
    )
    if save_path:
        adata.write(save_path)
        print(f"Embedded data saved to {save_path}")
    return adata

def compute_umap_and_leiden(adata, use_rep="custom", umap_key="custom", leiden_key="custom", metric="euclidean", n_neighbors=15, min_dist=0.3, random_state=0, resolution=1.0):
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
    sc.pp.neighbors(adata, use_rep=f"X_{use_rep}", n_neighbors=n_neighbors, metric=metric, random_state=0)

    # Compute UMAP
    sc.tl.umap(adata, min_dist=min_dist, random_state=random_state)

    # Save UMAP to a custom key to avoid overwriting
    adata.obsm[f"X_umap_{umap_key}"] = adata.obsm["X_umap"]

    # Run Leiden clustering
    sc.tl.leiden(
        adata,
        key_added=f"leiden_{leiden_key}",
        resolution=resolution,
    )

    return adata

def compute_clustering_metrics(adata, embedding_key='X_umap', cluster_key='leiden', label_key='cell_type'):
    """
    Compute Silhouette Score, ARI, and NMI for a given AnnData object.
    
    Parameters:
        adata (AnnData): Annotated data object with embedding and clustering.
        embedding_key (str): Key in adata.obsm for the 2D embedding (e.g., UMAP).
        cluster_key (str): Key in adata.obs for clustering labels (e.g., leiden).
        label_key (str): Key in adata.obs for ground truth labels (e.g., cell type).
        
    Returns:
        dict: Dictionary with silhouette, ARI, and NMI scores.
    """
    from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

    embedding = adata.obsm[f"X_{embedding_key}"]
    cluster_labels = adata.obs[f"leiden_{cluster_key}"].values
    true_labels = adata.obs[label_key].values

    # Silhouette Score: only makes sense if more than 1 cluster
    if len(set(cluster_labels)) > 1:
        sil_score = silhouette_score(embedding, cluster_labels)
    else:
        sil_score = float('nan')

    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)

    return {
        'silhouette': sil_score,
        'ARI': ari,
        'NMI': nmi
    }

# %% [markdown]
# ## scImmune - Embedding Generation

# %%
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import DataLoader
import tqdm

# --- your modules ---
from tokenizer import ScImmuneTokenizer
from config import ScImmuneConfig
from model import ScImmuneModel
from collator import ScImmuneDataCollator

# =======================
# User settings
# =======================
H5AD_PATH = "../data/inference/influenzaPBMC.h5ad"
MODEL_DIR = "final"     # folder with model.safetensors + config.json
VOCAB_FILE = "vocab_with_metadata.json"
NUM_META = 6
SCALE_FACTOR = 100.0                        # must match training
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Map obs fields -> token tag used in your vocab
# (Adjust the right-hand side strings so they exactly match your vocab’s token prefixes)

FIELD_TO_TAG = {
    "cell_type_ontology_term_id":           "cell_type",
    "self_reported_ethnicity_ontology_term_id": "self_reported_ethnicity",
    "tissue_general_ontology_term_id":      "tissue_general",
    "development_stage_ontology_term_id":   "development_stage",
    "sex_ontology_term_id":                 "sex",
    "disease_ontology_term_id":             "disease",
}
assert len(FIELD_TO_TAG) == NUM_META, "NUM_META must match the number of metadata fields."

# =======================
# Load data & model
# =======================
adata = sc.read_h5ad(H5AD_PATH)

for colname in list(adata.obs.columns):
    if pd.api.types.is_categorical_dtype(adata.obs[colname]):
        adata.obs[colname] = adata.obs[colname].cat.remove_unused_categories()

# %%
adata.obs["disease_ontology_term_id"] = adata.obs["disease_ontology_term_id"].astype(str)
adata.obs.loc[adata.obs["disease"] == "normal", "disease_ontology_term_id"] = "DOID:4"

# %%
X = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else np.asarray(adata.X)
gene_symbols = np.asarray(adata.var.feature_name)

gene_symbols

# %%
tok = ScImmuneTokenizer(vocab_file=VOCAB_FILE)
cfg = ScImmuneConfig.from_pretrained(MODEL_DIR)
model = ScImmuneModel.from_pretrained(MODEL_DIR, config=cfg).to(DEVICE).eval()

cls_id = tok.convert_tokens_to_ids(tok.cls_token)
unk_token = tok.unk_token
unk_id = tok.convert_tokens_to_ids(unk_token)

# %%
import tqdm
metadata_fields = [
                    "cell_type_ontology_term_id",
                    "self_reported_ethnicity_ontology_term_id", 
                    "tissue_general_ontology_term_id",
                    "development_stage_ontology_term_id",
                    "sex_ontology_term_id",
                    "disease_ontology_term_id"
]

gene_names = adata.var.feature_name.values # get gene names

tokenized_input_ids = []
tokenized_values = []

for i in range(adata.n_obs):
    
    # 1. Get dense expression vector
    row = adata.X[i]
    if not isinstance(row, np.ndarray):
        row = row.toarray().squeeze()

    # 2. Get metadata tokens
    obs_row = adata.obs.iloc[i]
    metadata_tokens = []
    for field in metadata_fields:
        val = obs_row.get(field)
        if isinstance(val, str) and val != "NA" and "=" not in val:
            if field == "tissue_ontology_term_id":
                token = f"<{field.split('_ontology_term_id')[0]}_general={val}>"
            else:
                token = f"<{field.split('_ontology_term_id')[0]}={val}>"
            metadata_tokens.append(token)

    # 3. Tokenize
    tokenized = tok.tokenize_cell_batch(
        data=np.expand_dims(row, axis=0),
        gene_ids=gene_names,
        metadata_tokens=metadata_tokens,
        append_cls=True,
        include_zero_gene=False
    )
    
    input_ids, values = tokenized[0]
    tokenized_input_ids.append(input_ids)
    tokenized_values.append(values)

# %%
from datasets import Dataset

# Padding and truncation happen in DataCollator, not here
hf_dataset = Dataset.from_dict({
    "genes": tokenized_input_ids,
    "expressions": tokenized_values
})

hf_dataset.set_format(type="torch", columns=["genes", "expressions"])

# %%
hf_dataset = hf_dataset.map(prepend_zeros)
hf_dataset = hf_dataset.map(move_cls_to_front)

# %%
hf_dataset["genes"][0][:10]

# %%
# import torch

# hf_dataset = hf_dataset.map(
#     lambda example: {
#         'genes': (
#             lambda g: g[:1] + [60697]*6 + g[7:] # replace all metadata tokens with <unk> value
#         )(example['genes'].tolist())
#     }
# )

# %%
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
adata.obs["cell_type_label"] = le.fit_transform(adata.obs["cell_type_ontology_term_id"])

labels = torch.tensor(adata.obs["cell_type_label"].values, dtype=torch.long)
hf_dataset = hf_dataset.add_column("label", labels.tolist())

# %%
import torch.nn as nn

class ScImmuneForClassification(ScImmuneModel):
    def __init__(self, config, num_classes):
        super().__init__(config)
        self.classifier = nn.Linear(config.hidden_size, num_classes)

    def forward(
        self, input_ids=None, values=None, attention_mask=None,
        labels=None, generative_training=False
    ):
        output = super().forward(
            input_ids=input_ids,
            values=values,
            attention_mask=attention_mask,
            generative_training=generative_training,
        )

        logits = self.classifier(output["cell_emb"])
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"logits": logits, "loss": loss, "cell_emb": output["cell_emb"]}

# %%
import torch.nn.functional as F

# =======================
# Collator for PCPT inference
# =======================
# We already scaled values; we just need padding + attention_mask.
collator = ScImmuneDataCollator(
    do_padding=True,
    pad_token_id=tok.pad_token_id,
    pad_value=-2,
    do_mlm=False,
    do_binning=False,
    mlm_probability=0.0,
    max_length=getattr(cfg, "max_seq_len", 2000),
    sampling=False,
    keep_first_n_tokens= 1 + NUM_META,
    data_style="pcpt",
    scale_factor=100.0,
)
# # If your collator tries to scale on pcpt, disable:
# setattr(collator, "scale_factor")

dl = DataLoader(hf_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

# =======================
# Embed
# =======================
embs = []
with torch.no_grad():
    for batch in dl:
        # collator._call_pcpt likely returns {"gene","expr"} in your codebase.
        # If you already patched it to return ("input_ids","values","attention_mask"), adapt here.
        input_ids = batch.get("input_ids", batch.get("gene")).to(DEVICE)
        values    = batch.get("values",    batch.get("expr")).to(DEVICE)

        # HF attention mask: 1=keep, 0=pad
        attention_mask = (input_ids != tok.pad_token_id).long().to(DEVICE)

        out = model(
            input_ids=input_ids,
            values=values,
            attention_mask=attention_mask,
            generative_training=False,  # force perceptual path
        )
        embs.append(out["cell_emb"].detach().cpu())

embs = torch.cat(embs, dim=0)  # shape: (n_cells, d)

# # L2-normalize per cell
embs = F.normalize(embs, p=2, dim=1)  # keeps magnitude=1

adata.obsm["X_scimmune"] = embs.numpy()

# %%
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, TensorDataset, random_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score, f1_score

# # =======================================
# # Step 1: Prepare Data
# # =======================================
# X = adata.obsm["X_scimmune"]
# y = adata.obs["cell_type_ontology_term_id"]  # or "cell_type_ontology_term_id"

# # Encode labels
# y_enc = LabelEncoder().fit_transform(y)

# # Torch tensors
# X_tensor = torch.tensor(X, dtype=torch.float32)
# y_tensor = torch.tensor(y_enc, dtype=torch.long)

# # Create dataset and split
# full_ds = TensorDataset(X_tensor, y_tensor)
# train_size = int(0.8 * len(full_ds))
# val_size = len(full_ds) - train_size
# train_ds, val_ds = random_split(full_ds, [train_size, val_size])

# train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
# val_loader = DataLoader(val_ds, batch_size=128)

# # =======================================
# # Step 2: MLP Model
# # =======================================
# class MLPClassifier(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_classes):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_dim, num_classes)
#         )

#     def forward(self, x):
#         return self.net(x)

# # =======================================
# # Step 3: Train
# # =======================================
# input_dim = X.shape[1]
# hidden_dim = 128
# num_classes = len(set(y_enc))
# device = "cuda" if torch.cuda.is_available() else "cpu"

# mlp_probe = MLPClassifier(input_dim, hidden_dim, num_classes).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# criterion = nn.CrossEntropyLoss()

# for epoch in range(30):
#     model.train()
#     for xb, yb in train_loader:
#         xb, yb = xb.to(device), yb.to(device)
#         logits = model(xb)
#         loss = criterion(logits, yb)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     # Eval
#     model.eval()
#     y_true, y_pred = [], []
#     with torch.no_grad():
#         for xb, yb in val_loader:
#             xb = xb.to(device)
#             logits = model(xb)
#             preds = torch.argmax(logits, dim=1).cpu()
#             y_true.extend(yb.numpy())
#             y_pred.extend(preds.numpy())

#     acc = accuracy_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred, average="macro")
#     print(f"Epoch {epoch:02d} | Accuracy: {acc:.3f} | Macro F1: {f1:.3f}")

# %%
compute_pca_embeddings(adata)

# %%
compute_scvi_embeddings(adata, batch_key=None)

# %%
adata.obsm["X_scgpt"] = adata.obsm["X_scGPT"]

# %%
# compute_umap_and_leiden(adata, use_rep="pca", umap_key="pca", leiden_key="pca")
# compute_umap_and_leiden(adata, use_rep="scvi", umap_key="scvi", leiden_key="scvi")
# compute_umap_and_leiden(adata, use_rep="scgpt", umap_key="scgpt", leiden_key="scgpt")
compute_umap_and_leiden(adata, use_rep="scimmune", umap_key="scimmune", leiden_key="scimmune")

# %%
methods = ["pca","scvi","scgpt", "scimmune"]

for id in methods:
    print(id, compute_clustering_metrics(adata, embedding_key=id, cluster_key=id, label_key='cell_type'))

# %%
adata.obs.cell_type.unique()

# %%
import numpy as np

# Compute counts per cell type
cell_type_counts = adata.obs["cell_type"].value_counts()

# Keep only those with count > 1000
valid_cell_types = cell_type_counts[cell_type_counts > 1000].index

# Filter AnnData
adata = adata[adata.obs["cell_type"].isin(valid_cell_types)].copy()

print(f"Remaining cell types: {len(valid_cell_types)}")
print(adata.obs["cell_type"].value_counts())

# %%
adata.obsm.

# %%
X = np.asarray(adata.obsm[], dtype=np.float32)

# %%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ==== config ====
LABEL_COL = "cell_type"        # change if needed
REP_KEY   = "X_scgpt"       # e.g., "X_scimmune", "X_scgpt", "X_scvi", "X_pca"
TEST_SIZE = 0.20
RANDOM_SEED = 0

# ==== 1) pull embeddings + labels ====
X = np.asarray(adata.obsm[REP_KEY], dtype=np.float32)
labels = adata.obs[LABEL_COL].astype("category")

# drop cells with missing labels (just in case)
keep = ~labels.isna().values
X = X[keep]
labels = labels[keep]

y = labels.cat.codes.values
class_names = list(labels.cat.categories)

# sanity
assert X.shape[0] == y.shape[0] and X.ndim == 2, f"X shape {X.shape}, y shape {y.shape}"

# ==== 2) split (stratified) ====
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)

# ==== 3) build a simple pipeline: Standardize -> Logistic Regression ====
# Standardizing helps the linear classifier (even if X is L2‑normalized already).
clf = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("logreg", LogisticRegression(
        penalty="l2",
        C=4.0,                 # tune if under/over-regularized
        max_iter=2000,
        n_jobs=-1,
        class_weight="balanced",  # good default if label distribution is skewed
        solver="lbfgs",
        multi_class="multinomial",  # proper softmax for multi-class
        verbose=0,
    )),
])

# ==== 4) train ====
clf.fit(Xtr, ytr)

# ==== 5) evaluate ====
yp = clf.predict(Xte)

print("== Linear probe (Logistic Regression) on", REP_KEY, "==")
print("Accuracy:", accuracy_score(yte, yp))
print("Macro‑F1:", f1_score(yte, yp, average="macro"))
print("\nClassification report:")
print(classification_report(yte, yp, target_names=class_names, digits=3))

cm = confusion_matrix(yte, yp)
print("\nConfusion matrix shape:", cm.shape)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Compute confusion matrix
cm = confusion_matrix(yte, yp, normalize='true')

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("scImmune Disease Deconvolution - CM")
plt.show()

# %%
sc.pl.embedding(adata, basis="umap_scgpt", color="cell_type")

# %%
import umap
from sklearn.preprocessing import normalize
import scanpy as sc
import pandas as pd

# Add predictions to AnnData
adata.obs["pred_cell_type"] = pd.Categorical([class_names[i] for i in yp])

# Compute UMAP from embeddings (normalize for cosine distance if needed)
Xn = normalize(X_scimmune, norm="l2", axis=1)
um = umap.UMAP(n_neighbors=20, min_dist=0.3, metric="cosine", random_state=0)
adata.obsm["X_umap_scimmune"] = um.fit_transform(Xn)

# Plot true labels
sc.pl.embedding(
    adata,
    basis="X_umap_scimmune",
    color=LABEL_COL,
    title="True Cell Types",
    frameon=False,
    legend_loc='on data'
)

# Plot predicted labels
sc.pl.embedding(
    adata,
    basis="X_umap_scimmune",
    color="pred_cell_type",
    title="Predicted Cell Types",
    frameon=False,
    legend_loc='on data'
)


