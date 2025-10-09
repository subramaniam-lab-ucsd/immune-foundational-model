def compute_scgpt_embeddings(
    data_path: str,
    model_dir: str,
    gene_col: str = "feature_name",
    batch_size: int = 128,
    return_new_adata: bool = False,
    save_path: str = None
):
    """
    Computes scGPT embeddings for a given h5ad file.

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

file = "covid_cells_chunk0"

adata = compute_scgpt_embeddings(
    data_path=f"../data/inference/{file}.h5ad",
    model_dir="../models/zero-shot/scgpt_human",
    save_path=f"../data/inference/{file}.h5ad"
)