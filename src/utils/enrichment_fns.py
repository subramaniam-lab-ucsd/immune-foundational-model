import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix
import seaborn as sns
import scvi
from scvi.model import SCVI

import gseapy as gp


def get_top_marker_genes(
    adata, 
    groupby, 
    logfc_threshold=1.0, 
    pval_threshold=0.05, 
    top_n=5, 
    method="wilcoxon"
):
    """
    Compute and return top marker genes per cluster for an AnnData object.

    Parameters
    ----------
    adata : AnnData
        The AnnData object.
    groupby : str
        The column in adata.obs to group by (e.g., cluster labels).
    logfc_threshold : float
        Minimum log fold change for marker selection.
    pval_threshold : float
        Maximum adjusted p-value for marker selection.
    top_n : int
        Number of top genes per cluster to return.
    method : str
        Method for differential expression (default: "wilcoxon").

    Returns
    -------
    marker_genes : list
        List of unique top marker gene names.
    top_genes_per_cluster : pd.DataFrame
        DataFrame of top marker genes per cluster.
    """
    if "log1p" not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    
    sc.tl.rank_genes_groups(adata, groupby=groupby, method=method)

    ranked_df = sc.get.rank_genes_groups_df(adata, None)
    
    filtered_df = ranked_df[
        (ranked_df["logfoldchanges"] > logfc_threshold) &
        (ranked_df["pvals_adj"] < pval_threshold)
    ]
    top_genes_per_cluster = (
        filtered_df
        .groupby("group")
        .apply(lambda g: g.sort_values("scores", ascending=False).head(top_n))
        .reset_index(drop=True)
    )
    marker_genes = top_genes_per_cluster["names"].unique().tolist()
    return marker_genes, top_genes_per_cluster, ranked_df

def run_gsea_fast_serial(
    ranked_df,
    libraries=None,
    ranking_metric="logfoldchanges",
    
):
    if libraries is None:
        libraries = ["COVID-19_Related_Gene_Sets_2021", "Reactome_2022"]

    # Step 1: Gene symbol mapping
    mg = MyGeneInfo()
    gene_ids = ranked_df["names"].unique().tolist()
    query_result = mg.querymany(
        gene_ids, scopes="symbol", fields="symbol", species="human", as_dataframe=True
    )
    query_result = query_result[~query_result.index.duplicated(keep="first")]
    mapped = query_result["symbol"].dropna().to_dict()

    ranked_df = ranked_df.copy()
    ranked_df["mapped_name"] = ranked_df["names"].map(mapped)
    ranked_df = ranked_df.dropna(subset=["mapped_name"])
    ranked_df["mapped_name"] = ranked_df["mapped_name"].str.upper()

    # Step 2: GSEA loop
    results = {}
    for cluster in ranked_df["group"].unique():
        cluster_df = ranked_df[ranked_df["group"] == cluster]
        rnk = cluster_df[["mapped_name", ranking_metric]]
        rnk.columns = ["gene_name", "score"]

        for lib in libraries:
            try:
                result = prerank(
                    rnk=rnk,
                    gene_sets=lib,
                    outdir=None,      # no file writing
                    no_plot=True,     # no plotting
                    format="png",
                    min_size=3,
                    max_size=1000,
                    permutation_num=1000,
                    seed=42,
                    verbose=False,
                )
                results[(cluster, lib)] = result.res2d
            except Exception as e:
                print(f"Failed GSEA for cluster {cluster} on {lib}: {e}")

    return results

def gsea_results_to_df(results_dict, method_name):
    rows = []
    for (cluster, library), df in results_dict.items():
        required_cols = {"NES", "FDR q-val", "Term"}
        if not required_cols.issubset(df.columns):
            print(f"⚠️ Skipping {cluster} - {library}, missing columns: {required_cols - set(df.columns)}")
            continue

        for _, row in df.iterrows():
            rows.append({
                "method": method_name,
                "cluster": cluster,
                "library": library,
                "pathway": row["Term"],
                "NES": row["NES"],
                "FDR_qval": row["FDR q-val"],
            })
    return pd.DataFrame(rows)

def run_ora_and_dotplot(
    gene_list,
    gene_sets="MSigDB_Hallmark_2020",
    organism="Human",
    title_prefix="",
    outdir=None,
    cutoff=0.1,
    top_n=10
):
    """
    Run ORA (Over-Representation Analysis) using gseapy.enrichr and plot dotplots for up and down genes.

    Parameters
    ----------
    gene_list : list
        List of gene symbols (e.g., marker_genes).
    gene_sets : str or list
        Gene set library name(s) (default: MSigDB_Hallmark_2020).
    organism : str
        Organism name for enrichr (default: "Human").
    title_prefix : str
        Prefix for plot titles.
    outdir : str or None
        Directory to save plots (optional).
    cutoff : float
        Adjusted p-value cutoff for significance.
    top_n : int
        Number of top terms to plot.
    """
    enr = gp.enrichr(
        gene_list=gene_list,
        gene_sets=gene_sets,
        organism=organism,
        outdir=None,
        cutoff=cutoff
    )
    df = enr.results

    # Up-regulated: terms with positive combined score
    up_df = df.sort_values("Combined Score", ascending=False).head(top_n)
    # Down-regulated: terms with negative Z-score (if available)
    if "Z-score" in df.columns:
        down_df = df.sort_values("Z-score").head(top_n)
    else:
        down_df = df.sort_values("Combined Score").tail(top_n)

    # Dotplot for up-regulated
    plt.figure(figsize=(8, 6))
    gp.dotplot(up_df, title=f"{title_prefix} Top {top_n} Enriched Terms (Up)", cutoff=cutoff, size=10)
    if outdir:
        plt.savefig(f"{outdir}/{title_prefix}_ora_up_dotplot.png", bbox_inches="tight", dpi=200)
    plt.show()

    # Dotplot for down-regulated
    plt.figure(figsize=(8, 6))
    gp.dotplot(down_df, title=f"{title_prefix} Top {top_n} Enriched Terms (Down)", cutoff=cutoff, size=10)
    if outdir:
        plt.savefig(f"{outdir}/{title_prefix}_ora_down_dotplot.png", bbox_inches="tight", dpi=200)
    plt.show()

    return df