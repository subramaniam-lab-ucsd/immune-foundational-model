import seaborn as sns


def plot_embedding_quality(models, silhouette, ari, nmi, title="Embedding Quality Comparison", figsize=(5,5), dpi=300):
    """
    Plot bar chart comparing silhouette, ARI, and NMI for different models.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    bars1 = ax.bar(x - width, silhouette, width, label='Silhouette', color='#6baed6')
    bars2 = ax.bar(x, ari, width, label='ARI', color='#74c476')
    bars3 = ax.bar(x + width, nmi, width, label='NMI', color='#fd8d3c')

    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)

    ax.set_ylabel('Score', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.show()

def palette_generator(cell_types):
    """
    Generate a consistent color palette for a list of cell types, grouped by biological class.

    Parameters
    ----------
    cell_types : list or pd.Index
        List of unique cell type names (strings).

    Returns
    -------
    palette : dict
        Dictionary mapping cell type names to colors.
    """
    # Group cell types by biological class
    monocyte_ct = [ct for ct in cell_types if "monocyte" in ct.lower()]
    t_cell_ct = [ct for ct in cell_types if "t cell" in ct.lower()]
    b_cell_ct = [ct for ct in cell_types if "b cell" in ct.lower()]
    nk_cell_ct = [ct for ct in cell_types if "nk" in ct.lower()]
    dc_ct = [ct for ct in cell_types if "dendritic" in ct.lower()]
    platelet_ct = [ct for ct in cell_types if "platelet" in ct.lower() or "megakaryocyte" in ct.lower()]
    erythro_ct = [ct for ct in cell_types if "erythro" in ct.lower()]
    stem_ct = [ct for ct in cell_types if "stem" in ct.lower()]
    other_ct = [ct for ct in cell_types if ct not in monocyte_ct + t_cell_ct + b_cell_ct + nk_cell_ct + dc_ct + platelet_ct + erythro_ct + stem_ct]

    # Assign color palettes to each group
    palette = {}
    palette.update(dict(zip(monocyte_ct, sns.color_palette("dark:salmon_r", len(monocyte_ct)))))
    palette.update(dict(zip(t_cell_ct, sns.color_palette("RdPu", len(t_cell_ct)))))
    palette.update(dict(zip(b_cell_ct, sns.color_palette("YlGn", len(b_cell_ct)))))
    palette.update(dict(zip(nk_cell_ct, sns.color_palette("BuPu", len(nk_cell_ct)))))
    palette.update(dict(zip(dc_ct, sns.color_palette("YlOrBr", len(dc_ct)))))
    palette.update(dict(zip(platelet_ct, sns.color_palette("cividis", len(platelet_ct)))))
    palette.update(dict(zip(erythro_ct, sns.color_palette("flare", len(erythro_ct)))))
    palette.update(dict(zip(stem_ct, sns.color_palette("crest", len(stem_ct)))))
    palette.update(dict(zip(other_ct, sns.color_palette("BrBG", len(other_ct)))))
    return palette

# Example usage:
# all_labels = pd.Index(flu_data_preds.obs["true_coarse"].unique()).union(flu_data_preds.obs["pred_coarse"].unique())
# palette = make_celltype_palette(all_labels)

# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

# # Compute raw confusion matrix
# cm = pd.crosstab(
#     covid_data_preds.obs["true_coarse"],
#     covid_data_preds.obs["pred_coarse"],
#     rownames=["True Cell Type"],
#     colnames=["Predicted Cell Type"],
#     dropna=False
# )

# # Reorder columns to match the true label order
# true_order = cm.index.tolist()
# column_order = [col for col in true_order if col in cm.columns] + [col for col in cm.columns if col not in true_order]
# cm = cm[column_order]

# # Normalize row-wise to get percentages
# cm_percent = cm.div(cm.sum(axis=1), axis=0) * 100

# # Plot confusion matrix
# plt.figure(figsize=(30, 20), dpi=300)
# sns.set(style="whitegrid")

# ax = sns.heatmap(
#     cm_percent,
#     cmap="YlGnBu",
#     annot=True,
#     fmt=".1f",
#     linewidths=0.5,
#     linecolor="gray",
#     square=True,
#     cbar_kws={'label': 'Percentage (%)'},
#     annot_kws={"fontsize": 12}
# )

# # Titles and labels
# ax.set_title("Confusion Matrix (in %) – scGPT Predicted vs True Coarse Cell Types - Influenza Validation Data", fontsize=16, pad=20)
# ax.set_xlabel("Predicted Cell Type", fontsize=14, labelpad=10)
# ax.set_ylabel("True Cell Type", fontsize=14, labelpad=10)

# # Tick formatting
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
# ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

# plt.tight_layout()
# plt.savefig("confusion_matrix_scgpt_coarse_aligned.png", dpi=300, bbox_inches="tight")
# plt.show()

# true_label_coarse_map = {
#     "classical monocyte": "CD14+ monocyte",
#     "non-classical monocyte": "CD16+ monocyte",
#     "intermediate monocyte": "Intermediate monocyte",

#     "CD8-positive, alpha-beta T cell": "CD8+ T cell",
#     "effector CD8-positive, alpha-beta T cell": "CD8+ T cell",

#     "CD4-positive helper T cell": "CD4+ T cell",
#     "effector CD4-positive, alpha-beta T cell": "CD4+ T cell",

#     "natural killer cell": "NK cell",

#     "IgG-negative class switched memory B cell": "Memory B cell",
#     "IgG memory B cell": "Memory B cell",

#     "dendritic cell": "Dendritic cell",

#     "erythrocyte": "Erythrocyte",
#     "platelet": "Platelet",

#     "blood cell": "Unspecified blood cell"
# }

# pred_label_coarse_map = {
#     # Monocytes
#     "classical monocyte": "CD14+ monocyte",
#     "CD14-positive monocyte": "CD14+ monocyte",
#     "CD14-positive, CD16-negative classical monocyte": "CD14+ monocyte",
#     "CD14-positive, CD16-positive monocyte": "Intermediate monocyte",
#     "non-classical monocyte": "CD16+ monocyte",
#     "CD14-low, CD16-positive monocyte": "CD16+ monocyte",
#     "monocyte": "Unspecified monocyte",
    
#     # CD8+ T cells
#     "CD8-positive, alpha-beta T cell": "CD8+ T cell",
#     "effector CD8-positive, alpha-beta T cell": "CD8+ T cell",
#     "effector memory CD8-positive, alpha-beta T cell": "CD8+ T cell",
#     "activated CD8-positive, alpha-beta T cell, human": "CD8+ T cell",
#     "naive thymus-derived CD8-positive, alpha-beta T cell": "CD8+ T cell",

#     # CD4+ T cells
#     "central memory CD4-positive, alpha-beta T cell": "CD4+ T cell",
#     "CD4-positive, alpha-beta T cell": "CD4+ T cell",
#     "CD4-positive, alpha-beta memory T cell": "CD4+ T cell",
#     "naive thymus-derived CD4-positive, alpha-beta T cell": "CD4+ T cell",
#     "activated CD4-positive, alpha-beta T cell, human": "CD4+ T cell",
#     "CD4-positive, alpha-beta cytotoxic T cell": "CD4+ T cell",
#     "regulatory T cell": "Regulatory T cell",
#     "T cell": "Unspecified T cell",
#     "mature alpha-beta T cell": "Unspecified T cell",

#     # Gamma-delta / MAIT / NKT
#     "gamma-delta T cell": "Gamma-delta T cell",
#     "mucosal invariant T cell": "MAIT cell",
#     "mature NK T cell": "NKT cell",

#     # NK cells
#     "natural killer cell": "NK cell",
#     "CD16-negative, CD56-bright natural killer cell, human": "NK cell",
#     "CD16-positive, CD56-dim natural killer cell, human": "NK cell",

#     # B cells
#     "naive B cell": "Naive B cell",
#     "memory B cell": "Memory B cell",
#     "transitional stage B cell": "Naive B cell",
#     "B cell": "Unspecified B cell",

#     # Plasma cells / Plasmablasts
#     "plasmablast": "Plasma cell",
#     "IgG plasmablast": "Plasma cell",
#     "IgA plasmablast": "Plasma cell",

#     # Dendritic cells
#     "conventional dendritic cell": "Dendritic cell",
#     "plasmacytoid dendritic cell, human": "Dendritic cell",
#     "dendritic cell": "Dendritic cell",

#     # Neutrophils
#     "neutrophil": "Neutrophil",
#     "immature neutrophil": "Neutrophil",

#     # Other cell types
#     "erythrocyte": "Erythrocyte",
#     "platelet": "Platelet",
#     "megakaryocyte": "Megakaryocyte",
#     "megakaryocyte-erythroid progenitor cell": "MEP",
#     "hematopoietic stem cell": "Hematopoietic stem cell",
#     "CD34-positive, CD38-negative hematopoietic stem cell": "Hematopoietic stem cell",
#     "myeloid cell": "Unspecified myeloid",
#     "lymphocyte": "Unspecified lymphoid",
#     "blood cell": "Unspecified blood cell",
# }
