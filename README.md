# scImmune: Foundation Model for Immune-Specific sc-RNAseq Data

This repository provides code and pipelines for fine-tuning and evaluating foundational models (such as scGPT) on single-cell RNA-seq data, with a focus on cell-type annotation tasks. It includes scripts and Jupyter notebooks for data preprocessing, model training (including PEFT/LoRA), and downstream analysis/visualization.

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
  - [1. Data Import and Preprocessing](#1-data-import-and-preprocessing)
  - [2. Data Integration](#2-data-integration)
  - [3. Model Fine-tuning](#3-model-fine-tuning)
  - [4. Inference and Evaluation](#4-inference-and-evaluation)
  - [5. Visualization](#5-visualization)
- [Scripts and Notebooks](#scripts-and-notebooks)
- [References](#references)

---

## Repository Structure

```
.
├── full_finetuning_annotation.py         # End-to-end fine-tuning script
├── full_finetuning_annotation_minimal.py # Minimal fine-tuning script
├── output.png
├── requirements.txt
├── test_download.sh
├── test.h5ad
├── test2.h5ad
├── src/
│   ├── code/
│   │   ├── 01_immune_data_import.ipynb
│   │   ├── 02_immune_data_combine.ipynb
│   │   ├── 03_scgpt_czi_finetuning.ipynb
│   │   ├── 03_scgpt_czi_finetuning.py
│   │   ├── 03_scgpt_finetuning_lora.ipynb
│   │   ├── 03_scgpt_finetuning_native.ipynb
│   │   ├── 03_scgpt_inference.ipynb
│   │   ├── 04_scgpt_czi_finetuning_peft.ipynb
│   │   ├── 04_scgpt_czi_finetuning_peft.py
│   │   ├── 05_visualization.ipynb
│   │   └── Tutorial_Annotation_PEFT.ipynb
│   ├── data/
│   ├── models/
│   └── utils/
└── README.md
```

---

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/YOUR_USERNAME/immune-foundational-model.git
   cd immune-foundational-model
   ```

2. **Set up a Python environment (recommended: Python 3.9):**
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Install CellxGene Census v1.15.0 and dependencies:**
   ```sh
   pip install -U cellxgene-census
   pip uninstall -y somacore tiledbsoma
   pip install -U somacore tiledbsoma
   ```

5. **Restart your Python kernel or shell after installation.**

---

## Data Preparation

- Download the required datasets (e.g., Multiple Sclerosis, COVID PBMC) from the provided links or Google Drive.
- Place `.h5ad` files in the appropriate `src/data/` or `../data/` directory as referenced in the scripts.
- Example datasets:
  - `c_data.h5ad`
  - `filtered_ms_adata.h5ad`
  - `czi_covid_pbmc_5pct.h5ad`
  - `czi_covid_pbmc_2pct.h5ad`
---

## Usage

### 1. Data Import and Preprocessing

- Use [src/code/01_immune_data_import.ipynb](src/code/01_immune_data_import.ipynb) to query and import data from CellxGene Census.
- Preprocessing steps include normalization, log1p transformation, and highly variable gene selection.

### 2. Data Integration

- [src/code/02_immune_data_combine.ipynb](src/code/02_immune_data_combine.ipynb) demonstrates how to combine multiple datasets and perform integration (e.g., using scVI or PCA).

### 3. Model Fine-tuning

- **Standard Fine-tuning:**  
  Use [src/code/03_scgpt_czi_finetuning.ipynb](src/code/03_scgpt_czi_finetuning.ipynb) or [full_finetuning_annotation.py](full_finetuning_annotation.py) for end-to-end fine-tuning on cell-type annotation tasks.
- **Parameter-Efficient Fine-tuning (PEFT/LoRA):**  
  Use [src/code/04_scgpt_czi_finetuning_peft.ipynb](src/code/04_scgpt_czi_finetuning_peft.ipynb) and [src/code/03_scgpt_finetuning_lora.ipynb](src/code/03_scgpt_finetuning_lora.ipynb) for LoRA-based PEFT.

### 4. Inference and Evaluation

- [src/code/03_scgpt_inference.ipynb](src/code/03_scgpt_inference.ipynb) provides scripts for running inference and evaluating model performance.

### 5. Visualization

- [src/code/05_visualization.ipynb](src/code/05_visualization.ipynb) contains code for visualizing results, including UMAPs and confusion matrices.

---

## Scripts and Notebooks

- **[full_finetuning_annotation.py](full_finetuning_annotation.py):**  
  End-to-end script for fine-tuning and evaluation.
- **[src/code/03_scgpt_czi_finetuning.py](src/code/03_scgpt_czi_finetuning.py):**  
  Python script version of the fine-tuning pipeline.
- **[src/code/04_scgpt_czi_finetuning_peft.py](src/code/04_scgpt_czi_finetuning_peft.py):**  
  Script for PEFT/LoRA fine-tuning.
- **[src/code/Tutorial_Annotation_PEFT.ipynb](src/code/Tutorial_Annotation_PEFT.ipynb):**  
  Tutorial notebook for PEFT-based annotation.

---

## Example: Running Fine-tuning

```sh
python full_finetuning_annotation.py --dataset_name ms --epochs 10 --batch_size 32 --lr 1e-4
```

Or run the corresponding Jupyter notebook for step-by-step execution and visualization.

---

## References

- [scGPT download docs](https://github.com/bowang-lab/scGPT/tree/main/data/cellxgene)
- [SOMA (Stack of Matrices, Annotated) docs](https://github.com/single-cell-data/SOMA/blob/main/abstract_specification.md)
- [CellxGene Census documentation](https://chanzuckerberg.github.io/cellxgene-census/)
- [PEFT/LoRA for Transformers](https://github.com/huggingface/peft)

---

## Notes

- For best results, ensure all dependencies match the versions in `requirements.txt`.
- GPU is recommended for model training.
- For questions or issues, please open an issue in this repository.

---