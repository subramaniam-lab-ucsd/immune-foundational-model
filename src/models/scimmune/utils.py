import os
import json
import numpy as np

import numpy as np
from typing import Any, Dict, List
import torch 

import torch
from gensim.models import Word2Vec
from transformers import PreTrainedTokenizer, PreTrainedModel


def assign_ontology_embeddings(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    node2vec_model_path: str,
    tag: str,
):
    """
    Assign ontology-based Node2Vec embeddings to all metadata tokens with a given tag.

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer with metadata tokens added.
        model (PreTrainedModel): HF-compatible model with get_input_embeddings().
        node2vec_model_path (str): Path to trained gensim Word2Vec model for the ontology (e.g. DOID, CL).
        tag (str): Metadata tag to match (e.g., "cell_type", "tissue", "disease").
    """
    # Load the Node2Vec model
    node2vec_model = Word2Vec.load(node2vec_model_path)
    embedding_dim = model.get_input_embeddings().embedding_dim
    embedding_layer = model.get_input_embeddings()

    num_assigned = 0
    for token, token_id in tokenizer.get_vocab().items():
        # Check if token is in metadata format: <tag=ONTOLOGY_ID>
        if not (token.startswith("<") and token.endswith(">") and "=" in token):
            continue

        token_tag, ontology_id = token[1:-1].split("=")
        if token_tag != tag:
            continue  # Skip tokens from other tags

        if ontology_id in node2vec_model.wv:
            vec = node2vec_model.wv[ontology_id]
            assert vec.shape[0] == embedding_dim, f"Embedding dim mismatch for {ontology_id}"
            with torch.no_grad():
                embedding_layer.weight[token_id] = torch.tensor(vec, dtype=torch.float32)
            num_assigned += 1
        else:
            print(f"[WARN] No Node2Vec embedding found for {ontology_id}, skipping.")

    print(f"[INFO] Assigned {num_assigned} Node2Vec embeddings for tag <{tag}=...>")


def generate_metadata_embeddings(node2vecModel: Any) -> Dict[str, np.ndarray]:
    """
    Generate a dictionary of ontology node embeddings from a trained Node2Vec model.

    Args:
        node2vecModel (gensim.models.Word2Vec): A trained Node2Vec model where each ontology node 
            (e.g., CL:0000084) is a key in the model's vocabulary.

    Returns:
        Dict[str, np.ndarray]: A dictionary mapping ontology node IDs to their embedding vectors.
            Example: { "CL:0000084": np.array([...]), ... }
    """
    # Retrieve all node keys from the model
    nodes = node2vecModel.wv.index_to_key  
    
    # Build embedding dictionary
    return {node: np.array(node2vecModel.wv[node]) for node in nodes}  


def generate_metadata_tokens(ontology_ids: List[str], tag: str) -> List[str]:
    """
    Construct token strings for metadata by prepending a tag to each ontology ID.

    Args:
        ontology_ids (List[str]): A list of ontology term IDs (e.g., ["CL:0000084", "CL:0000236"]).
        tag (str): A metadata tag indicating the ontology type (e.g., "cell_type", "tissue").

    Returns:
        List[str]: A list of formatted token strings.
            Example: ["<cell_type=CL:0000084>", "<cell_type=CL:0000236>"]
    """
    return [f"<{tag}={oid}>" for oid in ontology_ids]
    
    
def load_vocab(vocab_path):
    """
    Load a vocab mapping from a JSON file and return as a dictionary.

    Parameters
    ----------
    name : str
        Base name of the vocab file (without extension).
    path : str
        Directory containing the vocab file.

    Returns
    -------
    dict
        Dictionary mapping gene names to vocab indices.
    """
    if not os.path.isfile(vocab_path):
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
    with open(vocab_path, 'r') as f:
        vocab_map = json.load(f)
    return vocab_map

def load_model(name, path):
    pass

