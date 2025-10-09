import os
import obonet
from node2vec import Node2Vec
import argparse
import json

def make_node2vec_model(graph, dimensions=64, walk_length=30, num_walks=200, workers=4, window=10, min_count=1, batch_words=4):
    node2vec = Node2Vec(
        graph,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=workers
    )
    model = node2vec.fit(window=window, min_count=min_count, batch_words=batch_words)
    return model

def load_ontology_graph(obo_folder, ontology_name):
    obo_path = os.path.join(obo_folder, f"{ontology_name}.obo")
    if not os.path.isfile(obo_path):
        raise FileNotFoundError(f"OBO file not found: {obo_path}")
    graph = obonet.read_obo(obo_path)
    return graph

def save_model_and_metadata(model, outdir, ontology_name, params):
    os.makedirs(outdir, exist_ok=True)
    model_path = os.path.join(outdir, f"{ontology_name}_node2vec.model")
    meta_path = os.path.join(outdir, f"{ontology_name}_node2vec_metadata.json")
    model.save(model_path)
    with open(meta_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Model saved to {model_path}")
    print(f"Metadata saved to {meta_path}")

def get_cl_subtree(graph, root):
    """
    Traverse successors starting from `root`, 
    but only keep nodes with IDs starting with 'CL:'.
    Returns a subgraph containing the CL-only subtree.
    """
    nodes = set()
    edges = []

    def dfs(node):
        if node in nodes:
            return
        if not node.startswith("CL:"):
            return  # skip non-CL terms
        nodes.add(node)
        for child in graph.successors(node):
            if child.startswith("CL:"):   # recurse only into CL children
                edges.append((node, child))
                dfs(child)

    dfs(root)
    return graph.subgraph(nodes).copy()

def main():
    parser = argparse.ArgumentParser(description="Train and save Node2Vec model for an ontology OBO file.")
    parser.add_argument("--obo_folder", type=str, required=True, help="Folder containing OBO files.")
    parser.add_argument("--ontology_name", type=str, required=True, help="Ontology name (e.g., 'cl' for cl.obo).")
    parser.add_argument("--outdir", type=str, default="models", help="Output directory for model and metadata.")
    parser.add_argument("--dimensions", type=int, default=64, help="Embedding dimensions.")
    parser.add_argument("--walk_length", type=int, default=30, help="Length of each random walk.")
    parser.add_argument("--num_walks", type=int, default=200, help="Number of walks per node.")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads.")
    parser.add_argument("--window", type=int, default=10, help="Word2Vec window size.")
    parser.add_argument("--min_count", type=int, default=1, help="Word2Vec min_count.")
    parser.add_argument("--batch_words", type=int, default=4, help="Word2Vec batch_words.")

    args = parser.parse_args()

    graph = load_ontology_graph(args.obo_folder, args.ontology_name)
    model = make_node2vec_model(
        graph,
        dimensions=args.dimensions,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        workers=args.workers,
        window=args.window,
        min_count=args.min_count,
        batch_words=args.batch_words
    )
    params = vars(args)
    save_model_and_metadata(model, args.outdir, args.ontology_name, params)

if __name__ == "__main__":
    main()