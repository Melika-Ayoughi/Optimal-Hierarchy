import os
import pandas as pd
import json
import math
import pickle
import csv
from networkx.readwrite import json_graph
import torch
import networkx as nx
from hypll.manifolds.poincare_ball import PoincareBall, Curvature
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout


def tree_height(graph, root=0):
    lengths = nx.single_source_shortest_path_length(graph, root)
    return max(lengths.values())


def depth_measures(G, root=0):
    # Identify leaf nodes (nodes with degree 1, excluding the root)
    leaf_nodes = [n for n in G.nodes() if G.degree(n) == 1 and n != root]
    # Calculate depth of each leaf
    depths = [nx.shortest_path_length(G, root, leaf) for leaf in leaf_nodes]
    # Calculate average depth
    sakin_index = sum(depths)
    average_depth = sum(depths) / len(depths)  # normalize by number of leaf nodes
    variance_depth = sum((d - average_depth) ** 2 for d in depths) / len(depths)
    return sakin_index, average_depth, variance_depth, math.sqrt(variance_depth)


def degree_measures(G):
    degree_counts = nx.degree_histogram(G)
    degrees = range(len(degree_counts))
    max_degree = max(dict(G.degree).values())
    average_degree = sum(dict(G.degree).values()) / G.number_of_nodes()
    return degrees, degree_counts, max_degree, average_degree


def average_vertex_depth(G, root=0):
    depths = nx.single_source_shortest_path_length(G, root).values()
    # Calculate average depth
    average_depth = sum(depths) / len(depths)
    return average_depth


def smallest_path_from_root(G, root=0):
    shortest_path_lengths = nx.single_source_shortest_path_length(G, root)
    # Exclude the root and find the minimum shortest path length
    min_path_length = min(length for node, length in shortest_path_lengths.items() if node != root)
    return min_path_length


def save(tree, degree_hist_address, fig_address, tree_address, adjc_address, root=0):
    os.makedirs("/".join(fig_address.split('/')[:-1]), exist_ok=True)
    print("Graph is a tree:", nx.is_tree(tree))
    # Root node: [0] ?
    print(f"Number of leaf nodes: {len([x for x in tree.nodes() if tree.degree(x) == 1])}")
    # print(f"Diameter: {nx.diameter(tree)}")
    height = tree_height(tree, root)
    print(f"Height: {height}")
    sakin_index, average_depth, variance_depth, std_depth = depth_measures(tree, root)
    print(f"Sackin index: {sakin_index}")
    print(f"Average depth {average_depth} +- std {std_depth}")
    print(f"Variance depth: {variance_depth}")
    print(f"Average vertex depth: {average_vertex_depth(tree, root)}")
    degrees, degree_counts, max_degree, average_degree = degree_measures(tree)
    print(f"Max degree: {max_degree}, Average degree: {average_degree}")
    print(f"Normalized height imbalance: {(height - smallest_path_from_root(tree, root)) / height}")

    # Visualize the degree histogram
    plt.bar(degrees, degree_counts, color='#1B4C68')
    plt.title("Degree Histogram")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    # plt.ylim(0, 300)
    plt.savefig(degree_hist_address)
    plt.clf()

    # Visualize the tree
    plt.figure(figsize=(15, 10))
    nx.nx_agraph.write_dot(tree, 'test.dot')
    # nx.draw(tree, pos, with_labels=True, arrows=True, font_size=8, node_size=150, node_color="#4b9396")
    pos = nx.nx_agraph.graphviz_layout(tree, prog="dot")
    nx.draw(tree, pos, with_labels=False, node_size=150, font_size=8, node_color="#4b9396")
    # for node, (x, y) in pos.items():
    #     plt.text(
    #         x, y, s=node, #node["name"]
    #         horizontalalignment='center',
    #         verticalalignment='top',
    #         rotation=90,  # Rotate 90 degrees
    #         fontsize=8  # Smaller text
    #     )
    plt.savefig(fig_address)
    plt.clf()

    # Save tree
    data = nx.node_link_data(tree)  # Convert graph to node-link data
    with open(tree_address, "w") as f:
        json.dump(data, f)

    # Save adjacency file
    save_adjacency(tree, adjc_file=adjc_address)


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


def dist(x: torch.Tensor, y: torch.Tensor, c: float | torch.Tensor) -> torch.Tensor:
    mx2 = (1 - x.square().sum(dim=-1))
    my2 = (1 - y.square().sum(dim=-1))
    xmy2 = (x - y).square().sum(dim=-1)
    return (1 + 2 * xmy2 / (mx2 * my2)).acosh()


def distortion(
    embeddings: torch.Tensor,
    graph: nx.DiGraph | nx.Graph,
    # graph_name: str,
    ball: PoincareBall,
    tau: float,
) -> torch.Tensor:
    # Compute pairwise distances of embeddings
    embedding_dists = dist(embeddings, embeddings[:, None, :], ball.c())
    # Set some stuff up for computing target distances
    undirected_graph = graph.to_undirected()
    number_of_nodes = embeddings.size(0)
    # Compute the target distances as lengths of shortest paths
    # try:
    #     metric_file = os.path.join(
    #         os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    #         "results",
    #     )
    # except:
    target_dists = torch.empty([number_of_nodes, number_of_nodes])
    for dist_tuple in nx.shortest_path_length(undirected_graph, weight="weight"):
        distances_sorted_by_node_id = [d for n, d in sorted(dist_tuple[1].items())]
        target_dists[dist_tuple[0], :] = torch.tensor(distances_sorted_by_node_id)
    # Scale by tau
    target_dists = tau * target_dists
    # Compute and return the relative distortion
    rel_distortion = (embedding_dists - target_dists).abs() / target_dists
    rel_distortion.fill_diagonal_(0.0)
    true_dist = embedding_dists / target_dists
    true_dist.fill_diagonal_(1.0)
    return rel_distortion, true_dist


def mean_average_precision(
    embeddings: torch.Tensor,
    graph: nx.DiGraph | nx.Graph,
    ball: PoincareBall,
) -> torch.Tensor:
    n = len(graph.nodes())
    if graph.is_directed():
        graph = graph.to_undirected()
    # Compute pairwise distances of embeddings
    embedding_dists = dist(embeddings, embeddings[:, None, :], ball.c())
    embedding_dists.fill_diagonal_(float("inf"))
    embedding_dists_neighbours = embedding_dists.clone()
    # Grab indices of neighbourhood nodes for each row
    non_neighbourhood_index = torch.ones_like(embedding_dists, dtype=bool)
    for node in graph:
        # neighbourhood = (
        #     list(graph.predecessors(node))
        #     + list(graph.successors(node))
        # )
        neighbourhood = list(graph.neighbors(node))
        non_neighbourhood_index[node, neighbourhood] = False
    # Set all non-neighbouring nodes distances to inf in the cloned distance tensor
    embedding_dists_neighbours[non_neighbourhood_index] = float("inf")
    # Argsort the rows for both distance tensors
    argsorted_all = embedding_dists.argsort(stable=True)
    argsorted_neighbourhood = embedding_dists_neighbours.argsort(stable=True)
    # Rank each column node according to its proximity to the row node
    rank_all = torch.empty_like(embedding_dists)
    rank_neighbourhood = rank_all.clone()
    src_vals = torch.arange(1, n + 1, dtype=rank_all.dtype).expand(n, n)
    rank_all.scatter_(dim=1, index=argsorted_all, src=src_vals)
    rank_neighbourhood.scatter_(dim=1, index=argsorted_neighbourhood, src=src_vals)
    # Check fraction of rank within neighbourhood to rank overall
    prec = rank_neighbourhood / rank_all
    # Scale the values by the degree of the row node. This avoids having to take the inside mean
    # of the formula, which would otherwise require manual looping.
    weighted = prec / (~non_neighbourhood_index).sum(dim=1, keepdim=True)
    # Sum and divide by the number of nodes to obtain the mean average precision
    return weighted[~non_neighbourhood_index].sum() / n


# def reconstruct_tree_from_transitive_closure(transitive_closure_file):
#     transitive_closure_df = pd.read_csv(transitive_closure_file, usecols=['id1', 'id2'], engine='c', index_col=False)
#     # ravel() is an array method that returns a view (if possible) of a multidimensional array.
#     # The argument 'K' tells the method to flatten the array
#     nodes = pd.unique(transitive_closure_df.values.ravel('K'))
#
#     # Create a new directed graph for the original tree
#     tree = nx.Graph()
#     for _, row in transitive_closure_df.iterrows():
#         u, v = row['id1'], row['id2']
#         # Check if (u, v) is a direct edge by ensuring there's no intermediary node w
#         is_direct_edge = True
#         for w in nodes:
#             if w != u and w != v:
#                 # Check if (u, w) and (w, v) exist in the DataFrame
#                 if ((transitive_closure_df['id1'] == u) & (transitive_closure_df['id2'] == w)).any() and \
#                         ((transitive_closure_df['id1'] == w) & (transitive_closure_df['id2'] == v)).any():
#                     is_direct_edge = False
#                     break
#
#         if is_direct_edge:
#             tree.add_edge(u, v)
#
#     return tree

def save_adjacency(G, adjc_file):
    edges = list(G.edges())

    # Create a DataFrame and add the weight column
    df = pd.DataFrame(edges, columns=["id1", "id2"])
    df["weight"] = 1  # Add the weight column with value 1

    # Save to a CSV file without row indices
    df.to_csv(adjc_file, index=False)


def generate_trees_and_adjacency(tree_types, rs, Ns):

    for N in Ns:
        for tree_type in tree_types:
            if tree_type == "binomial":
                adr = f"./tree_properties/{tree_type}/{tree_type}{N}"
                fig_address = f'{adr}.pdf'  # visualization of the tree
                degree_hist_address = f'{adr}_hist.pdf'  # visualization of the degree histogram
                tree_address = f"{adr}.json"  # The tree in json format
                adjc_address = f"{adr}_adjacency.csv"  # The adjacency matrix of the tree
                # if file exists skip
                if os.path.isfile(tree_address):
                    continue
                # if not, generate it and save it
                tree = nx.binomial_tree(n=int(math.log(N, 2)))
                save(tree, degree_hist_address, fig_address, tree_address, adjc_address)

            elif tree_type == "full_rary_tree":
                for r in rs:
                    adr = f"./tree_properties/{tree_type}/{tree_type}{N}_r{r}"
                    fig_address = f'{adr}.pdf'  # visualization of the tree
                    degree_hist_address = f'{adr}_hist.pdf'  # visualization of the degree histogram
                    tree_address = f"{adr}.json"  # The tree in json format
                    adjc_address = f"{adr}_adjacency.csv"  # The adjacency matrix of the tree
                    # if file exists skip
                    if os.path.isfile(tree_address):
                        continue
                    tree = nx.full_rary_tree(r=r, n=N)
                    print(f"r is  {r}")
                    save(tree, degree_hist_address, fig_address, tree_address, adjc_address)

            elif tree_type == "barabasi_albert_graph":
                adr = f"./tree_properties/{tree_type}/{tree_type}{N}"
                fig_address = f'{adr}.pdf'  # visualization of the tree
                degree_hist_address = f'{adr}_hist.pdf'  # visualization of the degree histogram
                tree_address = f"{adr}.json"  # The tree in json format
                adjc_address = f"{adr}_adjacency.csv"  # The adjacency matrix of the tree
                # if file exists skip
                if os.path.isfile(tree_address):
                    continue
                # if not, generate it and save it
                tree = nx.barabasi_albert_graph(n=N, m=1, seed=30)
                save(tree, degree_hist_address, fig_address, tree_address, adjc_address)

            elif tree_type == "star_graph":
                adr = f"./tree_properties/{tree_type}/{tree_type}{N}"
                fig_address = f'{adr}.pdf'  # visualization of the tree
                degree_hist_address = f'{adr}_hist.pdf'  # visualization of the degree histogram
                tree_address = f"{adr}.json"  # The tree in json format
                adjc_address = f"{adr}_adjacency.csv"  # The adjacency matrix of the tree
                # if file exists skip
                if os.path.isfile(tree_address):
                    continue
                # if not, generate it and save it
                tree = nx.star_graph(n=N)
                save(tree, degree_hist_address, fig_address, tree_address, adjc_address)

            elif tree_type == "path_graph":
                adr = f"./tree_properties/{tree_type}/{tree_type}{N}"
                fig_address = f'{adr}.pdf'  # visualization of the tree
                degree_hist_address = f'{adr}_hist.pdf'  # visualization of the degree histogram
                tree_address = f"{adr}.json"  # The tree in json format
                adjc_address = f"{adr}_adjacency.csv"  # The adjacency matrix of the tree
                # if file exists skip
                if os.path.isfile(tree_address):
                    continue
                # if not, generate it and save it
                tree = nx.path_graph(n=N)
                save(tree, degree_hist_address, fig_address, tree_address, adjc_address)


def generate_tsv_for_entailment(tree_types, rs, Ns):
    def save_transitive_closures(tree, closure_address):
        closure = nx.transitive_closure(tree)

        with open(f'{closure_address}', 'w', newline='') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t')
            writer.writerows([(v, u) for u, v in closure.edges()])  # child-parent

    for N in Ns:
        for tree_type in tree_types:
            if tree_type in ["binomial", "barabasi_albert_graph", "star_graph", "path_graph"]:
                adr = f"./tree/{tree_type}/{tree_type}{N}"
                tree_address = f"{adr}.json"  # The tree in json format
                closure_address = f"{adr}_closure.tsv"
                tree = load_tree(tree_address)
                save_transitive_closures(tree, closure_address)

            elif tree_type == "full_rary_tree":
                for r in rs:
                    adr = f"./tree/{tree_type}/{tree_type}{N}_r{r}"
                    tree_address = f"{adr}.json"  # The tree in json format
                    closure_address = f"{adr}_closure.tsv"
                    tree = load_tree(tree_address)
                    save_transitive_closures(tree, closure_address)


if __name__ == '__main__':
    def load_tree(tree_address):
        # Load tree
        with open(tree_address, "r") as f:
            data = json.load(f)
        return json_graph.node_link_graph(data)

    def load_embedding(model_path, model_type):
        # Load embedding
        if model_type == "poincare":
            chkpnt = torch.load(model_path)
            embeddings_best = chkpnt['model']['lt.weight']
            embeddings_check = chkpnt['embeddings']
            hierarchy_objects = chkpnt['objects']
        elif model_type == "entailment":
            chkpnt = unpickle(model_path)
            embeddings_best = chkpnt['embeddings']
            hierarchy_objects = chkpnt['objects']
            all_edges = chkpnt['edges']
            all_relations = chkpnt['all_relations']
            adjacent_nodes = chkpnt['adjacent_nodes']
            transitive_closures = chkpnt['train_data']
        return embeddings_best

    def evaluate_tree(tree, embeddings_best):
        if any(isinstance(node, str) for node in tree.nodes()):
            # relabel the nodes by mapping them to integers
            mapping = {node: idx for idx, node in enumerate(sorted(tree.nodes()))}
            tree = nx.relabel_nodes(tree, mapping)
        # Evaluate tree
        rel_distortion, true_dist = distortion(embeddings_best, tree, PoincareBall(c=Curvature(1.0, constraining_strategy=lambda x: x)), tau=1)
        rel_dist_mean = (rel_distortion.sum() / (rel_distortion.size(0) * (rel_distortion.size(0) - 1))).item()
        worst_case_dist = true_dist.max().item() / true_dist.min().item()
        print(f"avg distortion: {rel_dist_mean}"
              f", Worst-case distortion: {worst_case_dist}"
              f", MAP: {mean_average_precision(embeddings_best, tree, PoincareBall(c=Curvature(1.0, constraining_strategy=lambda x: x)))}")


    # Hyperparameters
    model_type = "poincare"
    tree_types = ["imagenet_withDAGs", "imagenet", "imagenet_reorganized"]
    # tree_types = ["pizza_with_DAG", "pizza", "pizza_1lesshight", "pizza_2lesshight"]
    # tree_types = ["binomial", "full_rary_tree", "barabasi_albert_graph", "star_graph", "path_graph"]
    rs = [2, 3, 4, 5]
    Ns = [256, 512, 1024]
    # Ns = [16]
    # dims = [10, 20, 130]
    # dims = [40]
    dims = [70]

    # Generate and visualize tree
    # generate_trees_and_adjacency(tree_types, rs, Ns)
    # generate_tsv_for_entailment(tree_types, rs, Ns)

    # Real-world tree:
    if any("imagenet" in string for string in tree_types) or any("pizza" in string for string in tree_types):
        for dim in dims:
            for tree_type in tree_types:
                adr = f"./tree/real-world/{tree_type}"
                tree_address = f"{adr}.json"  # The tree in json format
                model_path = f"./tree/real-world/{model_type}/{tree_type}_dim{dim}.bin.best"
                tree = load_tree(tree_address)
                embeddings_best = load_embedding(model_path, model_type)
                print(f"Tree type: {tree_type}, Model: {model_type}, embedding dimension {dim}")
                evaluate_tree(tree, embeddings_best)

    # Generated tree:
    else:
        for N in Ns:
            for dim in dims:
                for tree_type in tree_types:
                    if tree_type in ["binomial", "barabasi_albert_graph", "star_graph", "path_graph"]:
                        adr = f"./tree/{tree_type}/{tree_type}{N}"
                        tree_address = f"{adr}.json"  # The tree in json format
                        model_path = f"./tree/{tree_type}/{model_type}/{tree_type}{N}_dim{dim}.bin.best"
                        tree = load_tree(tree_address)
                        embeddings_best = load_embedding(model_path, model_type)
                        print(f"Tree type: {tree_type}, #Nodes: {N}, Model: {model_type}, embedding dimension {dim}")
                        evaluate_tree(tree, embeddings_best)

                    elif tree_type == "full_rary_tree":
                        for r in rs:
                            adr = f"./tree/{tree_type}/{tree_type}{N}_r{r}"
                            tree_address = f"{adr}.json"  # The tree in json format
                            model_path = f"./tree/{tree_type}/{model_type}/{tree_type}{N}_r{r}_dim{dim}.bin.best"
                            tree = load_tree(tree_address)
                            embeddings_best = load_embedding(model_path, model_type)
                            print(f"Tree type: {tree_type}, #Nodes: {N}, Model: {model_type}, r: {r}, embedding dimension {dim}")
                            evaluate_tree(tree, embeddings_best)












