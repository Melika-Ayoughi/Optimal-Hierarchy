import networkx as nx
import json
from evaluate import save
from networkx.drawing.nx_agraph import to_agraph
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def find_extra_edges(root, G):
    spanning_tree = nx.dfs_tree(G, source=root)
    # Find edges that are not part of the spanning tree
    all_edges = set(G.edges())
    tree_edges = set(spanning_tree.edges())
    non_tree_edges = all_edges - tree_edges
    return non_tree_edges, tree_edges


def trim_imagenet_DAGs(G, adr):
    non_tree_edges, tree_edges = find_extra_edges('0', G)

    # Visualization
    plt.figure(figsize=(15, 10))
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")  # Use 'dot' layout for hierarchy
    nx.draw_networkx_edges(G, pos, edgelist=list(tree_edges), edge_color="black", label="Tree Edges", alpha=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=list(non_tree_edges), edge_color="red", style="dashed", label="Non-Tree Edges")
    nx.draw_networkx_nodes(G, pos, node_size=150, node_color="#4b9396")

    # Legend and title
    tree_legend = mlines.Line2D([], [], color="black", linestyle="-", label="Spanning Tree Edges")
    non_tree_legend = mlines.Line2D([], [], color="red", linestyle="--", label="Non-Tree Edges")

    plt.legend(handles=[tree_legend, non_tree_legend], loc="best")
    plt.title("ImageNet based on WordNet Tree and Non-Tree Edges")
    plt.axis('off')
    # plt.show()
    plt.savefig(f"{adr}_before_trim.pdf")
    plt.clf()

    #######################################################
    G.remove_edges_from(non_tree_edges)
    # G = G.to_undirected()
    non_tree_edges_new, tree_edges_new = find_extra_edges('0', G)
    if not non_tree_edges_new:
        print("Now Imagenet is a tree")
        print("Graph is a tree:", nx.is_tree(G))
    else:
        print("Graph is a tree:", nx.is_tree(G))

    plt.figure(figsize=(15, 10))
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")  # Use 'dot' layout for hierarchy
    nx.draw_networkx_edges(G, pos, edgelist=list(tree_edges_new), edge_color="black", label="Tree Edges", alpha=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=list(non_tree_edges_new), edge_color="red", style="dashed", label="Non-Tree Edges")
    nx.draw_networkx_nodes(G, pos, node_size=150, node_color="#4b9396")
    tree_legend = mlines.Line2D([], [], color="black", linestyle="-", label="Spanning Tree Edges")

    plt.legend(handles=[tree_legend], loc="best")
    plt.title("ImageNet Spanning Tree")
    plt.axis('off')
    # plt.show()
    plt.savefig(f"{adr}_after_trim.pdf")
    plt.clf()
    return G


def load_original_imagenet_from_website_to_tree():
    adr = "./tree/real-world/imagenet_directed"
    data = load_json("./tree_properties/real-world/imagenet_original.json")
    G = nx.DiGraph()  # Create a directed graph

    def add_node_and_edges(graph, parent, node):
        # Add the current node
        graph.add_node(node['id'], name=node['name'])  # , sift=node.get('sift'), index=node.get('index'))

        # Add an edge from parent to this node if there is a parent
        if parent:
            graph.add_edge(parent, node['id'])  # node['id'])

        # If the node has children, recursively add them
        if 'children' in node:
            for child in node['children']:
                add_node_and_edges(graph, node['id'], child)  # node['id'], child)

    # Add the root node (this is the starting point, has no parent)
    add_node_and_edges(G, None, data)

    G = trim_imagenet_DAGs(G, adr)

    # Now G contains the tree structure
    # print("Nodes:", G.nodes)  # (data=True))
    print(f" Number of nodes: {len(G.nodes)}, NUmber of edges: {len(G.edges)}")

    # G = G.to_undirected()
    root = '0'
    fig_address = f'{adr}.pdf'  # visualization of the tree
    degree_hist_address = f'{adr}_hist.pdf'  # visualization of the degree histogram
    tree_address = f"{adr}.json"  # The tree in json format
    adjc_address = f"{adr}_adjacency.csv"  # The adjacency matrix of the tree

    save(G, degree_hist_address, fig_address, tree_address, adjc_address, root)


def load_imagenet_after_owl_to_tree(which_imagenet="imagenet"):
    adr = f"./tree_properties/real-world/{which_imagenet}"
    data = load_json(f"./real-world/{which_imagenet}_hierarchy.json")
    G = nx.DiGraph()  # Create a directed graph


    def add_to_graph(graph, node, parent_path="root", sibling_index=0):
        # Create a unique identifier using the hierarchical path and sibling index
        current_path = f"{parent_path}/{node['name']}_{sibling_index}"
        graph.add_node(current_path, name=node["name"])  # Add the node with its name as an attribute

        # Add an edge from the parent to the current node if it's not the root
        if parent_path != "root":
            graph.add_edge(parent_path, current_path)

        # Traverse subclasses with their sibling index
        for index, subclass in enumerate(node.get("subclasses", [])):
            add_to_graph(graph, subclass, current_path, index)

    add_to_graph(G, data)
    # DiGraph with 1763 nodes and 1776 edges
    G = G.to_undirected()
    root = 'root/Thing_0'
    fig_address = f'{adr}.pdf'  # visualization of the tree
    degree_hist_address = f'{adr}_hist.pdf'  # visualization of the degree histogram
    tree_address = f"{adr}.json"  # The tree in json format
    adjc_address = f"{adr}_adjacency.csv"  # The adjacency matrix of the tree

    save(G, degree_hist_address, fig_address, tree_address, adjc_address, root)


if __name__ == '__main__':
    # First:
    # load_original_imagenet_from_website_to_tree()
    # Second:
    load_imagenet_after_owl_to_tree(which_imagenet="imagenet_reorganized")