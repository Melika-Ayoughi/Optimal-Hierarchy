import json
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
import matplotlib.pyplot as plt
import math
import os
from evaluate import save
import matplotlib.lines as mlines


# Function to recursively add nodes and edges to the graph
def add_to_graph(graph, node, parent=None):
    graph.add_node(node["name"])
    if parent:
        graph.add_edge(parent, node["name"])
    for subclass in node.get("subclasses", []):
        add_to_graph(graph, subclass, node["name"])

# Load JSON data (replace 'json_data' with your actual JSON string or file path)
address = "./real-world/pizza_with_DAG_hierarchy.json"
with open(address, "r") as f:
    data = json.load(f)

G = nx.DiGraph()
add_to_graph(G, data)


adr = f"./tree/real-world/pizza_with_DAG"
fig_address = f'{adr}.pdf'  # visualization of the tree
degree_hist_address = f'{adr}_hist.pdf'  # visualization of the degree histogram
tree_address = f"{adr}.json"  # The tree in json format
adjc_address = f"{adr}_adjacency.csv"  # The adjacency matrix of the tree
# G.remove_edge('CheeseTopping', 'CheeseyVegetableTopping')
# G = G.to_undirected()
root = 'Thing'


def find_extra_edges(json_data, G):
    root = json_data["name"]
    spanning_tree = nx.dfs_tree(G, source=root)
    # Find edges that are not part of the spanning tree
    all_edges = set(G.edges())
    tree_edges = set(spanning_tree.edges())
    non_tree_edges = all_edges - tree_edges
    return non_tree_edges, tree_edges

non_tree_edges, tree_edges = find_extra_edges(data, G)
# Visualization
plt.figure(figsize=(15, 10))
pos = nx.nx_agraph.graphviz_layout(G, prog="dot")  # Use 'dot' layout for hierarchy
nx.draw_networkx_edges(G, pos, edgelist=list(tree_edges), edge_color="black", label="Tree Edges", alpha=0.5)
nx.draw_networkx_edges(G, pos, edgelist=list(non_tree_edges), edge_color="red", style="dashed", label="Non-Tree Edges")
nx.draw_networkx_nodes(G, pos, node_size=150, node_color="#4b9396")

# Legend and title
tree_legend = mlines.Line2D([], [], color="black", linestyle="-", label="Spanning Tree Edges")
non_tree_legend = mlines.Line2D([], [], color="red", linestyle="--", label="Non-Tree Edges")
for node, (x, y) in pos.items():
        plt.text(
            x, y, s=node, #node["name"]
            horizontalalignment='center',
            verticalalignment='top',
            rotation=90,  # Rotate 90 degrees
            fontsize=8  # Smaller text
        )
plt.legend(handles=[tree_legend, non_tree_legend], loc="best")
plt.title("Pizza Tree and Non-Tree Edges")
plt.axis('off')
plt.savefig(fig_address)


# if not, generate it and save it
save(G, degree_hist_address, fig_address, tree_address, adjc_address, root)

# Draw the tree

