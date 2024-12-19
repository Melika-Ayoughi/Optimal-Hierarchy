import networkx as nx
from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef, Literal
from networkx.readwrite import json_graph
import json


def tree_to_owl(tree, root, output_file):
    """
    Convert a networkx tree into an OWL file.

    Args:
        tree (networkx.DiGraph): The networkx tree (directed graph).
        root (any): The root node of the tree.
        output_file (str): The path to save the OWL file.
    """
    # Initialize OWL graph and namespace
    g = Graph()
    ns = Namespace("http://example.com/tree#")
    g.bind("ex", ns)  # Bind prefix

    def get_node_name(node):
        return tree.nodes[node].get("name", f"Node_{node}")

    def add_class_and_children(node):
        node_name = get_node_name(node)
        # node_uri = ns[node_name.replace(" ", "_")]
        node_uri = ns[str(node)]
        g.add((node_uri, RDF.type, OWL.Class))
        g.add((node_uri, RDFS.label, Literal(node_name)))

        # Add children as Subclasses
        for child in tree.successors(node):
            child_name = get_node_name(child)
            # child_uri = ns[child_name.replace(" ", "_")]
            child_uri = ns[str(child)]
            g.add((child_uri, RDF.type, OWL.Class))
            g.add((child_uri, RDFS.subClassOf, node_uri))
            g.add((child_uri, RDFS.label, Literal(child_name)))
            add_class_and_children(child)

    # Start recursion from the root node
    add_class_and_children(root)
    print(f"number of nodes: {len(set(g.subjects()))} and objects {len(set(g.objects()))}")
    # Save the OWL graph to a file
    g.serialize(destination=output_file, format="turtle")
    print(f"OWL file saved to {output_file}")


# def tree_to_owl(tree, root, output_file):
#     """
#     Convert a networkx tree into an OWL file using node 'name' attributes.
#
#     Args:
#         tree (networkx.Graph): The networkx tree (undirected graph).
#         root (any): The root node of the tree.
#         output_file (str): The path to save the OWL file.
#     """
#     # Initialize OWL graph and namespace
#     g = Graph()
#     ns = Namespace("http://example.com/tree#")
#
#     # Bind prefix
#     g.bind("ex", ns)
#
#     # Track visited nodes to avoid cycles
#     visited = set()
#
#     # Helper to get node name or fallback to ID
#     def get_node_name(node):
#         return tree.nodes[node].get("name", f"Node_{node}")
#
#     # Recursive function to add classes and edges
#     def add_class_and_children(node, parent_uri=None):
#         if node in visited:
#             return
#         visited.add(node)
#
#         # Get node name and create a URI
#         node_name = get_node_name(node)
#         node_uri = ns[node_name.replace(" ", "_")]  # Replace spaces with underscores
#         g.add((node_uri, RDF.type, OWL.Class))
#         g.add((node_uri, RDFS.label, Literal(node_name)))
#
#         # Add subclass relationship if parent exists
#         if parent_uri:
#             g.add((node_uri, RDFS.subClassOf, parent_uri))
#
#         # Process neighbors (children) recursively
#         for child in tree.neighbors(node):
#             if child != parent_uri:  # Avoid revisiting the parent
#                 add_class_and_children(child, node_uri)
#
#     # Start recursion from the root node
#     add_class_and_children(root)
#
#     # Save the OWL graph to a file
#     g.serialize(destination=output_file, format="turtle")
#     print(f"OWL file saved to {output_file}")


def load_tree(tree_address):
    # Load tree
    with open(tree_address, "r") as f:
        data = json.load(f)
    return json_graph.node_link_graph(data, directed=True)


# Example Usage
if __name__ == "__main__":
    adr = f"./tree/real-world/imagenet_directed"
    tree_address = f"{adr}.json"  # The tree in json format
    tree = load_tree(tree_address)  # imagenet: 1778 nodes and 1777 edges

    # Convert the tree to OWL
    root_node = '0'
    output_owl_file = "./real-world/imagenet_uri.owl"
    tree_to_owl(tree, root_node, output_owl_file)
