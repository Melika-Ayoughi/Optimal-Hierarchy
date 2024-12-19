from rdflib import Graph, RDFS, OWL, Namespace
from rdflib import Graph, RDFS, OWL, Literal
import json


def build_hierarchy(graph, cls):
    """
    Recursively build a JSON hierarchy from an OWL class.
    """
    label = graph.value(cls, RDFS.label) or cls.split("#")[-1]
    subclasses = [
        build_hierarchy(graph, sub)
        for sub in graph.subjects(RDFS.subClassOf, cls)
    ]
    return {
        "name": str(label),
        "subclasses": subclasses
    }


def owl_to_json(owl_file):
    """
    Convert an OWL ontology to a JSON hierarchy using rdflib.
    """
    graph = Graph()
    graph.parse(owl_file, format="turtle")

    # Namespace for OWL terms
    owl_ns = Namespace("http://www.w3.org/2002/07/owl#")
    thing = owl_ns.Thing  # `owl:Thing` as the starting point

    hierarchy = build_hierarchy(graph, thing)
    return json.dumps(hierarchy, indent=4)


owl_file = "./real-world/imagenet_reorganized.owl"
json_file = "./real-world/imagenet_reorganized_hierarchy.json"

# Load ontology
json_hierarchy = owl_to_json(owl_file)

# Save to a file
with open(json_file, "w") as json_file:
    json_file.write(json_hierarchy)

