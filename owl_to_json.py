from owlready2 import *
import json
import json

owl_file = "./real-world/pizza_original.owl"
json_file = "./real-world/pizza_with_DAG_hierarchy.json"


# default_world.close()
# default_world = World()  # Reinitialize the world
# Load ontology
onto = get_ontology(owl_file).load()

# Function to recursively build hierarchy
def build_hierarchy(cls):
    return {
        "name": cls.name,
        "subclasses": [build_hierarchy(sub) for sub in cls.subclasses()]
    }

# Start from the top-level class (Thing)
ontology_hierarchy = build_hierarchy(Thing)

# Save hierarchy as JSON
with open(json_file, "w") as json_file:
    json.dump(ontology_hierarchy, json_file, indent=4)


