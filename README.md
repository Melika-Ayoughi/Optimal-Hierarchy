# Designing Hierarchies for Optimal Hyperbolic Embedding

PyTorch implementation of *Designing Hierarchies for Optimal Hyperbolic Embedding* paper.

## Generating Trees
The trees that we used in this paper are all in the `./tree` folder. However, if you want to generate and visualize them yourself, you can run `evaluate.py`:

```sh
tree_types = ["binomial", "full_rary_tree", "barabasi_albert_graph", "star_graph", "path_graph"]
rs = [2, 3, 4, 5]
Ns = [256, 512, 1024]    
dims = [10, 20, 130]

# Generate and visualize trees
generate_trees_and_adjacency(tree_types, rs, Ns)
```

## Ontology Data and Hyperbolic Embeddings
You can find the real-world ontology that we are using under the `./real-world/` directory. You can also find the final best embedding of each method under `./tree/{tree_name}/{method_name}/`.

## Example: Embedding Generated Trees
To get the Poincare embeddings of the generated trees (Binomial, rary, and Barabasi-Albert), run:

```sh
bash run_poincare.sh
```

## Example: Embedding Real-world Trees
To get the Poincare embeddings of the real-world trees (ImageNet and Pizza ontology), run:

```sh
bash run_poincare_realworld.sh
```

## Evaluation
To get the Average distortion, Worst-case distortion, and MAP of the hyperbolic embeddings, run:

```sh
python evaluate.py
```

This calls the `evaluate_tree(tree, embeddings)` function.

## Dependencies
You can find the dependencies of our project in:
- `requirements_optimal_hierarchy.txt` (for the main project)
- `requirements_poincare.txt` (for the Poincare code)

## License
This code is licensed under [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

![License Badge](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)
