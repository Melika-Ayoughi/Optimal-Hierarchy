#!/bin/bash

trees=("imagenet_withDAGs" "imagenet" "imagenet_reorganized")
#trees=("pizza_with_DAG" "pizza" "pizza_1lesshight" "pizza_2lesshight")
dims=(70)
#dims=(40)

# Constant arguments
other_args="-lr 1 -epochs 10000 -negs 50 -burnin 20 -ndproc 4 -model distance -manifold poincare -batchsize 50 -eval_each 50 -fresh -sparse -train_threads 1 -gpu -1 -debug -dampening 0.75 -burnin_multiplier 0.01 -neg_multiplier 0.1 -lr_type constant -dampening 1.0"

# Loop over the datasets and run the Python script with the dynamic argument
for dim in "${dims[@]}"; do
    for tree in "${trees[@]}"; do
        dim_arg="-dim ${dim}"
        dset_arg="-dset ./tree/real-world/${tree}_adjacency.csv"
        checkpoint_arg="-checkpoint ./tree/real-world/poincare/${tree}_dim${dim}.bin"
        echo "Running Python script with argument: $dim_arg $dset_arg $checkpoint_arg $other_args"
        python3 ./embed.py $dim_arg $dset_arg $checkpoint_arg $other_args
        echo "______________________________________________________________"

    done
done

