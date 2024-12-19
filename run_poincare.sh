#!/bin/bash

trees=("binomial" "full_rary_tree" "barabasi_albert_graph" "star_graph" "path_graph")
rs=("2" "3" "4" "5")
Ns=(256 512 1024)
dims=(10 20 130)

# Constant arguments
other_args="-lr 1 -epochs 10000 -negs 50 -burnin 20 -ndproc 4 -model distance -manifold poincare -batchsize 50 -eval_each 50 -fresh -sparse -train_threads 1 -gpu -1 -debug -dampening 0.75 -burnin_multiplier 0.01 -neg_multiplier 0.1 -lr_type constant -dampening 1.0"

# Loop over the datasets and run the Python script with the dynamic argument
for N in "${Ns[@]}"; do
    for dim in "${dims[@]}"; do
        for tree in "${trees[@]}"; do
            dim_arg="-dim ${dim}"
            if [ "$tree" == "full_rary_tree" ]; then
                for r in "${rs[@]}"; do
                    dset_arg="-dset ./tree/${tree}/${tree}${N}_r${r}_adjacency.csv"
                    checkpoint_arg="-checkpoint ./tree/${tree}/poincare/${tree}${N}_r${r}_dim${dim}.bin"
                    echo "Running Python script with argument: $dim_arg $dset_arg $checkpoint_arg $other_args"
                    python3 ./embed.py $dim_arg $dset_arg $checkpoint_arg $other_args
                    echo "______________________________________________________________"
                done

            else
                dset_arg="-dset ./tree/${tree}/${tree}${N}_adjacency.csv"
                checkpoint_arg="-checkpoint ./tree/${tree}/poincare/${tree}${N}_dim${dim}.bin"
                echo "Running Python script with argument: $dim_arg $dset_arg $checkpoint_arg $other_args"
                python3 ./embed.py $dim_arg $dset_arg $checkpoint_arg $other_args
                echo "______________________________________________________________"
            fi
        done
    done
done
