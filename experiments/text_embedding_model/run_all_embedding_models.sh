#!/bin/bash

# Define arrays of model names and dataset names
model_names=("e5" "roberta" "splade++" "spladev2" "simcse" "roberta")
dataset_names=("doris-mae" "arguana" "wtb" "clinical-trial" "relic")

# Loop over each model name
for model in "${model_names[@]}"; do
    # Loop over each dataset name
    for dataset in "${dataset_names[@]}"; do
        # Execute the command with the current combination of model and dataset
        python3 -u rank_by_embedding_model.py -cuda "0,1,2,3" -model_name "$model" -dataset_name "$dataset" -bs 4096
    done
done
