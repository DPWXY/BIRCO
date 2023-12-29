#!/bin/bash
# For models already with ranking result
model_name=("score_O" "reason_O" "subtask_O" "monot5" "rank_llama" "rankgpt" "rankgpt_O" "tart" "tart_O" "e5" "simcse" "roberta" "spladev2" "splade++")
dataset_name=("doris-mae" "arguana" "wtb" "clinical-trial" "relic")
set_option="test"

for dataset in "${dataset_name[@]}"; do
    # Inner loop for actions
    for model in "${model_name[@]}"; do
        echo "Dataset: $dataset, Experiment: $model"
        python3 -u run_evaluation_metric.py -cuda "0,1,2,3,4,5,6,7" -model_name $model -dataset_name $dataset -dataset_option $set_option -existing_rank ./rankings/${dataset}/test_rank_${model}.pickle
        echo "================================================="
    done
    echo ""
    echo "================================================="

done

# Add model names for models without ranking result
model_name=()
# Models can calculate ranks directly
# model_name=("e5" "simcse" "roberta" "spladev2" "splade++")
dataset_name=("doris-mae" "arguana" "wtb" "clinical-trial" "relic")
set_option="test"

for dataset in "${dataset_name[@]}"; do
    # Inner loop for actions
    for model in "${model_name[@]}"; do
        echo "Dataset: $dataset, Experiment: $model"
        python3 -u run_evaluation_metric.py -cuda "0,1,2,3,4,5,6,7" -model_name $model -dataset_name $dataset -dataset_option $set_option -existing_rank None
        echo "================================================="
    done
    echo ""
    echo "================================================="

done



