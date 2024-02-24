#!/bin/bash
# echo $$ > all_pid_file.txt

# Define variables
dataset_name=("doris-mae" "arguana" "wtb" "clinical-trial" "relic")
dataset_option="test"
prompt_names=("score_O" "reason_O" "subtask_O")
model_name="e5"
llm_names=("llama_7b" "llama_13b" "llama_70b" "StripedHyena_7B" "GPT-4")

echo "Starting the evaluation..."

for prompt_name in "${prompt_names[@]}"; do
    echo "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-="
    
    for llm_name in "${llm_names[@]}"; do
        echo "====================================================="
        
        for dataset in "${dataset_name[@]}"; do
            result_prefix="generation_data/output/${llm_name}_${dataset}_${dataset_option}_set_${prompt_name}"
            name="${llm_name}_${dataset}_${dataset_option}_set_${prompt_name}"
            rank_path="ranks/${dataset}/${llm_name}_${dataset_option}_rank_${prompt_name}.pickle"
            echo $prompt_name
            echo "$llm_name"
            echo "$dataset"
            echo "$rank_path"
            python3 rank2metric.py --dataset_name $dataset --rank_path $rank_path --boot "True"
            echo "----------------------------------------"
        done
        echo "====================================================="
    done
    echo "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-="
done

echo "FINISH ALL EVALUATION"