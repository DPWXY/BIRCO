#!/bin/bash
echo $$ > score_pid_file.txt

# Define variables
dataset_name=("arguana" "doris-mae" "wtb" "clinical-trial" "relic")
dataset_option="test"
prompt_name="score_O"
model_name="e5"

result_prefix="generation_data/output/${dataset_name}_${dataset_option}_set_${prompt_name}"
name="${dataset_name}_${dataset_option}_set_${prompt_name}"

echo "Collecting result"
for dataset in "${dataset_name[@]}"; do
    python3 annotation_to_rank_score.py -result_name $name -model_name e5 -dataset_name $dataset_name -dataset_option $dataset_option 
done

echo "Done!"