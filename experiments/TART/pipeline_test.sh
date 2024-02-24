#!/bin/bash

dataset="doris-mae"

python prepare_data.py $dataset

for i in {0..1}
do
    python generate_passage_embeddings.py --model_name_or_path facebook/contriever-msmarco --output_dir output/$dataset/$i --passages ./input_for_TART/$dataset/corpus_$i.jsonl --num_shards 1

    python eval_cross_task.py --passages ./input_for_TART/$dataset/corpus_$i.jsonl --passages_embeddings output/$dataset/$i/passages_* --qrels ./input_for_TART/$dataset/qrels.tsv --output_dir output --model_name_or_path facebook/contriever-msmarco --ce_model facebook/tart-full-flan-t5-xl --data ./input_for_TART/$dataset/queries_$i.jsonl --prompt  "The query consists of users’ needs, leading to several research questions that span a paragraph. Each candidate passage is an abstract from a scientific paper. The objective of this information retrieval task is to identify the abstract that most effectively meets the user's needs in the query." --dataset $dataset
done


dataset="doris-mae"
for i in {0..1}
do
    python eval_cross_task_empty.py --passages ./input_for_TART/$dataset/corpus_$i.jsonl --passages_embeddings output/$dataset/$i/passages_* --qrels ./input_for_TART/$dataset/qrels.tsv --output_dir output --model_name_or_path facebook/contriever-msmarco --ce_model facebook/tart-full-flan-t5-xl --data ./input_for_TART/$dataset/queries_$i.jsonl --prompt  "" --dataset $dataset
done

datasets=("doris-mae")

query_nums=(1)

length=${#datasets[@]}

for ((i=0; i<$length; i++)); do
    dataset=${datasets[$i]}
    query_num=${query_nums[$i]}
    
    python collect_data.py $dataset $query_num
done