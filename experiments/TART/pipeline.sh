#!/bin/bash

dataset="doris-mae"

python prepare_data.py $dataset

for i in {0..59}
do
    python generate_passage_embeddings.py --model_name_or_path facebook/contriever-msmarco --output_dir output/$dataset/$i --passages ./input_for_TART/$dataset/corpus_$i.jsonl --num_shards 1

    python eval_cross_task.py --passages ./input_for_TART/$dataset/corpus_$i.jsonl --passages_embeddings output/$dataset/$i/passages_* --qrels ./input_for_TART/$dataset/qrels.tsv --output_dir output --model_name_or_path facebook/contriever-msmarco --ce_model facebook/tart-full-flan-t5-xl --data ./input_for_TART/$dataset/queries_$i.jsonl --prompt  "The query consists of users’ needs, leading to several research questions that span a paragraph. Each candidate passage is an abstract from a scientific paper. The objective of this information retrieval task is to identify the abstract that most effectively meets the user's needs in the query." --dataset $dataset
done

dataset="arguana"

python prepare_dataset.py $dataset

for i in {0..99}
do
    python generate_passage_embeddings.py --model_name_or_path facebook/contriever-msmarco --output_dir output/$dataset/$i --passages ./input_for_TART/$dataset/corpus_$i.jsonl --num_shards 1

    python eval_cross_task.py --passages ./input_for_TART/$dataset/corpus_$i.jsonl --passages_embeddings output/$dataset/$i/passages_* --qrels ./input_for_TART/$dataset/qrels.tsv --output_dir output --model_name_or_path facebook/contriever-msmarco --ce_model facebook/tart-full-flan-t5-xl --data ./input_for_TART/$dataset/queries_$i.jsonl --prompt  "This information retrieval (IR) task has a debate format where a topic is given, and two directly opposing sides of arguments about this topic are formed.  A query is an argument that takes one side of this topic, focuses on a particular point about this topic, and takes a stance (i.e., opinion, position, view, perspective) about this particular point. A passage is an argument that takes the opposing side of the same topic, focuses on the same particular point about the same topic, and takes a directly opposing stance that directly (i.e., no implying or inferring) refutes and attacks the query’s stance regarding this particular point. Both query and passage might have citations in them but these citations should not be considered in the scope of this task. The overall goal of this specific information retrieval IR task is to identify the central topic of the debate, to articulate the query’s stance, and to find the passage that takes the opposing stance." --dataset $dataset
done

dataset="wtb"

python prepare_dataset.py $dataset

for i in {0..99}
do
    python generate_passage_embeddings.py --model_name_or_path facebook/contriever-msmarco --output_dir output/$dataset/$i --passages ./input_for_TART/$dataset/corpus_$i.jsonl --num_shards 1

    python eval_cross_task.py --passages ./input_for_TART/$dataset/corpus_$i.jsonl --passages_embeddings output/$dataset/$i/passages_* --qrels ./input_for_TART/$dataset/qrels.tsv --output_dir output --model_name_or_path facebook/contriever-msmarco --ce_model facebook/tart-full-flan-t5-xl --data ./input_for_TART/$dataset/queries_$i.jsonl --prompt  "The query consists of users’ needs, leading to several research questions that span a paragraph. Each candidate passage is an abstract from a scientific paper. The objective of this information retrieval task is to identify the abstract that most effectively meets the user's needs in the query." --dataset $dataset
done

dataset="clinic-trial"

python prepare_dataset.py $dataset

for i in {0..49}
do
    python generate_passage_embeddings.py --model_name_or_path facebook/contriever-msmarco --output_dir output/$dataset/$i --passages ./input_for_TART/$dataset/corpus_$i.jsonl --num_shards 1

    python eval_cross_task.py --passages ./input_for_TART/$dataset/corpus_$i.jsonl --passages_embeddings output/$dataset/$i/passages_* --qrels ./input_for_TART/$dataset/qrels.tsv --output_dir output --model_name_or_path facebook/contriever-msmarco --ce_model facebook/tart-full-flan-t5-xl --data ./input_for_TART/$dataset/queries_$i.jsonl --prompt  "The query consists of users’ needs, leading to several research questions that span a paragraph. Each candidate passage is an abstract from a scientific paper. The objective of this information retrieval task is to identify the abstract that most effectively meets the user's needs in the query." --dataset $dataset
done

dataset="relic"

python prepare_dataset.py $dataset

for i in {0..99}
do
    python generate_passage_embeddings.py --model_name_or_path facebook/contriever-msmarco --output_dir output/$dataset/$i --passages ./input_for_TART/$dataset/corpus_$i.jsonl --num_shards 1

    python eval_cross_task.py --passages ./input_for_TART/$dataset/corpus_$i.jsonl --passages_embeddings output/$dataset/$i/passages_* --qrels ./input_for_TART/$dataset/qrels.tsv --output_dir output --model_name_or_path facebook/contriever-msmarco --ce_model facebook/tart-full-flan-t5-xl --data ./input_for_TART/$dataset/queries_$i.jsonl --prompt  "The query consists of users’ needs, leading to several research questions that span a paragraph. Each candidate passage is an abstract from a scientific paper. The objective of this information retrieval task is to identify the abstract that most effectively meets the user's needs in the query." --dataset $dataset
done

dataset="doris-mae"
for i in {0..59}
do
    python eval_cross_task_empty.py --passages ./input_for_TART/$dataset/corpus_$i.jsonl --passages_embeddings output/$dataset/$i/passages_* --qrels ./input_for_TART/$dataset/qrels.tsv --output_dir output --model_name_or_path facebook/contriever-msmarco --ce_model facebook/tart-full-flan-t5-xl --data ./input_for_TART/$dataset/queries_$i.jsonl --prompt  "" --dataset $dataset
done

dataset="arguana"
for i in {0..99}
do
    python eval_cross_task_empty.py --passages ./input_for_TART/$dataset/corpus_$i.jsonl --passages_embeddings output/$dataset/$i/passages_* --qrels ./input_for_TART/$dataset/qrels.tsv --output_dir output --model_name_or_path facebook/contriever-msmarco --ce_model facebook/tart-full-flan-t5-xl --data ./input_for_TART/$dataset/queries_$i.jsonl --prompt  "" --dataset $dataset
done

dataset="wtb"
for i in {0..99}
do
    python eval_cross_task_empty.py --passages ./input_for_TART/$dataset/corpus_$i.jsonl --passages_embeddings output/$dataset/$i/passages_* --qrels ./input_for_TART/$dataset/qrels.tsv --output_dir output --model_name_or_path facebook/contriever-msmarco --ce_model facebook/tart-full-flan-t5-xl --data ./input_for_TART/$dataset/queries_$i.jsonl --prompt  "" --dataset $dataset
done

dataset="clinic-trial"
for i in {0..49}
do
    python eval_cross_task_empty.py --passages ./input_for_TART/$dataset/corpus_$i.jsonl --passages_embeddings output/$dataset/$i/passages_* --qrels ./input_for_TART/$dataset/qrels.tsv --output_dir output --model_name_or_path facebook/contriever-msmarco --ce_model facebook/tart-full-flan-t5-xl --data ./input_for_TART/$dataset/queries_$i.jsonl --prompt  "" --dataset $dataset
done

dataset="relic"
for i in {0..99}
do
    python eval_cross_task_empty.py --passages ./input_for_TART/$dataset/corpus_$i.jsonl --passages_embeddings output/$dataset/$i/passages_* --qrels ./input_for_TART/$dataset/qrels.tsv --output_dir output --model_name_or_path facebook/contriever-msmarco --ce_model facebook/tart-full-flan-t5-xl --data ./input_for_TART/$dataset/queries_$i.jsonl --prompt  "" --dataset $dataset
done

datasets=("doris-mae" "arguana" "wtb" "clinic-trial" "relic")

query_nums=(60 100 100 50 100)

length=${#datasets[@]}

for ((i=0; i<$length; i++)); do
    dataset=${datasets[$i]}
    query_num=${query_nums[$i]}
    
    python collect_data.py $dataset $query_num
done