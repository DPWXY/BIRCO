import numpy as np
import json
import pickle
import argparse


parser = argparse.ArgumentParser(description='Process the dataset.')
parser.add_argument('dataset', type=str, help='The name of the dataset')
args = parser.parse_args()

dataset = args.dataset

with open(f'../../datasets/{dataset}/test_set.pickle', 'rb') as f:
    test = pickle.load(f)

list_of_dict_query = []
for ele in test['query']:
    corpus_dict = {}
    corpus_dict['_id'] = ele
    corpus_dict['title'] = ''
    corpus_dict['text'] = test['query'][ele]
    corpus_dict['metadata'] = {}
    list_of_dict_query.append(corpus_dict)

list_of_dict_corpus = []
for ele in test['corpus']:
    corpus_dict = {}
    corpus_dict['_id'] = ele
    corpus_dict['title'] = ''
    corpus_dict['text'] = test['corpus'][ele]
    corpus_dict['metadata'] = {}
    list_of_dict_corpus.append(corpus_dict)

for i, ele in enumerate(list_of_dict_query):
    with open(f'./input_for_TART/{dataset}/queries_{i}.jsonl', 'w') as file:
        # Write each dictionary to the file
        record = ele
        json_record = json.dumps(record)
        file.write(json_record + '\n')

for i, q in enumerate(list_of_dict_query):
    with open(f'./input_for_TART/{dataset}/corpus_{i}.jsonl', 'w') as file:
        corpus_list = []
        for corpus in test['qrel'][q["_id"]]:
            corpus_list.append(corpus)
        record = [item for item in list_of_dict_corpus if item['_id'] in corpus_list]
        for re in record:
            json_record = json.dumps(re)
            file.write(json_record + '\n')

# Write data to TSV
with open(f"./input_for_TART/{dataset}/qrels.tsv", "w") as file:
    for query_id, inner_dict in test['qrel'].items():
        for corpus_id, score in inner_dict.items():
            score = int(score)
            file.write(f"{query_id}\t{corpus_id}\t{score}\n")