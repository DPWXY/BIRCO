from src.modeling_enc_t5 import EncT5ForSequenceClassification
from src.tokenization_enc_t5 import EncT5Tokenizer
import torch
import torch.nn.functional as F
import numpy as np
import json
import pickle
import os
import argparse


parser = argparse.ArgumentParser(description='Process the dataset.')
parser.add_argument('dataset', type=str, help='The name of the dataset')
parser.add_argument('query_num', type=int, help='The number of queries in the dataset')

args = parser.parse_args()

dataset = args.dataset
query_num = args.query_num

eval_dict = {}
for i in range(query_num):
    with open(f"./reranked_results/{dataset}/result_queries_{i}.pickle", "rb") as f:
        result = pickle.load(f)
    assert len(result) == 1
    for ele in result:
        sorted_keys = sorted(result[ele], key=result[ele].get, reverse=True)  
        eval_dict[ele] = sorted_keys

with open(f"./rankings/{dataset}/test_tart_O.pickle", "wb") as f:
    pickle.dump(eval_dict, f)


eval_dict = {}
for i in range(query_num):
    with open(f"./reranked_results/{dataset}_empty/result_queries_{i}.pickle", "rb") as f:
        result = pickle.load(f)
    assert len(result) == 1
    for ele in result:
        sorted_keys = sorted(result[ele], key=result[ele].get, reverse=True)  
        eval_dict[ele] = sorted_keys

with open(f"./rankings/{dataset}/test_tart_empty.pickle", "wb") as f:
    pickle.dump(eval_dict, f)