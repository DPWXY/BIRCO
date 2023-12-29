import json
import pickle
from tqdm import tqdm
import pickle
import argparse
import json
import os
import sys
import logging
import pathlib
import torch
import numpy as np
import re

def parse_score(output, k):
    if "SCORE" in output:
        score = float(output.split("SCORE: ")[1].split(" ")[0].split("\n")[0].split("/")[0])
    elif "Score" in output:
        score = float(output.split("Score: ")[1].split(" ")[0].split("\n")[0].split("/")[0])
    else:
        # print("No SCORE!, output : ")
        # print(output)
        score = 0
    return score

def get_req_rank_dict(gpt_anno, dataset):
    result_req = {}
    for k in dataset['query']:
        result_req[k] = {}
        for sub_k in dataset['qrel'][k]:
            result_req[k][sub_k] = []
    for k in gpt_anno:
        q = k[0]
        p = k[1]
        result_req[q][p].append(parse_score(gpt_anno[k], k))
    req_scores = {}
    for k in dataset['query']:
        req_scores[k] = {}

    for k in result_req:
        for sub_k in result_req[k]:
            req_scores[k][sub_k] = np.mean(result_req[k][sub_k])
    return req_scores

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)
    
def get_model_rank(model, path, dataset, dataset_name):
    get_all_e5_score = {}
    if path == None:
        with open(f"All_embedding_results/{dataset_name}_embedding_result/embedding_result_of_{model}_pretrain.pickle", 'rb') as f:
            e5_emb = pickle.load(f)
    else:
        with open(f"All_embedding_results/{dataset_name}_embedding_result/embedding_result_of_{model}_{path}.pickle", 'rb') as f:
            e5_emb = pickle.load(f)
    for k in dataset['query']:
        cand = dataset['qrel'][k]
        candidates_score = {}
        for sub_k in cand:
            candidates_score[sub_k] = cosine_similarity(e5_emb[k], e5_emb[sub_k])
        get_all_e5_score[k] = candidates_score
    return get_all_e5_score
    
def rerank_scores(original_scores, rerank_scores, random=False):

    # Group IDs by their score
    score_groups = {}
    for id, score in original_scores.items():
        score_groups.setdefault(score, []).append(id)

    # Sort the IDs within each score group based on the rerank scores
    new_scores_dict = {}
    for score, ids in score_groups.items():
        if len(ids) > 1:
            ids.sort(key=lambda id: rerank_scores[id], reverse=False)
            for new_rank, id in enumerate(ids):
                if not random:
                    new_scores_dict[id] = score + new_rank * 0.01
                else:
                    new_scores_dict[id] = score + 0.01 * np.random.random()
        else:
            new_scores_dict[ids[0]] = score
    new_rank_dict = dict(sorted(new_scores_dict.items(), key=lambda x: x[1], reverse=True))
    return new_rank_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-result_name", "--result_name", type = str, required = True, help = "the collected result")
    parser.add_argument("-model_name", "--model_name", type = str, required = True, help = "pretrained model, such as e5")
    parser.add_argument("-dataset_name", "--dataset_name", type = str, required = True, help = "Name of the dataset")
    parser.add_argument("-dataset_option", "--dataset_option", type = str, required = True, help = "should be one of [dev/test/all]")

    args = parser.parse_args()

    result_name = args.result_name
    model_name = args.model_name
    random = False
    dataset_name = args.dataset_name
    set_option = args.dataset_option

    with open(f"generation_data/output/{result_name}.pickle", 'rb') as f:
        generated_output = pickle.load(f)
    with open(f"./datasets/{dataset_name}/{set_option}_set.pickle", 'rb') as f:
        rank_dataset = pickle.load(f)

    rank_scores = get_req_rank_dict(generated_output, rank_dataset)
    model_ranks = get_model_rank(model_name, None, rank_dataset, dataset_name)
    rerank_result = {}
    for k in rank_scores:
        s = dict(sorted(rank_scores[k].items(), key=lambda x: x[1], reverse=True))
        rerank_result[k] = list(rerank_scores(s, model_ranks[k], random).keys())
        
    with open(f"./rankings/{dataset_name}/{set_option}_resaon_O_rank.pickle", 'wb') as f:
        pickle.dump(rerank_result, f)
        
    print("Finish...")