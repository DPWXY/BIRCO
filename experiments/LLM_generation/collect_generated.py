import pickle
import json
import numpy as np
import os
import random
import re
from tqdm import tqdm
from nltk import word_tokenize
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
import re
import argparse

# This file is used to collect results from ChatGPT (or other LLMs).
# Example code: python3 collect_generated.py -result_dir ./generation_result/generation_output -result_prefix relish_cluster_15_e5_triplet -name relish_cluster_15_e5_triplet


def find_files_with_prefix(folder_path, prefix):
    file_list = []
    
    for filename in os.listdir(folder_path):
        if filename.startswith(prefix):
            file_list.append(filename)
    
    return file_list

def get_all_generated_query(folder_path, folder_name):
    query_path = find_files_with_prefix(f"{folder_path}", f"{folder_name}")
    query_path_list = sorted([f"{folder_path}/{p}/" for p in query_path])
    generated_query = {}
    for p in tqdm(query_path_list):
        try:
            with open(f"{p}all_queries.pickle", 'rb') as f:
                part_query = pickle.load(f)
        except:
            print(f"No all_queries.pickle in path: {p}")
            continue
        for k in part_query:
            generated_query[k] = part_query[k]
    print(f"In total : {len(generated_query)} generated queries")
    return generated_query


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-result_dir', '--result_dir', required=True, help='path to the results')
    parser.add_argument('-result_prefix', '--result_prefix', required=True, help='prefix of the result dirs')
    parser.add_argument('-name', '--name', required=True, help='the name to store')
    

    args = parser.parse_args()
    folder_path = args.result_dir
    folder_prefix = args.result_prefix
    name = args.name

    generated_docs = get_all_generated_query(folder_path, folder_prefix)


    with open(f"./collected_result/{name}.pickle", 'wb') as f:
        pickle.dump(generated_docs, f)

    print(f"result stored at ./collected_result/{name}.pickle")