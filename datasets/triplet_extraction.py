import pickle
import numpy as np
from tqdm import tqdm

all_dataset_names = ["arguana", "wtb", "relic", "clinical-trial", "doris-mae"]

for dataset_name in all_dataset_names:
    with open(f"{dataset_name}/dev_set.pickle", 'rb') as f:
        dev_dataset = pickle.load(f)
    dev_qrel = dev_dataset['qrel']
    
    if dataset_name in ["arguana", "wtb", "relic"]:
        create_triplet = []
        for k in tqdm(dev_qrel):
            k_qrels = dev_qrel[k]
            rel_key = [key for key, value in k_qrels.items() if value != 0][0]
            pairs = [(rel_key, key) for key in k_qrels if k_qrels[key] == 0]
            for p in pairs:
                create_triplet.append((k, p[0], p[1]))
                
    elif dataset_name in ["clinical-trial"]:
        create_triplet = []
        for k in tqdm(dev_qrel):
            k_qrels = dev_qrel[k]
            key_with_1 = [key for key, value in k_qrels.items() if value == 1]
            key_with_2 = [key for key, value in k_qrels.items() if value == 2]
            key_with_0 = [key for key, value in k_qrels.items() if value == 0]
            for k_2 in key_with_2:
                for k_1 in key_with_1:
                    create_triplet.append((k, k_2, k_1))
                for k_0 in key_with_0:
                    create_triplet.append((k, k_2, k_0))
            for k_1 in key_with_1:
                for k_0 in key_with_0:
                    create_triplet.append((k, k_1, k_0))
                    
    elif dataset_name in ["doris-mae"]:
        create_triplet = []
        threshold = 0.25
        for k in tqdm(dev_qrel):
            k_qrels = dev_qrel[k]
            sorted_qrels = dict(sorted(k_qrels.items(), key=lambda x: x[1], reverse=True))
            highest_key = list(sorted_qrels.keys())[0]
            highest = sorted_qrels[highest_key]
            half = highest / 2
            for each_k in sorted_qrels:
                score = sorted_qrels[each_k]
                if score <= half:
                    break
                range_start = score - threshold
                for sub_k in sorted_qrels:
                    if half <= sorted_qrels[sub_k] <= range_start:
                        create_triplet.append((k, each_k, sub_k))
    
    else:
        raise ValueError("Dataset not defined")
    
    print(f"The total length of the created triplets for {dataset_name} is {len(create_triplet)}")
    with open(f"dev_triplets/{dataset_name}_dev_triplets.pickle", 'wb') as f:
        pickle.dump(create_triplet, f)