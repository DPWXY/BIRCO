import argparse
import sys
import numpy as np
import pickle
import os
from util import main, bootstrap_stats

def test_run(dataset_name, boot, rank_path):
    with open(f"../datasets/{dataset_name}/test_set.pickle", 'rb') as f:
        dataset = pickle.load(f)

    corpus = dataset['corpus']
    query = dataset['query']
    qrel = dataset['qrel']

    recall_list, r_10, r_20, r_100, ndcg, mrr, _map = main(corpus, query, qrel, dataset_name, rank_path = rank_path)

    if boot.lower() == 'true':

        print("The bootstrap default number is 1000...")
    
        r_at_5_mean, r_at_5_std = bootstrap_stats(recall_list, num_of_trial=1000, random_seed=42)
        print(f"R@5: {round(r_at_5_mean*100, 3)} +/- {round(r_at_5_std*100, 3)}")

        r_at_10_mean, r_at_10_std = bootstrap_stats(r_10, num_of_trial=1000, random_seed=42)
        print(f"R@100: {round(r_at_10_mean*100, 3)} +/- {round(r_at_10_std*100, 3)}")

        r_at_20_mean, r_at_20_std = bootstrap_stats(r_20, num_of_trial=1000, random_seed=42)
        print(f"R@20: {round(r_at_20_mean*100, 3)} +/- {round(r_at_20_std*100, 3)}")\

        r_at_100_mean, r_at_100_std = bootstrap_stats(r_100, num_of_trial=1000, random_seed=42)
        print(f"R@100: {round(r_at_100_mean*100, 3)} +/- {round(r_at_100_std*100, 3)}")

        ndcg_mean, ndcg_std = bootstrap_stats(ndcg, num_of_trial=1000, random_seed=42)
        print(f"NDCG: {round(ndcg_mean*100, 3)} +/- {round(ndcg_std*100, 3)}")

        mrr_mean, mrr_std = bootstrap_stats(mrr, num_of_trial=1000, random_seed=42)
        print(f"mrr: {round(mrr_mean*100, 3)} +/- {round(mrr_std*100, 3)}")

        map_mean, map_std = bootstrap_stats(_map, num_of_trial=1000, random_seed=42)
        print(f"map: {round(map_mean*100, 3)} +/- {round(map_std*100, 3)}")

    else:
        print(f"R@5: {round(np.mean(recall_list)*100, 3)}")
        print(f"R@10: {round(np.mean(r_10)*100, 3)}")
        print(f"R@20: {round(np.mean(r_20)*100, 3)}")
        print(f"R@100: {round(np.mean(r_100)*100, 3)}")
        print(f"NDCG@10: {round(np.mean(ndcg)*100, 3)}")
        print(f"MRR@10: {round(np.mean(mrr)*100, 3)}")
        print(f"MAP: {round(np.mean(_map)*100, 3)}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-dataset_name", "--dataset_name", type = str, required = True, help = "Name of the test dataset")
    parser.add_argument("-boot", "--boot", type = str, required = True, help = "Bootstrap option")
    parser.add_argument("-rank_path", "--rank_path", type = str, required = True, help = "Path to the rank")

    args = parser.parse_args()
    dataset_name = args.dataset_name
    rank_path = args.rank_path
    boot = args.boot

    with open(f"../datasets/{dataset_name}/test_set.pickle", 'rb') as f:
        dataset = pickle.load(f)

    corpus = dataset['corpus']
    query = dataset['query']
    qrel = dataset['qrel']

    recall_list, r_10, r_20, r_100, ndcg, mrr, _map = main(corpus, query, qrel, dataset_name, rank_path = rank_path)

    if boot.lower() == 'true':

        print("The bootstrap default number is 1000...")
    
        r_at_5_mean, r_at_5_std = bootstrap_stats(recall_list, num_of_trial=1000, random_seed=42)
        print(f"R@5: {round(r_at_5_mean*100, 3)} +/- {round(r_at_5_std*100, 3)}")

        r_at_10_mean, r_at_10_std = bootstrap_stats(r_10, num_of_trial=1000, random_seed=42)
        print(f"R@100: {round(r_at_10_mean*100, 3)} +/- {round(r_at_10_std*100, 3)}")

        r_at_20_mean, r_at_20_std = bootstrap_stats(r_20, num_of_trial=1000, random_seed=42)
        print(f"R@20: {round(r_at_20_mean*100, 3)} +/- {round(r_at_20_std*100, 3)}")\

        r_at_100_mean, r_at_100_std = bootstrap_stats(r_100, num_of_trial=1000, random_seed=42)
        print(f"R@100: {round(r_at_100_mean*100, 3)} +/- {round(r_at_100_std*100, 3)}")

        ndcg_mean, ndcg_std = bootstrap_stats(ndcg, num_of_trial=1000, random_seed=42)
        print(f"NDCG: {round(ndcg_mean*100, 3)} +/- {round(ndcg_std*100, 3)}")

        mrr_mean, mrr_std = bootstrap_stats(mrr, num_of_trial=1000, random_seed=42)
        print(f"mrr: {round(mrr_mean*100, 3)} +/- {round(mrr_std*100, 3)}")

        map_mean, map_std = bootstrap_stats(_map, num_of_trial=1000, random_seed=42)
        print(f"map: {round(map_mean*100, 3)} +/- {round(map_std*100, 3)}")

    else:
        print(f"R@5: {round(np.mean(recall_list)*100, 3)}")
        print(f"R@10: {round(np.mean(r_10)*100, 3)}")
        print(f"R@20: {round(np.mean(r_20)*100, 3)}")
        print(f"R@100: {round(np.mean(r_100)*100, 3)}")
        print(f"NDCG@10: {round(np.mean(ndcg)*100, 3)}")
        print(f"MRR@10: {round(np.mean(mrr)*100, 3)}")
        print(f"MAP: {round(np.mean(_map)*100, 3)}")
        