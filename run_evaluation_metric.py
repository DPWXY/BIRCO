import argparse
import sys
import numpy as np
import pickle
import os
from evaluation.rank_performance_util import *
from evaluation.model_embedding import load_model, get_embedding
import pdb
from evaluation_util import main, bootstrap_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-cuda", "--cuda", type = str, required = True, help = "list of gpus indices")
    parser.add_argument("-model_name", "--model_name", type = str, required = True, help = "pretrained model, such as e5")
    parser.add_argument("-dataset_name", "--dataset_name", type = str, required = True, help = "Name of the dataset")
    parser.add_argument("-dataset_option", "--dataset_option", type = str, required = True, help = "should be one of [dev/test/all]")
    parser.add_argument("-existing_rank", "--existing_rank", type = str, required = True, help = "the path to the ranking file (if exist), None otherwise")
    parser.add_argument("-boot", "--boot", type = str, required = False, default="True", help = "the path to the ranking file (if exist), None otherwise")

    args = parser.parse_args()
    
    cuda = args.cuda
    model_name = args.model_name
    dataset_name = args.dataset_name
    dataset_option = args.dataset_option
    rank_path = args.existing_rank
    boot= args.boot

    if dataset_option == "dev":
        with open(f"./datasets/{dataset_name}/dev_set.pickle", 'rb') as f:
            dataset = pickle.load(f)
    elif dataset_option == "test":
        with open(f"./datasets/{dataset_name}/test_set.pickle", 'rb') as f:
            dataset = pickle.load(f)
    elif dataset_option == "all":
        with open(f"./datasets/{dataset_name}/all_set.pickle", 'rb') as f:
            dataset = pickle.load(f)
    else:
        raise RuntimeError("Dataset Option is not [dev/test/all]")

    corpus = dataset['corpus']
    query = dataset['query']
    qrel = dataset['qrel']

    map, ndcg, recall_list, mrr, r_20 = main(corpus, query, qrel, dataset_name=dataset_name, dataset_option=dataset_option, model_name = model_name, cuda = cuda, rank_path = rank_path)

    if boot == "True":
        # map_mean, map_std = bootstrap_stats(map, num_of_trial=1000, random_seed=42)
        # print(f"MAP: {round(map_mean*100, 3)} +/- {round(map_std*100, 3)}")

        ndcg_mean, ndcg_std = bootstrap_stats(ndcg, num_of_trial=1000, random_seed=42)
        print(f"NDCG: {round(ndcg_mean*100, 3)} +/- {round(ndcg_std*100, 3)}")
    
        r_at_5_mean, r_at_5_std = bootstrap_stats(recall_list, num_of_trial=1000, random_seed=42)
        print(f"R@5: {round(r_at_5_mean*100, 3)} +/- {round(r_at_5_std*100, 3)}")
    else:
        print(f"NDCG: {round(np.mean(ndcg)*100, 3)}")
        print(f"R@5: {round(np.mean(recall_list)*100, 3)}")
        # print(f"MAP: {round(np.mean(map)*100, 3)}")
        # print(f"R@20: {round(np.mean(r_20)*100, 3)}")
        # print(f"mrr: {round(np.mean(mrr)*100, 3)}")

# python3 -u run_evaluation.py -cuda "0,1,2,3,4,5,6,7" -model_name e5 -dataset_name arguana -dataset_option test -existing_rank rankings/arguana/test_monot5.pickle
