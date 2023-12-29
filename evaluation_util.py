import argparse
import sys
import numpy as np
import pickle
import os
from evaluation.rank_performance_util import *
from evaluation.model_embedding import load_model, get_embedding
import pdb


# Precision at k: It measures the proportion of recommended items in the top-k set that are relevant
def precision_at_k(relevance_list, ranking_result, k=20):
    """
    Parameters:
        relevance_list (list): List of relevant document ids.
        ranking_result (list): List that shows the ranking of the documents.
        k (int): The 'k' in 'P@k'. Default is 20.

    Returns:
        float: The precision at k of the ranking result.
    """
    ranking_result = ranking_result[:k]
    relevance_set = set(relevance_list)
    ranking_set = set(ranking_result)
    inter_size = len(relevance_set.intersection(ranking_set))
    return inter_size / k


# Recall at k: It measures the proportion of relevant items found in the top-k recommendations
def recall_at_k(relevance_list, ranking_result, k=20):
    """
    Parameters:
        relevance_list (list): List of relevant document ids.
        ranking_result (list): List that shows the ranking of the documents.
        k (int): The 'k' in 'R@k'. Default is 20.

    Returns:
        float: The recall at k of the ranking result.
    """
    ranking_result = ranking_result[:k]
    relevance_set = set(relevance_list)
    ranking_set = set(ranking_result)

    inter_size = len(relevance_set.intersection(ranking_set))

    # return k/len(relevance_list)
    return inter_size / len(relevance_list)


def r_precision(relevance_list, ranking_result):
    """
    Parameters:
        relevance_list (list): List of relevant document ids.
        ranking_result (list): List that shows the ranking of the documents.

    Returns:
        float: R-precision of the ranking result.
    """
    relevance_len = len(relevance_list)
    ranking_result = ranking_result[:relevance_len]
    relevance_set = set(relevance_list)
    ranking_set = set(ranking_result)

    inter_size = len(relevance_set.intersection(ranking_set))
    return inter_size / relevance_len


# Normalized Discounted Cumulative Gain (NDCG): A measure of ranking quality. It uses the graded relevance of a query result set and discounts the relevance of documents lower down in the result list.
def ndcg_normal(ground_truth_ranking, ground_truth_score, ranking_result, p):
    """
    Parameters:
        ground_truth_ranking (list): List that shows the correct ranking of the documents.
        ground_truth_score (list): List that shows the correct score of the documents.
        ranking_result (list): List that shows the ranking of the documents.
        p (float): A float between 0 and 1

    Returns:
        float: NDCG of the ranking result given the ground truth ranking and score.
    """

    consider_length = 10
    dcg = 0
    for ind in range(consider_length):
        curr_id = ranking_result[ind]
        pos_in_ground_truth = ground_truth_ranking.index(curr_id)
        rel_i = ground_truth_score[pos_in_ground_truth]
        dcg += rel_i / math.log2(ind + 2)
    idcg = 0
    for ind in range(consider_length):
        rel_i = ground_truth_score[ind]
        idcg += rel_i / math.log2(ind + 2)
    return dcg / idcg


# Alternate version of NDCG: Similar to NDCG, but this version uses a exponential gain to emphasise the importance of relevance.
def ndcg_exp(ground_truth_ranking, ground_truth_score, ranking_result, p):
    """
    Parameters:
        ground_truth_ranking (list): List that shows the correct ranking of the documents.
        ground_truth_score (list): List that shows the correct score of the documents.
        ranking_result (list): List that shows the ranking of the documents.
        p (float): A float between 0 and 1

    Returns:
        float: NDCG (exponential) of the ranking result given the ground truth ranking and score.
    """
    consider_length = 10
    dcg = 0
    for ind in range(consider_length):
        curr_id = ranking_result[ind]
        pos_in_ground_truth = ground_truth_ranking.index(curr_id)
        rel_i = ground_truth_score[pos_in_ground_truth]
        # print(rel_i)
        dcg += (2 ** rel_i) / math.log2(ind + 2)
        # dcg += rel_i/math.log2(ind + 2)
    idcg = 0
    for ind in range(consider_length):
        rel_i = ground_truth_score[ind]
        idcg += (2 ** rel_i) / math.log2(ind + 2)
        # idcg += rel_i / math.log2(ind + 2)
    return dcg / idcg


# Reciprocal Rank: The reciprocal of the rank of the first relevant document
def reciprocal_rank(ground_truth_rank, ground_truth_score, ranking_result, k):
    """
    Parameters:
        ground_truth_rank (list): List that shows the correct ranking of the documents.
        ground_truth_score (list): List that shows the correct score of the documents.
        ranking_result (list): List that shows the ranking of the documents.
        k (int): An integer between 1 and the length of the ranking result

    Returns:
        float: Reciprocal rank of the ranking result.
    """

    visible_ranking_result = ranking_result[:k]
    max_score = ground_truth_score[0]
    for ind in range(len(ground_truth_rank)):
        if ground_truth_score[ind] < max_score:
            break
    relevance_docs = ground_truth_rank[:ind]
    for i in range(len(visible_ranking_result)):
        if visible_ranking_result[i] in relevance_docs:
            return 1 / (i + 1)
    return 0


def average_precision(relevance_list, ranking_result):
    """
    Parameters:
        relevance_list (list): List of relevant document ids.
        ranking_result (list): List that shows the ranking of the documents.

    Returns:
        float: R-precision of the ranking result.
    """
    relevant_docs = set(relevance_list)
    num_relevant_docs = len(relevant_docs)
    if num_relevant_docs == 0:
        return 0.0
    cum_sum_precisions = 0.0
    num_hits = 0
    for i, doc_id in enumerate(ranking_result):
        if doc_id in relevant_docs:
            num_hits += 1
            precision_at_i = num_hits / (i + 1.0)
            cum_sum_precisions += precision_at_i
    ret = cum_sum_precisions / num_relevant_docs
    return ret


def cosine_similarity(u, vectors):
    """
    Calculate the cosine similarity between a single vector u and a list of vectors.

    Parameters:
    u (np.ndarray): A 1-D array.
    vectors (list of np.ndarray): A list of 1-D arrays.

    Returns:
    np.ndarray: An array containing the cosine similarity between u and each vector in vectors.
    """
    # Normalize the input vector
    u_norm = u / np.linalg.norm(u)
    
    # Stack all the vectors in the list to create a 2D array
    vectors_stack = np.vstack(vectors)
    
    # Normalize the vectors along the rows (axis=1)
    vectors_norm = vectors_stack / np.linalg.norm(vectors_stack, axis=1, keepdims=True)
    
    # Calculate the dot product
    similarity = np.dot(vectors_norm, u_norm)
    
    return similarity

def calculate_l2_distances(ref_array, array_list):
    """
    Calculate the L2 distances between a reference array and a list of arrays.
    
    :param ref_array: A NumPy array of shape (n_features,)
    :param array_list: A list of NumPy arrays of shape (N, n_features)
    :return: A NumPy array of L2 distances of shape (N,)
    """
    # Stack the list of arrays to create a 2D NumPy array
    arrays_stack = np.stack(array_list)
    
    # Subtract the reference array from every array in the list and square the differences
    diff_squared = np.square(arrays_stack - ref_array)
    
    # Sum the squared differences along the features axis and take the square root
    distances = -np.sqrt(np.sum(diff_squared, axis=1))
    
    return distances

def evaluate_performance(model_rank, q_rels, relevance_threshold=1):
    average_p_list = []
    for doc_id in q_rels:
        curr_relevant_list = []
        for cand_id in q_rels[doc_id]:
            if q_rels[doc_id][cand_id] >= relevance_threshold:
                curr_relevant_list.append(cand_id)
        if len(curr_relevant_list) == 0:
            continue
        curr_model_rank = model_rank[doc_id]

        average_p_list.append(average_precision(curr_relevant_list, curr_model_rank))

    MAP = average_p_list

    ndcg_list = []
    # ndcg_exp_list = []
    for doc_id in q_rels:
        curr_candidate_and_score = q_rels[doc_id]
        sorted_candidates = dict(sorted(curr_candidate_and_score.items(), key=lambda item: item[1], reverse=True))
        curr_ground_truth_ranking = list(sorted_candidates.keys())
        curr_ground_truth_score = list(sorted_candidates.values())

        # print(curr_ground_truth_score)

        curr_model_rank = model_rank[doc_id]

        ndcg_list.append(ndcg_normal(curr_ground_truth_ranking, curr_ground_truth_score, curr_model_rank, 10))
        # ndcg_exp_list.append(ndcg_exp(curr_ground_truth_ranking, curr_ground_truth_score, curr_model_rank, 0.1))

    
    

    recall_list = []
    for doc_id in q_rels:
        curr_relevant_list = []
        for cand_id in q_rels[doc_id]:
            if q_rels[doc_id][cand_id] >= relevance_threshold:
                curr_relevant_list.append(cand_id)
        if len(curr_relevant_list) == 0:
            continue
        curr_model_rank = model_rank[doc_id]

        recall_list.append(recall_at_k(curr_relevant_list, curr_model_rank, k=5))
        
    r_20 = []
    for doc_id in q_rels:
        curr_relevant_list = []
        for cand_id in q_rels[doc_id]:
            if q_rels[doc_id][cand_id] >= relevance_threshold:
                curr_relevant_list.append(cand_id)
        if len(curr_relevant_list) == 0:
            continue
        curr_model_rank = model_rank[doc_id]

        r_20.append(recall_at_k(curr_relevant_list, curr_model_rank, k=20))

    mrr = []
    # ndcg_exp_list = []
    for doc_id in q_rels:
        curr_candidate_and_score = q_rels[doc_id]
        sorted_candidates = dict(sorted(curr_candidate_and_score.items(), key=lambda item: item[1], reverse=True))
        curr_ground_truth_ranking = list(sorted_candidates.keys())
        curr_ground_truth_score = list(sorted_candidates.values())

        # print(curr_ground_truth_score)

        curr_model_rank = model_rank[doc_id]

        mrr.append(reciprocal_rank(curr_ground_truth_ranking, curr_ground_truth_score, curr_model_rank, 10))

    return MAP, ndcg_list, recall_list, mrr, r_20

# This is the function that returns a dict of ranked candidates. The inputs are qrel (from load_dataset_for_validation for load_dataset_for_validation_triplets); docid2embedding is a dict that uses doc_id as key and their corresponding embedding as value; similarity_metric is a string that is either 'cosine' or 'l2'; dataset is relish or other datasets (implemented) from scirepeval.
def rank_candidate_pool(qrel, docid2embedding, similarity_metric):
    ranked_ret = {}
    for doc_id in qrel:
        query_embedding = docid2embedding[doc_id]
        candidate_embeddings = [docid2embedding[candidate_id] for candidate_id in qrel[doc_id]]
        similarity_list = []
        if similarity_metric == "cos":
            similarity_list = cosine_similarity(query_embedding, candidate_embeddings)
        elif similarity_metric == "l2":
            similarity_list = calculate_l2_distances(query_embedding, candidate_embeddings)
        else:
            raise ValueError("invalid similarity metric")
        similarity_dict = {}
        for ind, candidate_id in enumerate(qrel[doc_id]):
            similarity_dict[candidate_id] = similarity_list[ind]
        sorted_dict = dict(sorted(similarity_dict.items(), key=lambda item: item[1], reverse = True))
        ranked_ret[doc_id] = list(sorted_dict.keys())

    # with open("./All_embedding_results/relic_embedding_result/embedding_result_of_e5_pretrain.pickle",'wb') as f:
    #     pickle.dump(ranked_ret, f)
    # print(f"Stored at ./All_embedding_results/wtb_embedding_result/embedding_result_of_e5_pretrain.pickle")

    return ranked_ret

def get_final_embedding(text_list, cuda, my_model, batch_size, model_path):
    """
    This function should return the embedding of the text_list by the model.
    The output of the function should be a dictionary that uses the text as the key and the embedding vector as the value
    """
    model_name = my_model

    pre_encoded_text = text_list

    print(f"loading model: {model_name}....")

    model, tokenizer = load_model(model_name, cuda, model_path)

    embedding_dict = get_embedding(model_name, model, tokenizer, pre_encoded_text, cuda, batch_size)

    return embedding_dict

def main(corpus, query, qrel, model_name, cuda, dataset_name, dataset_option, rank_path = None):

    docs = []
    for ele in corpus:
        docs.append(corpus[ele])
    for ele in query:
        docs.append(query[ele])
        
    model_path = None

    if rank_path.lower() == "none":

        rank_path = f"./rankings/{dataset_name}/{dataset_option}_{model_name}_rank_none.pickle"
        if os.path.exists(rank_path):
            print(f"The file at {rank_path} exists.")
        else:
            print(f"The file at {rank_path} does not exist.")

        
            embeddings = get_final_embedding(docs, cuda, model_name, 10240, model_path = model_path)
            docid2embedding = {}

            for doc_id in corpus:
                docid2embedding[doc_id] = embeddings[corpus[doc_id]]

            for doc_id in query:
                docid2embedding[doc_id] = embeddings[query[doc_id]]
    else:
        # DO NOT need to do embedding under this scenerio
        embeddings = None
        docid2embedding = {}

    if "specter" not in model_name.lower():
        similarity_metric = "cos"
    else:
        similarity_metric = "l2"

    if not os.path.exists(rank_path):

        model_rank = rank_candidate_pool(qrel, docid2embedding, similarity_metric)

        with open(rank_path, 'wb') as f:
            pickle.dump(model_rank, f)
        print(f"Stored to {rank_path}")
    else:
        
        print(f"Loading model_rank from {rank_path}")
    
        with open(rank_path, 'rb') as f:
            model_rank = pickle.load(f)
            
    relevance_threshold = 1
            
    map, ndcg, recall_list, mrr, r_20 = evaluate_performance(model_rank, qrel, relevance_threshold = relevance_threshold)

    return map, ndcg, recall_list, mrr, r_20


def bootstrap_stats(result_list, num_of_trial=1000, random_seed=42):
    np.random.seed(random_seed)
    bs_result = []
    for ind in range(num_of_trial):
        curr_sample_ind = list(np.random.choice(range(len(result_list)), len(result_list), replace=True))
        curr_sample = [result_list[i] for i in curr_sample_ind]
        bs_result.append(np.mean(curr_sample))

    return np.mean(bs_result), np.std(bs_result)



# python3 -u evaluation_util.py -cuda "0,1,2,3,4,5,6,7" -model_name e5 -dataset_name wtb -dataset_option test