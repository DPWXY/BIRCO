import argparse
import sys
import numpy as np
import pickle
import os
from model_embedding import load_model, get_embedding

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

    return ranked_ret

def get_final_embedding(text_list, cuda, my_model, batch_size):
    """
    This function should return the embedding of the text_list by the model.
    The output of the function should be a dictionary that uses the text as the key and the embedding vector as the value
    """
    model_name = my_model

    pre_encoded_text = text_list

    print(f"loading model: {model_name}....")

    model, tokenizer = load_model(model_name, cuda)

    embedding_dict = get_embedding(model_name, model, tokenizer, pre_encoded_text, cuda, batch_size)

    return embedding_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-cuda", "--cuda", type = str, required = True, help = "list of gpus indices")
    parser.add_argument("-model_name", "--model_name", type = str, required = True, help = "pretrained model, such as e5")
    parser.add_argument("-dataset_name", "--dataset_name", type = str, required = True, help = "Name of the dataset")
    parser.add_argument("-bs", "--bs", type = int, required = True, help = "batch size")

    args = parser.parse_args()
    
    cuda = args.cuda
    model_name = args.model_name
    dataset_name = args.dataset_name
    bs = args.bs

    with open(f"../../datasets/{dataset_name}/test_set.pickle", 'rb') as f:
        dataset = pickle.load(f)

    corpus = dataset['corpus']
    query = dataset['query']
    qrel = dataset['qrel']

    docs = []
    for ele in corpus:
        docs.append(corpus[ele])
    for ele in query:
        docs.append(query[ele])
    
    embeddings = get_final_embedding(docs, cuda, model_name, bs)
    docid2embedding = {}
    
    for doc_id in corpus:
        docid2embedding[doc_id] = embeddings[corpus[doc_id]]
    
    for doc_id in query:
        docid2embedding[doc_id] = embeddings[query[doc_id]]
    
    if "specter" not in model_name.lower():
        similarity_metric = "cos"
    else:
        similarity_metric = "l2"
    model_rank = rank_candidate_pool(qrel, docid2embedding, similarity_metric)

    with open(f"./text_embedding_model_results/{dataset_name}/{model_name}_rank.pickle", 'wb') as f:
        pickle.dump(model_rank, f)