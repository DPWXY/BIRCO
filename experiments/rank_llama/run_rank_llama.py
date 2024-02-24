import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig
import time
from tqdm import tqdm
import pickle
import argparse
import json
import pdb



def get_model(peft_model_name):
    config = PeftConfig.from_pretrained(peft_model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, num_labels=1)
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    model.eval()
    return model

def rank_keys_by_values(input_dict):
    # Sort the dictionary by its values in descending order
    sorted_items = sorted(input_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Extract the keys from the sorted items
    ranked_keys = [item[0] for item in sorted_items]
    
    return ranked_keys

def call_rank_llama(query_text, candidate_text):
    query = query_text
    passage = candidate_text
    # Tokenize the query-passage pair
    
    inputs = tokenizer(f'query: {query}', f'document: {passage}', return_tensors='pt')
    # Run the model forward
    with torch.no_grad():
    
        outputs = model(**inputs)
        logits = outputs.logits
        score = logits[0][0]
    return score
        # print(score)
        # print(time.time() - curr_time)

# Load the tokenizer and model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument("-dataset_name", "--dataset_name", type = str, required = True, help = "Name of the dataset")
    



    args = parser.parse_args()
    dataset_name = args.dataset_name


    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    model = get_model('castorini/rankllama-v1-7b-lora-passage')



    with open(f"../../datasets/{dataset_name}/test_set.pickle", 'rb') as f:
        dataset = pickle.load(f)

    query = dataset['query']
    corpus = dataset['corpus']
    qrel = dataset['qrel']

    pairs = {}

    for qid in qrel:
        for cid in qrel[qid]:
            pairs[(qid, cid)] = {'query_text': query[qid], 'candidate_text': corpus[cid]}



    result_dict = {}

    for k in tqdm(pairs):
        curr_q_text = pairs[k]['query_text']
        curr_c_text = pairs[k]['candidate_text']
        curr_rankllama_result = call_rank_llama(curr_q_text, curr_c_text)
        # pdb.set_trace()
        result_dict[k] = float(curr_rankllama_result)

    with open(f"./rank_llama_result/{dataset_name}_query_candiate_scores.pickle", 'wb') as f:
        pickle.dump(result_dict, f)

    print("Pickle Stored")

    score_dict = {}
    for k in result_dict:
        qid = k[0]
        cid = k[1]
        if qid not in score_dict:
            score_dict[qid] = {}
        score_dict[qid][cid] = float(result_dict[k])

    rank = {}
    for qid in score_dict:
        rank[qid] = rank_keys_by_values(score_dict[qid])

    with open(f"./rank_llama_result/{dataset_name}_rank_llama_rank.pickle", 'wb') as f:
        json.dump(rank, f)
