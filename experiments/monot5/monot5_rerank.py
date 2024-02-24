import argparse
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5

from transformers import T5ForConditionalGeneration
import pickle

model = T5ForConditionalGeneration.from_pretrained('castorini/monot5-3b-msmarco-10k')
reranker = MonoT5(model=model)

def rerank_one_query(dataset, query_id):
    corpus = dataset['corpus']
    query = dataset['query']
    qrel = dataset['qrel']

    query_text = query[query_id]
    query = Query(query_text)

    passages = []
    for cor in qrel[query_id]:
        passage = []
        passage.append(cor)
        passage.append(corpus[cor])
        passages.append(passage)
    texts = [ Text(p[1], {'docid': p[0]}, 0) for p in passages] # Note, pyserini scores don't matter since T5 will ignore them.

    reranked = reranker.rerank(query, texts)

    ret = [i.metadata['docid'] for i in reranked]

    return ret

def rerank_all_query(dataset):
    query = dataset['query']
    reranked_dict = {}
    count = 0
    for query_id in query:
        reranked = rerank_one_query(dataset, query_id)
        reranked_dict[query_id] = reranked
        count += 1
        print(f"finish running {count} queries")
    return reranked_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="created by LEI")
        
    
    parser.add_argument("-dataset_name", "--dataset_name", type = str, required = True, help = "Name of the dataset")
    parser.add_argument("-dataset_option", "--dataset_option", type = str, required = True, help = "should be one of [dev/test/all]")

    args = parser.parse_args()
    
    # cuda = args.cuda
    dataset_name = args.dataset_name
    dataset_option = args.dataset_option

    
    if dataset_option == "test":
        with open(f"../../../datasets/{dataset_name}/test_set.pickle", 'rb') as f:
            dataset = pickle.load(f)
    
    else:
        raise RuntimeError("Dataset Option is not [dev/test/all]")

    reranked_dict = rerank_all_query(dataset)

    with open(f"./{dataset_name}_{dataset_option}_monot5_rank.pickle", "wb") as f:
        pickle.dump(reranked_dict, f)

# To run this script, clone github repo from monot5
# please create a new env using requirements.txt
# Then run the following command: python3 monot5_rerank.py -dataset_name <dataset_name> -dataset_option test