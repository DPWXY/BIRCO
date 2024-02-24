import argparse
import openai
import re
import time
import pickle
import threading
import random
import pdb


openai.api_key = 'replace-with-your-api-key'


def gpt_permute(input_conversation, temperature=0, pp=0, max_tokens=100):
    """
    Input: The prompt temperature (default 0) and presence penalty (default 0)
    Output: A list of dict that contains the user prompt and ChatGPT's answer to the prompt
    """
    try:
        # conversation = [{'role': 'user', 'content': prompt}]
        conversation = input_conversation
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=conversation,
            temperature=temperature,
            request_timeout=90,
            presence_penalty=pp,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0)
        content = response.choices[0]['message']['content']
        # Remove the first sentence of the answer
        role = response.choices[0]['message']['role']
        conversation.append({'role': role, 'content': content})
        return conversation[-1]['content']

    except Exception as e:  # Capture the exception as 'e'
        print(f"*** Error: {e} ***")  # Print the error message
        time.sleep(2)
        return gpt_permute(input_conversation, temperature)
    
def parse_and_validate_string(s):
    # Extract the sequence of '[number] > [number] > ...' from the string
    match = re.search(r'(\[\d+\]( > \[\d+\])*)', s)
    if match:
        # Get the matched sequence
        sequence = match.group(0)
        # Split the sequence by ' > ' and extract numbers
        segments = sequence.split(' > ')
        numbers = [int(re.search(r'\[(\d+)\]', segment).group(1)) for segment in segments]
        # Reverse the list of numbers
        if len(numbers)!=10:
            rand_l = list(range(10))
            random.shuffle(rand_l)
            print("Invalid format or sequence not found")
            return rand_l
        return numbers[::-1]
    else:
        rand_l = list(range(10))
        random.shuffle(rand_l)
        print("Invalid format or sequence not found")
        return rand_l


    
def preprocess_for_rankgpt(text):
    return text.replace("[", "(").replace("]", ")")

def create_conversation(query_text, candidate_text_list, dataset_name):
    with open(f"./prompt_template/{dataset_name}/prompt_rankGPT.pickle", 'rb') as f:
        curr_prompt = pickle.load(f)

    prev_prompt = curr_prompt['prev']
    post_prompt = curr_prompt['post']
    
    initial_conversation = [{'role': 'system',
             'content': prev_prompt},
            {'role': 'user',
             'content': f"I will provide you with {len(candidate_text_list)} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query_text}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]

    messages = initial_conversation

    for ind, cand_text in enumerate(candidate_text_list):
        messages.append({'role': 'user', 'content': f"[{ind}] {cand_text}"})
        messages.append({'role': 'assistant', 'content': f'Received passage [{ind}].'})

    

    messages.append({'role': 'user', 'content': post_prompt.format(query_text=query_text, length=len(candidate_text_list)) })
    # print(messages)
    return messages

# def create_conversation(query_text, candidate_text_list, dataset_name):
#     # print("Creating Conversation without Description")
    
#     # with open(f"./prompt_template/{dataset_name}/prompt_rankGPT.pickle", 'rb') as f:
#     #     curr_prompt = pickle.load(f)

#     prev_prompt = "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."
#     post_prompt = "Search Query: {query_text}. \nRank the {length} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."
    
#     initial_conversation = [{'role': 'system',
#              'content': prev_prompt},
#             {'role': 'user',
#              'content': f"I will provide you with {len(candidate_text_list)} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query_text}."},
#             {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]

#     messages = initial_conversation

#     for ind, cand_text in enumerate(candidate_text_list):
#         messages.append({'role': 'user', 'content': f"[{ind}] {cand_text}"})
#         messages.append({'role': 'assistant', 'content': f'Received passage [{ind}].'})

    

#     messages.append({'role': 'user', 'content': post_prompt.format(query_text=query_text, length=len(candidate_text_list)) })
#     return messages

def permute_the_window(query_text, candidate_id_list, cand_id2text, window_size, dataset_name):

    # This function is going to output a list with number

    candidate_text_list = []
    for cid in candidate_id_list:
        candidate_text_list.append(cand_id2text[cid])

    permutation_prompt = create_conversation(query_text, candidate_text_list, dataset_name)
    gpt_answer = gpt_permute(permutation_prompt, temperature=0, pp=0, max_tokens=100)

    return gpt_answer

def rank_candidate_pool_for_a_query(query_id, candidate_id_list, query_from_dataset, corpus_from_dataset, dataset_name, window_size=10):

    query_text = preprocess_for_rankgpt(query_from_dataset[query_id])
    cand_id2text = {}
    for cid in candidate_id_list:
        cand_id2text[cid] = preprocess_for_rankgpt(corpus_from_dataset[cid])

    window_start_ind = 0
    window_end_ind = window_start_ind + window_size
    reranked_candidate_id_list = candidate_id_list
    flag = True
    while window_start_ind + window_size <= len(candidate_id_list) and flag:
        if window_start_ind + window_size == len(candidate_id_list):
            flag = False
        # print(f"curr_window: {window_start_ind}")
        prev_cand_id_list = reranked_candidate_id_list[:window_start_ind]
        curr_cid_list = reranked_candidate_id_list[window_start_ind:window_end_ind]
        post_cand_id_list = reranked_candidate_id_list[window_end_ind:]

        ind2cand_id = {}
        for ind ,cid in enumerate(curr_cid_list):
            ind2cand_id[ind] = cid
        
        gpt_output = permute_the_window(query_text, curr_cid_list, cand_id2text, window_end_ind-window_start_ind, dataset_name)
        curr_window_rank = parse_and_validate_string(gpt_output)
        if len(curr_window_rank) != window_size:
            print(curr_window_rank)
            print(gpt_output)
            raise("Parse Failure")
        
        rank_in_cand_id = []
        for i in curr_window_rank:
            rank_in_cand_id.append(ind2cand_id[i])

        reranked_candidate_id_list = prev_cand_id_list + rank_in_cand_id + post_cand_id_list
        # print(reranked_candidate_id_list)
        
        window_start_ind = int(min(window_start_ind + window_size/2, len(candidate_id_list) - window_size))
        window_end_ind = int(min(len(candidate_id_list), window_end_ind+window_size/2))

    with open(f"./rankGPT_output/{dataset_name}/{query_id}.pickle", 'wb') as f:
        pickle.dump(reranked_candidate_id_list, f)
        
    return reranked_candidate_id_list
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="created by LEI")
    

    parser.add_argument("-dataset_name", "--dataset_name", type = str, required = True, help = "Name of the dataset")
    parser.add_argument("-dataset_option", "--dataset_option", type = str, required = True, help = "should be one of [dev/test/all]")

    

    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    dataset_option = args.dataset_option

    with open(f"./processed_dataset/{dataset_name}/test_set.pickle", 'rb') as f:
        dataset = pickle.load(f)

    query = dataset['query']
    corpus = dataset['corpus']
    qrel = dataset['qrel']

    thread_list = []

    random.seed(42)
    count = 0
    prob_list = list(query.keys())
    # prob_list = ['test-environment-chbwtlgcc-pro01a']
    with open(f"./rankings/{dataset_name}/test_rank_e5.pickle", 'rb') as f:
        e5_ranking = pickle.load(f)

    for q in query:
        if count < 10:

            candidate_id_list = list(qrel[q].keys())
            ranked_candidate = e5_ranking[q]
            ranked_candidate.reverse()
            for cid in candidate_id_list:
                assert cid in ranked_candidate

            candidate_id_list = ranked_candidate
            
            # random.shuffle(candidate_id_list)

            if q in prob_list:
                thread_list.append(threading.Thread(target=rank_candidate_pool_for_a_query, args=(q, candidate_id_list, query, corpus, dataset_name, 10)))
            
                count += 1
        
        if count == 10 or q == prob_list[-1]:
        # if count == 3:
            # pdb.set_trace()
            print("10 Start")
            curr_time = time.time()
            for t in thread_list:
                t.start()

            for t in thread_list:
                t.join()

            print("10 Done")
            print(time.time() - curr_time)
            thread_list = []
            count = 0


    rank_result = {}
    for q in query:
        curr_path = f"./rankGPT_output/{dataset_name}/{q}.pickle"
        with open(curr_path, 'rb') as f:
            curr_rank = pickle.load(f)
        curr_rank.reverse()
        rank_result[q] = curr_rank

    with open(f"./rankings/{dataset_name}/test_rankgpt_with_description.pickle", 'wb') as f:
        pickle.dump(rank_result, f)

    print("Done")

# nohup python3 -u run_exp_RankGPT.py -dataset_name wtb -dataset_option test > log/wtb_rankGPT.txt 2>&1 &

        
        