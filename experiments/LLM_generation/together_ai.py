import together
import time
import threading
import pickle
import argparse
import os
import pdb
import math
import time

together.api_key = "Replace_with_your_together_ai_API_key"

def call_chat_llama_7b(prompt, model_name, max_tokens = 1024, temperature = 0, top_k = 20, top_p = 1,repetition_penalty = 1.0):

    chat_llama_prompt = f"[INST] {prompt} [/INST]"
    assert "llama" in model_name
    assert "chat" in model_name

    try:

        output = together.Complete.create(
        prompt = chat_llama_prompt, 
        model = f"{model_name}", 
        max_tokens = max_tokens,
        temperature = temperature,
        top_k = 20,
        top_p = 1,
        repetition_penalty = 1.0,
        stop = ['[/INST]', '</s>']
        )
        return output['output']['choices'][0]['text']

    except Exception as e:  # Capture the exception as 'e'
        print(f"*** Error: {e} ***")  # Print the error message
        time.sleep(10)
        return call_chat_llama_7b(prompt, model_name, max_tokens=max_tokens, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)

def call_chat_llama_13b(prompt, model_name, max_tokens = 1024, temperature = 0, top_k = 20, top_p = 1,repetition_penalty = 1.0):

    chat_llama_prompt = f"[INST] {prompt} [/INST]"
    assert "llama" in model_name
    assert "chat" in model_name

    try:

        output = together.Complete.create(
        prompt = chat_llama_prompt, 
        model = f"{model_name}", 
        max_tokens = max_tokens,
        temperature = temperature,
        top_k = 20,
        top_p = 1,
        repetition_penalty = 1.0,
        stop = ['[/INST]', '</s>']
        )
        return output['output']['choices'][0]['text']

    except Exception as e:  # Capture the exception as 'e'
        print(f"*** Error: {e} ***")  # Print the error message
        time.sleep(10)
        return call_chat_llama(prompt, model_name, max_tokens=max_tokens, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)

def call_chat_llama_70b(prompt, model_name, max_tokens = 1024, temperature = 0, top_k = 20, top_p = 1,repetition_penalty = 1.0):

    chat_llama_prompt = f"[INST] {prompt} [/INST]"
    assert "llama" in model_name
    assert "chat" in model_name

    try:

        output = together.Complete.create(
        prompt = chat_llama_prompt, 
        model = f"{model_name}", 
        max_tokens = max_tokens,
        temperature = temperature,
        top_k = 20,
        top_p = 1,
        repetition_penalty = 1.0,
        stop = ['[/INST]', '</s>']
        )
        return output['output']['choices'][0]['text']

    except Exception as e:  # Capture the exception as 'e'
        print(f"*** Error: {e} ***")  # Print the error message
        time.sleep(10)
        return call_chat_llama_70b(prompt, model_name, max_tokens=max_tokens, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)

def call_sh_7b(prompt, model_name, max_tokens = 1024, temperature = 0, top_k = 20, top_p = 1,repetition_penalty = 1.0):

    chat_SH_prompt = f"###Intruction:{prompt} ###Response:"
    assert "stripedhyena" in model_name.lower()

    try:

        output = together.Complete.create(
        prompt = chat_SH_prompt, 
        model = f"togethercomputer/StripedHyena-Nous-7B", 
        max_tokens = max_tokens,
        temperature = temperature,
        top_k = 20,
        top_p = 1,
        repetition_penalty = 1.0,
        stop = ['###', '</s>']
        )
        return output['output']['choices'][0]['text']

    except Exception as e:  # Capture the exception as 'e'
        print(f"*** Error: {e} ***")  # Print the error message
        time.sleep(10)
        return call_sh_7b(prompt, model_name, max_tokens=max_tokens, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def gen_query(prompt_dict, path_to_store, model_name):
    curr_id_list = list(prompt_dict.keys())
    curr_prompt_list = []
    for i in curr_id_list:
        curr_prompt_list.append(prompt_dict[i])

    generated_queries = {}

    if model_name == "togethercomputer/StripedHyena-Nous-7B":
        chat_func = call_sh_7b
    elif model_name == "togethercomputer/llama-2-7b-chat":
        chat_func = call_chat_llama_7b
    elif model_name == "togethercomputer/llama-2-13b-chat":
        chat_func = call_chat_llama_13b
    elif model_name == "meta-llama/Llama-2-70b-chat-hf":
        chat_func = call_chat_llama_70b
    else:
        raise ValueError("Invalid model_name")

    for ind, prompt in enumerate(curr_prompt_list):

        curr_generated_query = chat_func(prompt, model_name, max_tokens = 1024, temperature = 0, top_k = 20, top_p = 1,repetition_penalty = 1.0)
        generated_queries[curr_id_list[ind]] = curr_generated_query

    create_directory_if_not_exists(f"{path_to_store}/")

    with open(f"{path_to_store}/query.pickle", 'wb') as f:
        pickle.dump(generated_queries, f)


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--output_dir', required=True, help='Output file to write responses to')
    parser.add_argument('-t', '--thread_num', required=True, help='Number of threads used in the program')
    parser.add_argument('-s', '--index_to_start', required=True, help='The index to start')
    parser.add_argument('-e', '--index_to_end', required=True, help='The index to end')
    parser.add_argument('-prompt', '--prompt_file_to_read', required=True, help='The file that contains the prompts')
    parser.add_argument('-model_name', '--model_name', required=True, help='Name of the model')

    args = parser.parse_args()

    file_name = args.prompt_file_to_read
    
    print(f"curr_filename: {file_name}")

    with open(file_name, 'rb') as f:
        whole_prompt_dict = pickle.load(f)

    output_path = args.output_dir
    thread_num = int(args.thread_num)
    start_ind = int(args.index_to_start)
    end_ind = int(args.index_to_end)
    model_name = args.model_name
    num_of_queries = end_ind - start_ind
    path_head = output_path
    abs_id_list = []

    thread_length = math.ceil((num_of_queries / thread_num))
    thread_list = []
    thread_id = 0
    path_list = []
    for i in range(thread_num):

        start = int(thread_length * i)
        end = int(thread_length * (i + 1))

        curr_path = f"./generation_result/{path_head}/{start_ind + start}_{start_ind + end}" #edit folder name
        list_of_ids = list(whole_prompt_dict.keys())[start_ind + start: start_ind + end]

        prompt_dict = {}
        for i in list_of_ids:
            prompt_dict[i] = whole_prompt_dict[i]

        thread_list.append(
            threading.Thread(target=gen_query,
                             args=(prompt_dict, curr_path, model_name)))
        thread_id += 1
        path_list.append(curr_path)

    for t in thread_list:
        t.start()

    for t in thread_list:
        t.join()

    all_queries = {}
    all_prompts = {}
    for p in path_list:
        with open(f"{p}/query.pickle", 'rb') as f:
            curr_queries = pickle.load(f)
        for k in curr_queries:
            all_queries[k] = curr_queries[k]
            
    with open(f"./generation_result/{path_head}/all_queries.pickle", 'wb') as f: # edit folder name
        pickle.dump(all_queries, f)

    print("Generation Ended")
    end_time = time.time()