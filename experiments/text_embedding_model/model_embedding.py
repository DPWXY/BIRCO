import pickle
import torch
import os
from math import *
import re
import json
import sys
import numpy as np
from itertools import chain
from tqdm import tqdm
from nltk import word_tokenize, sent_tokenize
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from transformers import RobertaTokenizer, RobertaModel
from torch import Tensor
import torch.nn.functional as F


# The function for preprocessing
def preprocessing(text):
    '''
    Note, this preprocessing function should be applied to any text before getting its embedding. 
    Input: text as string
    
    Output: removing newline, latex $$, and common website urls. 
    '''
    pattern = r"(((https?|ftp)://)|(www.))[^\s/$.?#].[^\s]*"
    try:
        text = re.sub(pattern, '', text)
    except:
        print(text)
    return text.replace("\n", " ").replace("$","")

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# The function to load the model and tokenizer. if model_path is None, then the pretrained model is loaded
def load_model(model_name, cuda =None, model_path=None):
    """
    loading models/tokenizers based on model_name, also based on cuda option specify whether DataParallel.
    Input: cuda option is a string, e.g. "1,3,5" specify cuda1, cuda3, and cuda5 will be used, store parameters on cuda1. 
    """
        
    if model_name == "e5":
        tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large-v2')
        if model_path is None:
            model = AutoModel.from_pretrained('intfloat/e5-large-v2')
        else:
            model_parallel = torch.load(f"./training/model_checkpoints/{model_path}.pth", map_location='cpu')
            try:
                model = model_parallel.module
            except:
                model = model_parallel
        model.eval()
        
    elif model_name == "simcse":
        tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
        if model_path is None:
            model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
        else:
            # Change the model path if you want to load your own trained model
            model_parallel = torch.load(f"./training/model_checkpoints/{model_path}.pth", map_location='cpu')
            model = model_parallel.module
        model.eval()
        
    elif model_name == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        if model_path is None:
            model = RobertaModel.from_pretrained('roberta-large')
        else:
            # Change the model path if you want to load your own trained model
            model_parallel = torch.load(f"./training/model_checkpoints/{model_path}.pth", map_location='cpu')
            model = model_parallel.module
        model.eval()
        
    elif model_name == "spladev2":
        tokenizer = AutoTokenizer.from_pretrained('naver/splade_v2_distil')
        if model_path is None:
            model = AutoModel.from_pretrained('naver/splade_v2_distil')
        else:
            # Change the model path if you want to load your own trained model
            model_parallel = torch.load(f"./training/model_checkpoints/{model_path}.pth", map_location='cpu')
            model = model_parallel.module
        model.eval()

    elif model_name == "splade++":
        tokenizer = AutoTokenizer.from_pretrained('naver/splade-cocondenser-ensembledistil')
        if model_path is None:
            model = AutoModel.from_pretrained('naver/splade-cocondenser-ensembledistil')
        else:
            raise ValueError("Not implemented for splade++, please check the file: model_embedding.py")
        

    elif model_name == "specterv2":
        tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
        if model_path is None:
            model = AutoModel.from_pretrained('allenai/specter2_base')
            model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
        else:
            model_parallel = torch.load(f"./training/model_checkpoints/{model_path}.pth")
            model = model_parallel.module
            print(f"Loading : ./training/model_checkpoints/{model_path}_adapter")
            model.load_adapter(f"./training/model_checkpoints/{model_path}_adapter", load_as="specter2", set_active=True)
        model.eval()
        
    if cuda!= "cpu":
        torch.cuda.set_device(int(cuda.split(",")[0]))
        
        model.to("cuda")
        cuda_list = cuda.split(",")
        cuda_num = len(cuda_list)
        if cuda_num>1:
            model = torch.nn.DataParallel(model, device_ids = [int(idx) for idx in cuda_list])
    else:
        print("Running model on CPU...")
    num_params= sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model_name} model parameters: {int(num_params/1000000)} millions.")
    return model, tokenizer


def get_embedding(model_name, model, tokenizer, text, cuda= "cpu", batch_size= 100):
    """
    Input model: loaded model
          tokenizer: associated tokenizer
          text: a list of strings, each string is either a query or an abstract
          cuda: in the format of "0,1,6,7" or "0", by default, cpu option is used
          batch_size: if not specified, then an optimal batch_size is found by system, else, 
                       the user specified batch_size is used, may run into OOM error.
    Return:  the embedding dictionary, where the key is a string (e.g. an abstract, query/subquery), and the value
             is np.ndarray of the vector, usually 1 or 2 dimensions. 
    """

    if cuda != "cpu":
        cuda_list = cuda.split(",")
        cuda_num = len(cuda_list)
        
        batch_size = batch_size
        
        length = ceil(len(text)/batch_size)
    else:
        batch_size = batch_size
    
    ret = {}  
    length = ceil(len(text)/batch_size)    
    for i in tqdm(range(length)):
        curr_batch = text[i*batch_size:(i+1)*batch_size]
        curr_batch_cleaned = [preprocessing(t) for t in curr_batch]
        if model_name == "ance": 
            inputs = tokenization(model_name, tokenizer, curr_batch_cleaned)
        else:
            inputs = tokenizer(curr_batch_cleaned, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
        embedding = encoding(model_name, model, inputs, cuda)
        for t, v in zip(curr_batch, embedding):
            ret[t] = v
    return ret


def tokenization(model_name, tokenizer, text):
    '''
    Different tokenization procedures based on different models.
    
    Input: text as list of strings, if cpu option then list has length 1.
    Return: tokenized inputs, could be dictionary for BERT models. 
    '''
    if model_name in ["e5", "spladev2", "specterv2", "e5", "simcse", "splade++", "roberta"]:
        inputs = tokenizer(text, padding=True, truncation=True, max_length=1000, return_tensors="pt")
    else:
        raise ValueError("Model name not defined")

    return inputs 


def encoding(model_name, model, inputs, cuda):
    '''
    Different encoding procedures based on different models. 
    Input: inputs are tokenized inputs in specific form
    Return: a numpy ndarray embedding on cpu. 
    
    '''
    if cuda != "cpu":
        device = "cuda"
    else:
        device = "cpu"
    with torch.no_grad():
        if model_name in ["e5"]:
            input_ids = inputs['input_ids'].to(device)
            assert input_ids.shape[1]<=512
            token_type_ids = inputs['token_type_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            new_batch_dict={}
            new_batch_dict["input_ids"] = input_ids
            new_batch_dict["token_type_ids"] = token_type_ids
            new_batch_dict["attention_mask"] = attention_mask

            outputs = model(**new_batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, new_batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
   
            output = embeddings.detach().cpu()

        elif model_name in ["simcse"]:
            input_ids = inputs['input_ids'].to(device)
            assert input_ids.shape[1]<=512
            attention_mask = inputs['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask = attention_mask)
            outputs = outputs.pooler_output
            outputs = F.normalize(outputs, p=2, dim=1)
            
            output = outputs.detach().cpu()
            
        elif model_name in ["roberta"]:
            input_ids = inputs['input_ids'].to(device)
            assert input_ids.shape[1]<=512
            attention_mask = inputs['attention_mask'].to(device)

            new_batch_dict={}
            new_batch_dict["input_ids"] = input_ids
            new_batch_dict["attention_mask"] = attention_mask

            outputs = model(**new_batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, new_batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
   
            output = embeddings.detach().cpu()

        elif model_name in ["specterv2"]:
            input_ids =inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            embeddings = model(input_ids, attention_mask = attention_mask)
            outputs = embeddings.last_hidden_state[:, 0, :]
            outputs = F.normalize(outputs, p=2, dim=1)
            
            output = outputs.detach().cpu()

        elif model_name in ["spladev2", "splade++"]:
            input_ids =inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            embeddings = model(input_ids, attention_mask = attention_mask)
            outputs = (torch.sum(embeddings.last_hidden_state * attention_mask.unsqueeze(-1), dim=1) /\
                                                      torch.sum(attention_mask, dim=-1, keepdim=True))
            outputs = F.normalize(outputs, p=2, dim=1)
            
            output = outputs.detach().cpu()

    return output.numpy()
    

    