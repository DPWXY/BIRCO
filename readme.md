# BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives

## Table of Contents
* [Abstract](#paper-abstract)
* [Setup](#setup)
* [Datasets](#datasets)
   * [Test Set](#test-set)
   * [Dev Set](#dev-set)
* [Generation Process for LLM System](#generation-process-for-llm-ir-systems)
    * [Generation](#generation)
    * [Ranking](#ranking)
* [Generation Process for Text Embedding Model](#generation-process-for-text-embedding-models)
* [Evaluation](#evaluation)

## Paper abstract
We present the **B**enchmark of **I**nformation **R**etrieval (IR) tasks with **C**omplex **O**bjectives (BIRCO). BIRCO evaluates the ability of IR systems to retrieve documents given multi-faceted user objectives. The benchmark's complexity and compact size make it suitable for evaluating large language model (LLM)-based information retrieval systems. We present a modular framework for investigating factors that may influence LLM performance on retrieval tasks, and identify a simple baseline model which matches or outperforms existing approaches and more complex alternatives. No approach achieves satisfactory performance on all benchmark tasks, suggesting that stronger models and new retrieval protocols are necessary to address complex user needs. 


## Setup
We highly recommend creating a new conda environment for the following steps by:
```
conda create -n birco python=3.10
conda activate birco
```
To start, please run
```
bash setup.sh
```
Running the above line will create a new conda environment with name `birco`, download the needed packages and datasets, and unzip the needed files.

## Datasets
The datasets with test and dev sets are in `datasets/`. 
### Test set
We provide information about test sets for five datasets in BIRCO. 
1. **DORIS-MAE**:
60 queries that are complex research questions from computer scientists. The query communicates specific requirements from research papers. Each query has a candidate pool sized approximately 110.
2. **ArguAna**:
100 queries, each with a candidate pool of around 50 passages. Queries and passages are both complex one-paragraph arguments about current affairs. The objective is to find matching counter-arguments.
3. **Clinical-Trial**:
100 queries that are paragraph-length patient case-reports. Each query has a candidate pool comprising 30-110 passages that are paragraph-length descriptions of clinical trials. The objective is to find the most suitable clinical trial for a patient.
4. **WhatsThatBook (WTB)**:
100 queries, with each query describing a book in an ambiguous manner. Each query has a pool of 50 passages, which are book descriptions.
5. **RELIC**:
100 queries which are excerpts from scholars analyzing classic English-language literature. Passages are sentences from a novel that have been extracted from the queries. The objective is to match a literary analysis with its missing quotations. 
### Dev set
We provide dev set for each dataset. You can run `triplet_extraction.py` to extract the triplets of dev set for training or testing. 

## Generation Process for LLM IR Systems
Using the prompts as shown in `experiments/LLM_generation/prompts/` for each specific dataset and method, the generated outputs will be stored in `collected_result/`.

### Generation by GPT

To generate full results from gpt, use the prompt in `experiments/LLM_generation/prompts/gpt`:

1. **Set up your API Key and OpenAI model**:
   - Open the `gpt_generation.py` file.
   - Locate the line `openai.api_key = None` and replace `None` with your OpenAI API key.
   - Our default model is `"gpt-4-0613"`, change this according to your requirement.

2. **Configure the Shell Script**:
   - Open the `run_generation.sh` script in a text editor.
   - Set the following variables according to your requirements:
     - `thread_num`: Number of threads, usually `20`
     - `prompt_path`: Path to the prompt file, this should be a pickle that stores a dictionary of format id string to prompt string.
     - `name`: prefix of name for the output file, e.g., `"arguana_test_set_reason_O"`.
     - `start`: Starting index for processing, usually `0`.
     - `end`: Ending index for processing. This depends on the length of the prompt file.

3. **Run the Generation Script**:
   - Execute the `run_generation.sh` script to start the generation process.
   - We recommend using nohup to run the generation process
    ```bash
    cd generation
    nohup bash run_generation.sh > generation_log/<LOG_FILE> 2>&1 &  
    ```
     remember to replace <LOG_FILE> with name of the log file.

### Generation by Llama-2 and StripedHyena
Similar to the setup for GPT generation, use the prompt in `experiments/LLM_generation/prompts/llama_stripedhyena` and run the following command to get the LLM output:
```bash
bash run_together_ai_generation.sh
```

### Ranking
After generation, please collect the generated ouput and get scores and rankings by LLM model. And save the result in `rankings/` for running evaluation.

## Generation Process for Text Embedding Models
The code for this process is in `experiments/text_embedding_model`
### Generation and Ranking
To get the ranking for text-embedding models(E5-L-v2, RoBERTa-L, SPLADE-v2, SPLADE++, SPECTER-v2, SIMCSE), run the following code and the rank results will be saved in `text_embeeding_model_results`:
```bash
embedding_model_names = ['e5', 'roberta', 'spladev2', 'splade++', 'specterv2', 'simcse']
dataset_names = ['doris-mae', 'arguana', 'clinical-trial', 'wtb', 'relic']

python3 rank_by_embedding_model.py --cuda [cuda_number] --model_name [model_name] --dataset_name [dataset_name] --bs [batch_size]
```

## Evaluation
To evaluate various models and methods on datasete, run the following code with your ranking path:
```bash
dataset_names = ['doris-mae', 'arguana', 'clinical-trial', 'wtb', 'relic']

# Run the evaluation with bootstrap
python3 rank2metric.py --dataset_name [dataset_name] --boot True --rank_path [path_to_rank]
```
The result metrics will be printed.