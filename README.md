# BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives

## Table of Contents
* [Abstract](#paper-abstract)
* [Setup](#setup)
* [Datasets](#datasets)
* [Generation Process](#generation-process-for-scoring-based-llm-ir-systems)
    * [Generation](#generation)
    * [Ranking](#ranking)
* [Evaluation](#evaluation)

## Paper abstract
We present the **B**enchmark of **IR** tasks with **C**omplex **O**bjectives (BIRCO) to evaluate the ability of Information Retrieval (IR) models to follow multi-faceted task objectives. We study the performance of various embedding, distilled and fine-tuned IR models on BIRCO, and find them lacking. We provide a unified framework for investigating the performance of large language models (LLMs) on these tasks. The proposed framework consists of 3 modular components: task-objective awareness; chain-of-thought reasoning; and task decomposition. We investigate the effects of these factors on LLM performance, and identify a simple baseline model which matches or outperforms existing approaches and more complex alternatives. No approach achieves satisfactory performance on all benchmark tasks, suggesting that stronger models and new retrieval protocols are necessary to address complex user needs. 


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

## Generation Process for Scoring-based LLM IR Systems
Using the prompts as shown in `generation/generation_data/prompts/` for each specific dataset and method, the sample generated outputs are stored in `generation/generation_data/output/`.


### Generation

To generate full results from gpt:

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

### Ranking
After generation, to get the ranking result, run the following for corresponding scoring-based method:
```bash
bash run_rank_score.sh # For score+Ogpt
bash run_rank_reason.sh # For reason+Ogpt
bash run_rank_subtask.sh # For subtask+Ogpt
```
The results will be saved in `rankings/`.

## Evaluation
To evaluate various models and methods on datasets:
1. Configure the Shell Script:
   - Open the `get_scores.sh` script in a text editor.
   - Set the following variables according to your requirements:
     - `model_name`: Model names
     - `dataset_name`: The names of the dataset to evaluate.
     - `set_option`: The evaluation option for the datasets.
    - The first loop is for models already have ranking result, the second loop is for models without ranking result, put the model names in corresponding loop areas for evaluation. 
2. Run the following:
```bash
nohup bash get_scores.sh > evaluation_log/<LOG_FILE> 2>&1 &  
```
The result will be printed in log file, and remember to replace <LOG_FILE> with name of the log file.
