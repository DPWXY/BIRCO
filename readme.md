# BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives

Our dataset is freely and publicly available at [zenodo](https://zenodo.org/records/10850865).

All human annotations are also available at [zenodo](https://zenodo.org/records/10738479).

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

## Evaluate Your own IR System
### Rank format

TLDR: Create your own rank result file (a **pickle** file for each dataset) for your model, run this command, get your model's evaluation result. 
```bash
# dataset_names should be one of ['doris-mae', 'arguana', 'clinical-trial', 'wtb', 'relic']

# Run the evaluation with bootstrap
python3 rank2metric.py --dataset_name [dataset_name] --boot True --rank_path [path_to_your_own_rank_pickle]
```


More details:

To assess the performance of your model, you should organize the ranking results for each search query. This involves ranking the candidates for each query based on their relevance and saving these rankings in a specific format. Here's how the ranking results should be formatted:

```bash
{
  "qid_1": ["cid_1_from_qid_1_candidate_pool", "cid_2_from_qid_1_candidate_pool", ...],
  "qid_2": ["cid_1_from_qid_2_candidate_pool", "cid_2_from_qid_2_candidate_pool", ...],
  ...
}
```

In this format, `cid_1_from_qid_1_candidate_pool`, `cid_2_from_qid_1_candidate_pool`, and so on, represent the IDs of the candidate documents in the candidate pool associated with the first query. These IDs come from the qrel dict associated with your dataset ([detail of the dataset](https://github.com/BIRCO-benchmark/BIRCO/blob/main/datasets/readme.md)). The candidates should be listed in order of decreasing relevance; meaning, the first ID in each list is considered the most relevant document to the query among all candidates in the pool.

Please note, for each query, the entirety of its candidate pool needs to be ranked. Also note, different query can have different candidate pool of different size.

We provide rank results in `./evaluation/ranks/{dataset_name}/`. To evaluate various models and methods on datasete, run the following code with rank file path:
```bash
# dataset_names should be one of ['doris-mae', 'arguana', 'clinical-trial', 'wtb', 'relic']

# Run the evaluation with bootstrap
python3 rank2metric.py --dataset_name [dataset_name] --boot True --rank_path [path_to_rank]
```
The result metrics will be printed.

## Datasets

The datasets with test and dev sets are in `datasets/`. 

We provide brief information about test sets for five datasets in BIRCO here. The detailed information about the dataset is available [here](https://github.com/BIRCO-benchmark/BIRCO/blob/main/datasets/readme.md).
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


## Reproduce the experimental results in the paper

We provide rank results of all methods reported in our paper in `evaluation/ranks/`.

To reproduce the experimental results from scratch, go to `experiments/`. The details to reproduce the ranks are provided [here](https://github.com/BIRCO-benchmark/BIRCO/blob/main/experiments/readme.md). 

## Knowledge Contamination
The scanned images of human annotations across ten datasets from BEIR are avaliable [here](https://zenodo.org/records/10738479). 
