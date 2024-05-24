# BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives

[![arXiv](https://img.shields.io/badge/arXiv-2402.14151-<COLOR>.svg)](https://arxiv.org/abs/2402.14151)&nbsp;

Our dataset is freely and publicly available at [zenodo](https://zenodo.org/records/10850865).

All human annotations are also available at [zenodo](https://zenodo.org/records/10738479).

## Paper abstract
We present the **B**enchmark of **I**nformation **R**etrieval (IR) tasks with **C**omplex **O**bjectives (BIRCO). BIRCO evaluates the ability of IR systems to retrieve documents given multi-faceted user objectives. The benchmark's complexity and compact size make it suitable for evaluating large language model (LLM)-based information retrieval systems. We present a modular framework for investigating factors that may influence LLM performance on retrieval tasks, and identify a simple baseline model which matches or outperforms existing approaches and more complex alternatives. No approach achieves satisfactory performance on all benchmark tasks, suggesting that stronger models and new retrieval protocols are necessary to address complex user needs. 

## Complex Task Objectives

We provide a reference version of each of the task objective in Appendix B1-B5 of our paper. All of our evaluated LLMs and embedding models have access to these task objectives (i.e. instructions), whenever there is +O notation. In order to evaluate your own model, you can refer to these task objectives so that your model have access to proper task instruction, since each task objective is complex and goes beyond traditional semantic relevance. We reproduce each task objective here.

### DORIS-MAE:

The query consists of users’ needs, leading to several research questions that span a paragraph. Each candidate passage is an abstract from a scientific paper. The objective of this information retrieval task is to identify the abstract that most effectively meets the user's needs in the query.

### ArguAna:

This information retrieval (IR) task has a debate format where a topic is given, and two directly opposing sides of arguments about this topic are formed.  A query is an argument that takes one side of this topic, focuses on a particular point about this topic, and takes a stance (i.e., opinion, position, view, perspective) about this particular point. A passage is an argument that takes the opposing side of the same topic, focuses on the same particular point about the same topic, and takes a directly opposing stance that directly (i.e., no implying or inferring) refutes and attacks the query’s stance regarding this particular point. Both query and passage might have citations in them but these citations should not be considered in the scope of this task. The overall goal of this specific information retrieval IR task is to identify the central topic of the debate, to articulate the query’s stance, and to find the passage that takes the opposing stance.

### WhatsThatBook:

The query has this format: a user is trying to remember the name of a specific book. The user only remembers some details about the book, such as places, events, and some characters’ names. Some of the details are described using informal language. The passage is a book description or summary of a specific book. The passage typically describes the overall storyline of the book and contains some details about the book. The objective of this information retrieval IR task is for you to find the passage that has details or components that holistically best match, explicitly or implicitly, the details or components raised in the query. In other words, you need to find the book description (i.e., the passage) that is most likely the book the user is looking for in the query.

### RELIC:

The query is a piece of literary analysis written by a scholar. In the query (i.e., the excerpt from a literary analysis), one or more quotations from a classic English novel is used as evidence to support the claims made by the literary analysis. Quotations are identified by quotation marks. Now, one quotation is being intentionally masked from the literary analysis (i.e., the query), and replaced by the symbol [masked sentence(s)]. An important claim is made in the preceding context and another important point is made in the subsequent context surrounding the [masked sentence(s)]. The objective of this information retrieval task is to find the most suitable passage that can be used to directly support at least one claim made in the query (i.e., the claim that is made in the preceding or the claim subsequent context surrounding the [masked sentence(s)]) and is very natural to be plugged into the [masked sentence(s)] part of the query. Obviously the most suitable passage should NOT REPEAT or be contained in any part of the query. It does not make sense to repeat the same or very similar things twice in literary analysis.

### Clinical-Trial:

The motivation of the Information Retrieval task is that clinical trials are experiments conducted in the development of new medical treatments, drugs or devices, and recruiting candidates for a trial is often a time-consuming and resource-intensive effort. A query is a patient case report (either in the form of electronic patient records or ad-hoc queries). A passage is a clinical trial. This Information Retrieval task is to improve patient recruitment for clinical trials. The overall goal of this specific information retrieval IR task is to match eligible patients (the query) to clinical trials (the passage) for recruitment.



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
