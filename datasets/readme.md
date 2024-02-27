# BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives

## Datasets

### Test Data

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

All of the datasets are in the following format:

- **query**
  - `qid_1` (text_qid_1)
  - `qid_2` (text_qid_2)
  - `...`

- **corpus**
  - `cid_1` (text_cid_1)
  - `cid 2` (text_cid_2)
  - `...`

- **qrel**
  - `qid_1`
    - `cid_1` : relevance score `(qid_1, cid_1)`
    - `cid 2` : relevance score `(qid_1, cid_2)`
    - `...`
  - `qid_2`
    - `cid_3` : relevance score `(qid_2, cid_3)`
    - `...`
  - `...`

### Dev Data

We also provide a development set for each dataset, which follows the same format as the test set. However, we prioritize the quality of the candidate pools in the test set and there is no overlap between the test set and the dev set, we advise utilizing the triplets from the development set for prompt tuning or training purposes, rather than the entire candidate pool. To extract these triplets, execute the following command:

```
python3 triplet_extraction.py
```

The list of triplets will be stored in `dev_triplets`.



