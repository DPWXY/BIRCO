# BIRCO Experiments

The content within this directory enables the reproduction of the experimental results presented in the paper. All inferences were conducted on a machine with 8 80GB Nvidia H100 GPUs.

## Text Embedding Models

The code to run the text embedding models is stored in `text_embedding_model/`.

To reproduce the experimental results reported in our paper, we recommend executing `run_all_embedding_models.sh`. Please note that one may need to change the arguments `-cuda` and `-bs` when using different GPU/TPU configurations.

## TART

To run TART, please visit https://github.com/facebookresearch/tart to configure the environment. Then, go to `TART/`, and execute `pipeline.sh`

## MonoT5

To run MonoT5, please go to `monoT5/`, clone repo from https://github.com/castorini/pygaggle, and move `monot5_rerank.py` in `pygaggle` and run the following:

```
python3 monot5_rerank.py -dataset_name <dataset_name> -dataset_option test
```

## RankGPT

To run RankGPT, go to `rankGPT/`, and use the following command:

```
python3 run_exp_RankGPT.py -dataset_name <dataset_name> -dataset_option test
```

## Scoring-Based Models

To run the scoring-based models, go to `LLM_generation/`. We provide the prompts and code for `GPT-4-0613`, `LLaMA2`(7B, 13B, 70B), and `StripedHyena`. Then, make changes in `run_together_ai_generation.sh` to generate the corresponding outputs.