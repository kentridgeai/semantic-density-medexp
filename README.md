# semantic-density-paper
This repo contains the source code for reproducing the experimental results reported in paper: "Semantic Density: Uncertainty Quantification for Large Language Models through Confidence Measurement in Semantic Space", which is accepted to Neurips 2024 (Arxiv link: https://arxiv.org/abs/2405.13845).

## Environment Setup

Below are the step-by-step guideline for setting up the experiment environment:

(a) Use the ```environment_llama2_mistral_mixtral.yml``` file to create an anaconda environment for all the experiments with ```Llama-2-13B```, ```Llama-2-70B```, ```Mistral-7B```, ```Mixtral-8x7B``` and ```Mixtral-8x22B```. 

(b) Replace ```anaconda3/envs/{your_env_name_for_llama2_mistral_mixtral}/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py``` with ```modeling_llama2.py``` in folder ```huggingface_replacement``` (change the file name to ```modeling_llama.py```). 

(c) Replace ```anaconda3/envs/{your_env_name_for_llama2_mistral_mixtral}/lib/python3.10/site-packages/transformers/models/mistral/modeling_mistral.py``` with ```modeling_mistral.py``` in folder ```huggingface_replacement```. 

(d) Replace ```anaconda3/envs/{your_env_name_for_llama2_mistral_mixtral}/lib/python3.10/site-packages/transformers/models/mixtral/modeling_mixtral.py``` with ```modeling_mixtral.py``` in folder ```huggingface_replacement```. 

(e) Use the ```environment_llama3.yml``` file to create an anaconda environment for all the experiments with ```Llama-3-8B``` and ```Llama-3-70B```. 

(f) Replace ```anaconda3/envs/{your_env_name_for_llama3}/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py``` with ```modeling_llama3.py attached``` (change the file name to ```modeling_llama.py```).

## Running Experiments

Please read the detailed step-by-step guideline inside folder ```experiment_code``` to generate the experimental results.

## Citation

If you find semantic density useful, please cite it using the following BibTeX entry:
```
@inproceedings{qiu2024semantic,
title={Semantic Density: Uncertainty Quantification for Large Language Models through Confidence Measurement in Semantic Space},
author={Qiu, Xin and Miikkulainen, Risto},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
}
```
