# Read generation results
import argparse
import os
import pickle
import random

parser = argparse.ArgumentParser()
parser.add_argument('--generation_model', type=str, default='opt-1.3b')
parser.add_argument('--run_id_for_few_shot_prompt', type=str, default='run_1')
parser.add_argument('--run_id_for_evaluation', type=str, default='run_1')
parser.add_argument('--dataset', type=str, default='NQ')
parser.add_argument('--cuda_device', type=str, default="0")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda_device

import accelerate
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score

import config
from config import device_map

# Set a seed value
seed_value = 3
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value

os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value

random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value

np.random.seed(seed_value)

device = torch.device('cuda')

#Fix torch random seed
torch.manual_seed(seed_value)

os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache

model_name = f'{args.generation_model}'

mistralai_models = ['Mistral-7B-v0.1', 'Mixtral-8x7B-v0.1', 'Mixtral-8x22B-v0.1']
llama_models = ['Llama-2-13b-hf', 'Llama-2-70b-hf', 'Meta-Llama-3-8B', 'Meta-Llama-3-70B']


if f"{args.generation_model}" in mistralai_models:
    hf_model_dir = 'mistralai/' + f"{args.generation_model}"

if f"{args.generation_model}" in llama_models:
    hf_model_dir = 'meta-llama/' + f"{args.generation_model}"

generation_tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, use_fast=False, cache_dir=config.data_dir)

model = AutoModelForCausalLM.from_pretrained(hf_model_dir,
                                             torch_dtype=torch.float16,
                                             cache_dir=config.hf_cache_dir, device_map="auto")

with open(f'{config.output_dir}/{model_name}_generations_all_{args.dataset}.pkl', 'rb') as infile:
    sequences_for_few_shot_prompt = pickle.load(infile)


# Build few shot prompt

subset_of_sequences_for_few_shot_prompt = sequences_for_few_shot_prompt[-10:]
number_of_few_shot_samples = 5

prompt_template = 'Question: {} \n Here are some ideas that were brainstormed:{}\n Possible answer:{}\n Is the possible answer:\n (A) True\n (B) False\n The possible answer is:'
few_shot_promopt = ''
for sequence in subset_of_sequences_for_few_shot_prompt:
    question = sequence['question']
    question = question.split('Question: ')[-1].split('Answer: ')[0]
    prompt = sequence['prompt']
    generated_texts = '\n'.join(sequence['cleaned_generated_texts'][:number_of_few_shot_samples])

    most_likely_answer = sequence['most_likely_generation']
    correct = ' True' if sequence['rougeL_beam_search_to_target_0'] > 0.3 else ' False'
    few_shot_promopt += prompt_template.format(question, generated_texts, most_likely_answer) + correct + '\n'

# Build prompt for question
labels_across_beams = []
p_trues_across_beams = []
p_trues_auroc_list = []

n_samples_to_use = 10
beam_num = 10

with torch.no_grad():

    for beam_id in range(beam_num):
        aurocs = []
        p_trues = []
        corrects = []
        for sequence in tqdm(sequences_for_few_shot_prompt):

            question = sequence['question']
            if 'Question: ' in question:
                question = question.split('Question: ')[-1].split('Answer: ')[0]
            else:
                question = question.split('Q: ')[-1].split('A: ')[0]

            generated_texts = '\n'.join(sequence['cleaned_generated_texts'][:number_of_few_shot_samples])
            most_likely_answer = sequence['cleaned_beam_search_generation_{}'.format(beam_id)]
            correct = 1.0 if sequence['rougeL_beam_search_to_target_{}'.format(beam_id)] > 0.3 else 0.0
            base_prompt = prompt_template.format(question, generated_texts, most_likely_answer)
            prompt_true = few_shot_promopt + prompt_template.format(question, generated_texts, most_likely_answer) + ' True'

            # This computation of the negative log likelihoods follows this tutorial: https://huggingface.co/docs/transformers/perplexity
            tokenized_base_prompt = generation_tokenizer(base_prompt)['input_ids']
            tokenized_prompt_true = torch.tensor(generation_tokenizer(prompt_true)['input_ids'], device=device)

            target_ids_true = tokenized_prompt_true.clone()
            target_ids_true[:len(tokenized_base_prompt)] = -100

            model_output_true = model(torch.reshape(tokenized_prompt_true, (1, -1)), labels=target_ids_true)
            loss_true = model_output_true.loss

            p_trues.append(loss_true.item())
            corrects.append(correct)

        labels_across_beams.append(corrects)
        p_trues_across_beams.append(p_trues)

        p_true_auroc = roc_auc_score(1 - torch.tensor(corrects), torch.tensor(p_trues))
        p_trues_auroc_list.append(p_true_auroc)
    p_trues_all = []
    corrects_all = []
    for i in range(len(p_trues_across_beams)):
        p_trues_all += p_trues_across_beams[i]
        corrects_all += labels_across_beams[i]
    p_true_auroc_all = roc_auc_score(1 - torch.tensor(corrects_all), torch.tensor(p_trues_all))
    print("p_true_auroc_all: {}".format(p_true_auroc_all))
    print("p_trues_auroc_list: {}".format(p_trues_auroc_list))

    # Store p_true aurocs in a pickle file
    with open(f'{config.output_dir}/{model_name}_p_true_{args.dataset}.pkl', 'wb') as outfile:
        pickle.dump((p_trues_across_beams, labels_across_beams), outfile)
