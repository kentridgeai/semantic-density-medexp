import argparse
import csv
import os
import pickle
import random

parser = argparse.ArgumentParser()
parser.add_argument('--generation_model', type=str, default='opt-350m')
parser.add_argument('--run_id', type=str, default='run_1')
parser.add_argument('--dataset', type=str, default='NQ')
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--cuda_device', type=str, default="0")
args = parser.parse_args()
print(args)

import evaluate
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import scipy

import config
import wandb
import math

device = f"cuda:{args.cuda_device}" if torch.cuda.is_available() and args.cuda_device != '-1' else 'cpu'

print(torch.cuda.device_count())
print(torch.cuda.is_available())

# Set a seed value
seed_value = 3
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value

os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value

random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value

np.random.seed(seed_value)

#Fix torch random seed
torch.manual_seed(seed_value)

os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache

mistralai_models = ['Mistral-7B-v0.1', 'Mixtral-8x7B-v0.1', 'Mixtral-8x22B-v0.1']
llama_models = ['Llama-2-13b-hf', 'Llama-2-70b-hf', 'Meta-Llama-3-8B', 'Meta-Llama-3-8B-Instruct', 'Meta-Llama-3-70B', 'Llama-2-7b-hf']
SUPPORTED_OTHER_LMS = ['phi3', 'meerkat7b']

if f"{args.generation_model}" in mistralai_models:
    hf_model_dir = 'mistralai/' + f"{args.generation_model}"

if f"{args.generation_model}" in llama_models:
    hf_model_dir = 'meta-llama/' + f"{args.generation_model}"
    
if f"{args.generation_model}" in SUPPORTED_OTHER_LMS:
    hf_model_dir = "dmis-lab/meerkat-7b-v1.0" if "meerkat" in args.generation_model else "microsoft/phi-3-mini-4k-instruct"

generation_tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, use_fast=False, cache_dir=config.data_dir)

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
model0 = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to(device)

# with open(f'{config.output_dir}/{args.generation_model}_generations_all_{args.dataset}.pkl', 'rb') as infile:
with open(f'{config.output_dir}/{args.generation_model}_generations_all_{args.dataset}.pkl', 'rb') as infile:
    sequences = pickle.load(infile)

# with open(f'{config.output_dir}/{args.generation_model}_generations_{args.generation_model}_likelihoods_all_{args.dataset}_temperature{args.temperature}.pkl', 'rb') as infile:
with open(f'{config.output_dir}/{args.generation_model}_generations_{args.generation_model}_likelihoods_all_{args.dataset}_temperature{args.temperature}.pkl', 'rb') as infile:
    likelihoods = pickle.load(infile)
    
# print(len(likelihoods))
# pass

result_dict = {}

meteor = evaluate.load('meteor')

deberta_predictions = []

def clean_text(generated_text):
    strings_to_filter_on = [
        '.', '\n', 'Q:', 'A:', 'question:', 'answer:', 'Question:', 'Answer:', 'Questions:', 'questions:', 'QUESTION:',
        'ANSWER:'
    ]
    for string in strings_to_filter_on:
        if string in generated_text:
            generated_text = generated_text.split(string)[0]
    return generated_text

num_beams = 10

average_contradict_prob_list_beam = []
semantic_density_list_beam = []

for beam_id in range(0, num_beams):
    average_contradict_prob_list = []
    semantic_density_list = []
    index = 0
    for sample in tqdm(sequences):
        question = sample['question']
        if 'cleaned_generated_texts' in sample:
            generated_texts = sample['cleaned_generated_texts']
        else:
            generated_texts = sample['generated_texts']

        id_ = sample['id']
#         print(id_)
#         print(likelihoods[index]["id"])
#         print([x for x in range(500) if likelihoods[x]['id'] == id_])
#         print("llindex", likelihoods[index],  likelihoods[index] == likelihoods[id_])
#         print([likelihoods[index]['average_neg_log_likelihood_of_beam_search_gen_{}'.format(beam_index)] for beam_index in range(num_beams)])
#         print([likelihoods[id_]['average_neg_log_likelihood_of_beam_search_gen_{}'.format(beam_index)] for beam_index in num_beams])
#         break
        most_likely_text = clean_text(sample['cleaned_beam_search_generation_{}'.format(beam_id)])
        contradict_prob_list = []
        likelihood_sum = 0
        semantic_density = 0
        # Evalauate semantic similarity
        unique_cleaned_beam_search_generation = set()
        unique_beam_index = []
        for beam_index in range(num_beams):
            if sample['cleaned_beam_search_generation_{}'.format(beam_index)] not in unique_cleaned_beam_search_generation:
                unique_cleaned_beam_search_generation.add(sample['cleaned_beam_search_generation_{}'.format(beam_index)])
                unique_beam_index.append(beam_index)

        for beam_index in unique_beam_index:
            qa_1 = question + ' ' + sample['cleaned_beam_search_generation_{}'.format(beam_index)]
            qa_2 = question + ' ' + most_likely_text
            average_likelihood = math.exp(-likelihoods[index]['average_neg_log_likelihood_of_beam_search_gen_{}'.format(beam_index)])
            origin_input = qa_1 + ' [SEP] ' + qa_2
            encoded_input = tokenizer.encode(origin_input, padding=True)
            prediction = model0(torch.tensor(torch.tensor([encoded_input]), device=device))['logits'][0]
            prediction_softmax = scipy.special.softmax(prediction.cpu().detach().numpy())
            contradict_prob_1 = 1-prediction_softmax[2]
            semantic_distance = prediction_softmax[0] + 0.5*prediction_softmax[1]
            semantic_density += 0.5*(1.0-semantic_distance)*average_likelihood

            reverse_input = qa_2 + ' [SEP] ' + qa_1
            encoded_reverse_input = tokenizer.encode(reverse_input, padding=True)
            reverse_prediction = model0(torch.tensor(torch.tensor([encoded_reverse_input]), device=device))['logits'][0]
            reverse_prediction_softmax = scipy.special.softmax(reverse_prediction.cpu().detach().numpy())
            contradict_prob_2 = 1-reverse_prediction_softmax[2]
            reverse_semantic_distance = reverse_prediction_softmax[0] + 0.5*reverse_prediction_softmax[1]
            semantic_density += 0.5*(1.0-reverse_semantic_distance)*average_likelihood
            likelihood_sum += average_likelihood

            contradict_prob_list.append((contradict_prob_1+contradict_prob_2)/2.0)
        average_contradict_prob_list.append(np.mean(contradict_prob_list))
        semantic_density_list.append(semantic_density/likelihood_sum)
        index += 1

    average_contradict_prob_list_beam.append(average_contradict_prob_list)
    semantic_density_list_beam.append(semantic_density_list)

with open(f'{config.output_dir}/{args.generation_model}_generations_average_contradict_prob_beam_all_{args.dataset}_temperature{args.temperature}.pkl', 'wb') as outfile:
    pickle.dump(average_contradict_prob_list_beam, outfile)

with open(f'{config.output_dir}/{args.generation_model}_generations_semantic_density_beam_all_{args.dataset}_temperature{args.temperature}.pkl', 'wb') as outfile:
    pickle.dump(semantic_density_list_beam, outfile)
