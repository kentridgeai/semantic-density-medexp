import argparse
import os
import pickle
import random
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--evaluation_model', type=str, default='opt-350m')
parser.add_argument('--generation_model', type=str, default='opt-350m')
parser.add_argument('--run_id', type=str, default='run_1')
parser.add_argument('--dataset', type=str, default='NQ')
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--cuda_device', type=str, default="0")
args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda_device


import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
import math


device = 'cuda'
import config

num_beams = 10
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
llama_models = ['Llama-2-13b-hf', 'Llama-2-70b-hf', 'Meta-Llama-3-8B', 'Meta-Llama-3-8B-Instruct', 'Meta-Llama-3-70B', 'Meta-Llama-3-70B-Instruct']


if f"{args.evaluation_model}" in mistralai_models:
    hf_model_dir = 'mistralai/' + f"{args.evaluation_model}"

if f"{args.evaluation_model}" in llama2_models:
    hf_model_dir = 'meta-llama/' + f"{args.evaluation_model}"


model = AutoModelForCausalLM.from_pretrained(hf_model_dir,
                                             torch_dtype=torch.float16,
                                             cache_dir=config.hf_cache_dir, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(hf_model_dir,
                                          use_fast=False,
                                          cache_dir=config.data_dir)

tokenizer.pad_token_id = 1

with open(f'{config.output_dir}/{args.generation_model}_generations_all_{args.dataset}.pkl', 'rb') as infile:
    sequences = pickle.load(infile)

with open(f'{config.output_dir}/{args.generation_model}_generations_similarities_all_{args.dataset}.pkl', 'rb') as infile:
    similarities_dict = pickle.load(infile)


def get_neg_loglikelihoods(model, sequences):

    with torch.no_grad():
        result = []
        for sample in tqdm.tqdm(sequences):
            result_dict = {}
            prompt = sample['prompt']
            if 'cleaned_generations' in sample:
                generations = sample['cleaned_generations'].to(device)
            else:
                generations = sample['generations'].to(device)
            id_ = sample['id']

            average_neg_log_likelihoods = torch.zeros((generations.shape[0],))
            average_unconditioned_neg_log_likelihoods = torch.zeros((generations.shape[0],))
            neg_log_likelihoods = torch.zeros((generations.shape[0],))
            neg_unconditioned_log_likelihoods = torch.zeros((generations.shape[0],))
            pointwise_mutual_information = torch.zeros((generations.shape[0],))
            sequence_embeddings = []

            for generation_index in range(generations.shape[0]):
                prompt = prompt[prompt != tokenizer.pad_token_id]
                generation = generations[generation_index][generations[generation_index] != tokenizer.pad_token_id]
                target_ids = generation.clone()
                target_ids[:len(prompt)] = -100
                if target_ids[-1] == 4:
                    target_ids = target_ids[:-1]
                    generation = generation[:-1]
                model_output = model(torch.reshape(generation, (1, -1)), labels=target_ids, output_hidden_states=True, temperature=args.temperature)
                generation_only = generation.clone()[(len(prompt) - 1):]
                unconditioned_model_output = model(torch.reshape(generation_only, (1, -1)),
                                                   labels=generation_only,
                                                   output_hidden_states=True,
                                                   temperature=args.temperature)
                hidden_states = model_output['hidden_states']
                average_neg_log_likelihood = model_output['loss']

                average_unconditioned_neg_log_likelihood = unconditioned_model_output['loss']
                average_neg_log_likelihoods[generation_index] = average_neg_log_likelihood
                average_unconditioned_neg_log_likelihoods[generation_index] = average_unconditioned_neg_log_likelihood
                neg_log_likelihoods[generation_index] = average_neg_log_likelihood * (len(generation) - len(prompt))
                neg_unconditioned_log_likelihoods[generation_index] = average_unconditioned_neg_log_likelihood * (
                    len(generation) - len(prompt))
                pointwise_mutual_information[generation_index] = -neg_log_likelihoods[
                    generation_index] + neg_unconditioned_log_likelihoods[generation_index]

                average_of_last_layer_token_embeddings = torch.mean(hidden_states[-1], dim=1)
                sequence_embeddings.append(average_of_last_layer_token_embeddings)

            most_likely_generation = sample['most_likely_generation_ids'][sample['most_likely_generation_ids'] != tokenizer.pad_token_id].to(device)
            target_ids = most_likely_generation.clone()
            target_ids[:len(prompt)] = -100
            if target_ids[-1] == 4:
                target_ids = target_ids[:-1]
                most_likely_generation = most_likely_generation[:-1]
            model_output = model(torch.reshape(most_likely_generation, (1, -1)),
                                 labels=target_ids,
                                 output_hidden_states=True,
                                 temperature=args.temperature)
            hidden_states = model_output['hidden_states']
            average_neg_log_likelihood_of_most_likely_gen = model_output['loss']
            most_likely_generation_embedding = torch.mean(hidden_states[-1], dim=1)
            second_most_likely_generation = sample['second_most_likely_generation_ids'][sample['second_most_likely_generation_ids'] != tokenizer.pad_token_id].to(device)
            target_ids = second_most_likely_generation.clone()
            target_ids[:len(prompt)] = -100
            if target_ids[-1] == 4:
                target_ids = target_ids[:-1]
                second_most_likely_generation = second_most_likely_generation[:-1]
            model_output = model(torch.reshape(second_most_likely_generation, (1, -1)),
                                 labels=target_ids,
                                 output_hidden_states=True,
                                 temperature=args.temperature)
            hidden_states = model_output['hidden_states']
            average_neg_log_likelihood_of_second_most_likely_gen = model_output['loss']
            second_most_likely_generation_embedding = torch.mean(hidden_states[-1], dim=1)
            neg_log_likelihood_of_most_likely_gen = average_neg_log_likelihood_of_most_likely_gen * (
                len(most_likely_generation) - len(prompt))

            average_neg_log_likelihood_of_beam_search = torch.zeros((num_beams,))
            for beam_index in range(num_beams):
                beam_search_generation = sample['cleaned_beam_search_generation_{}_ids'.format(beam_index)][sample['cleaned_beam_search_generation_{}_ids'.format(beam_index)] != tokenizer.pad_token_id].to(device)
                target_ids = beam_search_generation.clone()
                target_ids[:len(prompt)] = -100
                while target_ids[-1] == 4 or target_ids[-1] == 1437:
                    target_ids = target_ids[:-1]
                    beam_search_generation = beam_search_generation[:-1]
                model_output = model(torch.reshape(beam_search_generation, (1, -1)),
                                     labels=target_ids,
                                     output_hidden_states=True,
                                     temperature=args.temperature)
                hidden_states = model_output['hidden_states']
                result_dict['average_neg_log_likelihood_of_beam_search_gen_{}'.format(beam_index)]= model_output['loss']
                result_dict['beam_search_generation_embedding_{}'.format(beam_index)] = torch.mean(hidden_states[-1], dim=1)
                average_neg_log_likelihood_of_beam_search[beam_index] = model_output['loss']
            sequence_embeddings = torch.stack(sequence_embeddings)
            result_dict['prompt'] = prompt
            result_dict['generations'] = generations
            result_dict['average_neg_log_likelihoods'] = average_neg_log_likelihoods
            result_dict['neg_log_likelihoods'] = neg_log_likelihoods
            result_dict['sequence_embeddings'] = most_likely_generation_embedding
            result_dict['most_likely_sequence_embedding'] = most_likely_generation
            result_dict['average_unconditioned_neg_log_likelihoods'] = average_unconditioned_neg_log_likelihoods
            result_dict['neg_unconditioned_log_likelihoods'] = neg_unconditioned_log_likelihoods
            result_dict['pointwise_mutual_information'] = pointwise_mutual_information
            result_dict['average_neg_log_likelihood_of_most_likely_gen'] = average_neg_log_likelihood_of_most_likely_gen
            result_dict['average_neg_log_likelihood_of_second_most_likely_gen'] = average_neg_log_likelihood_of_second_most_likely_gen
            result_dict['neg_log_likelihood_of_most_likely_gen'] = neg_log_likelihood_of_most_likely_gen
            result_dict['semantic_set_ids'] = torch.tensor(similarities_dict[id_[0]]['semantic_set_ids'], device=device)
            result_dict['average_neg_log_likelihood_of_beam_search'] = average_neg_log_likelihood_of_beam_search
            result_dict['id'] = id_
            result.append(result_dict)

        return result


likelihoods = get_neg_loglikelihoods(model, sequences)

with open(f'{config.data_dir}/{args.generation_model}_generations_{args.evaluation_model}_likelihoods_all_{args.dataset}_temperature{args.temperature}.pkl',
          'wb') as outfile:
    pickle.dump(likelihoods, outfile)
