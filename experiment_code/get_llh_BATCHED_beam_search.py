import argparse
import os
import pickle
import random
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='opt-350m')
parser.add_argument('--run_id', type=str, default='run_1')
parser.add_argument('--dataset', type=str, default='NQ')
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--cuda_device', type=str, default="0")
parser.add_argument('--batch_size', type=str, default="8")
args = parser.parse_args()
print(args)

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
import math
import config

device = f"cuda:{args.cuda_device}" if torch.cuda.is_available() and args.cuda_device != '-1' else 'cpu'

print(torch.cuda.device_count())
print(torch.cuda.is_available())

# Set a seed value
seed_value = 10
num_beams = 10
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random

random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

#Fix torch random seed
torch.manual_seed(seed_value)

os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache

mistralai_models = ['Mistral-7B-v0.1', 'Mixtral-8x7B-v0.1', 'Mixtral-8x22B-v0.1']
llama_models = ['Llama-2-13b-hf', 'Llama-2-70b-hf', 'Meta-Llama-3-8B', 'Meta-Llama-3-8B-Instruct', 'Meta-Llama-3-70B, Llama-2-7b-hf']
SUPPORTED_OTHER_LMS = ['phi3', 'meerkat7b']

if f"{args.model}" in mistralai_models:
    hf_model_dir = 'mistralai/' + f"{args.model}"

if f"{args.model}" in llama_models:
    hf_model_dir = 'meta-llama/' + f"{args.model}"

print("before loading model")
isInstructMdl = False
if f"{args.model}" in llama_models or f"{args.model}" in mistralai_models:
    # tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, use_fast=False, cache_dir=config.hf_cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, use_fast=False, cache_dir=config.hf_cache_dir, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(hf_model_dir,
                                                 torch_dtype=torch.float16,
                                                 cache_dir=config.hf_cache_dir)
    #                                              cache_dir=config.hf_cache_dir, device_map="auto")
    model.config.pad_token_id = model.config.eos_token_id

elif f"{args.model}" in SUPPORTED_OTHER_LMS:
    isInstructMdl = True
    if 'phi3' in f"{args.model}":
        tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-3-mini-4k-instruct',
                        truncation_side="left")
        model = Phi3ForCausalLM.from_pretrained("microsoft/phi-3-mini-4k-instruct")
#     elif 'mistral' in f"{args.model}":
#         tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1",
#                         truncation_side="left")
#         tokenizer.pad_token = self._tokenizer.eos_token
#         model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")                
    elif 'meerkat' in f"{args.model}":
        tokenizer = AutoTokenizer.from_pretrained("dmis-lab/meerkat-7b-v1.0",
                        truncation_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained("dmis-lab/meerkat-7b-v1.0") 

model.to(device)

with open(f'{config.output_dir}/{args.model}_generations_all_{args.dataset}.pkl', 'rb') as infile:
    sequences = pickle.load(infile)

with open(f'{config.output_dir}/{args.model}_generations_similarities_all_{args.dataset}.pkl', 'rb') as infile:
    similarities_dict = pickle.load(infile)


import torch
from tqdm import tqdm

def get_neg_loglikelihoods(model, sequences, batch_size=8):
    results = []
    
    with torch.no_grad():
        # Split sequences into batches
        batched_sequences = [sequences[i:i + batch_size] for i in range(0, len(sequences), batch_size)]
        
        for batch in tqdm(batched_sequences):
            result_batch = []
            batch_prompts = []
            batch_generations = []
            batch_ids = []

            for sample in batch:
                batch_prompts.append(sample['prompt'])
                batch_ids.append(sample['id'])
                
                if 'cleaned_generations' in sample:
                    batch_generations.append(sample['cleaned_generations'].to(device))
                else:
                    batch_generations.append(sample['generations'].to(device))

            batch_prompts = torch.stack(batch_prompts)
            batch_generations = torch.stack(batch_generations)  # (batch_size, num_generations, seq_len)

            # Prepare tensors for likelihood calculations
            avg_neg_log_likelihoods = torch.zeros((batch_generations.shape[0], batch_generations.shape[1]))
            avg_uncond_neg_log_likelihoods = torch.zeros((batch_generations.shape[0], batch_generations.shape[1]))
            neg_log_likelihoods = torch.zeros((batch_generations.shape[0], batch_generations.shape[1]))
            neg_uncond_log_likelihoods = torch.zeros((batch_generations.shape[0], batch_generations.shape[1]))
            pmi = torch.zeros((batch_generations.shape[0], batch_generations.shape[1]))
            sequence_embeddings = []

            for i in range(batch_generations.shape[0]):  # Iterate over samples in batch
                prompt = batch_prompts[i][batch_prompts[i] != tokenizer.pad_token_id]
                generations = batch_generations[i]

                for j in range(generations.shape[0]):  # Iterate over generations
                    generation = generations[j][generations[j] != tokenizer.pad_token_id]
                    target_ids = generation.clone()
                    target_ids[:len(prompt)] = -100

                    if target_ids[-1] == 4:
                        target_ids = target_ids[:-1]
                        generation = generation[:-1]

                    # Model forward pass (BATCHED)
                    model_output = model(generation.unsqueeze(0), labels=target_ids.unsqueeze(0), output_hidden_states=True)
                    unconditioned_output = model(generation[len(prompt)-1:].unsqueeze(0), labels=generation[len(prompt)-1:].unsqueeze(0), output_hidden_states=True)

                    # Compute likelihoods
                    avg_neg_log_likelihoods[i, j] = model_output['loss']
                    avg_uncond_neg_log_likelihoods[i, j] = unconditioned_output['loss']
                    neg_log_likelihoods[i, j] = avg_neg_log_likelihoods[i, j] * (len(generation) - len(prompt))
                    neg_uncond_log_likelihoods[i, j] = avg_uncond_neg_log_likelihoods[i, j] * (len(generation) - len(prompt))
                    pmi[i, j] = -neg_log_likelihoods[i, j] + neg_uncond_log_likelihoods[i, j]

                    # Extract embeddings
                    avg_last_layer_embeddings = torch.mean(model_output['hidden_states'][-1], dim=1)
                    sequence_embeddings.append(avg_last_layer_embeddings)

            # Convert embeddings list to tensor
            sequence_embeddings = torch.stack(sequence_embeddings)

            # Process most likely generation (batch-wise)
            most_likely_generations = torch.stack([sample['most_likely_generation_ids'][sample['most_likely_generation_ids'] != tokenizer.pad_token_id].to(device) for sample in batch])
            most_likely_targets = most_likely_generations.clone()
            for i in range(len(batch)):
                most_likely_targets[i][:len(batch_prompts[i])] = -100
                if most_likely_targets[i][-1] == 4:
                    most_likely_targets[i] = most_likely_targets[i][:-1]
                    most_likely_generations[i] = most_likely_generations[i][:-1]

            model_output = model(most_likely_generations, labels=most_likely_targets, output_hidden_states=True)
            most_likely_embeds = torch.mean(model_output['hidden_states'][-1], dim=1)
            avg_neg_log_likelihood_most_likely = model_output['loss']
            neg_log_likelihood_most_likely = avg_neg_log_likelihood_most_likely * (most_likely_generations.shape[1] - batch_prompts.shape[1])

            # Process beam search generations (batch-wise)
            avg_neg_log_likelihood_beam_search = torch.zeros((len(batch), num_beams))

            for beam_idx in range(num_beams):
                beam_generations = torch.stack([sample[f'cleaned_beam_search_generation_{beam_idx}_ids'][sample[f'cleaned_beam_search_generation_{beam_idx}_ids'] != tokenizer.pad_token_id].to(device) for sample in batch])
                beam_targets = beam_generations.clone()
                for i in range(len(batch)):
                    beam_targets[i][:len(batch_prompts[i])] = -100
                    while beam_targets[i][-1] in [4, 1437]:
                        beam_targets[i] = beam_targets[i][:-1]
                        beam_generations[i] = beam_generations[i][:-1]

                model_output = model(beam_generations, labels=beam_targets, output_hidden_states=True)
                avg_neg_log_likelihood_beam_search[:, beam_idx] = model_output['loss']

            # Store batch results
            for i, sample in enumerate(batch):
                result_dict = {
                    'id': batch_ids[i],
                    'prompt': batch_prompts[i],
                    'generations': batch_generations[i],
                    'average_neg_log_likelihoods': avg_neg_log_likelihoods[i],
                    'neg_log_likelihoods': neg_log_likelihoods[i],
                    'sequence_embeddings': sequence_embeddings[i],
                    'most_likely_sequence_embedding': most_likely_generations[i],
                    'average_unconditioned_neg_log_likelihoods': avg_uncond_neg_log_likelihoods[i],
                    'neg_unconditioned_log_likelihoods': neg_uncond_log_likelihoods[i],
                    'pointwise_mutual_information': pmi[i],
                    'average_neg_log_likelihood_of_most_likely_gen': avg_neg_log_likelihood_most_likely[i],
                    'neg_log_likelihood_of_most_likely_gen': neg_log_likelihood_most_likely[i],
                    'average_neg_log_likelihood_of_beam_search': avg_neg_log_likelihood_beam_search[i],
                }
                result_batch.append(result_dict)

            results.extend(result_batch)

    return results

likelihoods = get_neg_loglikelihoods(model, sequences, int(args.batch_size))

with open(f'{config.data_dir}/{args.model}_generations_{args.model}_likelihoods_all_{args.dataset}_temperature{args.temperature}.pkl',
          'wb') as outfile:
    pickle.dump(likelihoods, outfile)
