import argparse
import os
import pickle
import random
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--generation_model', type=str, default='opt-350m')
parser.add_argument('--evaluation_model', type=str, default='opt-350m')
parser.add_argument('--run_id', type=str, default='run_1')
parser.add_argument('--verbose', type=bool, default=False)
parser.add_argument('--dataset', type=str, default='NQ')
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--cuda_device', type=str, default="0")
args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda_device

import config
import numpy as np
import torch
#import wandb


device = f"cuda:{args.cuda_device}" if torch.cuda.is_available() and args.cuda_device != '-1' else 'cpu'

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

llh_shift = torch.tensor(5.0)


def get_overall_log_likelihoods(list_of_results):
    """Compute log likelihood of all generations under their given context.

    list_of_results: list of dictionaries with keys:

    returns: dictionary with keys: 'neg_log_likelihoods', 'average_neg_log_likelihoods'
             that contains tensors of shape (num_models, num_generations, num_samples_per_generation)
    """

    result_dict = {}

    list_of_keys = ['neg_log_likelihoods', 'average_neg_log_likelihoods', 'sequence_embeddings',\
                    'pointwise_mutual_information', 'average_neg_log_likelihood_of_most_likely_gen',\
                    'neg_log_likelihood_of_most_likely_gen', 'semantic_set_ids',\
                    'average_neg_log_likelihood_of_second_most_likely_gen', 'average_neg_log_likelihood_of_beam_search',\
                    'neg_log_likelihood_of_beam_search']

    for key in list_of_keys:
        list_of_ids = []
        overall_results = []
        for model_size, result in list_of_results:
            results_per_model = []
            for sample in result:
                average_neg_log_likelihoods = sample[key]
                list_of_ids.append(sample['id'])
                results_per_model.append(average_neg_log_likelihoods)

            results_per_model = torch.stack(results_per_model)

            overall_results.append(results_per_model)

        if key != 'sequence_embeddings':
            overall_results = torch.stack(overall_results)

        result_dict[key] = overall_results

    result_dict['ids'] = list_of_ids
    return result_dict


def get_mutual_information(log_likelihoods):
    """Compute confidence measure for a given set of likelihoods"""

    mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - torch.log(torch.tensor(log_likelihoods.shape[0]))
    tiled_mean = mean_across_models.tile(log_likelihoods.shape[0], 1, 1)
    diff_term = torch.exp(log_likelihoods) * log_likelihoods - torch.exp(tiled_mean) * tiled_mean
    f_j = torch.div(torch.sum(diff_term, dim=0), diff_term.shape[0])
    mutual_information = torch.div(torch.sum(torch.div(f_j, mean_across_models), dim=1), f_j.shape[-1])

    return mutual_information


def get_log_likelihood_variance(neg_log_likelihoods):
    """Compute log likelihood variance of approximate posterior predictive"""
    mean_across_models = torch.mean(neg_log_likelihoods, dim=0)
    variance_of_neg_log_likelihoods = torch.var(mean_across_models, dim=1)

    return variance_of_neg_log_likelihoods


def get_log_likelihood_mean(neg_log_likelihoods):
    """Compute softmax variance of approximate posterior predictive"""
    mean_across_models = torch.mean(neg_log_likelihoods, dim=0)
    mean_of_neg_log_likelihoods = torch.mean(mean_across_models, dim=1)

    return mean_of_neg_log_likelihoods


def get_mean_of_poinwise_mutual_information(pointwise_mutual_information):
    """Compute mean of pointwise mutual information"""
    mean_across_models = torch.mean(pointwise_mutual_information, dim=0)
    return torch.mean(mean_across_models, dim=1)


def get_predictive_entropy(log_likelihoods):
    """Compute predictive entropy of approximate posterior predictive"""
    mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - torch.log(torch.tensor(log_likelihoods.shape[0]))
    entropy = -torch.sum(mean_across_models, dim=1) / torch.tensor(mean_across_models.shape[1])
    return entropy


def get_predictive_entropy_over_concepts(log_likelihoods, semantic_set_ids):
    """Compute the semantic entropy"""
    mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - torch.log(torch.tensor(log_likelihoods.shape[0]))
    # This is ok because all the models have the same semantic set ids
    semantic_set_ids = semantic_set_ids[0].to('cpu')
    entropies = []
    for row_index in range(mean_across_models.shape[0]):
        aggregated_likelihoods = []
        row = mean_across_models[row_index]
        semantic_set_ids_row = semantic_set_ids[row_index]
        for semantic_set_id in torch.unique(semantic_set_ids_row):
            if semantic_set_id!=-1:
                aggregated_likelihoods.append(torch.logsumexp(row[semantic_set_ids_row == semantic_set_id], dim=0))
        aggregated_likelihoods = torch.tensor(aggregated_likelihoods) - llh_shift
        entropy = - torch.sum(aggregated_likelihoods, dim=0) / torch.tensor(aggregated_likelihoods.shape[0])
        entropies.append(entropy)

    return torch.tensor(entropies)


def get_margin_probability_uncertainty_measure(log_likelihoods):
    """Compute margin probability uncertainty measure"""
    mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - torch.log(torch.tensor(log_likelihoods.shape[0]))
    topk_likelihoods, indices = torch.topk(mean_across_models, 2, dim=1, sorted=True)
    margin_probabilities = np.exp(topk_likelihoods[:, 0]) - np.exp(topk_likelihoods[:, 1])

    return margin_probabilities

def add_neg_log_likelihood_of_beam_search(sequences):
    with open(f'{config.output_dir}/{args.generation_model}_generations_all_{args.dataset}.pkl', 'rb') as infile:
        samples = pickle.load(infile)

    mistralai_models = ['Mistral-7B-v0.1', 'Mixtral-8x7B-v0.1', 'Mixtral-8x22B-v0.1']
    llama_models = ['Llama-2-13b-hf', 'Llama-2-70b-hf', 'Meta-Llama-3-8B', 'Meta-Llama-3-8B-Instruct', 'Meta-Llama-3-70B', 'Llama-2-7b-hf']
    SUPPORTED_OTHER_LMS = ['phi3', 'meerkat7b']
    if f"{args.evaluation_model}" in mistralai_models:
        hf_model_dir = 'mistralai/' + f"{args.evaluation_model}"

    if f"{args.evaluation_model}" in llama_models:
        hf_model_dir = 'meta-llama/' + f"{args.evaluation_model}"
    if f"{args.evaluation_model}" in SUPPORTED_OTHER_LMS:
        hf_model_dir = "dmis-lab/meerkat-7b-v1.0" if "meerkat" in args.evaluation_model else "microsoft/phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(hf_model_dir,
                                              use_fast=False,
                                              cache_dir=config.data_dir)
    tokenizer.pad_token = tokenizer.eos_token
    for i in range(len(samples)):
        sample = samples[i]
        sequence = sequences[i]

        prompt = sample['prompt']
        prompt = prompt[prompt != tokenizer.pad_token_id]

        num_beams = len(sequence['average_neg_log_likelihood_of_beam_search'])
        sequence['neg_log_likelihood_of_beam_search'] = torch.zeros((num_beams,))
        for beam_index in range(num_beams):
            beam_search_generation = sample['cleaned_beam_search_generation_{}_ids'.format(beam_index)][sample['cleaned_beam_search_generation_{}_ids'.format(beam_index)] != tokenizer.pad_token_id].to(device)
            sequence['neg_log_likelihood_of_beam_search'][beam_index] = sequence['average_neg_log_likelihood_of_beam_search'][beam_index] * (len(beam_search_generation) - len(prompt))

    return

list_of_results = []

with open(f'{config.output_dir}/{args.generation_model}_generations_{args.evaluation_model}_likelihoods_all_{args.dataset}_temperature{args.temperature}.pkl',
          'rb') as infile:
    sequences = pickle.load(infile)
    add_neg_log_likelihood_of_beam_search(sequences)
    list_of_results.append((args.evaluation_model, sequences))

overall_results = get_overall_log_likelihoods(list_of_results)
mutual_information = get_mutual_information(-overall_results['neg_log_likelihood_of_beam_search'])
predictive_entropy = get_predictive_entropy(-overall_results['neg_log_likelihood_of_beam_search'])
predictive_entropy_over_concepts = get_predictive_entropy_over_concepts(-overall_results['average_neg_log_likelihood_of_beam_search'],
                                                                        overall_results['semantic_set_ids'])
unnormalised_entropy_over_concepts = get_predictive_entropy_over_concepts(-overall_results['neg_log_likelihood_of_beam_search'],
                                                                          overall_results['semantic_set_ids'])

margin_measures = get_margin_probability_uncertainty_measure(-overall_results['average_neg_log_likelihood_of_beam_search'])
unnormalised_margin_measures = get_margin_probability_uncertainty_measure(-overall_results['neg_log_likelihood_of_beam_search'])


def get_number_of_unique_elements_per_row(tensor):
    assert len(tensor.shape) == 2
    tensor_copy = tensor.clone().detach()
    for tensor_row in tensor_copy:
        for i in range(len(tensor_row)):
            if tensor_row[i] == -1:
                tensor_row[i] = 0
    return torch.count_nonzero(torch.sum(torch.nn.functional.one_hot(tensor_copy), dim=1), dim=1)

number_of_semantic_sets = get_number_of_unique_elements_per_row(overall_results['semantic_set_ids'][0])
average_predictive_entropy = get_predictive_entropy(-overall_results['average_neg_log_likelihood_of_beam_search'])
average_predictive_entropy_on_subsets = []
predictive_entropy_on_subsets = []
semantic_predictive_entropy_on_subsets = []
num_predictions = overall_results['average_neg_log_likelihood_of_beam_search'].shape[-1]
number_of_semantic_sets_on_subsets = []
for i in range(1, num_predictions + 1):
    offset = num_predictions * (i / 100)
    average_predictive_entropy_on_subsets.append(
        get_predictive_entropy(-overall_results['average_neg_log_likelihood_of_beam_search'][:, :, :int(i)]))
    predictive_entropy_on_subsets.append(get_predictive_entropy(-overall_results['neg_log_likelihood_of_beam_search'][:, :, :int(i)]))
    semantic_predictive_entropy_on_subsets.append(
        get_predictive_entropy_over_concepts(-overall_results['average_neg_log_likelihood_of_beam_search'][:, :, :int(i)],
                                             overall_results['semantic_set_ids'][:, :, :int(i)]))
    number_of_semantic_sets_on_subsets.append(
        get_number_of_unique_elements_per_row(overall_results['semantic_set_ids'][0][:, :i]))

average_pointwise_mutual_information = get_mean_of_poinwise_mutual_information(
    overall_results['pointwise_mutual_information'])

overall_results['mutual_information'] = mutual_information
overall_results['predictive_entropy'] = predictive_entropy
overall_results['predictive_entropy_over_concepts'] = predictive_entropy_over_concepts
overall_results['unnormalised_entropy_over_concepts'] = unnormalised_entropy_over_concepts
overall_results['number_of_semantic_sets'] = number_of_semantic_sets
overall_results['margin_measures'] = margin_measures
overall_results['unnormalised_margin_measures'] = unnormalised_margin_measures

overall_results['average_predictive_entropy'] = average_predictive_entropy
for i in range(len(average_predictive_entropy_on_subsets)):
    overall_results[f'average_predictive_entropy_on_subset_{i + 1}'] = average_predictive_entropy_on_subsets[i]
    overall_results[f'predictive_entropy_on_subset_{i + 1}'] = predictive_entropy_on_subsets[i]
    overall_results[f'semantic_predictive_entropy_on_subset_{i + 1}'] = semantic_predictive_entropy_on_subsets[i]
    overall_results[f'number_of_semantic_sets_on_subset_{i + 1}'] = number_of_semantic_sets_on_subsets[i]
overall_results['average_pointwise_mutual_information'] = average_pointwise_mutual_information

with open(f'{config.output_dir}/aggregated_likelihoods_{args.generation_model}_generations_all_{args.dataset}_temperature{args.temperature}.pkl',
          'wb') as outfile:
    pickle.dump(overall_results, outfile)

if args.verbose:
    print('Margin measure', margin_measures)
    print('Number of semantic sets', number_of_semantic_sets)
    print('predicitve entropy shape: ', predictive_entropy.shape)
    print('predicitve entropy per concept shape: ', predictive_entropy_over_concepts.shape)
    print(overall_results['average_neg_log_likelihoods'].shape)
    print(len(number_of_semantic_sets_on_subsets))
    print(number_of_semantic_sets_on_subsets[0].shape)
    print('average predictive entropy on subsets: ', len(average_predictive_entropy_on_subsets))
    print(average_predictive_entropy_on_subsets[0].shape)
    print(overall_results['pointwise_mutual_information'])
    print(overall_results['margin_measures'])
