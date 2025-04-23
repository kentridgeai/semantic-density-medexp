# parse arguments
import argparse
import json
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--run_ids', nargs='+', default=[])
parser.add_argument('--verbose', type=bool, default=False)
parser.add_argument('--dataset', type=str, default='NQ')
parser.add_argument('--model', type=str, default='opt-350m')
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--cuda_device', type=str, default="0")
parser.add_argument('--sample_num', type=int, default=10)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda_device

import config
import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
import torch
import wandb

model_name = f'{args.model}'

with open(f'{config.output_dir}/{model_name}_generations_average_contradict_prob_beam_all_{args.dataset}_temperature{args.temperature}_sample_num.pkl', 'rb') as outfile:
    average_contradict_prob_list = pickle.load(outfile)

with open(f'{config.output_dir}/{model_name}_generations_semantic_density_beam_all_{args.dataset}_temperature{args.temperature}_sample_num.pkl', 'rb') as outfile:
    semantic_density_list = pickle.load(outfile)

run_ids_to_analyze = args.run_ids

result_dict_sample_num = []
for sample_num in range(args.sample_num):
    correct_all_list = []
    average_contradict_prob_all_list = []
    semantic_density_all_list = []
    overall_result_dict = {}
    for beam_id in range(10):
        for run_id in run_ids_to_analyze:
            print('beam_group_{}'.format(beam_id))

            def get_similarities_df():
                """Get the similarities df from the pickle file"""
                with open(f'{config.output_dir}/{model_name}_generations_similarities_all_{args.dataset}.pkl', 'rb') as f:
                    similarities = pickle.load(f)
                    similarities_df = pd.DataFrame.from_dict(similarities, orient='index')
                    similarities_df['id'] = similarities_df.index
                    similarities_df['has_semantically_different_answers'] = similarities_df[
                        'has_semantically_different_answers'].astype('int')
                    similarities_df['rougeL_among_generations'] = similarities_df['syntactic_similarities'].apply(
                        lambda x: x['rougeL'])

                    return similarities_df

            def get_generations_df():
                """Get the generations df from the pickle file"""
                with open(f'{config.output_dir}/{model_name}_generations_all_{args.dataset}.pkl', 'rb') as infile:
                    generations = pickle.load(infile)
                    generations_df = pd.DataFrame(generations)
                    generations_df['id'] = generations_df['id'].apply(lambda x: x[0])
                    generations_df['id'] = generations_df['id'].astype('object')
                    if not generations_df['semantic_variability_reference_answers'].isnull().values.any():
                        generations_df['semantic_variability_reference_answers'] = generations_df[
                            'semantic_variability_reference_answers'].apply(lambda x: x[0].item())

                    if not generations_df['rougeL_reference_answers'].isnull().values.any():
                        generations_df['rougeL_reference_answers'] = generations_df['rougeL_reference_answers'].apply(
                            lambda x: x[0].item())
                    generations_df['length_of_most_likely_generation'] = generations_df['most_likely_generation'].apply(
                        lambda x: len(str(x).split(' ')))
                    generations_df['length_of_answer'] = generations_df['answer'].apply(lambda x: len(str(x).split(' ')))
                    generations_df['variance_of_length_of_generations'] = generations_df['generated_texts'].apply(
                        lambda x: np.var([len(str(y).split(' ')) for y in x]))
                    generations_df['correct'] = (generations_df['rougeL_beam_search_to_target_{}'.format(beam_id)] > 0.3).astype('int')
                    generations_df['beam_search_results_list'] = [[] for r in range(generations_df.shape[0])]
                    for sample_index in range(generations_df.shape[0]):
                        for beam_index in range(10):
                            generations_df['beam_search_results_list'][sample_index].append(generations_df['cleaned_beam_search_generation_{}'.format(beam_index)][sample_index])

                    return generations_df

            def get_likelihoods_df(average_contradict_prob_list, semantic_density_list):
                """Get the likelihoods df from the pickle file"""

                with open(f'{config.output_dir}/aggregated_likelihoods_{model_name}_generations_all_{args.dataset}_temperature{args.temperature}.pkl', 'rb') as f:
                    likelihoods = pickle.load(f)

                    subset_keys = ['average_predictive_entropy_on_subset_' + str(i) for i in range(1, num_generations + 1)]
                    subset_keys += ['predictive_entropy_on_subset_' + str(i) for i in range(1, num_generations + 1)]
                    subset_keys += ['semantic_predictive_entropy_on_subset_' + str(i) for i in range(1, num_generations + 1)]
                    subset_keys += ['number_of_semantic_sets_on_subset_' + str(i) for i in range(1, num_generations + 1)]

                    keys_to_use = ('ids', 'predictive_entropy', 'mutual_information', 'average_predictive_entropy',\
                                    'average_pointwise_mutual_information', 'average_neg_log_likelihood_of_most_likely_gen',\
                                    'average_neg_log_likelihood_of_second_most_likely_gen', 'neg_log_likelihood_of_most_likely_gen',\
                                    'predictive_entropy_over_concepts', 'number_of_semantic_sets', 'unnormalised_entropy_over_concepts')

                    likelihoods_small = dict((k, likelihoods[k]) for k in keys_to_use + tuple(subset_keys))
                    for key in likelihoods_small:
                        if key == 'average_predictive_entropy_on_subsets':
                            likelihoods_small[key].shape
                        if type(likelihoods_small[key]) is torch.Tensor:
                            likelihoods_small[key] = torch.squeeze(likelihoods_small[key].cpu())
                    likelihoods_small['average_contradict_prob'] = torch.Tensor(np.array(average_contradict_prob_list[beam_id])[:,sample_num])

                    likelihoods_small['semantic_density'] = torch.Tensor(np.array(semantic_density_list[beam_id])[:,sample_num])

                    likelihoods_df = pd.DataFrame.from_dict(likelihoods_small)

                    likelihoods_df.rename(columns={'ids': 'id'}, inplace=True)

                    return likelihoods_df

            similarities_df = get_similarities_df()
            generations_df = get_generations_df()
            num_generations = len(generations_df['generated_texts'][0])
            likelihoods_df = get_likelihoods_df(average_contradict_prob_list, semantic_density_list)
            result_df = generations_df.merge(similarities_df, on='id').merge(likelihoods_df, on='id').dropna(subset=['predictive_entropy_over_concepts', 'average_predictive_entropy', 'average_neg_log_likelihood_of_most_likely_gen', 'average_neg_log_likelihood_of_second_most_likely_gen'])

            # Begin analysis
            result_dict = {}
            result_dict['accuracy'] = result_df['correct'].mean()

            # Compute the auroc for semantic density
            average_contradict_prob_auroc = sklearn.metrics.roc_auc_score(1 - result_df['correct'],
                                                                        result_df['average_contradict_prob'])
            result_dict['average_contradict_prob_auroc'] = average_contradict_prob_auroc
            semantic_density_auroc = sklearn.metrics.roc_auc_score(1 - result_df['correct'],
                                                                    1 - result_df['semantic_density'])
            result_dict['semantic_density_auroc'] = semantic_density_auroc
            overall_result_dict['beam_{}'.format(beam_id)] = result_dict

            torch.cuda.empty_cache()

            correct_all_list+=result_df['correct'].to_list()
            average_contradict_prob_all_list+=result_df['average_contradict_prob'].to_list()
            semantic_density_all_list+=result_df['semantic_density'].to_list()

    result_dict = {}
    result_dict['accuracy'] = np.mean(correct_all_list)
    result_dict['average_contradict_prob_auroc'] = sklearn.metrics.roc_auc_score(1 - np.array(correct_all_list),
                                                                np.array(average_contradict_prob_all_list))
    result_dict['semantic_density_auroc'] = sklearn.metrics.roc_auc_score(1 - np.array(correct_all_list),
                                                                1 - np.array(semantic_density_all_list))
    overall_result_dict['beam_all'] = result_dict

    result_dict_sample_num.append(overall_result_dict)
print(result_dict_sample_num)
with open(f'results/overall_results_beam_search_{args.model}_{args.dataset}_temperature{args.temperature}_sample_num{args.sample_num}.pkl', 'wb') as f:
    pickle.dump(result_dict_sample_num, f)
