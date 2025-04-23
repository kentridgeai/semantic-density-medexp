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
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda_device

import config
import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
import torch
import wandb


overall_result_dict = {}

aurocs_across_models = []

sequence_embeddings_dict = {}

model_name = f'{args.model}'

with open(f'{config.output_dir}/{model_name}_generations_average_contradict_prob_beam_all_{args.dataset}_temperature{args.temperature}.pkl', 'rb') as outfile:
    average_contradict_prob_list = pickle.load(outfile)

with open(f'{config.output_dir}/{model_name}_generations_semantic_density_beam_all_{args.dataset}_temperature{args.temperature}.pkl', 'rb') as outfile:
    semantic_density_list = pickle.load(outfile)

correct_all_list = []
average_contradict_prob_all_list = []
average_neg_log_likelihood_beam_search_all_list = []
semantic_density_all_list = []
average_predictive_entropy_all_list = []
predictive_entropy_all_list = []
predictive_entropy_over_concepts_all_list = []
rougeL_among_generations_all_list = []


run_ids_to_analyze = args.run_ids
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
                likelihoods_small['average_contradict_prob'] = torch.Tensor(average_contradict_prob_list[beam_id])

                likelihoods_small['semantic_density'] = torch.Tensor(semantic_density_list[beam_id])

                sequence_embeddings = likelihoods['sequence_embeddings']

                likelihoods_df = pd.DataFrame.from_dict(likelihoods_small)

                likelihoods_df.rename(columns={'ids': 'id'}, inplace=True)

                return likelihoods_df, sequence_embeddings

        similarities_df = get_similarities_df()
        generations_df = get_generations_df()
        num_generations = len(generations_df['generated_texts'][0])
        likelihoods_df, sequence_embeddings = get_likelihoods_df(average_contradict_prob_list, semantic_density_list)
        result_df = generations_df.merge(similarities_df, on='id').merge(likelihoods_df, on='id').dropna(subset=['predictive_entropy_over_concepts', 'average_predictive_entropy', 'average_neg_log_likelihood_of_most_likely_gen', 'average_neg_log_likelihood_of_second_most_likely_gen'])

        n_samples_before_filtering = len(result_df)
        result_df['len_most_likely_generation_length'] = result_df['most_likely_generation'].apply(lambda x: len(x.split()))

        # Begin analysis
        result_dict = {}
        result_dict['accuracy'] = result_df['correct'].mean()

        # Compute the auroc for semantic density
        average_contradict_prob_auroc = sklearn.metrics.roc_auc_score(1 - result_df['correct'],
                                                                    result_df['average_contradict_prob'])
        result_dict['average_contradict_prob_auroc'] = average_contradict_prob_auroc
        with open(f'{config.data_dir}/{model_name}_generations_{model_name}_likelihoods_all_{args.dataset}_temperature{args.temperature}.pkl','rb') as infile:
            likelihoods_all = pickle.load(infile)
        average_neg_log_likelihood_beam_search_list = []
        for index_tmp in result_df.index:
            average_neg_log_likelihood_beam_search_list.append(likelihoods_all[index_tmp]['average_neg_log_likelihood_of_beam_search_gen_{}'.format(beam_id)].cpu())
        average_log_likelihood_auroc = sklearn.metrics.roc_auc_score(1 - result_df['correct'],
                                                                    average_neg_log_likelihood_beam_search_list)
        result_dict['average_neg_log_likelihood_auroc'] = average_log_likelihood_auroc
        roc_curve_list = []
        roc_curve_list.append(sklearn.metrics.roc_curve(1 - result_df['correct'], result_df['average_contradict_prob']))

        semantic_density_auroc = sklearn.metrics.roc_auc_score(1 - result_df['correct'],
                                                                1 - result_df['semantic_density'])
        result_dict['semantic_density_auroc'] = semantic_density_auroc
        roc_curve_list.append(sklearn.metrics.roc_curve(1 - result_df['correct'], 1 - result_df['semantic_density']))


        # Compute the auroc for the length normalized predictive entropy
        ln_predictive_entropy_auroc = sklearn.metrics.roc_auc_score(1 - result_df['correct'],
                                                                    result_df['average_predictive_entropy'])
        result_dict['ln_predictive_entropy_auroc'] = ln_predictive_entropy_auroc

        predictive_entropy_auroc = sklearn.metrics.roc_auc_score(1 - result_df['correct'], result_df['predictive_entropy'])
        result_dict['predictive_entropy_auroc'] = predictive_entropy_auroc

        entropy_over_concepts_auroc = sklearn.metrics.roc_auc_score(1 - result_df['correct'],
                                                                    result_df['predictive_entropy_over_concepts'])
        result_dict['entropy_over_concepts_auroc'] = entropy_over_concepts_auroc
        roc_curve_list.append(sklearn.metrics.roc_curve(1 - result_df['correct'], result_df['predictive_entropy_over_concepts']))
        with open('results/roc_curve_{}_10sample_beam5.pkl'.format(model_name), 'wb') as f:
            pickle.dump(roc_curve_list, f)

        if 'unnormalised_entropy_over_concepts' in result_df.columns:
            unnormalised_entropy_over_concepts_auroc = sklearn.metrics.roc_auc_score(
                1 - result_df['correct'], result_df['unnormalised_entropy_over_concepts'])
            result_dict['unnormalised_entropy_over_concepts_auroc'] = unnormalised_entropy_over_concepts_auroc

        aurocs_across_models.append(entropy_over_concepts_auroc)

        neg_llh_most_likely_gen_auroc = sklearn.metrics.roc_auc_score(1 - result_df['correct'],
                                                                      result_df['neg_log_likelihood_of_most_likely_gen'])
        result_dict['neg_llh_most_likely_gen_auroc'] = neg_llh_most_likely_gen_auroc

        number_of_semantic_sets_auroc = sklearn.metrics.roc_auc_score(1 - result_df['correct'],
                                                                      result_df['number_of_semantic_sets'])
        result_dict['number_of_semantic_sets_auroc'] = number_of_semantic_sets_auroc

        result_dict['number_of_semantic_sets_correct'] = result_df[result_df['correct'] ==
                                                                   1]['number_of_semantic_sets'].mean()
        result_dict['number_of_semantic_sets_incorrect'] = result_df[result_df['correct'] ==
                                                                     0]['number_of_semantic_sets'].mean()

        result_dict['average_rougeL_among_generations'] = result_df['rougeL_among_generations'].mean()
        result_dict['average_rougeL_among_generations_correct'] = result_df[result_df['correct'] ==
                                                                            1]['rougeL_among_generations'].mean()
        result_dict['average_rougeL_among_generations_incorrect'] = result_df[result_df['correct'] ==
                                                                              0]['rougeL_among_generations'].mean()
        result_dict['average_rougeL_auroc'] = sklearn.metrics.roc_auc_score(result_df['correct'],
                                                                            result_df['rougeL_among_generations'])

        average_neg_llh_most_likely_gen_auroc = sklearn.metrics.roc_auc_score(
            1 - result_df['correct'], result_df['average_neg_log_likelihood_of_most_likely_gen'])
        result_dict['average_neg_llh_most_likely_gen_auroc'] = average_neg_llh_most_likely_gen_auroc
        result_dict['rougeL_based_accuracy'] = result_df['correct'].mean()

        result_dict['margin_measure_auroc'] = sklearn.metrics.roc_auc_score(
            1 - result_df['correct'], result_df['average_neg_log_likelihood_of_most_likely_gen'] +
            result_df['average_neg_log_likelihood_of_second_most_likely_gen'])

        if args.verbose:
            print('ln_predictive_entropy_auroc', ln_predictive_entropy_auroc)
            print('semantci entropy auroc', entropy_over_concepts_auroc)
            print('average_log_likelihood_auroc', average_log_likelihood_auroc)
            print('average_contradict_prob_auroc', average_contradict_prob_auroc)
            print('semantic_density_auroc', semantic_density_auroc)
            print(
                'Semantic entropy +',
                sklearn.metrics.roc_auc_score(
                    1 - result_df['correct'],
                    result_df['predictive_entropy_over_concepts'] - 3 * result_df['rougeL_among_generations']))
            print('RougeL among generations auroc',
                  sklearn.metrics.roc_auc_score(result_df['correct'], result_df['rougeL_among_generations']))
            print('margin measure auroc:', result_dict['margin_measure_auroc'])
            print('accuracy:', result_dict['accuracy'])

        # Measure the AURROCs when using different numbers of generations to compute our uncertainty measures.
        ln_aurocs = []
        aurocs = []
        semantic_aurocs = []
        average_number_of_semantic_sets = []
        average_number_of_semantic_sets_correct = []
        average_number_of_semantic_sets_incorrect = []
        for i in range(1, num_generations + 1):
            ln_predictive_entropy_auroc = sklearn.metrics.roc_auc_score(
                1 - result_df['correct'], result_df['average_predictive_entropy_on_subset_{}'.format(i)])
            aurocs.append(
                sklearn.metrics.roc_auc_score(1 - result_df['correct'],
                                              result_df['predictive_entropy_on_subset_{}'.format(i)]))
            ln_aurocs.append(ln_predictive_entropy_auroc)
            semantic_aurocs.append(
                sklearn.metrics.roc_auc_score(1 - result_df['correct'],
                                              result_df['semantic_predictive_entropy_on_subset_{}'.format(i)]))
            average_number_of_semantic_sets.append(result_df['number_of_semantic_sets_on_subset_{}'.format(i)].mean())
            average_number_of_semantic_sets_correct.append(
                result_df[result_df['correct'] == 1]['number_of_semantic_sets_on_subset_{}'.format(i)].mean())
            average_number_of_semantic_sets_incorrect.append(
                result_df[result_df['correct'] == 0]['number_of_semantic_sets_on_subset_{}'.format(i)].mean())

        result_dict['ln_predictive_entropy_auroc_on_subsets'] = ln_aurocs
        result_dict['predictive_entropy_auroc_on_subsets'] = aurocs
        result_dict['semantic_predictive_entropy_auroc_on_subsets'] = semantic_aurocs
        result_dict['average_number_of_semantic_sets_on_subsets'] = average_number_of_semantic_sets
        result_dict['average_number_of_semantic_sets_on_subsets_correct'] = average_number_of_semantic_sets_correct
        result_dict['average_number_of_semantic_sets_on_subsets_incorrect'] = average_number_of_semantic_sets_incorrect
        result_dict['model_name'] = model_name


        overall_result_dict['beam_{}'.format(beam_id)] = result_dict
        sequence_embeddings_dict[run_id] = sequence_embeddings

        torch.cuda.empty_cache()

        correct_all_list+=result_df['correct'].to_list()
        average_contradict_prob_all_list+=result_df['average_contradict_prob'].to_list()
        average_neg_log_likelihood_beam_search_all_list+=average_neg_log_likelihood_beam_search_list
        semantic_density_all_list+=result_df['semantic_density'].to_list()
        average_predictive_entropy_all_list+=result_df['average_predictive_entropy'].to_list()
        predictive_entropy_all_list+=result_df['predictive_entropy'].to_list()
        predictive_entropy_over_concepts_all_list+=result_df['predictive_entropy_over_concepts'].to_list()
        rougeL_among_generations_all_list+=result_df['rougeL_among_generations'].to_list()

result_dict = {}
result_dict['accuracy'] = np.mean(correct_all_list)
result_dict['average_contradict_prob_auroc'] = sklearn.metrics.roc_auc_score(1 - np.array(correct_all_list),
                                                            np.array(average_contradict_prob_all_list))
result_dict['semantic_density_auroc'] = sklearn.metrics.roc_auc_score(1 - np.array(correct_all_list),
                                                            1 - np.array(semantic_density_all_list))
result_dict['ln_predictive_entropy_auroc'] = sklearn.metrics.roc_auc_score(1 - np.array(correct_all_list),
                                                            np.array(average_predictive_entropy_all_list))
result_dict['predictive_entropy_auroc'] = sklearn.metrics.roc_auc_score(1 - np.array(correct_all_list),
                                                            np.array(predictive_entropy_all_list))
result_dict['entropy_over_concepts_auroc'] = sklearn.metrics.roc_auc_score(1 - np.array(correct_all_list),
                                                            np.array(predictive_entropy_over_concepts_all_list))
result_dict['average_neg_log_likelihood_auroc'] = sklearn.metrics.roc_auc_score(1 - np.array(correct_all_list),
                                                            np.array(average_neg_log_likelihood_beam_search_all_list))
result_dict['average_rougeL_auroc'] = sklearn.metrics.roc_auc_score(1 - np.array(correct_all_list),
                                                            1 - np.array(rougeL_among_generations_all_list))
overall_result_dict['beam_all'] = result_dict
print(result_dict)
with open(f'results/overall_results_beam_search_{args.model}_{args.dataset}_temperature{args.temperature}.pkl', 'wb') as f:
    pickle.dump(overall_result_dict, f)

with open('sequence_embeddings.pkl', 'wb') as f:
    pickle.dump(sequence_embeddings_dict, f)

# Store data frame as csv
accuracy_verification_df = result_df[['most_likely_generation', 'answer', 'correct']]
accuracy_verification_df.to_csv('accuracy_verification.csv')
