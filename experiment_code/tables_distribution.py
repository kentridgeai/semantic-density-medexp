# parse arguments
import argparse
import json
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--run_ids', nargs='+', default=[])
parser.add_argument('--verbose', type=bool, default=False)
parser.add_argument('--dataset', type=str, default='pubmedqa')
parser.add_argument('--model', type=str, default='Mistral-7B-v0.1')
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--cuda_device', type=str, default="0")
args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda_device

import config
import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
import torch
import wandb
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

device = f"cuda:{args.cuda_device}" if torch.cuda.is_available() and args.cuda_device != '-1' else 'cpu'

overall_result_dict = {}

aurocs_across_models = []

sequence_embeddings_dict = {}

# model_name = f'{args.model}'
model_list = ['Llama-2-7b-hf','Mistral-7B-v0.1', 'Meta-Llama-3-8B'] #['meerkat7b',  ]#  ]#
dataset_list  = [ 'medexqa', 'pubmedqa']





for ds in dataset_list:
    # Create subplots (2 rows, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(8, 4), sharey=True)  # Adjust figure size as needed    
    # Flatten axes for easy indexing
    axes = axes.flatten()    
    datalist = []
    titles = []
    results = []
    for model_name in model_list:

        with open(f'{config.output_dir}/{model_name}_generations_average_contradict_prob_beam_all_{ds}_temperature{args.temperature}.pkl', 'rb') as outfile:
            average_contradict_prob_list = pickle.load(outfile)

        with open(f'{config.output_dir}/{model_name}_generations_semantic_density_beam_all_{ds}_temperature{args.temperature}.pkl', 'rb') as outfile:
            semantic_density_list = pickle.load(outfile)


        import datasets
        if ds == 'coqa':
            dataset = datasets.load_from_disk(f'{config.data_dir}/coqa_dataset')
            id_to_question_mapping = dict(zip(dataset['id'], dataset['question']))
        elif ds == 'pubmedqa':
            dataset = datasets.load_from_disk(f'{config.data_dir}/pubmedqa_test_dataset')
            id_to_question_mapping = dict(zip(dataset['id'], dataset['question']))
        elif ds == 'medexqa':
            dataset = datasets.load_from_disk(f'{config.data_dir}/medexqa_test_dataset')    
        dataset_df = dataset.to_pandas()

        correct_all_list = []
        average_contradict_prob_all_list = []
        average_neg_log_likelihood_beam_search_all_list = []
        semantic_density_all_list = []
        average_predictive_entropy_all_list = []
        predictive_entropy_all_list = []
        predictive_entropy_over_concepts_all_list = []
        rougeL_among_generations_all_list = []

        # Helper function to compute AUROC safely
        def safe_auroc(y_true, y_score):
            y_true = 1 - np.array(y_true)  # Inverting correctness
            if len(set(y_true)) < 2:  # If only one class exists
                return np.nan  # or return 0.5 as a fallback
            return sklearn.metrics.roc_auc_score(y_true, y_score)

        with open(f'{config.output_dir}/{model_name}_generations_similarities_all_{ds}.pkl', 'rb') as f:
            similarities = pickle.load(f)
        similarities_df = pd.DataFrame.from_dict(similarities, orient='index')
        similarities_df['id'] = similarities_df.index
        print(similarities_df.columns.tolist())
        #     similarities_df['has_semantically_different_answers'] = similarities_df[
        #         'has_semantically_different_answers'].astype('int')
        similarities_df['rougeL_among_generations'] = similarities_df['syntactic_similarities'].apply(
            lambda x: x['rougeL'])

        """Get the generations df from the pickle file"""
        with open(f'{config.output_dir}/{model_name}_generations_all_{ds}.pkl', 'rb') as infile:
            generations = pickle.load(infile)
        generations_df = pd.DataFrame(generations)
        if not generations_df['semantic_variability_reference_answers'].isnull().values.any():
            generations_df['semantic_variability_reference_answers'] = generations_df[
                'semantic_variability_reference_answers'].apply(lambda x: x[0].item())
        generations_df = generations_df.merge(dataset_df, on='id', how='outer', suffixes =[None, '_y'])
        #         generations_df['id'] = generations_df['id'].apply(lambda x: x[0])
        generations_df['id'] = generations_df['id'].astype('object')
        for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
            if rouge_type in generations_df.columns.tolist():
                generations_df[rouge_type + '_reference_answers'] = generations_df[rouge_type]

            else:
                generations_df[rouge_type + '_reference_answers'] = None   

        #         if not generations_df['rougeL_reference_answers'].isnull().values.any():
        #             generations_df['rougeL_reference_answers'] = generations_df['rougeL_reference_answers'].apply(
        #                 lambda x: x[0].item())
        generations_df['length_of_most_likely_generation'] = generations_df['most_likely_generation'].apply(
            lambda x: len(str(x).split(' ')))
        generations_df['length_of_answer'] = generations_df['answer'].apply(lambda x: len(str(x).split(' ')))
        generations_df['variance_of_length_of_generations'] = generations_df['generated_texts'].apply(
            lambda x: np.var([len(str(y).split(' ')) for y in x]))
        # generations_df['correct'] = (generations_df['rougeL_beam_search_to_target_{}'.format(beam_id)] > 0.3).astype('int')
        num_generations = len(generations_df['generated_texts'][0])

        """Get the likelihoods df from the pickle file"""
        with open(f'{config.output_dir}/aggregated_likelihoods_{model_name}_generations_all_{ds}_temperature{args.temperature}.pkl', 'rb') as f:
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
        # likelihoods_small['average_contradict_prob'] = torch.Tensor(average_contradict_prob_list[beam_id])

        likelihoods_small['semantic_density'] = list(map(list, zip(*semantic_density_list))) #torch.Tensor(semantic_density_list[beam_id][entries] for entries in len(generations_df))
        sequence_embeddings = likelihoods['sequence_embeddings']

        likelihoods_df = pd.DataFrame.from_dict(likelihoods_small)
        likelihoods_df['avg_semantic_density'] = likelihoods_df['semantic_density'].apply(lambda x: sum(x) / len(x))

        likelihoods_df.rename(columns={'ids': 'id'}, inplace=True)

        run_ids_to_analyze = args.run_ids if len(args.run_ids)!=0 else [0,]
        run_id = 0
        result_df_list = []

        # similarities_df = get_similarities_df()
        # generations_df = get_generations_df()
        # likelihoods_df, sequence_embeddings = get_likelihoods_df(average_contradict_prob_list, semantic_density_list)
        num_generations = len(generations_df['generated_texts'][0])
        result_df = generations_df.merge(similarities_df, on='id').merge(likelihoods_df, on='id').dropna(subset=['predictive_entropy_over_concepts', 'average_predictive_entropy', 'average_neg_log_likelihood_of_most_likely_gen', 'average_neg_log_likelihood_of_second_most_likely_gen'])
        # new_dict = result_df.set_index('id')[['A', 'B']+[f'rougeL_beam_search_to_target_{idz}' for idz in range(10)]].apply(tuple, axis=1).to_dict()
        #     for run_id in run_ids_to_analyze:
        #         print("GOOD MORNING")
#         results.append(sum([result_df[f"rougeL_beam_search_to_target_{beam_id}"].mean() for beam_id in range(10)])/10)
#         results.append(result_df[f"rougeL_to_target"].mean())

#     table_row = '{}&{:.3f}&{:.3f}&{:.3f}\\\\'.format(ds, *results) 

#     with open("{}/Table_ROUGE_BEST_{}.txt".format(config.paper_dir, ds), "a+") as text_file:
#         print(ds+"\n"+table_row, file=text_file)
        
        
        metric_X = "avg_semantic_density"
        threshold = 0.3
        sums = [sum((result_df[f"rougeL_beam_search_to_target_{beam_id}"] > threshold).astype('int')==1) for beam_id in range(10)]
        beam_id = sums.index(max(sums))
        thresholding_Y = f"rougeL_beam_search_to_target_{beam_id}"
        
#         if sum((result_df[thresholding_Y] > threshold).astype('int')==1) != 0:

        result_df['correct'] = (result_df[thresholding_Y] > threshold).astype('int')
        positive_class_values = result_df[[metric_X, thresholding_Y]][result_df[thresholding_Y] > threshold]
        negative_class_values = result_df[[metric_X, thresholding_Y]][result_df[thresholding_Y] <= threshold]
        print(len(positive_class_values)) #,len(positive_class_values[0]))
        print(len(negative_class_values)) #,len(negative_class_values[0]))
        all_class_values = result_df[[metric_X, thresholding_Y]]
# all_class_values = result_df[[metric_X, thresholding_Y,'corr']]
# Assuming you have the values for each class
# sns.kdeplot(positive_class_values, x = thresholding_Y, y = metric_X, label='Positive Class', fill=True, color='orange')
# sns.kdeplot(negative_class_values, x = thresholding_Y, y = metric_X, label='Negative Class', fill=True, color ='blue')
        result_df.rename(columns={"avg_semantic_density": "Mean SD", thresholding_Y: "ROUGE-L"}, inplace=True)
        datalist.append(result_df[["Mean SD", "ROUGE-L",'correct']])
        titles.append(f"{model_name} (Beam {beam_id})")
        # Paired t-test between positive class and entire sample
#         t_stat_pos, p_value_pos = stats.ttest_ind(positive_class_values, all_class_values, equal_var=False)

#         # Paired t-test between negative class and entire sample
#         t_stat_neg, p_value_neg = stats.ttest_ind(negative_class_values, all_class_values, equal_var=False)

#         print(f"Paired t-test (Positive vs Entire Sample): t-stat = {t_stat_pos}, p-value = {p_value_pos}")
#         print(f"Paired t-test (Negative vs Entire Sample): t-stat = {t_stat_neg}, p-value = {p_value_neg}")

    # Plot KDEs in subplots
    for i, ax in enumerate(axes):
#         print(len(datalist))
        positive_class_len = len(datalist[i][["Mean SD", "ROUGE-L"]][datalist[i]["correct"] == 1])
        if positive_class_len < 5:
            sns.kdeplot(datalist[i][["Mean SD", "ROUGE-L"]][datalist[i]["correct"] == 0], ax=ax, fill=True, x = "ROUGE-L", y = "Mean SD", label='0')
            sns.scatterplot(datalist[i], ax=ax, x = "ROUGE-L", y = "Mean SD", hue = 'correct', alpha=0.5)            
#             sns.scatterplot(datalist[i][["Mean SD", "ROUGE-L"]][datalist[i]["correct"] == 1], ax=ax, x = "ROUGE-L", y = "Mean SD",label='1')
#             sns.scatterplot(datalist[i][["Mean SD", "ROUGE-L"]][datalist[i]["correct"] == 0], ax=ax, x = "ROUGE-L", y = "Mean SD",label='1', color = '#1f77b4')
            orange_patch = mpatches.Patch(color='#ff7f0e', label='1')
            blue_patch = mpatches.Patch(color='#1f77b4', label='0')
            ax.legend(handles=[blue_patch,orange_patch],loc='upper right', title='correct')
        else:
            sns.kdeplot(datalist[i], ax=ax, fill=True, x = "ROUGE-L", y = "Mean SD", hue = 'correct')
            sns.scatterplot(datalist[i], ax=ax, x = "ROUGE-L", y = "Mean SD", hue = 'correct', alpha=0.5)
            orange_patch = mpatches.Patch(color='#ff7f0e', label='1')
            blue_patch = mpatches.Patch(color='#1f77b4', label='0')
            ax.legend(handles=[blue_patch,orange_patch],loc='upper right', title='correct')
        ax.set_title(f"{titles[i]}")
    if "pubmed" in ds:
        fig.suptitle("PubMedQA results")
    else:
        fig.suptitle("MedExQA results")
    plt.savefig(f'{config.paper_dir}/distall_{ds}.png')

# sns.kdeplot(result_df, x = thresholding_Y, y = metric_X, hue = 'correct', label='Entire Sample')

# plt.legend()
# plt.savefig(f'{config.paper_dir}/distribution_{thresholding_Y}_{metric_X}.png')

#normal


 