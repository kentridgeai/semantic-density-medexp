import argparse
import json
import pickle
import os
import torch
from sklearn.metrics import roc_auc_score
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--temperature', type=float, default=1.0)
args = parser.parse_args()

beam_num = 10
model_name_list = ['Llama-2-13b-hf', 'Llama-2-70b-hf', 'Meta-Llama-3-8B', 'Meta-Llama-3-70B',
        'Mistral-7B-v0.1', 'Mixtral-8x7B-v0.1', 'Mixtral-8x22B-v0.1']
dataset_list = ['coqa', 'trivia_qa', 'sciq', 'NQ']
key_list = ['entropy_over_concepts_auroc', 'average_contradict_prob_auroc', 'average_neg_log_likelihood_auroc',
        'ln_predictive_entropy_auroc', 'predictive_entropy_auroc', 'average_rougeL_auroc']
for model_name in model_name_list:
    for beam_id in range(beam_num):
        result_dict_datasets = {}
        for dataset in dataset_list:
            with open(f'results/overall_results_beam_search_{model_name}_{dataset}_temperature{args.temperature}.pkl', 'rb') as f:
                overall_result_dict_temperature = pickle.load(f)
            result_dict_temperature = overall_result_dict_temperature['beam_{}'.format(beam_id)]
            if 'semantic_density_auroc' not in result_dict_datasets.keys():
                result_dict_datasets['semantic_density_auroc'] = []
            result_dict_datasets['semantic_density_auroc'].append(result_dict_temperature['semantic_density_auroc'])

            with open(f'results_05172024/{model_name}_p_true_{dataset}.pkl', 'rb') as outfile:
                p_trues_across_beams, labels_across_beams = pickle.load(outfile)
            p_true_auroc = roc_auc_score(1 - torch.tensor(labels_across_beams[beam_id]), torch.tensor(p_trues_across_beams[beam_id]))
            if 'p_true_auroc' not in result_dict_datasets.keys():
                result_dict_datasets['p_true_auroc'] = []
            result_dict_datasets['p_true_auroc'].append(p_true_auroc)

            with open(f'results/overall_results_beam_search_{model_name}_{dataset}_temperature1.0.pkl', 'rb') as f:
                overall_result_dict = pickle.load(f)
            result_dict = overall_result_dict['beam_{}'.format(beam_id)]
            for key in key_list:
                if key not in result_dict_datasets.keys():
                    result_dict_datasets[key] = []
                result_dict_datasets[key].append(result_dict[key])


        table_row = '{}&{:.3f}&{:.3f}&{:.3f}&{:.3f}&{:.3f}&{:.3f}&{:.3f}&{:.3f}\\\\'.format(
                                                            model_name, np.mean(result_dict_datasets['semantic_density_auroc']), np.mean(result_dict_datasets['entropy_over_concepts_auroc']),
                                                            np.mean(result_dict_datasets['p_true_auroc']), np.mean(result_dict_datasets['average_contradict_prob_auroc']),
                                                            np.mean(result_dict_datasets['average_neg_log_likelihood_auroc']),
                                                            np.mean(result_dict_datasets['ln_predictive_entropy_auroc']), np.mean(result_dict_datasets['predictive_entropy_auroc']),
                                                            np.mean(result_dict_datasets['average_rougeL_auroc']))
        with open("paper_results/Table_auroc_temperature{}_group_average_over_datasets.txt".format(args.temperature), "a+") as text_file:
            print(beam_id, file=text_file)
            print(table_row, file=text_file)

