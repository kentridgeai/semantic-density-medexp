import argparse
import json
import pickle
import os
import torch
from sklearn.metrics import roc_auc_score
import scipy

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='NQ')
parser.add_argument('--model', type=str, default='opt-350m')
parser.add_argument('--temperature', type=float, default=1.0)
args = parser.parse_args()

dataset_list = ['coqa', 'trivia_qa', 'sciq', 'NQ']
model_name_list = ['Llama-2-13b-hf', 'Llama-2-70b-hf', 'Meta-Llama-3-8B', 'Meta-Llama-3-70B',
        'Mistral-7B-v0.1', 'Mixtral-8x7B-v0.1', 'Mixtral-8x22B-v0.1']
key_list = ['entropy_over_concepts_auroc', 'average_contradict_prob_auroc', 'average_neg_log_likelihood_auroc',
        'ln_predictive_entropy_auroc', 'predictive_entropy_auroc', 'average_rougeL_auroc']
result_dict_statistical = {}
for dataset in dataset_list:
    for model_name in model_name_list:
        with open(f'results/overall_results_beam_search_{model_name}_{dataset}_temperature{args.temperature}.pkl', 'rb') as f:
            overall_result_dict_temperature = pickle.load(f)
        result_dict_temperature = overall_result_dict_temperature['beam_all']
        if 'semantic_density_auroc' not in result_dict_statistical.keys():
            result_dict_statistical['semantic_density_auroc'] = []
        result_dict_statistical['semantic_density_auroc'].append(result_dict_temperature['semantic_density_auroc'])

        with open(f'results/{model_name}_p_true_{args.dataset}.pkl', 'rb') as outfile:
            p_trues_across_beams, labels_across_beams = pickle.load(outfile)

        p_trues_all = []
        corrects_all = []
        for i in range(len(p_trues_across_beams)):
            p_trues_all += p_trues_across_beams[i]
            corrects_all += labels_across_beams[i]
        p_true_auroc_all = roc_auc_score(1 - torch.tensor(corrects_all), torch.tensor(p_trues_all))

        if 'p_true_auroc' not in result_dict_statistical.keys():
            result_dict_statistical['p_true_auroc'] = []
        result_dict_statistical['p_true_auroc'].append(p_true_auroc_all)

        with open(f'results/overall_results_beam_search_{model_name}_{args.dataset}_temperature1.0.pkl', 'rb') as f:
            overall_result_dict = pickle.load(f)
        result_dict = overall_result_dict['beam_all']
        for key in key_list:
            if key not in result_dict_statistical.keys():
                result_dict_statistical[key] = []
            result_dict_statistical[key].append(result_dict[key])

print(scipy.stats.ttest_rel(result_dict_statistical['semantic_density_auroc'], result_dict_statistical['entropy_over_concepts_auroc']))
print(scipy.stats.ttest_rel(result_dict_statistical['semantic_density_auroc'], result_dict_statistical['p_true_auroc']))
print(scipy.stats.ttest_rel(result_dict_statistical['semantic_density_auroc'], result_dict_statistical['average_contradict_prob_auroc']))
print(scipy.stats.ttest_rel(result_dict_statistical['semantic_density_auroc'], result_dict_statistical['average_neg_log_likelihood_auroc']))
print(scipy.stats.ttest_rel(result_dict_statistical['semantic_density_auroc'], result_dict_statistical['ln_predictive_entropy_auroc']))
print(scipy.stats.ttest_rel(result_dict_statistical['semantic_density_auroc'], result_dict_statistical['predictive_entropy_auroc']))


table_row = '{}&{}&{}&{}&{}&{}\\\\'.format(scipy.stats.ttest_rel(result_dict_statistical['semantic_density_auroc'], result_dict_statistical['entropy_over_concepts_auroc'])[1],
                                                 scipy.stats.ttest_rel(result_dict_statistical['semantic_density_auroc'], result_dict_statistical['p_true_auroc'])[1],
                                                 scipy.stats.ttest_rel(result_dict_statistical['semantic_density_auroc'], result_dict_statistical['average_contradict_prob_auroc'])[1],
                                                 scipy.stats.ttest_rel(result_dict_statistical['semantic_density_auroc'], result_dict_statistical['average_neg_log_likelihood_auroc'])[1],
                                                 scipy.stats.ttest_rel(result_dict_statistical['semantic_density_auroc'], result_dict_statistical['ln_predictive_entropy_auroc'])[1],
                                                 scipy.stats.ttest_rel(result_dict_statistical['semantic_density_auroc'], result_dict_statistical['predictive_entropy_auroc'])[1])
print(table_row)
with open("paper_results/Table_auroc_statistical_test_temperature{}.txt".format(args.temperature), "a+") as text_file:
    print(table_row, file=text_file)

