import argparse
import json
import pickle
import os
import torch
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='NQ')
parser.add_argument('--model', type=str, default='opt-350m')
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--sample_num', type=int, default=10)
args = parser.parse_args()

model_name_list = ['Llama-2-13b-hf', 'Llama-2-70b-hf', 'Meta-Llama-3-8B', 'Meta-Llama-3-70B',
        'Mistral-7B-v0.1', 'Mixtral-8x7B-v0.1', 'Mixtral-8x22B-v0.1']
average_contradict_prob_dict = {}
semantic_density_dict = {}
for sample_num in range(args.sample_num):
    for model_name in model_name_list:
        with open(f'results/overall_results_beam_search_{model_name}_{args.dataset}_temperature{args.temperature}_sample_num{args.sample_num}.pkl', 'rb') as f:
            overall_result_dict_temperature = pickle.load(f)
        result_dict_temperature = overall_result_dict_temperature[sample_num]['beam_all']
        if model_name not in semantic_density_dict.keys():
            semantic_density_dict[model_name]=[]
            average_contradict_prob_dict[model_name]=[]
        semantic_density_dict[model_name].append(result_dict_temperature['semantic_density_auroc'])
        average_contradict_prob_dict[model_name].append(result_dict_temperature['average_contradict_prob_auroc'])
print(semantic_density_dict)
print(average_contradict_prob_dict)
