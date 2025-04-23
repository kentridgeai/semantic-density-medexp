import argparse
import pickle
import os
import matplotlib.pyplot as plt
import config
import sklearn
import sklearn.metrics
import torch

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='NQ')
parser.add_argument('--sample_num', type=int, default=10)
args = parser.parse_args()

# Define model names
model_name_list = ['Llama-2-7b-hf'] #'Mistral-7B-v0.1', 'Meta-Llama-3-8B','Llama-2-7b-hf'] #['meerkat7b',  ]#  ]#
dataset_list  = [ 'medexqa'] #['pubmedqa',

# Plot ROC Curves
plt.figure(figsize=(8, 6))

for dataset in dataset_list:
    for model_name in model_name_list:
        plt.clf()
        #         remove p_true metric
        with open(f'{config.output_dir}/{model_name}_p_true_{dataset}.pkl', 'rb') as outfile:
            p_trues_across_beams, labels_across_beams = pickle.load(outfile)

        p_trues_all = []
        corrects_all = []
        for i in range(len(p_trues_across_beams)):
            p_trues_all += p_trues_across_beams[i]
            corrects_all += labels_across_beams[i]
        roc_curve_path = f'{config.result_dir}/roc_curve_{model_name}_{dataset}_10sample_beam5.pkl'
        lables = ['Deg', 'SD', 'NE', 'PE', 'SE', 'NL', 'P(True)']
        if os.path.exists(roc_curve_path):
            with open(roc_curve_path, 'rb') as f:
                roc_curve_list = pickle.load(f)
            print(len(roc_curve_list))
            roc_curve_list.append(sklearn.metrics.roc_curve(1 - torch.tensor(corrects_all), torch.tensor(p_trues_all)))
            # Iterate through stored ROC curves
            for e, n in enumerate(roc_curve_list):
                print("plotted", f'{lables[e]} {model_name} on {dataset}')
                plt.plot(n[0], n[1], label=f'{lables[e]}')

        # Plot settings
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guessing')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves for {model_name} on {dataset}')
        plt.legend()
        plt.grid(True)
#         plt.show()
        plt.savefig(f'{config.paper_dir}/roccurves_{model_name}_{dataset}.png')





# import argparse
# import json
# import pickle
# import os
# import torch
# from sklearn.metrics import roc_auc_score
# import config

# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, default='NQ')
# parser.add_argument('--model', type=str, default='opt-350m')
# parser.add_argument('--temperature', type=float, default=1.0)
# parser.add_argument('--sample_num', type=int, default=10)
# args = parser.parse_args()

# model_name_list = ['Llama-2-13b-hf', 'Llama-2-70b-hf', 'Meta-Llama-3-8B', 'Meta-Llama-3-70B',
#         'Mistral-7B-v0.1', 'Mixtral-8x7B-v0.1', 'Mixtral-8x22B-v0.1']
# average_contradict_prob_dict = {}
# semantic_density_dict = {}
# for sample_num in range(args.sample_num):
#     for model_name in model_name_list:
#         with open(f'{config.result_dir}/roc_curve_{model_name}_{args.dataset}_10sample_beam5.pkl', 'rb') as f:
#             overall_result_dict_temperature = pickle.load(f)
#         result_dict_temperature = overall_result_dict_temperature[sample_num]['beam_all']
#         if model_name not in semantic_density_dict.keys():
#             semantic_density_dict[model_name]=[]
#             average_contradict_prob_dict[model_name]=[]
#         semantic_density_dict[model_name].append(result_dict_temperature['semantic_density_auroc'])
#         average_contradict_prob_dict[model_name].append(result_dict_temperature['average_contradict_prob_auroc'])
# print(semantic_density_dict)
# print(average_contradict_prob_dict)
