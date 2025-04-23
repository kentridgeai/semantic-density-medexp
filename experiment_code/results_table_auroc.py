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
args = parser.parse_args()

model_name_list = ['Llama-2-13b-hf', 'Llama-2-70b-hf', 'Meta-Llama-3-8B', 'Meta-Llama-3-70B',
        'Mistral-7B-v0.1', 'Mixtral-8x7B-v0.1', 'Mixtral-8x22B-v0.1']
with open("paper_results/Table_auroc_{}_temperature{}.txt".format(args.dataset, args.temperature), "a+") as text_file:
    print(args.dataset, file=text_file)
for model_name in model_name_list:
    with open(f'results/overall_results_beam_search_{model_name}_{args.dataset}_temperature{args.temperature}.pkl', 'rb') as f:
        overall_result_dict_temperature = pickle.load(f)
    result_dict_temperature = overall_result_dict_temperature['beam_all']

    with open(f'results/{model_name}_p_true_{args.dataset}.pkl', 'rb') as outfile:
        p_trues_across_beams, labels_across_beams = pickle.load(outfile)

    p_trues_all = []
    corrects_all = []
    for i in range(len(p_trues_across_beams)):
        p_trues_all += p_trues_across_beams[i]
        corrects_all += labels_across_beams[i]
    p_true_auroc_all = roc_auc_score(1 - torch.tensor(corrects_all), torch.tensor(p_trues_all))

    with open(f'results/overall_results_beam_search_{model_name}_{args.dataset}_temperature1.0.pkl', 'rb') as f:
        overall_result_dict = pickle.load(f)
    result_dict = overall_result_dict['beam_all']


    table_row = '{}&{:.3f}&{:.3f}&{:.3f}&{:.3f}&{:.3f}&{:.3f}&{:.3f}&{:.3f}\\\\'.format(
                                                        model_name, result_dict_temperature['semantic_density_auroc'], result_dict['entropy_over_concepts_auroc'],
                                                        p_true_auroc_all, result_dict['average_contradict_prob_auroc'], result_dict['average_neg_log_likelihood_auroc'],
                                                        result_dict['ln_predictive_entropy_auroc'], result_dict['predictive_entropy_auroc'], result_dict['average_rougeL_auroc'])
    with open("paper_results/Table_auroc_{}_temperature{}.txt".format(args.dataset, args.temperature), "a+") as text_file:
        print(table_row, file=text_file)

