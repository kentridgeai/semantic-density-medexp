import argparse
import json
import pickle
import os
import torch
from sklearn.metrics import roc_auc_score
import config

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='NQ')
parser.add_argument('--model', type=str, default='opt-350m')
parser.add_argument('--temperature', type=float, default=0.1)
args = parser.parse_args()

# model_name_list = ['Llama-2-13b-hf', 'Llama-2-70b-hf', 'Meta-Llama-3-8B', 'Meta-Llama-3-70B',
#         'Mistral-7B-v0.1', 'Mixtral-8x7B-v0.1', 'Mixtral-8x22B-v0.1']

model_name_list = ['Mistral-7B-v0.1', 'Meta-Llama-3-8B','Llama-2-7b-hf'] #['meerkat7b', 
dataset_names  = ['pubmedqa', 'medexqa']

                   
for model_name in model_name_list:
    for dataset_name in dataset_names:
#         with open("{}/Table_auroc_{}_temperature{}.txt".format(config.paper_dir, dataset_name, args.temperature), "a+") as text_file:
#             print(dataset_name, file=text_file)        
        with open(f'{config.result_dir}/overall_results_beam_search_{model_name}_{dataset_name}_temperature{args.temperature}.pkl', 'rb') as f:
            overall_result_dict_temperature = pickle.load(f)
        result_dict_temperature = overall_result_dict_temperature['beam_all']
#         print(model_name, dataset_name, result_dict_temperature['accuracy'])
        
#         remove p_true metric
        with open(f'{config.output_dir}/{model_name}_p_true_{dataset_name}.pkl', 'rb') as outfile:
            p_trues_across_beams, labels_across_beams = pickle.load(outfile)

        p_trues_all = []
        corrects_all = []
        for i in range(len(p_trues_across_beams)):
            p_trues_all += p_trues_across_beams[i]
            corrects_all += labels_across_beams[i]
        p_true_auroc_all = roc_auc_score(1 - torch.tensor(corrects_all), torch.tensor(p_trues_all))

#         with open(f'{config.result_dir}/overall_results_beam_search_{model_name}_{dataset_name}_temperature1.0.pkl', 'rb') as f:
#             overall_result_dict = pickle.load(f)
#         result_dict = overall_result_dict['beam_all']

#         &{:.3f}

        table_row = '{}&{:.3f}&{:.3f}&{:.3f}&{:.3f}&{:.3f}&{:.3f}&{:.3f}\\\\'.format(
                                                                                model_name, 
                                                                                result_dict_temperature['semantic_density_auroc'], 
                                                                                result_dict_temperature['entropy_over_concepts_auroc'],
                                                                                p_true_auroc_all,
                                                                                result_dict_temperature['average_contradict_prob_auroc'],
                                                                                result_dict_temperature['average_neg_log_likelihood_auroc'],
                                                                                result_dict_temperature['ln_predictive_entropy_auroc'], 
                                                                                result_dict_temperature['predictive_entropy_auroc']) 
                                                                                #esult_dict_temperature['average_rougeL_auroc'])
#                                                             p_true_auroc_all, result_dict['average_contradict_prob_auroc'], result_dict['average_neg_log_likelihood_auroc'],
                                                           
        with open("{}/Table_auroc_{}_temperature{}.txt".format(config.paper_dir, dataset_name, args.temperature), "a+") as text_file:
            print(table_row, file=text_file)

# average_contradict_prob semantic_density_auroc ln_predictive_entropy_auroc predictive_entropy_auroc entropy_over_concepts_auroc predictive_entropy_over_concepts
# ['Deg', 'SD', 'NE', 'PE', 'SE', 'NL', 'P(True)']