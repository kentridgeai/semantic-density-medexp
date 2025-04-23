import argparse
import json
import pickle
import os
import torch
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--temperature', type=float, default=0.1)
args = parser.parse_args()

beam_num = 10
model_name_list = ['Llama-2-13b-hf', 'Llama-2-70b-hf', 'Meta-Llama-3-8B', 'Meta-Llama-3-70B',
        'Mistral-7B-v0.1', 'Mixtral-8x7B-v0.1', 'Mixtral-8x22B-v0.1']
dataset_list = ['coqa', 'trivia_qa', 'sciq', 'NQ']
key_list = ['entropy_over_concepts_auroc', 'average_contradict_prob_auroc', 'average_neg_log_likelihood_auroc',
        'ln_predictive_entropy_auroc', 'predictive_entropy_auroc']
result_dict_models = {}

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 30}

plt.rc('font', size=20)  # controls default text sizes
plt.rc('axes', titlesize=30)     # fontsize of the axes title
plt.rc('axes', labelsize=23)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=23)    # fontsize of the tick labels
plt.rc('ytick', labelsize=23)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
def export_legend(ax, filename="legend.pdf"):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(*ax.get_legend_handles_labels(), frameon=False, loc='lower center', ncol=1,)
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

marker_list = ['o','v','^','d','s','*','x']
display_name = {'semantic_density_auroc': 'semantic density', 'p_true_auroc': 'P(True)',
                'entropy_over_concepts_auroc': 'semantic entropy', 'average_contradict_prob_auroc': 'Deg',
                'average_neg_log_likelihood_auroc': 'length-normalized likelihood',
                'ln_predictive_entropy_auroc': 'length-normalized entropy', 'predictive_entropy_auroc': 'predictive entropy'}

display_model_name = {'Llama-2-13b-hf': 'Llama-2-13B', 'Llama-2-70b-hf': 'Llama-2-70B',
                'Meta-Llama-3-8B': 'Llama-3-8B', 'Meta-Llama-3-70B': 'Llama-3-70B',
                'Mistral-7B-v0.1': 'Mistral-7B', 'Mixtral-8x7B-v0.1': 'Mixtral-8x7B',
                'Mixtral-8x22B-v0.1': 'Mixtral-8x22B'}

for model_name in model_name_list:
    result_dict_beams = {}
    for beam_id in range(beam_num):
        result_dict_datasets = {}
        for dataset in dataset_list:
            with open(f'results/overall_results_beam_search_{model_name}_{dataset}_temperature{args.temperature}.pkl', 'rb') as f:
                overall_result_dict_temperature = pickle.load(f)
            result_dict_temperature = overall_result_dict_temperature['beam_{}'.format(beam_id)]
            if 'semantic_density_auroc' not in result_dict_datasets.keys():
                result_dict_datasets['semantic_density_auroc'] = []
            result_dict_datasets['semantic_density_auroc'].append(result_dict_temperature['semantic_density_auroc'])

            with open(f'results/{model_name}_p_true_{dataset}.pkl', 'rb') as outfile:
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
        for key in result_dict_datasets.keys():
            if key not in result_dict_beams.keys():
                result_dict_beams[key] = []
            result_dict_beams[key].append(np.mean(result_dict_datasets[key]))
    result_dict_models[model_name] = result_dict_beams

    f, ax = plt.subplots()
    plt.title(display_model_name[model_name], fontweight="bold")
    marker_id = 0
    for key in result_dict_beams.keys():
        plt.plot(np.arange(1,beam_num+1,dtype=int),result_dict_beams[key],linestyle='--', marker=marker_list[marker_id], label=display_name[key])
        marker_id+=1
    plt.xlabel('group index', fontweight="bold")
    plt.ylabel('average AUROC score', fontweight="bold")
    plt.show()
    plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'plots','plot_auroc_group_{}.pdf'.format(model_name))
    f.savefig(plot_file_name, bbox_inches='tight')
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'plots','plot_auroc_group_legend.pdf'.format(model_name))
export_legend(ax, filename=plot_file_name)
