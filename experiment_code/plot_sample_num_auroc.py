import argparse
import json
import pickle
import os
import torch
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='NQ')
parser.add_argument('--model', type=str, default='opt-350m')
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--sample_num', type=int, default=10)
args = parser.parse_args()

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
    legend = ax2.legend(*ax.get_legend_handles_labels(), frameon=False, loc='lower center', ncol=7,)
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

model_name_list = ['Llama-2-13b-hf', 'Llama-2-70b-hf', 'Meta-Llama-3-8B', 'Meta-Llama-3-70B',
        'Mistral-7B-v0.1', 'Mixtral-8x7B-v0.1', 'Mixtral-8x22B-v0.1']
display_name = {'Llama-2-13b-hf': 'Llama-2-13B', 'Llama-2-70b-hf': 'Llama-2-70B',
                'Meta-Llama-3-8B': 'Llama-3-8B', 'Meta-Llama-3-70B': 'Llama-3-70B',
                'Mistral-7B-v0.1': 'Mistral-7B', 'Mixtral-8x7B-v0.1': 'Mixtral-8x7B',
                'Mixtral-8x22B-v0.1': 'Mixtral-8x22B'}
display_dataset_name = {'coqa': "CoQA", 'trivia_qa': 'TriviaQA', 'sciq': 'SciQ', 'NQ': 'NQ'}
marker_list = ['o','v','^','d','s','*','x']
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
f, ax = plt.subplots()
plt.title(display_dataset_name[args.dataset], fontweight="bold")
marker_id = 0
for key in semantic_density_dict.keys():
    plt.plot(np.arange(1,args.sample_num+1,dtype=int),semantic_density_dict[key], linestyle='--', marker=marker_list[marker_id], label=display_name[key])
    marker_id+=1
plt.xlabel('number of reference responses', fontweight="bold")
plt.ylabel('AUROC score', fontweight="bold")
plt.show()
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'plots','plot_auroc_sample_num{}_{}.pdf'.format(args.sample_num, args.dataset))
f.savefig(plot_file_name, bbox_inches='tight')
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'plots','plot_auroc_sample_num{}_legend.pdf'.format(args.sample_num))
export_legend(ax, filename=plot_file_name)
