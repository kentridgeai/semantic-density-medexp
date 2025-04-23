import argparse
import os
import pathlib
import pickle
from lib2to3.pgen2.tokenize import tokenize

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

# import accelerate
import config
import datasets
import evaluate
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

parser = argparse.ArgumentParser()
parser.add_argument('--type_of_question', type=str)
parser.add_argument('--num_generations_per_prompt', type=int, default=3)
parser.add_argument('--fraction_of_data_to_use', type=float, default=0.9)
parser.add_argument('--model', type=str, default='opt-350m')
parser.add_argument('--run_id', type=str, default='run_1')
parser.add_argument('--temperature', type=float, default='1.0')
parser.add_argument('--num_beams', type=int, default='3')
parser.add_argument('--decoding_method', type=str, default='beam_search')
parser.add_argument('--top_p', type=float, default=1.0)
parser.add_argument('--dataset', type=str, default='NQ')
parser.add_argument('--device', type=str, default='0')
args = parser.parse_args()

device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() and args.device != '-1' else torch.device('cpu')

print(torch.cuda.device_count())
print(torch.cuda.is_available())

# Set a seed value
seed_value = 10
num_beams = args.num_beams
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random

random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

#Fix torch random seed
torch.manual_seed(seed_value)

os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache

rouge = evaluate.load('rouge')
exact_match_metric = evaluate.load("exact_match")

print(args)
# with open(f'{config.output_dir}/{args.model}_generations_all_{args.dataset}.pkl', 'rb') as infile:
with open(f'{config.output_dir}/{args.model}_generations_all_{args.dataset}.pkl', 'rb') as infile:
    sequences = pickle.load(infile)

rouge_types = ['rouge1', 'rouge2', 'rougeL']

for sequence_dict in tqdm(sequences):
#     print(sequence_dict['answer'], sequence_dict['additional_answers'] )
    reference_answers = sequence_dict['answer'] if args.dataset == 'pubmedqa' else [sequence_dict['answer'], sequence_dict['additional_answers']]
    if type(reference_answers) == list:
        for answer in reference_answers:
            for beam_index in range(num_beams):
                predictions = [sequence_dict['cleaned_beam_search_generation_{}'.format(beam_index)].lstrip().rstrip()]
                references = [answer]
    #             print(sequence_dict['exact_match'])
    #             print(sequence_dict['exact_match_{}'.format(beam_index)])

                results = exact_match_metric.compute(predictions=predictions,
                                                     references=references,
                                                     ignore_case=True,
                                                     ignore_punctuation=True)
    #             print(results)

                sequence_dict['exact_match'] = max(results['exact_match'], sequence_dict['exact_match_{}'.format(beam_index)])
                rouge_results = rouge.compute(predictions=predictions, references=references)
                for rouge_type in rouge_types:
                    sequence_dict[rouge_type + '_beam_search_to_target_{}'.format(beam_index)] = 0.0
                    sequence_dict[rouge_type + '_beam_search_to_target_{}'.format(beam_index)] = max(rouge_results[rouge_type],
                                                                   sequence_dict[rouge_type + '_beam_search_to_target_{}'.format(beam_index)])
    else:
        for beam_index in range(num_beams):
            predictions = [sequence_dict['cleaned_beam_search_generation_{}'.format(beam_index)].lstrip().rstrip()]
            references = [reference_answers]
#             print(sequence_dict['exact_match'])
#             print(sequence_dict['exact_match_{}'.format(beam_index)])
#             print(predictions, references)
            results = exact_match_metric.compute(predictions=predictions,
                                                 references=references,
                                                 ignore_case=True,
                                                 ignore_punctuation=True)
#             print(results)

            sequence_dict['exact_match'] = max(results['exact_match'], sequence_dict['exact_match_{}'.format(beam_index)])
            rouge_results = rouge.compute(predictions=predictions, references=references)
            for rouge_type in rouge_types:
#                 print("rouge results:", rouge_results[rouge_type])
                sequence_dict[rouge_type + '_beam_search_to_target_{}'.format(beam_index)] = 0.0
                sequence_dict[rouge_type + '_beam_search_to_target_{}'.format(beam_index)] = max(rouge_results[rouge_type],
                                                               sequence_dict[rouge_type + '_beam_search_to_target_{}'.format(beam_index)])            

print(f'saving to {config.output_dir}/{args.model}_generations_all_{args.dataset}.pkl')
with open(f'{config.output_dir}/{args.model}_generations_all_{args.dataset}.pkl', 'wb') as outfile:
    pickle.dump(sequences, outfile)
