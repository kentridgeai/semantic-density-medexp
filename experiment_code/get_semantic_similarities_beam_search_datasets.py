import argparse
import csv
import os
import pickle
import random

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import evaluate
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config

parser = argparse.ArgumentParser()
parser.add_argument('--generation_model', type=str, default='opt-350m')
parser.add_argument('--run_id', type=str, default='run_1')
parser.add_argument('--dataset', type=str, default='NQ')
parser.add_argument('--device', type=str, default='0')
args = parser.parse_args()

device = torch.device('cuda:{}'.format(args.device)) if torch.cuda.is_available() and args.device!='-1' else torch.device('cpu')

# Set a seed value
seed_value = 3
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value

os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value

random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value

np.random.seed(seed_value)

#Fix torch random seed
torch.manual_seed(seed_value)

os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache

mistralai_models = ['Mistral-7B-v0.1', 'Mixtral-8x7B-v0.1', 'Mixtral-8x22B-v0.1']
llama_models = ['Llama-2-13b-hf', 'Llama-2-70b-hf', 'Meta-Llama-3-8B', 'Meta-Llama-3-8B-Instruct', 'Meta-Llama-3-70B, Llama-2-7b-hf']
SUPPORTED_OTHER_LMS = ['phi3', 'meerkat7b']

if f"{args.generation_model}" in SUPPORTED_OTHER_LMS:
    hf_model_dir = 'dmis-lab/meerkat-7b-v1.0' if 'meerkat' in args.generation_model else "microsoft/phi-3-mini-4k-instruct"
    
if f"{args.generation_model}" in mistralai_models:
    hf_model_dir = 'mistralai/' + f"{args.generation_model}"

if f"{args.generation_model}" in llama_models:
    hf_model_dir = 'meta-llama/' + f"{args.generation_model}"

# generation_tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, use_fast=False, cache_dir=config.data_dir)

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
model0 = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to(device)

with open(f'{config.output_dir}/{args.generation_model}_generations_all_{args.dataset}.pkl', 'rb') as infile:
    sequences = pickle.load(infile)

result_dict = {}

# meteor = evaluate.load('meteor')

deberta_predictions = []

beam_num = 10
count = 0

for sample in tqdm(sequences):
    count += 1
    question = sample['question']
    generated_texts = []
    for beam_index in range(beam_num):
        generated_texts.append(sample['cleaned_beam_search_generation_{}'.format(beam_index)])

    id_ = sample['id'][0]
    print(id_, sample['id'])
    unique_generated_texts = list(set(generated_texts))
    break
    answer_list_1 = []
    answer_list_2 = []
    has_semantically_different_answers = False
    inputs = []
    syntactic_similarities = {}
    rouge_types = ['rouge1', 'rouge2', 'rougeL']
    for rouge_type in rouge_types:
        syntactic_similarities[rouge_type] = 0.0

    semantic_set_ids = {}
    for index, answer in enumerate(unique_generated_texts):
        semantic_set_ids[answer] = index

    print('Number of unique answers:', len(unique_generated_texts))

    if len(unique_generated_texts) > 1:

        # Evalauate semantic similarity
        for i, reference_answer in enumerate(unique_generated_texts):
            for j in range(i + 1, len(unique_generated_texts)):

                answer_list_1.append(unique_generated_texts[i])
                answer_list_2.append(unique_generated_texts[j])

                qa_1 = question + ' ' + unique_generated_texts[i]
                qa_2 = question + ' ' + unique_generated_texts[j]

                origin_input = qa_1 + ' [SEP] ' + qa_2
                inputs.append(origin_input)
                encoded_input = tokenizer.encode(origin_input, padding=True)
                prediction = model0(torch.tensor(torch.tensor([encoded_input]), device=device))['logits']
                predicted_label = torch.argmax(prediction, dim=1)

                reverse_input = qa_2 + ' [SEP] ' + qa_1
                encoded_reverse_input = tokenizer.encode(reverse_input, padding=True)
                reverse_prediction = model0(torch.tensor(torch.tensor([encoded_reverse_input]), device=device))['logits']
                reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)

                deberta_prediction = 1
                if 0 in predicted_label or 0 in reverse_predicted_label:
                    has_semantically_different_answers = True
                    deberta_prediction = 0

                else:
                    semantic_set_ids[unique_generated_texts[j]] = semantic_set_ids[unique_generated_texts[i]]

                deberta_predictions.append([unique_generated_texts[i], unique_generated_texts[j], deberta_prediction])

        rouge = evaluate.load('rouge')

        # Evalauate syntactic similarity
        answer_list_1 = []
        answer_list_2 = []
        for i in generated_texts:
            for j in generated_texts:
                if i != j:
                    answer_list_1.append(i)
                    answer_list_2.append(j)

        results = rouge.compute(predictions=answer_list_1, references=answer_list_2)

        for rouge_type in rouge_types:
            syntactic_similarities[rouge_type] = results[rouge_type]

    result_dict[id_] = {
        'syntactic_similarities': syntactic_similarities,
        'has_semantically_different_answers': has_semantically_different_answers
    }
    list_of_semantic_set_ids = [semantic_set_ids[x] for x in generated_texts]
    result_dict[id_]['semantic_set_ids'] = list_of_semantic_set_ids
    print(len(result_dict))

with open('deberta_predictions_{}.csv'.format(args.run_id), 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(['qa_1', 'qa_2', 'prediction'])
    writer.writerows(deberta_predictions)


with open(f'{config.output_dir}/{args.generation_model}_generations_similarities_all_{args.dataset}.pkl', 'wb') as outfile:
    pickle.dump(result_dict, outfile)
