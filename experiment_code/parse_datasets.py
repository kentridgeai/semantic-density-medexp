import argparse
import pathlib
import pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import accelerate
import datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='NQ')
parser.add_argument('--model', type=str, default='opt-350m')
parser.add_argument('--few_shot_num', type=int, default=10)
args = parser.parse_args()
os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache

question_file = f'{config.data_dir}/{args.dataset}/test.txt'

mistralai_models = ['Mistral-7B-v0.1', 'Mixtral-8x7B-v0.1', 'Mixtral-8x22B-v0.1']
llama_models = ['Llama-2-13b-hf', 'Llama-2-70b-hf', 'Meta-Llama-3-8B', 'Meta-Llama-3-8B-Instruct', 'Meta-Llama-3-70B', 'Meta-Llama-3-70B-Instruct']


if f"{args.model}" in mistralai_models:
    hf_model_dir = 'mistralai/' + f"{args.model}"

if f"{args.model}" in llama_models:
    hf_model_dir = 'meta-llama/' + f"{args.model}"


tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, use_fast=False, cache_dir=config.hf_cache_dir)

seed_value = 10

print('Preprocessing dataset')
if question_file.endswith('.txt'):
    questions = []
    answers = []
    for item in open(question_file).read().split('\n\n'):
        if item:
            questions.append(item.split('\n')[0])
            answers.append(item.split('\n')[1].split(';'))
        answers[-1] = [answer.strip() for answer in answers[-1]]

elif question_file.endswith('.csv'):
    data_df = pd.read_csv(question_file)
    questions = data_df['question'].tolist()
    answers = data_df['answer'].tolist()
else:
    print('question file format must be txt or csv')
    exit()

few_shot_prompt = 'This is a bot that correctly answers questions. \n'
for i in range(args.few_shot_num):
    few_shot_prompt += 'Question: ' + questions[i] + ' Answer: ' + answers[i][0] + ' '

processed_dataset = []
for i in range(args.few_shot_num, len(questions)):
    processed_data = {}
    processed_data['question'] = questions[i]
    processed_data['answer'] = answers[i]
    prompt = few_shot_prompt + "Question: " + questions[i] + " Answer:"
    processed_data['prompt'] = prompt
    inputs = tokenizer(prompt, padding=False, truncation=False)
    outputs = tokenizer(answers[i], padding=False, truncation=False)
    processed_data["input_ids"] = inputs.input_ids
    processed_data["attention_mask"] = inputs.attention_mask
    processed_data["answer_input_ids"] = outputs.input_ids
    processed_data["answer_attention_mask"] = outputs.attention_mask
    processed_data["labels"] = outputs.input_ids.copy()
    processed_data["labels"] = [-100 if token == tokenizer.pad_token_id else token for token in processed_data["labels"]]

    processed_dataset.append(processed_data)

with open(f'{config.data_dir}/{args.dataset}_{args.model}.pkl', 'wb') as dataset_file:
    pickle.dump(processed_dataset,dataset_file)
