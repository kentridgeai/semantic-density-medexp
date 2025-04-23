import argparse
import pathlib
import pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import accelerate
import datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config


parser = argparse.ArgumentParser()
parser.add_argument('--type_of_question', type=str)
parser.add_argument('--num_generations_per_prompt', type=int, default=5)
parser.add_argument('--fraction_of_data_to_use', type=float, default=0.9)
parser.add_argument('--model', type=str, default='opt-350m')
parser.add_argument('--run_id', type=str, default='run_1')
parser.add_argument('--temperature', type=float, default='1.0')
parser.add_argument('--num_beams', type=int, default='5')
parser.add_argument('--decoding_method', type=str, default='beam_search')
parser.add_argument('--top_p', type=float, default=1.0)
args = parser.parse_args()
os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache

mistralai_models = ['Mistral-7B-v0.1', 'Mixtral-8x7B-v0.1', 'Mixtral-8x22B-v0.1']
llama_models = ['Llama-2-13b-hf', 'Llama-2-70b-hf', 'Meta-Llama-3-8B', 'Meta-Llama-3-8B-Instruct', 'Meta-Llama-3-70B', 'Meta-Llama-3-70B-Instruct']

if f"{args.model}" in mistralai_models:
    hf_model_dir = 'mistralai/' + f"{args.model}"

if f"{args.model}" in llama_models:
    hf_model_dir = 'meta-llama/' + f"{args.model}"


tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, use_fast=False, cache_dir=config.hf_cache_dir)
seed_value = 10

if not pathlib.Path(f'{config.data_dir}/trivia_qa_{args.model}').exists():

    print('Preprocessing dataset')
    val_data = datasets.load_dataset("trivia_qa", "rc.nocontext", split="validation")
    train_data = datasets.load_dataset("trivia_qa", "rc.nocontext", split="train")
    data_for_few_shot_prompt = train_data.select(range(0, 10))

    few_shot_prompt = 'This is a bot that correctly answers questions. \n'
    for sample in data_for_few_shot_prompt:
        few_shot_prompt += 'Question: ' + sample['question'] + ' Answer: ' + sample['answer']['value'] + ' '

    batch_size = 4  # change to 16 for full training
    encoder_max_length = 1024
    decoder_max_length = 128

    def process_data_to_model_inputs(batch):
        # tokenize the inputs and labels
        answers = [answer["value"] for answer in batch["answer"]]

        batch_with_prompt = [few_shot_prompt + "Question: " + question + " Answer:" for question in batch["question"]]
        inputs = tokenizer(batch_with_prompt, padding=False, truncation=False)
        outputs = tokenizer(answers, padding=False, truncation=False)

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["decoder_input_ids"] = outputs.input_ids
        batch["decoder_attention_mask"] = outputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()
        batch['answer'] = answers

        # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
        # We have to make sure that the PAD token is ignored
        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
        ]

        return batch

    val_data = val_data.map(process_data_to_model_inputs,
                            batched=True,
                            batch_size=batch_size,
                            remove_columns=["search_results", "question_source", "entity_pages"])
    val_data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
        output_all_columns=True)

    val_data.save_to_disk(f'{config.data_dir}/trivia_qa_{args.model}')
else:

    val_data = datasets.load_from_disk(f'{config.data_dir}/trivia_qa_{args.model}')
