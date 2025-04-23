import argparse
import os
import pathlib
import pickle
from lib2to3.pgen2.tokenize import tokenize

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
parser.add_argument('--dataset', type=str, default='coqa')
parser.add_argument('--cuda_device', type=str, default='0')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda_device

import accelerate
import config
import datasets
import evaluate
import numpy as np
import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig


device = 'cuda'

print(torch.cuda.device_count())
print(torch.cuda.is_available())

# Set a seed value
seed_value = 10
num_beams = 10
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

mistralai_models = ['Mistral-7B-v0.1', 'Mixtral-8x7B-v0.1', 'Mixtral-8x22B-v0.1']
llama_models = ['Llama-2-13b-hf', 'Llama-2-70b-hf', 'Meta-Llama-3-8B', 'Meta-Llama-3-8B-Instruct', 'Meta-Llama-3-70B']

if f"{args.model}" in mistralai_models:
    hf_model_dir = 'mistralai/' + f"{args.model}"

if f"{args.model}" in llama_models:
    hf_model_dir = 'meta-llama/' + f"{args.model}"

print("before loading model")
tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, use_fast=False, cache_dir=config.hf_cache_dir)
model = AutoModelForCausalLM.from_pretrained(hf_model_dir,
                                             torch_dtype=torch.float16,
                                             cache_dir=config.hf_cache_dir, device_map="auto")

if args.dataset == 'coqa':
    dataset = datasets.load_from_disk(f'{config.data_dir}/coqa_dataset')
    id_to_question_mapping = dict(zip(dataset['id'], dataset['question']))
elif args.dataset == 'trivia_qa':
    dataset = datasets.load_from_disk(f'{config.data_dir}/trivia_qa_{args.model}')

if args.fraction_of_data_to_use < 1.0:
    train_dataset = dataset.train_test_split(test_size=(1 - args.fraction_of_data_to_use), seed=seed_value)['train']
else:
    train_dataset = dataset


def encode(examples):
    if 'Meta-Llama-3' in args.model:
        return tokenizer(examples['story'] + ' Q: ' + examples['question'] + ' A: ', truncation=False, padding=False)
    else:
        return tokenizer(examples['story'] + ' Q: ' + examples['question'] + ' A:', truncation=False, padding=False)


def encode_and_format_dataset(dataset):
    dataset = dataset.map(encode, batched=False, load_from_cache_file=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)

    return dataset


if args.dataset == 'coqa':
    questions = encode_and_format_dataset(train_dataset)
elif args.dataset == 'trivia_qa':
    questions = train_dataset
    question_id_set = set()

dataloader = torch.utils.data.DataLoader(questions, batch_size=1)

eos_tokens = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:']
if args.model in ['Meta-Llama-3-8B', 'Meta-Llama-3-70B']:
    period_token_id = tokenizer('. ')['input_ids'][0]
    question_framing_ids = [[tokenizer(eos_token)['input_ids'][0]] for eos_token in eos_tokens]
else:
    period_token_id = tokenizer('. ')['input_ids'][1]
    question_framing_ids = [[tokenizer(eos_token)['input_ids'][1]] for eos_token in eos_tokens]
squad_metric = evaluate.load("squad")
rouge = evaluate.load('rouge')
exact_match_metric = evaluate.load("exact_match")

print(args)

def get_generations(model, dataloader, number_of_generations):
    """For a given model, produce a number of generation """

    with torch.no_grad():
        max_length_of_generated_sequence = 32
        sequences = []
        for batch in tqdm.tqdm(dataloader):
            if args.dataset == 'trivia_qa':
                if batch['question_id'][0] in question_id_set:
                    continue
                else:
                    question_id_set.add(batch['question_id'][0])
            input_ids = torch.cat(batch['input_ids']).to(device).reshape(
                1, -1) if args.dataset == 'trivia_qa' else batch['input_ids'].to(device)
            if args.decoding_method == 'beam_search':
                most_likely_generation = model.generate(input_ids,
                                                        use_cache=True,
                                                        num_beams=num_beams,
                                                        num_return_sequences=num_beams,
                                                        do_sample=False,
                                                        max_length=input_ids.shape[1] +
                                                        max_length_of_generated_sequence,
                                                        eos_token_id=period_token_id,
                                                        bad_words_ids=question_framing_ids,
                                                        num_beam_groups=num_beams,
                                                        diversity_penalty=1.0)
            elif args.decoding_method == 'greedy':
                most_likely_generation = model.generate(input_ids,
                                                        num_beams=1,
                                                        do_sample=False,
                                                        max_length=input_ids.shape[1] +
                                                        max_length_of_generated_sequence,
                                                        eos_token_id=period_token_id,
                                                        bad_words_ids=question_framing_ids)

            input_length = input_ids.shape[1] if args.dataset == 'trivia_qa' else batch['input_ids'].shape[1]
            generations = torch.ones((number_of_generations, input_length + max_length_of_generated_sequence),
                                     dtype=torch.long,
                                     device=device)
            for i in range(number_of_generations):

                generation = model.generate(input_ids,
                                            do_sample=True,
                                            num_return_sequences=1,
                                            num_beams=args.num_beams,
                                            max_length=input_ids.shape[1] + max_length_of_generated_sequence,
                                            eos_token_id=period_token_id,
                                            temperature=args.temperature,
                                            bad_words_ids=question_framing_ids,
                                            top_p=args.top_p)
                generations[i, :generation.shape[1]] = generation

            generations = torch.reshape(generations, (-1, number_of_generations, generations.shape[-1]))
            for i in range(generations.shape[0]):

                if args.dataset == 'coqa':
                    sequence_dict = {
                        'prompt': batch['input_ids'][i].to('cpu'),
                        'generations': generations[i].to('cpu'),
                        'id': batch['id'],
                        'question': id_to_question_mapping[batch['id'][0]]
                    }
                elif args.dataset == 'trivia_qa':
                    few_shot_question = tokenizer.decode(input_ids[0])
                    question = few_shot_question.split('Question: ')[-1].split('Answer: ')[0]
                    sequence_dict = {
                        'prompt': input_ids[0],
                        'generations': generations[i],
                        'id': batch['question_id'],
                        'few_shot_question': tokenizer.decode(input_ids[0]),
                        'question': question
                    }

                generated_texts = []
                for generation in generations[i]:
                    generated_texts.append(
                        tokenizer.decode(generation[input_length:], skip_special_tokens=True))

                sequence_dict['generated_texts'] = generated_texts
                sequence_dict['most_likely_generation_ids'] = most_likely_generation[0].to('cpu')
                sequence_dict['most_likely_generation'] = tokenizer.decode(
                    most_likely_generation[0][input_length:], skip_special_tokens=True)
                sequence_dict['second_most_likely_generation_ids'] = most_likely_generation[1].to('cpu')
                sequence_dict['second_most_likely_generation'] = tokenizer.decode(
                    most_likely_generation[1][input_length:], skip_special_tokens=True)

                sequence_dict['beam_search_generation_{}_ids'.format(0)] = sequence_dict['most_likely_generation_ids']
                sequence_dict['beam_search_generation_{}'.format(0)] = sequence_dict['most_likely_generation']
                sequence_dict['beam_search_generation_{}_ids'.format(1)] = sequence_dict['second_most_likely_generation_ids']
                sequence_dict['beam_search_generation_{}'.format(1)] = sequence_dict['second_most_likely_generation']
                for beam_index in range(2, num_beams):
                    sequence_dict['beam_search_generation_{}_ids'.format(beam_index)] = most_likely_generation[beam_index].to('cpu')
                    sequence_dict['beam_search_generation_{}'.format(beam_index)] = tokenizer.decode(
                                                most_likely_generation[beam_index][input_length:], skip_special_tokens=True)
                sequence_dict['semantic_variability_reference_answers'] = batch[
                    'semantic_variability'] if 'semantic_variability' in batch else None
                rouge_types = ['rouge1', 'rouge2', 'rougeL']
                for rouge_type in rouge_types:
                    if rouge_type in batch:
                        sequence_dict[rouge_type + '_reference_answers'] = batch[rouge_type]

                    else:
                        sequence_dict[rouge_type + '_reference_answers'] = None

                    sequence_dict[rouge_type + '_to_target'] = 0.0
                    sequence_dict[rouge_type + '_to_target_second'] = 0.0
                    for j in range(len(generated_texts)):
                        sequence_dict[rouge_type + '_to_target_{}'.format(j)] = 0.0

                sequence_dict['answer'] = batch['answer']['text'] if args.dataset == 'coqa' else batch['answer']
                sequence_dict['additional_answers'] = [x[0] for x in batch['additional_answers']
                                                      ] if args.dataset == 'coqa' else None

                sequence_dict['exact_match'] = 0.0
                sequence_dict['exact_match_second'] = 0.0
                for j in range(len(generated_texts)):
                    sequence_dict['exact_match_{}'.format(j)] = 0.0

                reference_answers = batch['answer']['text'] + [x[0] for x in batch['additional_answers']
                                                              ] if args.dataset == 'coqa' else batch['answer']

                for answer in reference_answers:
                    predictions = [sequence_dict['most_likely_generation'].lstrip()]
                    references = [answer]
                    results = exact_match_metric.compute(predictions=predictions,
                                                         references=references,
                                                         ignore_case=True,
                                                         ignore_punctuation=True)
                    sequence_dict['exact_match'] = max(results['exact_match'], sequence_dict['exact_match'])
                    rouge_results = rouge.compute(predictions=predictions, references=references)
                    for rouge_type in rouge_types:
                        sequence_dict[rouge_type + '_to_target'] = max(rouge_results[rouge_type].mid.fmeasure,
                                                                       sequence_dict[rouge_type + '_to_target'])
                    predictions = [sequence_dict['second_most_likely_generation'].lstrip()]
                    references = [answer]
                    results = exact_match_metric.compute(predictions=predictions,
                                                         references=references,
                                                         ignore_case=True,
                                                         ignore_punctuation=True)
                    sequence_dict['exact_match_second'] = max(results['exact_match'], sequence_dict['exact_match_second'])
                    rouge_results = rouge.compute(predictions=predictions, references=references)
                    for rouge_type in rouge_types:
                        sequence_dict[rouge_type + '_to_target_second'] = max(rouge_results[rouge_type].mid.fmeasure,
                                                                       sequence_dict[rouge_type + '_to_target_second'])
                    for j in range(len(generated_texts)):
                        predictions = [sequence_dict['generated_texts'][j].lstrip()]
                        references = [answer]
                        results = exact_match_metric.compute(predictions=predictions,
                                                             references=references,
                                                             ignore_case=True,
                                                             ignore_punctuation=True)
                        sequence_dict['exact_match'] = max(results['exact_match'], sequence_dict['exact_match_{}'.format(j)])
                        rouge_results = rouge.compute(predictions=predictions, references=references)
                        for rouge_type in rouge_types:
                            sequence_dict[rouge_type + '_to_target_{}'.format(j)] = max(rouge_results[rouge_type].mid.fmeasure,
                                                                           sequence_dict[rouge_type + '_to_target_{}'.format(j)])

                sequences.append(sequence_dict)

            del most_likely_generation
            del generations
            torch.cuda.empty_cache()


    return sequences


sequences = get_generations(model, dataloader, args.num_generations_per_prompt)

cleaned_sequences = []

for sample in tqdm.tqdm(sequences):
    cleaned_generations = torch.ones_like(sample['generations'])
    question = sample['question']
    generated_texts = sample['generated_texts']
    cleaned_generated_texts = []
    max_len_of_generations = cleaned_generations.shape[-1]

    strings_to_filter_on = [
        '.', '\n', 'Q:', 'A:', 'question:', 'answer:', 'Question:', 'Answer:', 'Questions:', 'questions:', 'QUESTION:',
        'ANSWER:', ':'
    ]
    if 'Meta-Llama-3' in f"{args.model}":
        strings_to_filter_on.append('000000')

    for i, generated_text in enumerate(generated_texts):
        for string in strings_to_filter_on:
            if string in generated_text:
                generated_text = generated_text.split(string)[0].rstrip()
        cleaned_generated_texts.append(generated_text)
        clean_ids = torch.cat(
            [sample['prompt'].to(device),
             torch.tensor(tokenizer(generated_text)['input_ids'][1:], device=device)])
        cleaned_generations[i, :min(len(clean_ids), max_len_of_generations)] = clean_ids[:max_len_of_generations]

    sample['cleaned_generated_texts'] = cleaned_generated_texts
    sample['cleaned_generations'] = cleaned_generations

    for i in range(num_beams):
        sample['cleaned_beam_search_generation_{}'.format(i)] = sample['beam_search_generation_{}'.format(i)]
        for string in strings_to_filter_on:
            if string in sample['cleaned_beam_search_generation_{}'.format(i)]:
                sample['cleaned_beam_search_generation_{}'.format(i)] = sample['cleaned_beam_search_generation_{}'.format(i)].split(string)[0].rstrip()
        clean_ids = torch.cat(
            [sample['prompt'].to(device),
             torch.tensor(tokenizer(sample['cleaned_beam_search_generation_{}'.format(i)])['input_ids'][1:], device=device)])
        sample['cleaned_beam_search_generation_{}_ids'.format(i)] = torch.ones_like(sample['beam_search_generation_{}_ids'.format(i)])
        max_len_of_generations = len(sample['beam_search_generation_{}_ids'.format(i)])
        sample['cleaned_beam_search_generation_{}_ids'.format(i)][:min(len(clean_ids), max_len_of_generations)] = clean_ids[:max_len_of_generations]

    sample['most_likely_generation_ids'] = sample['cleaned_beam_search_generation_0_ids']
    sample['second_most_likely_generation_ids'] = sample['cleaned_beam_search_generation_1_ids']
    sample['most_likely_generation'] = sample['cleaned_beam_search_generation_0']
    sample['second_most_likely_generation'] = sample['cleaned_beam_search_generation_1']

    cleaned_sequences.append(sample)

with open(f'{config.output_dir}/{args.model}_generations_all_{args.dataset}.pkl', 'wb') as outfile:
    pickle.dump(cleaned_sequences, outfile)
