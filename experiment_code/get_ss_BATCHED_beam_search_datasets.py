import argparse
import csv
import os
import pickle
import random

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

device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() and args.device != '-1' else torch.device('cpu')

# Set seed for reproducibility
seed_value = 3
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to(device)

# Load generated sequences
# with open(f'{config.output_dir}/{args.generation_model}_generations_all_{args.dataset}.pkl', 'rb') as infile:
with open(f'{config.output_dir}/{args.generation_model}_generations_all_{args.dataset}.pkl', 'rb') as infile:
    sequences = pickle.load(infile)

result_dict = {}
deberta_predictions = []
# **Evaluate syntactic similarity**
rouge = evaluate.load('rouge')

beam_num = 10
batch_size = 16  # Define batch size for processing

batch_size = 16  # Adjust based on available GPU memory

batched_sequences = [sequences[i:i + batch_size] for i in range(0, len(sequences), batch_size)]

for batch in tqdm(batched_sequences):
    batch_questions = [sample['question'] for sample in batch]
    batch_ids = [sample['id'] for sample in batch]
        
    batch_generated_texts = []
    for sample in batch:
        batch_generated_texts.append([sample[f'cleaned_beam_search_generation_{i}'] for i in range(beam_num)])

    result_batch = {}
    deberta_predictions_batch = []
#     break
    for idx, generated_texts in enumerate(batch_generated_texts):
        question = batch_questions[idx]
        id_ = batch_ids[idx]

        unique_generated_texts = list(set(generated_texts))
        syntactic_similarities = {rouge_type: 0.0 for rouge_type in ['rouge1', 'rouge2', 'rougeL']}
        has_semantically_different_answers = False
        semantic_set_ids = {answer: idx for idx, answer in enumerate(unique_generated_texts)}

        if len(unique_generated_texts) > 1:
            batch_inputs = []
            pairs = []
            
            for i in range(len(unique_generated_texts)):
                for j in range(i + 1, len(unique_generated_texts)):
                    qa_1 = question + ' ' + unique_generated_texts[i]
                    qa_2 = question + ' ' + unique_generated_texts[j]
                    pairs.append((unique_generated_texts[i], unique_generated_texts[j]))
                    batch_inputs.append(qa_1 + ' [SEP] ' + qa_2)

            for batch_start in range(0, len(batch_inputs), batch_size):
                batch_end = batch_start + batch_size
                batch_texts = batch_inputs[batch_start:batch_end]
                
                encoded_inputs = tokenizer.batch_encode_plus(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    predictions = model(**encoded_inputs).logits
                    predicted_labels = torch.argmax(predictions, dim=1).cpu().numpy()

                for idx, label in enumerate(predicted_labels):
                    qa_1, qa_2 = pairs[batch_start + idx]
                    deberta_prediction = 0 if label == 0 else 1
                    if label == 0:
                        has_semantically_different_answers = True
                    else:
                        semantic_set_ids[qa_2] = semantic_set_ids[qa_1]
                    deberta_predictions_batch.append([qa_1, qa_2, deberta_prediction])


            answer_list_1, answer_list_2 = zip(*[(i, j) for i in generated_texts for j in generated_texts if i != j])
            rouge_results = rouge.compute(predictions=answer_list_1, references=answer_list_2)
            
            for rouge_type in syntactic_similarities:
                syntactic_similarities[rouge_type] = rouge_results[rouge_type]

        result_batch[id_] = {
            'syntactic_similarities': syntactic_similarities,
            'has_semantically_different_answers': has_semantically_different_answers,
            'semantic_set_ids': [semantic_set_ids[x] for x in generated_texts]
        }

    # Save batched results
    result_dict.update(result_batch)
    deberta_predictions.extend(deberta_predictions_batch)

# Save DeBERTa predictions to CSV
with open(f'deberta_predictions_{args.run_id}.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['qa_1', 'qa_2', 'prediction'])
    writer.writerows(deberta_predictions)

# Save results to pickle file
with open(f'{config.output_dir}/{args.generation_model}_generations_similarities_all_{args.dataset}.pkl', 'wb') as outfile:
    pickle.dump(result_dict, outfile)
