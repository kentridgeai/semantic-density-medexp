import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import config
import evaluate

# Load and process all TSV files in a given folder
def load_and_process_tsv_files(folder_path):
    all_data = []
    
    # Iterate through all files in the given folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.tsv'):
            file_path = os.path.join(folder_path, file_name)
            # Read the TSV file into a pandas DataFrame
            df = pd.read_csv(file_path, sep='\t', header=None)
            
            # Process each row in the TSV file
            for _, row in df.iterrows():
                question = row[0]  # First column: the question
                story = f"A: {row[1]} B: {row[2]} C: {row[3]} D: {row[4]}"  # Columns 2-5: choices A, B, C, D
                explanation_pair_1 = f"Explanation 1: {row[5]}"  # Column 6: explanation pair 1
                explanation_pair_2 = f"Explanation 2: {row[6]}"  # Column 7: explanation pair 2
                correct_answer = row[7]  # Column 8: the correct answer choice (A, B, C, D)
                
                # Append the processed row to the data list
                all_data.append({
                    'story': story,
                    'question': question,
                    'answer': explanation_pair_1,  # First explanation
                    'additional_answers': explanation_pair_2,  # Second explanation
                    'label': correct_answer,  # Correct answer choice
                    'id': f"{file_name}_{_}",  # Unique ID
                })
    
    return all_data

# Initialize the ROUGE evaluator
rouge = evaluate.load('rouge')

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli")
device = torch.device(config.device)  # or "cpu"
model.to(device)

# Function to process the data and create a dataset
def process_data_with_semantic_variability_and_rouge(all_data):
    dataset = {
        'story': [],
        'question': [],
        'answer': [],
        'additional_answers': [],
        'label': [],
        'id': [],
        'semantic_variability': [],
        'rouge1': [],
        'rouge2': [],
        'rougeL': [],
    }

    for entry in all_data:
        question = entry['question']
        story = entry['story']
        explanation_pair_1 = entry['answer']
        explanation_pair_2 = entry['additional_answers']
        correct_answer = entry['label']
        
        # Add the answers into the list
        all_answers = [explanation_pair_1, explanation_pair_2]
        
#         print("batch answers", all_answers)
        answer_list_1 = []
        answer_list_2 = []
        has_semantically_different_answers = False
        inputs = []

        # This computes the syntactic similarity across the reference answers
        for i, reference_answer in enumerate(all_answers):
            for j in range(1):  # Only two answers, no real variability
                if i != j:
                    answer_list_1.append(all_answers[i])
                    answer_list_2.append(all_answers[j])

                    qa_1 = question + ' ' + all_answers[i]
                    qa_2 = question + ' ' + all_answers[j]

                    input_text = qa_1 + ' [SEP] ' + qa_2
                    inputs.append(input_text)

        # Tokenize inputs
        encoded_input = tokenizer.batch_encode_plus(inputs, padding=True, return_tensors="pt")
        encoded_input = {key: value.to(device) for key, value in encoded_input.items()}

        # Get model predictions
        prediction = model(**encoded_input)['logits']
        predicted_label = torch.argmax(prediction, dim=1)

        # Check if there's semantic variability
        if 0 in predicted_label:
            has_semantically_different_answers = True

        dataset['semantic_variability'].append(has_semantically_different_answers)
        
#         print(answer_list_1,answer_list_2)
#         break
        # Compute ROUGE scores
        results = rouge.compute(predictions=answer_list_1, references=answer_list_2)
        dataset['rouge1'].append(results['rouge1'])
        dataset['rouge2'].append(results['rouge2'])
        dataset['rougeL'].append(results['rougeL'])

        # Append other fields
        dataset['story'].append(story)
        dataset['question'].append(question)
        dataset['answer'].append({'text': explanation_pair_1, 'answer_start': 0})
        dataset['additional_answers'].append([explanation_pair_2,])
        dataset['label'].append(correct_answer)
        dataset['id'].append(entry['id'])
    
    return dataset

# Folder paths for dev and test
dev_folder = os.path.join(config.data_dir, '..', 'MedExQA', 'dev')
test_folder = os.path.join(config.data_dir, '..', 'MedExQA', 'test')

# Load and process the data
dev_data = load_and_process_tsv_files(dev_folder)
test_data = load_and_process_tsv_files(test_folder)

# Process dev and test data separately
processed_dev_dataset = process_data_with_semantic_variability_and_rouge(dev_data)
processed_test_dataset = process_data_with_semantic_variability_and_rouge(test_data)

# Convert to pandas DataFrame and then to HuggingFace Dataset
dev_df = pd.DataFrame.from_dict(processed_dev_dataset)
test_df = pd.DataFrame.from_dict(processed_test_dataset)

dev_dataset = Dataset.from_pandas(dev_df)
test_dataset = Dataset.from_pandas(test_df)

# Save the datasets to disk
dev_dataset.save_to_disk(f'{config.data_dir}/medexqa_dev_dataset')
test_dataset.save_to_disk(f'{config.data_dir}/medexqa_test_dataset')
