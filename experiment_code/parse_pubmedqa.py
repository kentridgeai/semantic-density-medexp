import json
import evaluate
import pandas as pd
from datasets import Dataset
import config

# Load data from separate files
def load_data(file_name):
    with open(f'{config.data_dir}/{file_name}', 'r') as infile:
        return json.load(infile)

# Function to process the data and create a dataset
def process_data(data):
    dataset = {}

    dataset['story'] = []
    dataset['question'] = []
    dataset['answer'] = []
    dataset['label'] = []
    dataset['id'] = []

    # Assuming data is in the structure similar to the schema example provided
    for pubmed_id, pubmed_entry in data.items():
        question = pubmed_entry['QUESTION']
        contexts = "\n".join(pubmed_entry['CONTEXTS'])
        labels = pubmed_entry['final_decision']
        long_answer = pubmed_entry['LONG_ANSWER']

        dataset['story'].append(contexts)
        dataset['question'].append(question)
        dataset['answer'].append({
            'text': long_answer,  # Assuming long_answer is the best answer
            'answer_start': 0  # Setting answer start to 0, as there is no span-based annotation in PubMedQA
        })
        dataset['id'].append(pubmed_id)
        dataset['label'].append(labels)
    return dataset

# Load datasets
train_data = load_data('pubmedqa_train_set.json')
dev_data = load_data('pubmedqa_dev_set.json')
test_data = load_data('pubmedqa_test_set.json')

# Process datasets
train_dataset = process_data(train_data)
dev_dataset = process_data(dev_data)
test_dataset = process_data(test_data)

# Convert to DataFrames and then to HuggingFace Datasets
train_df = pd.DataFrame.from_dict(train_dataset)
dev_df = pd.DataFrame.from_dict(dev_dataset)
test_df = pd.DataFrame.from_dict(test_dataset)

train_dataset = Dataset.from_pandas(train_df)
dev_dataset = Dataset.from_pandas(dev_df)
test_dataset = Dataset.from_pandas(test_df)

# Save datasets to disk
train_dataset.save_to_disk(f'{config.data_dir}/pubmedqa_train_dataset')
dev_dataset.save_to_disk(f'{config.data_dir}/pubmedqa_dev_dataset')
test_dataset.save_to_disk(f'{config.data_dir}/pubmedqa_test_dataset')