from transformers import AutoTokenizer
from datasets import load_dataset

def load_data():
    dataset = load_dataset('imdb')
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(['text'])
    tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
    tokenized_datasets.set_format(type='torch')
    return tokenized_datasets


# def load_data():
#     dataset = load_dataset('imdb')
#     tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
#     def tokenize_function(examples):
#         return tokenizer(examples['text'], padding='max_length', truncation=True)
#     tokenized_datasets = dataset.map(tokenize_function, batched=True)
#     tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
#     return tokenized_datasets