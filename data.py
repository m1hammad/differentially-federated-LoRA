import torch
from datasets import load_dataset
import logging

# from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)

def load_data(tokenizer, split):
    dataset = load_dataset("imdb", split=split)
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
            )
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    dataloader = torch.utils.data.DataLoader(tokenized_datasets, batch_size=8, shuffle=True)

        # Debug tensor shapes
    for batch in dataloader:
        logging.info(f"Batch input_ids shape: {batch['input_ids'].shape}")
        logging.info(f"Batch labels shape: {batch['label'].shape}")
        break  # Only log the first batch

    return dataloader

# models_to_test = [
#     "prajjwal1/bert-tiny",
#     "google/mobilebert-uncased",
#     "distilbert-base-uncased",
# ]

# for model_name in models_to_test:
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     training = load_data(tokenizer, split='train')
#     testing = load_data(tokenizer, split="test")