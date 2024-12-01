import torch
from transformers import AutoTokenizer
from datasets import load_dataset


def load_data(tokenizer, split):
    """
    Loads and tokenizes the IMDb dataset for the specified split.
    """
    dataset = load_dataset("imdb", split=split)
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    dataloader = torch.utils.data.DataLoader(tokenized_datasets, batch_size=8)
    return dataloader