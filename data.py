import torch
from datasets import load_dataset


def load_data(tokenizer, split):
    dataset = load_dataset("imdb", split=split)
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    dataloader = torch.utils.data.DataLoader(tokenized_datasets, batch_size=8)
    return dataloader