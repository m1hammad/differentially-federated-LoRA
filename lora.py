from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from device import move_to_device
import logging

# In lora.py
def create_lora_model(model_class=AutoModelForSequenceClassification, transformer_model='distilbert-base-uncased', rank=16, num_labels=2):
    try:
        model = model_class.from_pretrained(transformer_model, num_labels=num_labels)
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=rank,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        model = get_peft_model(model, peft_config)
        model = move_to_device(model)
        return model
    except Exception as e:
        logging.error(f"Error in create_lora_model: {e}")
        raise