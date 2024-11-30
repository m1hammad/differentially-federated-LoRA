from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType


def create_lora_model(model_class= AutoModelForSequenceClassification, transformer_model = 'distilbert-base-uncased'):
    model = model_class.from_pretrained(transformer_model, num_labels=2)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)
    return model

