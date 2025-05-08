from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig
from datasets import load_dataset
from huggingface_hub import login

# Authenticate Hugging Face access

# Model & tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Dataset from db.json
dataset = load_dataset("json", data_files="db.json", split="train")

# Chat-style formatting
def formatting_func(example):
    return tokenizer.apply_chat_template(example["messages"], tokenize=False)

# Quantization config (replaces deprecated `load_in_4bit=True`)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_quant_type="nf4"
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto"
)

# LoRA config
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

#model.add_adapter(peft_config)

# SFTConfig (includes formatting, tokenizer, LoRA)
sft_config = SFTConfig(
    output_dir="./mixtral-finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    fp16=True,
    save_total_limit=2,
    max_seq_length=2048,
    packing=False,
)



# SFTTrainer using config
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
    peft_config=peft_config
)

trainer.train()
trainer.save_model("./mixtral-finetuned")
