from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig
from huggingface_hub import login

# Authenticate

# Load tokenizer and model
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load local dataset
dataset = load_dataset("json", data_files="db.json", split="train")

# Define formatting function (Mistral chat-style)
def formatting(example):
    return tokenizer.apply_chat_template(example["messages"], tokenize=False)

# Training args
training_args = TrainingArguments(
    output_dir="./mixtral-finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    fp16=True,
    save_total_limit=2,
    report_to="none"
)

# LoRA config for quantized fine-tuning
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")

# SFTTrainer with LoRA and chat formatting
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
    formatting_func=formatting,
    peft_config=peft_config
)

# Start training
trainer.train()

# Save model (LoRA adapters only)
trainer.save_model("./mixtral-finetuned")
