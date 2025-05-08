from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset

from huggingface_hub import login


model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("json", data_files="db.json", split="train")

training_args = TrainingArguments(
    output_dir="./mixtral-finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    fp16=True,
    save_total_limit=2
)

training_args = SFTConfig(packing=True)

def formatting(example):
    return tokenizer.apply_chat_template(example["messages"], tokenize=False)

model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
#    tokenizer=tokenizer,
#    formatting_func=formatting,
)

trainer.train()
trainer.save_model("./mixtral-finetuned")
