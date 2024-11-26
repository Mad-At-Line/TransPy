import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

# Paths to your current model and dataset
model_checkpoint = r"C:\Users\mythi\OneDrive\Desktop\Working_model\Mode_4_current_best"
model_output_dir = r"C:\Users\mythi\OneDrive\Desktop\Working_model\Mode_4_current_best_finetuned"
dataset_path = r"C:\Users\mythi\OneDrive\Desktop\ai_training_database"

# Load tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, trust_remote_code=True).to(device)

# Load the dataset
print("Loading dataset...")
dataset = load_dataset("json", data_files={"train": os.path.join(dataset_path, "*.jsonl")})

# Split dataset into train and validation
dataset = dataset["train"].train_test_split(test_size=0.1)

# Preprocessing function for tokenizing the dataset
def preprocess_function(examples):
    inputs = examples["input"]
    targets = examples["output"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Preprocess the datasets
print("Tokenizing dataset...")
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Set fine-tuning arguments (with reduced learning rate)
training_args = TrainingArguments(
    output_dir=model_output_dir,
    evaluation_strategy="steps",
    eval_steps=500,
    learning_rate=5e-6,  # Reduced learning rate for fine-tuning
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    save_steps=500,
    logging_dir='./logs',
    logging_steps=50,
    push_to_hub=False,
    fp16=True,  # Mixed precision training
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"]
)

# Start fine-tuning the model
print("Starting fine-tuning...")
trainer.train()

# Save the fine-tuned model
print("Saving fine-tuned model...")
trainer.save_model(model_output_dir)
tokenizer.save_pretrained(model_output_dir)

print("Fine-tuning completed!")
