import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

# Set paths
model_checkpoint = "Salesforce/codet5p-220m"
model_output_dir = r"C:\Users\mythi\OneDrive\Desktop\Please work models"
dataset_path = r"C:\Users\mythi\OneDrive\Desktop\ai_training_database"

# Load tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, trust_remote_code=True).to(device)

# Load the dataset
dataset = load_dataset("json", data_files={"train": os.path.join(dataset_path, "*.jsonl")})

# Split dataset into train and validation
dataset = dataset["train"].train_test_split(test_size=0.1)

# Tokenization function for dataset
def preprocess_function(examples):
    inputs = examples["input"]
    targets = examples["output"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Preprocess the datasets
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Set training arguments
training_args = TrainingArguments(
    output_dir=model_output_dir,
    evaluation_strategy="steps",
    eval_steps=500,
    learning_rate=5e-5,
    per_device_train_batch_size=2,  # Reduced batch size to avoid OOM errors
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,  # Use gradient accumulation to effectively increase batch size
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    save_steps=500,
    logging_dir='./logs',
    logging_steps=50,
    push_to_hub=False,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"]
)

# Train the model
trainer.train()

# Save the trained model
trainer.save_model(model_output_dir)
tokenizer.save_pretrained(model_output_dir)
