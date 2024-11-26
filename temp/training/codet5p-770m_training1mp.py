import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from torch.optim import AdamW

# Set paths
model_checkpoint = "Salesforce/codet5p-770m"
model_output_dir = r"C:\Users\mythi\OneDrive\Desktop\PLease_work_models"
dataset_path = r"C:\Users\mythi\OneDrive\Desktop\ai_training_database"

# Load tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, trust_remote_code=True, use_cache=False).to(device)

# Create an optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)


# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Load the dataset
dataset = load_dataset("json", data_files={"train": os.path.join(dataset_path, "*.jsonl")})

# Split dataset into train and validation
dataset = dataset["train"].train_test_split(test_size=0.1)

# Tokenization function for dataset
def preprocess_function(examples):
    inputs = examples["input"]
    targets = examples["output"]
    model_inputs = tokenizer(inputs, truncation=True, padding=True)
    labels = tokenizer(targets, truncation=True, padding=True)

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
    per_device_train_batch_size=1,  # Reduced batch size to avoid OOM errors
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,  # Increase gradient accumulation to effectively increase batch size
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    save_steps=500,
    logging_dir='./logs',
    logging_steps=50,
    push_to_hub=False,
    fp16=True,  # Enable mixed precision training
    warmup_steps=500,  # Warmup steps for better convergence
    lr_scheduler_type="linear"  # Use a learning rate scheduler for better optimization
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    optimizers=(optimizer, None)  # Pass the optimizer to the Trainer
)

# Train the model
try:
    trainer.train()
finally:
    # Clear GPU cache to free memory
    torch.cuda.empty_cache()

# Save the trained model
trainer.save_model(model_output_dir)
tokenizer.save_pretrained(model_output_dir)
