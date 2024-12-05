import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from torch.optim import AdamW
from functools import partial  # For passing additional arguments

# Set paths
model_checkpoint = "Salesforce/codet5p-770m"
model_output_dir = r"C:\Users\mythi\OneDrive\Desktop\PLease_work_models"
dataset_path = r"C:\Users\mythi\OneDrive\Desktop\active_training"

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

# Calculate the number of steps for 1 week of training, assuming roughly 1 hour of training per day
steps_per_epoch = len(tokenized_datasets["train"]) // 1  # 1 sample per batch
total_steps = steps_per_epoch * 7 * 24  # Approx. number of steps in one week

# Set training arguments
training_args = TrainingArguments(
    output_dir=model_output_dir,
    evaluation_strategy="steps",
    eval_steps=500,  # Evaluate every 500 steps
    learning_rate=5e-5,  # Fine-tuning with a moderate learning rate
    per_device_train_batch_size=1,  # Reduced batch size to avoid OOM errors on 6GB VRAM
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,  # Simulates a larger batch size
    num_train_epochs=1000,  # High epoch number to simulate long training (can be adjusted)
    max_steps=total_steps,  # Approx. 1 week of training steps
    weight_decay=0.01,  # Regularization to prevent overfitting
    save_total_limit=2,  # Keep the last 2 checkpoints
    save_steps=500,  # Save model every 500 steps
    logging_dir='./logs',
    logging_steps=50,  # Log every 50 steps for more granular updates
    push_to_hub=False,
    fp16=True,  # Enable mixed precision training for better performance
    warmup_steps=500,  # Warmup steps to stabilize learning
    lr_scheduler_type="linear",  # Use a learning rate scheduler for better optimization
    load_best_model_at_end=True,
    label_smoothing_factor=0.1,  # Helps with noisy labels
    max_grad_norm=1.0  # Clip gradients to avoid explosion
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
print(f"Model and tokenizer saved to {model_output_dir}")
