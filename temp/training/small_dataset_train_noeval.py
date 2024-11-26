# This script is used to train CodeT5 on a small dataset to evaluate its performance on Python-to-C++ translation.
# The dataset consists of preprocessed JSONL files containing input-output pairs for training.

import os
import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
import tensorflow as tf

# Ensure CUDA is properly set up for TensorFlow
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("TensorFlow GPU setup successful.")
    except RuntimeError as e:
        print(f"Error setting up GPU for TensorFlow: {e}")
else:
    print("No GPU found or TensorFlow could not recognize GPU.")

# Ensure CUDA is properly set up for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"PyTorch is using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch is using CPU, GPU not found.")

# Set paths and hyperparameters
data_dir = r"C:\Users\mythi\OneDrive\Desktop\ai_training_database"
model_dir = r"C:\Users\mythi\OneDrive\Desktop\Important BTYSE Work\trained_model"
model_checkpoint = "Salesforce/codet5-base"
epochs = 2
batch_size = 2

# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)
model = T5ForConditionalGeneration.from_pretrained(model_checkpoint).to(device)

# Load dataset
def load_codet5_dataset(data_dir):
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".jsonl")]
    dataset = load_dataset('json', data_files=files)
    return dataset

# Tokenization function
def preprocess_function(examples):
    inputs = examples["input"]
    targets = examples["output"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length").input_ids

    # Replace padding token id's of the labels by -100 so it's ignored by the loss function
    labels = [-100 if token == tokenizer.pad_token_id else token for token in labels]
    model_inputs["labels"] = labels
    return model_inputs

# Load and preprocess the dataset
dataset = load_codet5_dataset(data_dir)
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Set training arguments
training_args = TrainingArguments(
    output_dir=model_dir,
    evaluation_strategy="no",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    save_total_limit=2,
    save_steps=500,
    logging_dir='./logs',
    logging_steps=50,
    push_to_hub=False,
)

# Use Trainer API for training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)

# Start training
trainer.train()

# Save the model
model.save_pretrained(model_dir)
print("Training completed and model saved at:", model_dir)