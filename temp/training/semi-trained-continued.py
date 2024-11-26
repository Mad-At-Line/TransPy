import os
import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from datasets import load_dataset

# Set paths
model_dir = r"C:\Users\mythi\OneDrive\Desktop\Important BTYSE Work\trained_model"
model_checkpoint = "Salesforce/codet5-base"

# Load tokenizer and trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)
model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device)

# Updated paths
dataset_path = r"C:\Users\mythi\OneDrive\Desktop\ai_training_database"

# Load dataset (loading all JSONL files from the folder)
dataset = load_dataset("json", data_files={"train": os.path.join(dataset_path, "*.jsonl")})

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

# Set training arguments for further training
training_args = TrainingArguments(
    output_dir="./fine_tuned_model_continued",  # New output directory for the further-trained model
    evaluation_strategy="no",
    learning_rate=5e-5,
    per_device_train_batch_size=2,  # Reduced batch size for limited GPU memory
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    save_steps=500,
    logging_dir='./logs',
    logging_steps=50,
    push_to_hub=False,
)

# Initialize Trainer for further training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=None,
)

# Continue training the model
trainer.train()

# Save the updated model
trainer.save_model("./fine_tuned_model_continued")
tokenizer.save_pretrained("./fine_tuned_model_continued")