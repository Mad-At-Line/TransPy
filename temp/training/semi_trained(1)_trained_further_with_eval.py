import os
import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from datasets import load_dataset
import signal

# Set paths
model_dir = r"C:\Users\mythi\OneDrive\Desktop\already_trained_on\fine_tuned_model_continued_pt.1"
dataset_path = r"C:\Users\mythi\OneDrive\Desktop\ai_training_database"

# Load tokenizer and trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device)

# Load dataset (loading all JSONL files from the folder)
dataset = load_dataset("json", data_files={"train": os.path.join(dataset_path, "*.jsonl")})

# Splitting dataset into train and eval
dataset = dataset['train'].train_test_split(test_size=0.1)
train_dataset = dataset['train']
eval_dataset = dataset['test']

# Tokenization function for dataset
def preprocess_function(examples):
    inputs = examples["input"]
    targets = examples["output"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Preprocess the datasets
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# Set training arguments for further training
training_args = TrainingArguments(
    output_dir="./fine_tuned_model_continued",  # New output directory for the further-trained model
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,  # Reduced batch size for limited GPU memory
    per_device_eval_batch_size=2,
    num_train_epochs=6,
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
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
)

# Function to handle pause signal
def signal_handler(sig, frame):
    print("\nPausing training...")
    trainer.save_model("./fine_tuned_model_continued_checkpoint")
    tokenizer.save_pretrained("./fine_tuned_model_continued_checkpoint")
    exit(0)

# Register signal handler for pausing
signal.signal(signal.SIGINT, signal_handler)

# Continue training the model
trainer.train()

# Save the updated model
trainer.save_model("./fine_tuned_model_continued")
tokenizer.save_pretrained("./fine_tuned_model_continued")