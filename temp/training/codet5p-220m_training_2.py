import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, DatasetDict
import os

# Paths for model, tokenizer, and data
model_dir = r"C:\Users\mythi\OneDrive\Desktop\Working_model\Working_model_1"
dataset_path = r"C:\Users\mythi\OneDrive\Desktop\ai_training_database"
output_dir = r"C:\Users\mythi\OneDrive\Desktop\Please_work_models"

# Main function to avoid multiprocessing issues on Windows
if __name__ == "__main__":
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset('json', data_files=os.path.join(dataset_path, '*.jsonl'))

    # Split dataset into train and validation
    split_datasets = dataset['train'].train_test_split(test_size=0.1, shuffle=True, seed=42)
    dataset = DatasetDict({
        'train': split_datasets['train'],
        'validation': split_datasets['test']
    })

    # Load model and tokenizer
    print("Loading tokenizer and model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)

    # Tokenize the dataset
    def preprocess_function(examples):
        inputs = [ex for ex in examples["input"]]
        targets = [ex for ex in examples["output"]]
        model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding='max_length')
        labels = tokenizer(targets, max_length=256, truncation=True, padding='max_length')
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(preprocess_function, batched=True, num_proc=1, remove_columns=dataset['train'].column_names)

    # Set training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,  # Adjusted learning rate for more stable training
        per_device_train_batch_size=16,  # Increased batch size for better GPU utilization
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=5,  # Increased epochs to allow more training iterations
        predict_with_generate=True,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=50,
        save_strategy="epoch",
        gradient_accumulation_steps=1,  # Reduced for faster training with larger batch size
        fp16=True,  # Enable mixed precision for better GPU utilization
        warmup_steps=500,  # Increased warmup steps for smoother training start
        lr_scheduler_type="cosine"  # Changed scheduler for potentially better convergence
    )

    # Use DataCollatorForSeq2Seq
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Save the final model
    print("Saving the model...")
    trainer.save_model(output_dir)
    print("Training complete and model saved at", output_dir)
