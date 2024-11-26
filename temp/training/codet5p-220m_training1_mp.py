import os
import torch
import subprocess
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback, DataCollatorForSeq2Seq
from datasets import load_dataset

if __name__ == '__main__':
    # Set paths
    model_checkpoint = "Salesforce/codet5p-220m"
    model_output_dir = r"C:\Users\mythi\OneDrive\Desktop\PLease_work_models"
    dataset_path = r"C:\Users\mythi\OneDrive\Desktop\ai_training_database"
    eval_dataset_path = r"C:\Users\mythi\OneDrive\Desktop\eval_dataset"

    # Load tokenizer and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, trust_remote_code=True).to(device)

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    # Load the training and evaluation datasets
    dataset = load_dataset("json", data_files={"train": os.path.join(dataset_path, "*.jsonl")})
    eval_dataset = load_dataset("json", data_files={"eval": os.path.join(eval_dataset_path, "*.jsonl")})

    # Calculate average token length of dataset
    def calculate_average_token_length(dataset, tokenizer):
        total_length = 0
        total_examples = 0

        for example in dataset:
            input_tokens = tokenizer(example["input"], truncation=False)["input_ids"]
            output_tokens = tokenizer(example["output"], truncation=False)["input_ids"]
            total_length += len(input_tokens) + len(output_tokens)
            total_examples += 1

        average_length = total_length / (2 * total_examples)  # Average for both input and output
        return average_length

    average_token_length = calculate_average_token_length(dataset["train"], tokenizer)
    print(f"Average token length: {average_token_length}")

    # Tokenization function for dataset
    def preprocess_function(examples):
        model_inputs = tokenizer(examples["input"], max_length=768, truncation=True, padding="max_length")
        labels = tokenizer(examples["output"], max_length=768, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Preprocess the datasets
    tokenized_datasets = dataset.map(preprocess_function, batched=True, num_proc=1)
    tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True, num_proc=1)

    # Set training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=model_output_dir,
        evaluation_strategy="steps",
        eval_steps=1200,
        learning_rate=2.7e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=4,
        weight_decay=0.01,
        save_total_limit=2,
        save_steps=1200,
        logging_dir='./logs',
        logging_steps=200,
        push_to_hub=False,
        fp16=True,
        warmup_steps=1500,
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        label_smoothing_factor=0.05,
        max_grad_norm=1.0
    )

    # Initialize Seq2SeqTrainer with EarlyStoppingCallback
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_eval_dataset["eval"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    # Train the model
    trainer.train()

    # Save the trained model
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
