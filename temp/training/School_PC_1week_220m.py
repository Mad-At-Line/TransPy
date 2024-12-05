import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback, DataCollatorForSeq2Seq
from datasets import load_dataset
from functools import partial  # Import for passing additional arguments

if __name__ == '__main__':
    # Set paths
    model_checkpoint = "Salesforce/codet5p-220m"
    model_output_dir = r"C:\Users\mythi\OneDrive\Desktop\V2_TransPy\models_testing"
    dataset_path = r"C:\Users\mythi\OneDrive\Desktop\active_training"

    # Load tokenizer and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, trust_remote_code=True).to(device)

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    # Load the training and evaluation datasets
    dataset = load_dataset("json", data_files={"train": os.path.join(dataset_path, "*.jsonl")})

    # Preprocessing function for tokenization
    def preprocess_function(examples, model_checkpoint):
        local_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
        model_inputs = local_tokenizer(examples["input"], max_length=512, truncation=True, padding="max_length")
        labels = local_tokenizer(examples["output"], max_length=512, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Tokenize datasets with multiprocessing (using partial function)
    print("Tokenizing datasets...")
    tokenized_datasets = dataset.map(
        partial(preprocess_function, model_checkpoint=model_checkpoint),  # Pass `model_checkpoint`
        batched=True,
        num_proc=4  # Enable multiprocessing
    )

    # Calculate the number of steps for 1 week of training, assuming roughly 1 hour of training per day
    steps_per_epoch = len(tokenized_datasets["train"]) // 1  # 1 sample per batch
    total_steps = steps_per_epoch * 7 * 24  # Approx. number of steps in one week

    # Set training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=model_output_dir,
        evaluation_strategy="steps",
        eval_steps=500,  # Evaluate every 500 steps for detailed progress tracking
        learning_rate=3e-5,  # Slightly higher learning rate for fine-tuning
        per_device_train_batch_size=1,  # Minimized for 6 GB VRAM
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,  # Simulates an effective batch size of 16
        num_train_epochs=1000,  # High epoch number to simulate long training (can be adjusted)
        max_steps=total_steps,  # Approx. 1 week of training steps
        weight_decay=0.01,  # Regularization to prevent overfitting
        save_total_limit=3,  # Keep the last 3 checkpoints
        save_steps=1000,  # Save model frequently
        logging_dir='./logs',
        logging_steps=100,
        push_to_hub=False,
        fp16=True,  # Mixed precision for better performance
        warmup_steps=1000,  # Longer warmup for smoother learning rate adaptation
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        label_smoothing_factor=0.1,  # Handles noisy labels
        max_grad_norm=1.0
    )

    # Initialize data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Initialize Seq2SeqTrainer with EarlyStoppingCallback
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["train"].select(range(500)),  # Subset for validation
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]  # Stops training early if no improvement
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Save the trained model and tokenizer
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    print(f"Model and tokenizer saved to {model_output_dir}")
