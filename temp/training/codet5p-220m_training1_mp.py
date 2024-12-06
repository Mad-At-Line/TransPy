import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import load_dataset
from functools import partial

if __name__ == '__main__':
    # Set paths
    model_checkpoint = "Salesforce/codet5p-220m"
    model_output_dir = r"C:\Users\mythi\OneDrive\Desktop\V2_TransPy\models_testing"
    train_dataset_path = r"C:\Users\mythi\OneDrive\Desktop\active_training"

    # Load tokenizer and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, trust_remote_code=True).to(device)

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    # Load the training dataset
    train_dataset = load_dataset("json", data_files={"train": os.path.join(train_dataset_path, "*.jsonl")})

    # Preprocessing function for tokenization
    def preprocess_function(examples, model_checkpoint):
        local_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
        model_inputs = local_tokenizer(examples["input_code"], max_length=512, truncation=True, padding="max_length")
        labels = local_tokenizer(examples["output_code"], max_length=512, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Tokenize datasets with multiprocessing
    print("Tokenizing training dataset...")
    tokenized_train_dataset = train_dataset["train"].map(
        partial(preprocess_function, model_checkpoint=model_checkpoint),
        batched=True,
        num_proc=4
    )

    # Set training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=model_output_dir,
        learning_rate=3e-5,
        per_device_train_batch_size=2,  # Reduced batch size for 6GB VRAM
        gradient_accumulation_steps=16,  # Simulates effective batch size of 32
        num_train_epochs=8,  # Adequate epochs for small dataset
        weight_decay=0.01,
        save_total_limit=2,  # Save fewer checkpoints
        save_steps=1000,
        logging_dir='./logs',
        logging_steps=100,
        push_to_hub=False,
        fp16=True,  # Mixed precision training for GPU efficiency
        warmup_steps=500,  # Adjusted warmup steps for smaller dataset
        lr_scheduler_type="cosine",
        label_smoothing_factor=0.1,
        max_grad_norm=1.0
    )

    # Initialize data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Initialize Seq2SeqTrainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Save the trained model and tokenizer
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    print(f"Model and tokenizer saved to {model_output_dir}")

