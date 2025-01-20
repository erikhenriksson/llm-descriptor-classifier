import os

os.environ["HF_HOME"] = ".hf"
import json
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import numpy as np
from sklearn.metrics import f1_score, classification_report
import torch
from labels import labels  # Import labels from labels.py


# Load and prepare the data
def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


# Function to convert labels to multi-hot encoding
def convert_to_label_ids(example):
    # Initialize array of zeros
    label_array = np.zeros(len(labels))
    # Set 1s for present labels
    for label in example["labels"]:
        if label in labels:
            label_array[labels.index(label)] = 1
    return {"labels": label_array}


# Load datasets
train_data = load_jsonl("splits/train.jsonl")
dev_data = load_jsonl("splits/dev.jsonl")
test_data = load_jsonl("splits/test.jsonl")

# Convert to HuggingFace datasets
train_dataset = Dataset.from_list(train_data)
dev_dataset = Dataset.from_list(dev_data)
test_dataset = Dataset.from_list(test_data)

# Convert labels to multi-hot encoding
train_dataset = train_dataset.map(convert_to_label_ids)
dev_dataset = dev_dataset.map(convert_to_label_ids)
test_dataset = test_dataset.map(convert_to_label_ids)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")


# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=512
    )


# Tokenize datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_dev = dev_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Set format for pytorch
tokenized_train.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)
tokenized_dev.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)
tokenized_test.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)


# Compute metrics function for evaluation
def compute_metrics(eval_pred):
    predictions, labels_ids = eval_pred
    predictions = (predictions > 0).astype(int)

    # Calculate different types of F1 scores
    micro_f1 = f1_score(labels_ids, predictions, average="micro")
    macro_f1 = f1_score(labels_ids, predictions, average="macro")
    weighted_f1 = f1_score(labels_ids, predictions, average="weighted")

    # Generate classification report as string
    report = classification_report(
        labels_ids, predictions, target_names=labels, zero_division=0, output_dict=True
    )

    # Print detailed classification report
    print("\nClassification Report:")
    print(
        classification_report(
            labels_ids, predictions, target_names=labels, zero_division=0
        )
    )

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
    }


# Initialize model
model = AutoModelForSequenceClassification.from_pretrained(
    "answerdotai/ModernBERT-base",
    problem_type="multi_label_classification",
    num_labels=len(labels),
)

# Training arguments with early stopping configuration
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=1000,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    load_best_model_at_end=True,
    metric_for_best_model="micro_f1",
    greater_is_better=True,  # For F1 score, higher is better
    save_strategy="steps",  # Save at each evaluation step
    save_steps=1000,  # Save at same frequency as evaluation
    save_total_limit=1,  # Keep only the best model
)

# Initialize trainer with early stopping
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_dev,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

# Train the model
trainer.train()

# Save the best model
trainer.save_model("./best_model")

# Save the tokenizer alongside the model
tokenizer.save_pretrained("./best_model")

# Evaluate on test set using the best model
print("\nFinal Test Set Evaluation:")
test_results = trainer.evaluate(tokenized_test)

# Print final metrics
print("\nFinal Test Metrics:")
for metric, value in test_results.items():
    if metric != "classification_report":
        print(f"{metric}: {value:.4f}")

print("\nBest model saved to ./best_model")
