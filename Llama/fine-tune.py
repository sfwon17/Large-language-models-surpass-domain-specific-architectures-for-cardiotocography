import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, AutoProcessor, TrainingArguments, Llama4ForConditionalGeneration, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
import torch
import torch.nn as nn
import csv
import os
from datetime import datetime
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, confusion_matrix, balanced_accuracy_score
from scipy.special import softmax
import random
from collections import Counter
from transformers import DataCollatorWithPadding
from utils import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--access_token", type=str, default=None)

    parser.add_argument("--output_dir", type=str, default="./ctg-lora-classifier_1500_2")
    parser.add_argument("--num_train_epochs", type=int, default=15)
    parser.add_argument("--per_device_train_batch_size", type=int, default=3)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=12)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=15)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label_smoothing_factor", type=float, default=0.1)

    return parser.parse_args()

args = parse_args()

# your data path
CONTROL_DATA_PATH = "control_data.npy"
ADVERSE_DATA_PATH = "adverse_data.npy"
CONTROL_VAL_PATH = "control_val.npy"
ADVERSE_VAL_PATH = "adverse_val.npy"

control_data = np.load(CONTROL_DATA_PATH, allow_pickle=True)
adverse_data = np.load(ADVERSE_DATA_PATH, allow_pickle=True)
control_data_val = np.load(CONTROL_VAL_PATH, allow_pickle=True)
adverse_data_val = np.load(ADVERSE_VAL_PATH, allow_pickle=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForSequenceClassification.from_pretrained(
    args.model_id,
    quantization_config=bnb_config,
    device_map="auto",
    num_labels=2,
    ignore_mismatched_sizes=True,
    use_auth_token=args.access_token,
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(
    args.model_id, use_auth_token=args.access_token,)

classifier_layer = None
if hasattr(model, "classifier"):
    classifier_layer = model.classifier
elif hasattr(model, "score"):
    classifier_layer = model.score
elif hasattr(model, "classification_head"):
    classifier_layer = model.classification_head

if classifier_layer is not None:
    if hasattr(classifier_layer, "weight"):
        nn.init.xavier_uniform_(classifier_layer.weight, gain=0.1)
    if hasattr(classifier_layer, "bias") and classifier_layer.bias is not None:
        nn.init.zeros_(classifier_layer.bias)

for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        if "classifier" in name or "score" in name or module.out_features == 2:
            nn.init.xavier_uniform_(module.weight, gain=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


control_downsampled_fhr = downsample_4hz_to_half_hz(control_data.item()["fhr_segments"])
control_downsampled_toco = downsample_4hz_to_half_hz(control_data.item()["toco_segments"])

adverse_downsampled_fhr = downsample_4hz_to_half_hz(adverse_data.item()["fhr_segments"])
adverse_downsampled_toco = downsample_4hz_to_half_hz(adverse_data.item()["toco_segments"])

adverse_downsampled_fhr_val = downsample_4hz_to_half_hz(adverse_data_val.item()["fhr_segments"])
control_downsampled_fhr_val = downsample_4hz_to_half_hz(control_data_val.item()["fhr_segments"])

adverse_downsampled_toco_val = downsample_4hz_to_half_hz(adverse_data_val.item()["toco_segments"])
control_downsampled_toco_val = downsample_4hz_to_half_hz(control_data_val.item()["toco_segments"])

data_records = []
data_records += build_dataset_records(control_downsampled_fhr, control_downsampled_toco, label=0)
data_records += build_dataset_records(adverse_downsampled_fhr, adverse_downsampled_toco, label=1)

data_records_val = []
data_records_val += build_dataset_records(adverse_downsampled_fhr_val, adverse_downsampled_toco_val, label=1)
data_records_val += build_dataset_records(control_downsampled_fhr_val, control_downsampled_toco_val, label=0)

random.shuffle(data_records)

dataset = Dataset.from_list(data_records)
dataset_val = Dataset.from_list(data_records_val)

train_dataset = dataset
val_dataset = dataset_val

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model.config.pad_token_id = tokenizer.pad_token_id

tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names,
)

tokenized_val = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=val_dataset.column_names,
)

csv_file = "validation_results.csv"
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["timestamp", "accuracy", "f1", "recall", "specificity", "auc"]
        )


lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    inference_mode=False,
)

model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    warmup_steps=args.warmup_steps,
    learning_rate=args.learning_rate,
    logging_steps=args.logging_steps,
    save_steps=args.save_steps,
    eval_steps=args.eval_steps,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_total_limit=args.save_total_limit,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=args.bf16,
    max_grad_norm=args.max_grad_norm,
    remove_unused_columns=False,
    seed=args.seed,
    label_smoothing_factor=args.label_smoothing_factor,
)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    pad_to_multiple_of=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

trainer.train()
