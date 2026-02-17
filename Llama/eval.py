# cleaned evaluate.py
import os
import argparse
import csv
import numpy as np
import torch
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from peft import PeftModel
from scipy.special import softmax
from tqdm import tqdm
from utils import *
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-8B")
    parser.add_argument("--access_token", type=str, default=None)
    parser.add_argument("--lora_checkpoint", type=str, default="path/to/lora_checkpoint")
    parser.add_argument("--tokenizer_path", type=str, default=None)

    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="float16")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    parser.add_argument("--bnb_4bit_use_double_quant", action="store_true", default=True)

    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=2500)
    parser.add_argument("--pad_to_multiple_of", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--control_val", type=str, default="control_data_val.npy")
    parser.add_argument("--adverse_val", type=str, default="adverse_data_val.npy")

    parser.add_argument("--output_dir", type=str, default="./eval_tmp")
    parser.add_argument("--csv_file", type=str, default="evaluation_results.csv")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()

def tokenize_dataset(dataset, tokenizer, max_length):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def tf(examples):
        out = tokenizer(examples["text"], truncation=True, max_length=max_length)
        out["labels"] = examples["label"]
        return out

    return dataset.map(tf, batched=True, remove_columns=dataset.column_names)

def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    control_dict = np.load(args.control_val, allow_pickle=True).item()
    adverse_dict = np.load(args.adverse_val, allow_pickle=True).item()

    # use utils to build records (replaces the four loops)
    control_records = build_dataset_records(control_dict["fhr_segments"], control_dict["toco_segments"], label=0)
    adverse_records = build_dataset_records(adverse_dict["fhr_segments"], adverse_dict["toco_segments"], label=1)

    # create evaluation dataset from both control + adverse validation sets
    records = adverse_records + control_records
    dataset_eval = Dataset.from_list(records)

    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16 if args.bnb_4bit_compute_dtype == "float16" else torch.bfloat16,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        )
        base_model = AutoModelForSequenceClassification.from_pretrained(
            args.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            num_labels=2,
            ignore_mismatched_sizes=True,
            use_auth_token=args.access_token,
        )
    else:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            args.model_id,
            device_map="auto",
            num_labels=2,
            ignore_mismatched_sizes=True,
            use_auth_token=args.access_token,
        )

    tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.lora_checkpoint
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_auth_token=args.access_token)

    model = PeftModel.from_pretrained(base_model, args.lora_checkpoint)
    model.eval()

    tokenized_eval = tokenize_dataset(dataset_eval, tokenizer, args.max_length)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, pad_to_multiple_of=args.pad_to_multiple_of)

    eval_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        remove_unused_columns=False,
        do_train=False,
    )

    compute_metrics = compute_metrics_factory(args.threshold)

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    results = trainer.evaluate()
    log_results(args.csv_file, os.path.basename(args.lora_checkpoint), results)

    print("Evaluation results:", results)

if __name__ == "__main__":
    main()
