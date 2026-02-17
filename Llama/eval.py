# Cleaned version of our code. No the most efficient code, but it will work! 

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
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, confusion_matrix
from scipy.special import softmax
from tqdm import tqdm

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

    parser.add_argument("--per_device_eval_batch_size", type=int, default=4) # if you have 32gb or below, lower it 
    parser.add_argument("--max_length", type=int, default=2500) # increase it if you increase data length 
    parser.add_argument("--pad_to_multiple_of", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--control_val", type=str, default="control_data_val.npy")
    parser.add_argument("--adverse_val", type=str, default="adverse_data_val.npy")

    parser.add_argument("--output_dir", type=str, default="./eval_tmp")
    parser.add_argument("--csv_file", type=str, default="evaluation_results.csv")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()

def remove_trailing_nans(signal):
    non_nan = np.where(~np.isnan(signal))[0]
    if non_nan.size == 0:
        return np.array([], dtype=signal.dtype)
    return signal[: non_nan[-1] + 1]

def create_eval_dataset_from_dicts(control_dict, adverse_dict):
    sys_instr = "You are an expert in obstetrics and women's health. Analyze the following CTG data and decide if it is 'healthy' (0) or 'abnormal' (1)."
    records = []
    adverse_fhr = adverse_dict["fhr_segments"]
    adverse_toco = adverse_dict["toco_segments"]
    control_fhr = control_dict["fhr_segments"]
    control_toco = control_dict["toco_segments"]

    for fhr, toco in zip(adverse_fhr, adverse_toco):
        f = remove_trailing_nans(fhr)
        t = remove_trailing_nans(toco)
        f = np.nan_to_num(f, nan=0).astype(np.int16)
        t = np.nan_to_num(t, nan=0).astype(np.int16)
        text = f"fhr: {','.join(map(str, f))}, toco: {','.join(map(str, t))}"
        records.append({"text": sys_instr + "\n\n" + text, "label": 1})

    for fhr, toco in zip(control_fhr, control_toco):
        f = remove_trailing_nans(fhr)
        t = remove_trailing_nans(toco)
        f = np.nan_to_num(f, nan=0).astype(np.int16)
        t = np.nan_to_num(t, nan=0).astype(np.int16)
        text = f"fhr: {','.join(map(str, f))}, toco: {','.join(map(str, t))}"
        records.append({"text": sys_instr + "\n\n" + text, "label": 0})

    return Dataset.from_list(records)

def tokenize_dataset(dataset, tokenizer, max_length):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def tf(examples):
        out = tokenizer(examples["text"], truncation=True, max_length=max_length)
        out["labels"] = examples["label"]
        return out

    return dataset.map(tf, batched=True, remove_columns=dataset.column_names)

def compute_metrics_factory(threshold):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = softmax(logits, axis=1)
        pos_probs = probs[:, 1]
        preds = (pos_probs >= threshold).astype(int)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="weighted")
        recall = recall_score(labels, preds)
        auc = roc_auc_score(labels, pos_probs)
        cm = confusion_matrix(labels, preds, labels=[0,1])
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        return {"accuracy": acc, "f1": f1, "recall": recall, "specificity": specificity, "auc": auc}
    return compute_metrics

def log_results(csv_path, model_name, results):
    header = ["timestamp", "model", "accuracy", "auc", "recall", "specificity"]
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_name,
            f"{results.get('accuracy', 0):.4f}",
            f"{results.get('auc', 0):.4f}",
            f"{results.get('recall', 0):.4f}",
            f"{results.get('specificity', 0):.4f}",
        ])

def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    control_dict = np.load(args.control_val, allow_pickle=True).item()
    adverse_dict = np.load(args.adverse_val, allow_pickle=True).item()

    dataset_eval = create_eval_dataset_from_dicts(control_dict, adverse_dict)

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
