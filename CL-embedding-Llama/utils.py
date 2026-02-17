import os
import csv
from datetime import datetime
import numpy as np
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

def downsample_4hz_to_half_hz(data_array):
    n_samples, n_timesteps = data_array.shape
    trimmed_timesteps = (n_timesteps // 8) * 8
    trimmed = data_array[:, :trimmed_timesteps]
    reshaped = trimmed.reshape(n_samples, -1, 8)
    return np.mean(reshaped, axis=2)

def take_60_skip_16_vec(signal, chunk=60, skip=15):
    total_len = len(signal)
    step = chunk + skip
    starts = np.arange(0, total_len, step)
    starts = starts[starts + chunk <= total_len]
    chunks = signal[np.arange(chunk)[None, :] + starts[:, None]]
    return chunks

def remove_trailing_nans(signal):
    non_nan = np.where(~np.isnan(signal))[0]
    if non_nan.size == 0:
        return np.array([], dtype=signal.dtype)
    return signal[: non_nan[-1] + 1]

def build_record(fhr, toco, label, system_instruction):
    f = remove_trailing_nans(fhr)
    t = remove_trailing_nans(toco)
    f = np.nan_to_num(f, nan=0).astype(np.int16)
    t = np.nan_to_num(t, nan=0).astype(np.int16)
    text = f"fhr: {','.join(map(str, f))}, toco: {','.join(map(str, t))}"
    return {"text": system_instruction + "\n\n" + text, "label": label}

def build_dataset_records(fhr_array, toco_array, label, system_instruction=None):
    if system_instruction is None:
        system_instruction = "You are an expert in obstetrics and women's health. Analyze the following CTG data and decide if it is 'healthy' (0) or 'abnormal' (1)."
    records = []
    for fhr, toco in zip(fhr_array, toco_array):
        records.append(build_record(fhr, toco, label, system_instruction))
    return records

def build_numeric_records(fhr_array, toco_array, label):
    records = []
    for fhr, toco in zip(fhr_array, toco_array):
        f = np.nan_to_num(fhr, nan=0).astype(np.float32)
        t = np.nan_to_num(toco, nan=0).astype(np.float32)
        records.append({"fhr": f, "toco": t, "label": int(label)})
    return records

def tokenize_dataset(dataset, tokenizer, max_length=2500):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    def tf(examples):
        out = tokenizer(examples["text"], truncation=True, max_length=max_length)
        out["labels"] = examples["label"]
        return out
    return dataset.map(tf, batched=True, remove_columns=dataset.column_names)

def compute_metrics_factory(threshold=0.5):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = softmax(logits, axis=1)
        pos = probs[:, 1]
        preds = (pos >= threshold).astype(int)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="weighted")
        recall = recall_score(labels, preds)
        auc = roc_auc_score(labels, pos)
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
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
