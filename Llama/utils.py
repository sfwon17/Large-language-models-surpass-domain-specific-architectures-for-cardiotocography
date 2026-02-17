def downsample_4hz_to_1hz(data_array):
    n_samples, n_timesteps = data_array.shape
    trimmed_timesteps = (n_timesteps // 4) * 4
    trimmed_data = data_array[:, :trimmed_timesteps]
    reshaped = trimmed_data.reshape(n_samples, -1, 4)
    downsampled = np.mean(reshaped, axis=2)
    return downsampled


def downsample_4hz_to_half_hz(data_array):
    n_samples, n_timesteps = data_array.shape
    trimmed_timesteps = (n_timesteps // 8) * 8
    trimmed_data = data_array[:, :trimmed_timesteps]
    reshaped = trimmed_data.reshape(n_samples, -1, 8)
    downsampled = np.mean(reshaped, axis=2)
    return downsampled

# Only use this if you have limited gpu memory
def take_60_skip_16_vec(signal, chunk=60, skip=15):
    total_len = len(signal)
    step = chunk + skip
    starts = np.arange(0, total_len, step)
    starts = starts[starts + chunk <= total_len]
    chunks = signal[np.arange(chunk)[None, :] + starts[:, None]]
    return chunks


def remove_trailing_nans(signal):
    non_nan_indices = np.where(~np.isnan(signal))[0]
    if len(non_nan_indices) == 0:
        return None
    last_valid_idx = non_nan_indices[-1]
    return signal[: last_valid_idx + 1]


def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=2500,
    )
    tokenized["labels"] = examples["label"]
    return tokenized


def compute_metrics_factory(threshold=0.5):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred

        probs = softmax(logits, axis=1)
        pos_probs = probs[:, 1]
        preds = (pos_probs >= threshold).astype(int)

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="weighted")
        recall = recall_score(labels, preds)
        auc = roc_auc_score(labels, pos_probs)

        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
        specificity = tn / (tn + fp)

        return {
            "accuracy": acc,
            "f1": f1,
            "recall": recall,
            "specificity": specificity,
            "auc": auc,
        }

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

def build_record(fhr, toco, label, system_instruction):
    fhr = remove_trailing_nans(fhr)
    toco = remove_trailing_nans(toco)

    fhr = np.nan_to_num(fhr, nan=0).astype(np.int16)
    toco = np.nan_to_num(toco, nan=0).astype(np.int16)

    text_fhr = ",".join(map(str, fhr))
    text_toco = ",".join(map(str, toco))

    user_input = f"fhr: {text_fhr}, toco: {text_toco}"
    full_text = system_instruction + "\n\n" + user_input

    return {"text": full_text, "label": label}

def build_dataset_records(fhr_array, toco_array, label):
    system_instruction = (
        "You are an expert in obstetrics and women's health. "
        "Analyze the following CTG data and decide if it is 'healthy' (0) or 'abnormal' (1)."
    )

    records = []
    for fhr, toco in zip(fhr_array, toco_array):
        records.append(build_record(fhr, toco, label, system_instruction))

    return records
