import os
import argparse
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from utils import (
    downsample_4hz_to_half_hz,
    build_numeric_records,
    tokenize_dataset,
    compute_metrics_factory,
    log_results,
)
from models import (
    ContrastiveCTGModel,
    SupervisedContrastiveLoss,
    CTGClassifierWrapper,
    PatchCNNFlattenMLPEmbedding,
)
from torch.utils.data import DataLoader
import torch.nn as nn

class CTGDataCollator:
    def __call__(self, features):
        fhr = torch.tensor([f['fhr'] for f in features], dtype=torch.float32)
        toco = torch.tensor([f['toco'] for f in features], dtype=torch.float32)
        labels = torch.tensor([f['label'] for f in features], dtype=torch.long)
        return {'fhr': fhr, 'toco': toco, 'labels': labels}

class CTGTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        device = next(model.parameters()).device
        fhr = inputs['fhr'].to(device)
        toco = inputs['toco'].to(device)
        labels = inputs['labels'].to(device)
        outputs = model(fhr=fhr, toco=toco, labels=labels)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        device = next(model.parameters()).device
        fhr = inputs['fhr'].to(device)
        toco = inputs['toco'].to(device)
        labels = inputs['labels'].to(device)
        with torch.no_grad():
            outputs = model(fhr=fhr, toco=toco, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
        return (loss, logits, labels)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--control_train", type=str, default="control_train.npy")
    p.add_argument("--adverse_train", type=str, default="adverse_train.npy")
    p.add_argument("--control_val", type=str, default="control_val.npy")
    p.add_argument("--adverse_val", type=str, default="adverse_val.npy")
    p.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B")
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--pretrain_epochs", type=int, default=50)
    p.add_argument("--pretrain_bs", type=int, default=64)
    p.add_argument("--stage2_epochs", type=int, default=30)
    p.add_argument("--stage2_bs", type=int, default=8)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def pretrain_supervised_contrastive(model, train_dataset, epochs, batch_size, lr, device, save_dir, save_every=25):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = SupervisedContrastiveLoss(temperature=0.5)
    collator = CTGDataCollator()
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for batch in loader:
            fhr = batch['fhr'].to(device)
            toco = batch['toco'].to(device)
            labels = batch['labels'].to(device)
            inputs = torch.stack([fhr, toco], dim=-1)
            projections = model(inputs)
            loss = criterion(projections, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item()
        if (epoch + 1) % save_every == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(save_dir, f"contrastive_epoch_{epoch+1}.pt"))
    return model

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    control_train = np.load(args.control_train, allow_pickle=True).item()
    adverse_train = np.load(args.adverse_train, allow_pickle=True).item()
    control_val = np.load(args.control_val, allow_pickle=True).item()
    adverse_val = np.load(args.adverse_val, allow_pickle=True).item()
    c_fhr = downsample_4hz_to_half_hz(control_train["fhr_segments"])
    c_toco = downsample_4hz_to_half_hz(control_train["toco_segments"])
    a_fhr = downsample_4hz_to_half_hz(adverse_train["fhr_segments"])
    a_toco = downsample_4hz_to_half_hz(adverse_train["toco_segments"])
    c_fhr_val = downsample_4hz_to_half_hz(control_val["fhr_segments"])
    c_toco_val = downsample_4hz_to_half_hz(control_val["toco_segments"])
    a_fhr_val = downsample_4hz_to_half_hz(adverse_val["fhr_segments"])
    a_toco_val = downsample_4hz_to_half_hz(adverse_val["toco_segments"])
    train_numeric = build_numeric_records(c_fhr, c_toco, 0) + build_numeric_records(a_fhr, a_toco, 1)
    val_numeric = build_numeric_records(a_fhr_val, a_toco_val, 1) + build_numeric_records(c_fhr_val, c_toco_val, 0)
    train_dataset = Dataset.from_list(train_numeric)
    val_dataset = Dataset.from_list(val_numeric)
    contrastive_model = ContrastiveCTGModel(patch_size=64, llm_hidden_dim=4096, projection_dim=256)
    contrastive_model = pretrain_supervised_contrastive(
        model=contrastive_model,
        train_dataset=train_dataset,
        epochs=args.pretrain_epochs,
        batch_size=args.pretrain_bs,
        lr=3e-4,
        device=device,
        save_dir=os.path.join(args.output_dir, "contrastive"),
        save_every=25
    )
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        num_labels=2,
        ignore_mismatched_sizes=True,
        use_auth_token=None,
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    lora_cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.lora_rank, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj"], lora_dropout=0.05, bias="none", inference_mode=False)
    base_model = get_peft_model(base_model, lora_cfg)
    classification_model = CTGClassifierWrapper(contrastive_model.encoder, base_model)
    if True:
        for p in classification_model.patch_cnn_mlp_embedding.parameters():
            p.requires_grad = False
    classification_model = classification_model.to(device)
    lora_params = [p for n, p in classification_model.llm.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=1e-4, weight_decay=0.01)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.stage2_epochs,
        per_device_train_batch_size=args.stage2_bs,
        per_device_eval_batch_size=max(1, args.stage2_bs // 2),
        gradient_accumulation_steps=8,
        warmup_steps=50,
        logging_steps=25,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer = CTGTrainer(
        model=classification_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=CTGDataCollator(),
        compute_metrics=compute_metrics_factory(threshold=0.5),
        tokenizer=tokenizer,
        optimizers=(optimizer, None),
    )
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    classification_model.llm.save_pretrained(os.path.join(args.output_dir, "lora_adapter"))
    log_results(os.path.join(args.output_dir, "eval_log.csv"), "ctg_lora", {"accuracy":0.0,"auc":0.0,"recall":0.0,"specificity":0.0})

if __name__ == "__main__":
    main()
