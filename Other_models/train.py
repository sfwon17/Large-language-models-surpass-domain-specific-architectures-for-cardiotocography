import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, balanced_accuracy_score
from models.seresnet import create_seresnet152d_model
from utils import *

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

control_downsampled_fhr = downsample_4hz_to_1hz(control_data.item()["fhr_segments"])
control_downsampled_toco = downsample_4hz_to_1hz(control_data.item()["toco_segments"])

adverse_downsampled_fhr = downsample_4hz_to_1hz(adverse_data.item()["fhr_segments"])
adverse_downsampled_toco = downsample_4hz_to_1hz(adverse_data.item()["toco_segments"])

adverse_downsampled_fhr_val = downsample_4hz_to_1hz(adverse_data_val.item()["fhr_segments"])
control_downsampled_fhr_val = downsample_4hz_to_1hz(control_data_val.item()["fhr_segments"])

adverse_downsampled_toco_val = downsample_4hz_to_1hz(adverse_data_val.item()["toco_segments"])
control_downsampled_toco_val = downsample_4hz_to_1hz(control_data_val.item()["toco_segments"])

fhr_train = np.concatenate([control_downsampled_fhr, adverse_downsampled_fhr], axis=0)
toco_train = np.concatenate([control_downsampled_toco, adverse_downsampled_toco], axis=0)
labels_train = np.concatenate([np.zeros(control_downsampled_fhr.shape[0], dtype=np.int64),
                               np.ones(adverse_downsampled_fhr.shape[0], dtype=np.int64)], axis=0)

fhr_val = np.concatenate([control_downsampled_fhr_val, adverse_downsampled_fhr_val], axis=0)
toco_val = np.concatenate([control_downsampled_toco_val, adverse_downsampled_toco_val], axis=0)
labels_val = np.concatenate([np.zeros(control_downsampled_fhr_val.shape[0], dtype=np.int64),
                             np.ones(adverse_downsampled_fhr_val.shape[0], dtype=np.int64)], axis=0)

perm = np.random.permutation(fhr_train.shape[0])
fhr_train = fhr_train[perm]
toco_train = toco_train[perm]
labels_train = labels_train[perm]

class CTGDataset(Dataset):
    def __init__(self, fhr_array, toco_array, labels_array):
        self.fhr = fhr_array.astype(np.float32)
        self.toco = toco_array.astype(np.float32)
        self.labels = labels_array.astype(np.int64)
    def __len__(self):
        return self.fhr.shape[0]
    def __getitem__(self, idx):
        f = self.fhr[idx]
        t = self.toco[idx]
        x = np.stack([f, t], axis=1)
        y = int(self.labels[idx])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

batch_size = 32
train_dataset = CTGDataset(fhr_train, toco_train, labels_train)
val_dataset = CTGDataset(fhr_val, toco_val, labels_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = create_seresnet152d_model(num_classes=2, dropout=0.1)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
  
best_auc = 0.0 # or loss 
patience = 3 # or 5
patience_counter = 0
num_epochs = 100

for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss = 0.0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]"):
        x = x.to(device)
        y = y.to(device).long()

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]"):
            x = x.to(device)
            y = y.to(device).long()

            outputs = model(x)
            loss = criterion(outputs, y)
            val_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)

    accuracy, auc, sensitivity, specificity = calculate_metrics(
        all_labels, all_preds, all_probs
    )

    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    optimal_preds = (np.array(all_probs) >= best_threshold).astype(int)
    balanced_acc = balanced_accuracy_score(all_labels, optimal_preds)

    print(
        f"Epoch {epoch}/{num_epochs} | "
        f"train_loss {avg_train_loss:.4f} | "
        f"val_loss {avg_val_loss:.4f} | "
        f"acc {accuracy:.4f} | "
        f"auc {auc:.4f} | "
        f"sens {sensitivity:.4f} | "
        f"spec {specificity:.4f} | "
        f"thr {best_threshold:.4f} | "
        f"bal_acc {balanced_acc:.4f}"
    )

    if auc > best_auc:
        best_auc = auc
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}. Best AUC: {best_auc:.4f}")
            break

print(f"Training finished. Best AUC: {best_auc:.4f}")
