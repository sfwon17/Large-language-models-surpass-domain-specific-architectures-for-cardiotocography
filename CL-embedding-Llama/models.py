import torch
import torch.nn as nn
import torch.nn.functional as F

# refactored and cleaned, simplified for improved readability 

class PatchCNNFlattenMLPEmbedding(nn.Module):
    def __init__(self, patch_size=16, in_channels=2, llm_hidden_dim=4096, dropout=0.1, max_patches=1024):
        super().__init__()
        self.patch_size = patch_size
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.res1_conv1 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.res1_bn1 = nn.BatchNorm1d(64)
        self.res1_conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.res1_bn2 = nn.BatchNorm1d(64)
        self.res2_conv1 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.res2_bn1 = nn.BatchNorm1d(128)
        self.res2_conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.res2_bn2 = nn.BatchNorm1d(128)
        self.res2_downsample = nn.Conv1d(64, 128, kernel_size=1)
        self.activation = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, llm_hidden_dim),
            nn.LayerNorm(llm_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_patches, llm_hidden_dim))
        nn.init.normal_(self.positional_embedding, std=0.02)

    def forward(self, x):
        batch_size, seq_len, channels = x.shape
        num_patches = seq_len // self.patch_size
        x = x[:, :num_patches * self.patch_size, :]
        x = x.reshape(batch_size, num_patches, self.patch_size, channels)
        x = x.reshape(batch_size * num_patches, self.patch_size, channels)
        x = x.transpose(1, 2)
        x = self.activation(self.bn1(self.conv1(x)))
        identity = x
        out = self.activation(self.res1_bn1(self.res1_conv1(x)))
        out = self.res1_bn2(self.res1_conv2(out))
        out += identity
        out = self.activation(out)
        x = out
        identity = self.res2_downsample(x)
        out = self.activation(self.res2_bn1(self.res2_conv1(x)))
        out = self.res2_bn2(self.res2_conv2(out))
        out += identity
        out = self.activation(out)
        x = out
        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        x = x.reshape(batch_size, num_patches, -1)
        x = x + self.positional_embedding[:, :num_patches, :]
        return x

class ContrastiveCTGModel(nn.Module):
    def __init__(self, patch_size=64, llm_hidden_dim=4096, projection_dim=256):
        super().__init__()
        self.encoder = PatchCNNFlattenMLPEmbedding(
            patch_size=patch_size,
            in_channels=2,
            llm_hidden_dim=llm_hidden_dim,
            dropout=0.1
        )
        self.projection = nn.Sequential(
            nn.Linear(llm_hidden_dim, llm_hidden_dim),
            nn.ReLU(),
            nn.Linear(llm_hidden_dim, projection_dim)
        )

    def forward(self, x):
        embeddings = self.encoder(x)
        pooled = embeddings.mean(dim=1)
        projections = self.projection(pooled)
        return projections

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, dim=1)
        similarity = torch.matmul(features, features.T) / self.temperature
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.ones_like(mask).fill_diagonal_(0)
        mask = mask * logits_mask
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        loss = -mean_log_prob_pos.mean()
        return loss

class CTGClassifierWrapper(nn.Module):
    def __init__(self, encoder, llm):
        super().__init__()
        self.patch_cnn_mlp_embedding = encoder
        self.llm = llm
        self.llm_input_device = next(llm.parameters()).device

    def forward(self, fhr, toco, labels=None):
        time_series = torch.stack([fhr, toco], dim=-1)
        llm_embeds = self.patch_cnn_mlp_embedding(time_series)
        llm_embeds = llm_embeds.to(self.llm_input_device)
        if labels is not None:
            labels = labels.to(self.llm_input_device)
        outputs = self.llm(inputs_embeds=llm_embeds, labels=labels, return_dict=True)
        return outputs
