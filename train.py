import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # <--- Added
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics.pairwise import cosine_distances
from timm.layers import trunc_normal_, DropPath
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ==============================================================================
# 1. MODEL DEFINITION (RFNet)
# ==============================================================================

class Block(nn.Module):
    def __init__(self, dim, expansion, kernel_size, stride=1, padding=0, drop_path=0.0):
        super(Block, self).__init__()
        inner_dim = dim * expansion
        self.f_in = nn.Conv1d(dim, inner_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.gate = nn.Conv1d(dim, inner_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.depthwise_conv = nn.Conv1d(inner_dim, inner_dim, kernel_size=kernel_size,
                                       stride=stride, padding=padding, groups=inner_dim, bias=False)
        self.project = nn.Conv1d(inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.ln = nn.LayerNorm(dim, eps=1e-6)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.ln(x.permute(0, 2, 1)).permute(0,2,1)
        x = self.depthwise_conv(self.f_in(x)) * F.silu(self.gate(x))      
        x = self.drop_path(self.project(x)) + residual
        return x
    
class Reduction(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Reduction, self).__init__()
        self.conv = nn.Conv1d(dim_in, dim_out, kernel_size=2, stride=2, padding=0, bias=False)
        self.ln = nn.LayerNorm(dim_in, eps=1e-6)
    def forward(self, x):
        x = self.ln(x.permute(0, 2, 1)).permute(0,2,1) 
        x = self.conv(x)
        return x
    
class RFNet(nn.Module):
    def __init__(self, num_channel=2, layers=[3, 3, 9, 3], base_dim=48, expansion=2, dim_embedding=128):
        super(RFNet, self).__init__()

        self.stem = nn.Conv1d(num_channel, base_dim, kernel_size=7, stride=1, padding=3, bias=False)
        
        dims = [base_dim * (2 ** i) for i in range(len(layers))]
        self.blocks = nn.ModuleList()
        
        for i in range(len(layers)):
            for _ in range(layers[i]):
                self.blocks.append(Block(dims[i], expansion, kernel_size=5, padding=2))
            if i < len(layers) - 1:
                self.blocks.append(Reduction(dims[i], dims[i+1]))
        
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        
        # Calculate Flatten Dim
        final_spatial_dim = 117 // (2**(len(layers)-1)) 
        final_feature_dim = dims[-1] * final_spatial_dim

        self.classifier = nn.Linear(final_feature_dim, dim_embedding, bias=False)
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x.permute(0, 2, 1)).permute(0,2,1)
        x = self.classifier(x.flatten(1))
        return F.normalize(x, p=2, dim=1) 
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

# ==============================================================================
# 2. LOSS FUNCTION (VarCon)
# ==============================================================================

class VarConLoss(nn.Module):
    def __init__(self, tau1: float = 0.1, epsilon: float = 0.02):
        super().__init__()
        self.tau1 = tau1
        self.epsilon = epsilon
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean', log_target=False)

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        unique_labels, labels_remapped = torch.unique(labels, return_inverse=True)
        num_classes_in_batch = unique_labels.shape[0]

        one_hot = F.one_hot(labels_remapped, num_classes=num_classes_in_batch).float()
        class_counts = one_hot.sum(dim=0).unsqueeze(-1)
        class_sums = one_hot.T @ feats
        class_means = class_sums / class_counts.clamp(min=1.0)
        w = F.normalize(class_means, p=2, dim=1)

        logits = torch.matmul(feats, w.T) / self.tau1
        log_p_theta = F.log_softmax(logits, dim=1)

        probs = log_p_theta.exp()
        p_r_z = torch.gather(probs, 1, labels_remapped.view(-1, 1))

        tau_2 = (self.tau1 - self.epsilon) + 2 * self.epsilon * p_r_z
        tau_2 = tau_2.clamp(min=1e-4)

        exp_term = torch.exp(1.0 / tau_2)
        denominator = exp_term + (num_classes_in_batch - 1.0)
        q_phi_true = exp_term / denominator
        q_phi_other = 1.0 / denominator

        q_phi = torch.full_like(probs, fill_value=0.0)
        for i in range(num_classes_in_batch):
             q_phi[:, i] = q_phi_other.squeeze()
        q_phi.scatter_(1, labels_remapped.view(-1, 1), q_phi_true)

        loss_kl = self.kl_div_loss(log_p_theta, q_phi)
        log_p_r_z = torch.gather(log_p_theta, 1, labels_remapped.view(-1, 1))
        loss_log_post = -log_p_r_z.mean()
        
        return loss_kl + loss_log_post

# ==============================================================================
# 3. DATA LOADING & PROCESSING
# ==============================================================================

def load_json_file(file_path):
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        pass
    return data

def load_dataset(root_dir, device_list, distance_list):
    xs = []
    ys = []
    print(f"Scanning {root_dir} for devices: {device_list}...")
    
    for device in device_list:
        for distance in distance_list:
            candidates = [
                f"{root_dir}/Device{device}/Device{device}_{distance}m.json",
                f"{root_dir}/Device{device}/device{device}_{distance}m.json",
                f"{root_dir}/device{device}/device{device}_{distance}m.json",
                f"{root_dir}/Device{device}_{distance}m.json"
            ]
            file_name = None
            for path in candidates:
                if os.path.exists(path):
                    file_name = path
                    break
            
            if not file_name: continue

            data = load_json_file(file_name)
            for record in data:
                raw_csi = record.get("csi", [])
                if raw_csi and len(raw_csi) == 234:
                    x = np.array(raw_csi, dtype=float)
                    x = np.nan_to_num(x, nan=0.0)
                    if len(x) % 2 == 0:
                        x_complex = x[0::2] + 1j * x[1::2]
                        xs.append(x_complex)
                        try:
                            label = device_list.index(device)
                            ys.append(label)
                        except ValueError:
                            pass 
    return np.array(xs), np.array(ys)

def process_csi(X_complex):
    # 1. Magnitude
    X_abs = np.abs(X_complex)
    # 2. Phase (Unwrapped)
    X_phase = np.angle(X_complex)
    X_phase_unwrapped = np.unwrap(X_phase, axis=1)
    
    # Shape: (N, 2, 117)
    X_combined = np.stack([X_abs, X_phase_unwrapped], axis=1)
    return X_combined

def get_dataloaders(root_dir, seen_devices, unseen_devices, batch_size):
    distances = [3, 6, 10, 20, 30, 40, 50, 60, 71]
    
    # 1. Load SEEN Data
    print(f"--- Loading SEEN Data {seen_devices} ---")
    X_seen, y_seen = load_dataset(root_dir, seen_devices, distances)
    if len(X_seen) == 0: raise ValueError("No data found for seen devices.")

    # Stratified Split: 80% Train, 20% Test (Seen)
    X_train, X_test_seen, y_train, y_test_seen = train_test_split(
        X_seen, y_seen, test_size=0.2, random_state=42, stratify=y_seen
    )

    # 2. Load UNSEEN Data
    print(f"--- Loading UNSEEN Data {unseen_devices} ---")
    X_unseen, y_unseen_raw = load_dataset(root_dir, unseen_devices, distances)
    # Remap unseen labels (e.g. 0,1,2 -> 7,8,9)
    y_unseen = y_unseen_raw + len(seen_devices)

    # 3. Combine Test Sets
    X_test_combined = np.concatenate([X_test_seen, X_unseen])
    y_test_combined = np.concatenate([y_test_seen, y_unseen])

    print(f"Train Size: {len(X_train)} | Test Size: {len(X_test_combined)} (Seen: {len(X_test_seen)}, Unseen: {len(X_unseen)})")

    # 4. Processing
    print("Processing CSI Features (Abs + Phase)...")
    X_train_proc = process_csi(X_train)          # (N_train, 2, 117)
    X_test_proc = process_csi(X_test_combined)   # (N_test, 2, 117)

    # 5. Standardization using StandardScaler
    print("Applying StandardScaler...")
    
    # Reshape to (Samples, Features) -> (N, 234)
    # We treat every subcarrier in every channel as a separate feature to normalize
    N_train, C, L = X_train_proc.shape
    X_train_flat = X_train_proc.reshape(N_train, -1)
    
    scaler = StandardScaler()
    X_train_scaled_flat = scaler.fit_transform(X_train_flat)
    
    # Transform Test Data using Train Statistics
    N_test = X_test_proc.shape[0]
    X_test_flat = X_test_proc.reshape(N_test, -1)
    X_test_scaled_flat = scaler.transform(X_test_flat)
    
    # Reshape back to (N, 2, 117)
    X_train_norm = X_train_scaled_flat.reshape(N_train, C, L)
    X_test_norm = X_test_scaled_flat.reshape(N_test, C, L)

    print(f"Scaler Mean Shape: {scaler.mean_.shape}") # Should be (234,)

    # 6. Create DataLoaders
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_norm), torch.LongTensor(y_train)), 
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test_norm), torch.LongTensor(y_test_combined)), 
                             batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, len(seen_devices) + len(unseen_devices)

# ==============================================================================
# 4. TRAINING & EVALUATION UTILS
# ==============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for signals, labels in pbar:
        signals, labels = signals.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(signals)
        loss = criterion(embeddings, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    return running_loss / len(train_loader)

def evaluate_few_shot(model, loader, device, n_shots=5, n_episodes=5):
    """N-Way K-Shot Evaluation."""
    model.eval()
    all_feats, all_labels = [], []
    
    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            feats = model(signals) 
            all_feats.append(feats.cpu())
            all_labels.append(labels.cpu())
            
    all_feats = torch.cat(all_feats)
    all_labels = torch.cat(all_labels)
    unique_classes = torch.unique(all_labels).numpy()
    
    accuracies = []
    
    # Episode Loop
    for _ in range(n_episodes):
        support_centroids, valid_classes = [], []
        query_feats_list, query_labels_list = [], []
        
        for c in unique_classes:
            indices = (all_labels == c).nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(indices))
            indices = indices[perm]
            
            if len(indices) <= n_shots: continue
            valid_classes.append(c)
            
            # Split
            support_idx = indices[:n_shots]
            query_idx = indices[n_shots:]
            
            # Centroid
            centroid = all_feats[support_idx].mean(dim=0)
            centroid = F.normalize(centroid, p=2, dim=0)
            support_centroids.append(centroid)
            
            query_feats_list.append(all_feats[query_idx])
            query_labels_list.append(all_labels[query_idx])
            
        if not support_centroids: continue
            
        support_centroids = torch.stack(support_centroids).to(device)
        query_feats = torch.cat(query_feats_list).to(device)
        query_labels = torch.cat(query_labels_list).to(device)
        
        sims = torch.matmul(query_feats, support_centroids.T)
        preds_local = torch.argmax(sims, dim=1)
        
        valid_classes_tensor = torch.tensor(valid_classes, device=device)
        preds_global = valid_classes_tensor[preds_local]
        
        acc = (preds_global == query_labels).float().mean().item()
        accuracies.append(acc)
        
        # Keep last episode for plotting
        last_preds = preds_global.cpu().numpy()
        last_labels = query_labels.cpu().numpy()
        last_centroids = support_centroids.cpu().numpy()
        
    return np.mean(accuracies), last_preds, last_labels, last_centroids, unique_classes

# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================

def main():
    # --- Config ---
    ROOT_DIR = 'dataset-new'
    SEEN = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    UNSEEN = ['01', '03', '09']
    BATCH_SIZE = 256
    EPOCHS = 10
    LR = 0.0005
    SAVE_PATH = 'best_rfnet_model.pth'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    train_loader, test_loader, num_total_classes = get_dataloaders(ROOT_DIR, SEEN, UNSEEN, BATCH_SIZE)

    # --- Model ---
    model = RFNet(dim_embedding=128).to(device)
    criterion = VarConLoss(tau1=0.1, epsilon=0.02)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # --- Loop ---
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate (Few-Shot on Full Test Set)
        acc, _, _, _, _ = evaluate_few_shot(model, test_loader, device, n_shots=50, n_episodes=1)

        print(f"Loss: {train_loss:.4f} | Test Acc (50-shot): {acc*100:.2f}%")

        scheduler.step()
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"--> Best Model Saved! ({best_acc*100:.2f}%)")

    print(f"\nTraining Finished. Best Accuracy: {best_acc*100:.2f}%")

    # --- Final Visualization ---
    print("\nGenerating Final Plots...")
    model.load_state_dict(torch.load(SAVE_PATH))
    acc, preds, labels, centroids, classes = evaluate_few_shot(model, test_loader, device, n_shots=5, n_episodes=1)

    # 1. Confusion Matrix
    all_device_names = SEEN + UNSEEN
    tick_labels = [all_device_names[i] for i in classes]
    
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=tick_labels, yticklabels=tick_labels)
    plt.title(f"Confusion Matrix (Test Acc: {acc*100:.2f}%)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    
    # 2. Feature Distance
    dist_matrix = cosine_distances(centroids)
    plt.figure(figsize=(10, 8))
    sns.heatmap(dist_matrix, annot=True, fmt=".2f", cmap="viridis", xticklabels=tick_labels, yticklabels=tick_labels)
    plt.title("Feature Space Distance (Cosine)")
    plt.tight_layout()
    plt.savefig("feature_distance.png")
    
    print("Plots saved: confusion_matrix.png, feature_distance.png")

if __name__ == "__main__":
    main()