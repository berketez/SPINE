"""
SPINE-AI: Eğitilebilir Yapısal Analiz Modeli

Fizik-bilgili neural network ile kritik yük tahmini.

Girdi: E, nu, h, L, W, m, n
Çıktı: N_cr (kritik burkulma yükü)

Kullanım:
    from spine_ai import SpineAI, train_model, test_model

    model = SpineAI()
    train_model(model, "path/to/dataset.csv", epochs=100)
    test_model(model, "path/to/dataset.csv")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import math
import csv
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# DEVICE
# =============================================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()


# =============================================================================
# DATASET
# =============================================================================

@dataclass
class BucklingExample:
    """Tek bir burkulma örneği."""
    E: float        # Young modülü (Pa)
    nu: float       # Poisson oranı
    h: float        # Kalınlık (m)
    L: float        # Uzunluk (m)
    W: float        # Genişlik (m)
    m: int          # x modu
    n: int          # y modu
    N_cr: float     # Kritik yük (N/m)
    theory: str     # KL veya Mindlin


class BucklingDataset(Dataset):
    """Burkulma veri seti."""

    def __init__(self, csv_path: str):
        self.examples: List[BucklingExample] = []
        self.load_csv(csv_path)
        self.compute_normalization()

    def load_csv(self, path: str):
        """CSV dosyasını yükle."""
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.examples.append(BucklingExample(
                    E=float(row['E_GPa']) * 1e9,
                    nu=float(row['nu']),
                    h=float(row['h_mm']) / 1000,
                    L=float(row['L_m']),
                    W=float(row['W_m']),
                    m=int(float(row['mode_m'])),
                    n=int(float(row['mode_n'])),
                    N_cr=float(row['N_cr_N_per_m']),
                    theory=row.get('theory_used', 'KL')
                ))

    def compute_normalization(self):
        """Min-max normalizasyon parametrelerini hesapla."""
        if not self.examples:
            return

        self.norm = {
            'E': {'min': min(ex.E for ex in self.examples),
                  'max': max(ex.E for ex in self.examples)},
            'nu': {'min': min(ex.nu for ex in self.examples),
                   'max': max(ex.nu for ex in self.examples)},
            'h': {'min': min(ex.h for ex in self.examples),
                  'max': max(ex.h for ex in self.examples)},
            'L': {'min': min(ex.L for ex in self.examples),
                  'max': max(ex.L for ex in self.examples)},
            'W': {'min': min(ex.W for ex in self.examples),
                  'max': max(ex.W for ex in self.examples)},
        }

        # N_cr için log-scale
        log_ncr = [math.log10(max(ex.N_cr, 1e-6)) for ex in self.examples]
        self.norm['log_N_cr'] = {'min': min(log_ncr), 'max': max(log_ncr)}

    def normalize(self, value: float, key: str) -> float:
        """Değeri [0, 1] aralığına normalize et."""
        vmin = self.norm[key]['min']
        vmax = self.norm[key]['max']
        if vmax - vmin < 1e-9:
            return 0.5
        return (value - vmin) / (vmax - vmin)

    def denormalize_log_ncr(self, value: float) -> float:
        """Log-scale N_cr'ı gerçek değere dönüştür."""
        vmin = self.norm['log_N_cr']['min']
        vmax = self.norm['log_N_cr']['max']
        log_ncr = vmin + value * (vmax - vmin)
        return 10 ** log_ncr

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]

        # Normalize inputs
        inputs = torch.tensor([
            self.normalize(ex.E, 'E'),
            self.normalize(ex.nu, 'nu'),
            self.normalize(ex.h, 'h'),
            self.normalize(ex.L, 'L'),
            self.normalize(ex.W, 'W'),
            float(ex.m) / 5.0,  # mod normalize (1-5 arası varsayıyoruz)
            float(ex.n) / 5.0,
        ], dtype=torch.float32)

        # Log-scale target
        log_ncr = math.log10(max(ex.N_cr, 1e-6))
        log_ncr_norm = self.normalize(log_ncr, 'log_N_cr')
        target = torch.tensor([log_ncr_norm], dtype=torch.float32)

        # Raw target for evaluation
        raw_target = torch.tensor([ex.N_cr], dtype=torch.float32)

        return {
            'inputs': inputs,
            'target': target,
            'raw_target': raw_target,
        }


# =============================================================================
# MODEL
# =============================================================================

class PhysicsLayer(nn.Module):
    """
    Fizik-bilgili katman.

    Plaka rijitliği D = Eh³/[12(1-ν²)] hesaplamasını içerir.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.physics_weight = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, physics_features: Optional[torch.Tensor] = None):
        out = self.linear(x)
        if physics_features is not None:
            out = out + self.physics_weight * physics_features
        return out


class SpineAI(nn.Module):
    """
    SPINE-AI: Fizik-bilgili Neural Network

    Mimari:
        Girdi (7) → [PhysicsEncoder] → [Hidden Layers] → [Output] → N_cr

    Girdi: [E_norm, nu_norm, h_norm, L_norm, W_norm, m_norm, n_norm]
    Çıktı: log10(N_cr) normalized

    ~10k parametre hedefi için:
        7 → 64 → 128 → 64 → 32 → 1
        Toplam: 7*64 + 64*128 + 128*64 + 64*32 + 32*1 = 448 + 8192 + 8192 + 2048 + 32 ≈ 19k

    Daha küçük versiyon (~5k):
        7 → 32 → 64 → 32 → 1
        Toplam: 7*32 + 32*64 + 64*32 + 32*1 = 224 + 2048 + 2048 + 32 ≈ 4.4k
    """

    def __init__(self, hidden_dims: List[int] = [64, 128, 64, 32],
                 use_physics_features: bool = True,
                 dropout: float = 0.1):
        super().__init__()

        self.use_physics_features = use_physics_features
        input_dim = 7  # E, nu, h, L, W, m, n

        # Physics feature dimension
        physics_dim = 3 if use_physics_features else 0  # D_approx, aspect_ratio, mode_factor

        # Build network
        layers = []
        prev_dim = input_dim + physics_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # [0, 1] output for normalized log N_cr

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def compute_physics_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fizik özelliklerini hesapla.

        x: [batch, 7] - [E, nu, h, L, W, m, n] normalized

        Returns:
            [batch, 3] - [D_approx, aspect_ratio, mode_factor]
        """
        E_norm = x[:, 0:1]
        nu_norm = x[:, 1:2]
        h_norm = x[:, 2:3]
        L_norm = x[:, 3:4]
        W_norm = x[:, 4:5]
        m_norm = x[:, 5:6]
        n_norm = x[:, 6:7]

        # Approximate D (normalized)
        # D ∝ E * h³ / (1 - ν²)
        D_approx = E_norm * (h_norm ** 3) / (1 - nu_norm ** 2 + 0.1)

        # Aspect ratio
        aspect = L_norm / (W_norm + 0.1)

        # Mode factor (higher modes = higher N_cr generally)
        mode_factor = m_norm ** 2 + n_norm ** 2

        return torch.cat([D_approx, aspect, mode_factor], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [batch, 7] normalized inputs

        Returns:
            [batch, 1] normalized log10(N_cr)
        """
        if self.use_physics_features:
            physics = self.compute_physics_features(x)
            x = torch.cat([x, physics], dim=1)

        return self.network(x)

    def count_parameters(self) -> int:
        """Toplam parametre sayısı."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# TRAINING
# =============================================================================

def train_model(
    model: SpineAI,
    dataset_path: str,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_split: float = 0.1,
    device: torch.device = DEVICE,
    print_every: int = 10,
) -> Dict[str, List[float]]:
    """
    Modeli eğit.

    Returns:
        history: {'train_loss': [...], 'val_loss': [...], 'val_rel_error': [...]}
    """
    print("=" * 60)
    print("SPINE-AI Eğitimi")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model parametreleri: {model.count_parameters():,}")
    print()

    # Load dataset
    dataset = BucklingDataset(dataset_path)
    print(f"Veri seti: {len(dataset)} örnek")

    # Split
    val_len = int(len(dataset) * val_split)
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    print(f"Eğitim: {train_len}, Doğrulama: {val_len}")
    print()

    # Move model to device
    model = model.to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Store normalization params in model
    model.norm = dataset.norm

    # Training history
    history = {'train_loss': [], 'val_loss': [], 'val_rel_error': []}
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            inputs = batch['inputs'].to(device)
            target = batch['target'].to(device)

            optimizer.zero_grad()
            pred = model(inputs)
            loss = F.mse_loss(pred, target)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= train_len
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        total_rel_error = 0.0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['inputs'].to(device)
                target = batch['target'].to(device)
                raw_target = batch['raw_target'].to(device)

                pred = model(inputs)
                loss = F.mse_loss(pred, target)
                val_loss += loss.item() * inputs.size(0)

                # Denormalize predictions
                pred_ncr = denormalize_predictions(pred, dataset.norm)
                rel_error = torch.abs(pred_ncr - raw_target) / (raw_target + 1e-6)
                total_rel_error += rel_error.sum().item()

        val_loss /= val_len
        val_rel_error = total_rel_error / val_len

        # History
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rel_error'].append(val_rel_error)

        # Best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

        # Print
        if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | Val RelErr: {val_rel_error:.4f}")

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    print()
    print(f"En iyi val loss: {best_val_loss:.6f}")
    print("=" * 60)

    return history


def denormalize_predictions(pred: torch.Tensor, norm: dict) -> torch.Tensor:
    """Normalize edilmiş tahminleri gerçek N_cr değerlerine dönüştür."""
    log_min = norm['log_N_cr']['min']
    log_max = norm['log_N_cr']['max']
    log_ncr = log_min + pred * (log_max - log_min)
    return 10 ** log_ncr


# =============================================================================
# TESTING
# =============================================================================

@torch.no_grad()
def test_model(
    model: SpineAI,
    dataset_path: str,
    device: torch.device = DEVICE,
    print_samples: int = 10,
) -> Dict[str, float]:
    """
    Modeli test et.

    Returns:
        metrics: {'mse': ..., 'mae': ..., 'rel_error': ..., 'max_error': ...}
    """
    print("=" * 60)
    print("SPINE-AI Test")
    print("=" * 60)

    dataset = BucklingDataset(dataset_path)

    # Use model's normalization if available
    if hasattr(model, 'norm'):
        dataset.norm = model.norm

    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    for batch in loader:
        inputs = batch['inputs'].to(device)
        raw_target = batch['raw_target']

        pred = model(inputs)
        pred_ncr = denormalize_predictions(pred, dataset.norm).cpu()

        all_preds.append(pred_ncr)
        all_targets.append(raw_target)

    preds = torch.cat(all_preds, dim=0).squeeze()
    targets = torch.cat(all_targets, dim=0).squeeze()

    # Metrics
    mse = F.mse_loss(preds, targets).item()
    mae = torch.abs(preds - targets).mean().item()
    rel_error = (torch.abs(preds - targets) / (targets + 1e-6)).mean().item()
    max_rel_error = (torch.abs(preds - targets) / (targets + 1e-6)).max().item()

    print(f"Toplam örnek: {len(dataset)}")
    print()
    print("Metrikler:")
    print(f"  MSE: {mse:.4e}")
    print(f"  MAE: {mae:.4e}")
    print(f"  Ortalama Bağıl Hata: {rel_error*100:.2f}%")
    print(f"  Maksimum Bağıl Hata: {max_rel_error*100:.2f}%")
    print()

    # Sample predictions
    if print_samples > 0:
        print(f"Örnek Tahminler (ilk {print_samples}):")
        print("-" * 50)
        print(f"{'Gerçek (kN/m)':<18} {'Tahmin (kN/m)':<18} {'Hata %':<10}")
        print("-" * 50)

        for i in range(min(print_samples, len(preds))):
            t = targets[i].item() / 1000
            p = preds[i].item() / 1000
            err = abs(p - t) / (t + 1e-6) * 100
            print(f"{t:<18.2f} {p:<18.2f} {err:<10.2f}")

        print("-" * 50)

    print("=" * 60)

    return {
        'mse': mse,
        'mae': mae,
        'rel_error': rel_error,
        'max_rel_error': max_rel_error,
    }


# =============================================================================
# SAVE/LOAD
# =============================================================================

def save_model(model: SpineAI, path: str):
    """Modeli kaydet."""
    torch.save({
        'state_dict': model.state_dict(),
        'norm': model.norm if hasattr(model, 'norm') else None,
        'config': {
            'use_physics_features': model.use_physics_features,
        }
    }, path)
    print(f"Model kaydedildi: {path}")


def load_model(path: str, device: torch.device = DEVICE) -> SpineAI:
    """Modeli yükle."""
    checkpoint = torch.load(path, map_location=device)

    config = checkpoint.get('config', {})
    model = SpineAI(use_physics_features=config.get('use_physics_features', True))
    model.load_state_dict(checkpoint['state_dict'])

    if checkpoint.get('norm'):
        model.norm = checkpoint['norm']

    model = model.to(device)
    print(f"Model yüklendi: {path}")
    return model


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SPINE-AI Eğitim ve Test")
    parser.add_argument("--train", type=str, help="Eğitim veri seti (CSV)")
    parser.add_argument("--test", type=str, help="Test veri seti (CSV)")
    parser.add_argument("--model", type=str, default="spine_ai_model.pth", help="Model dosyası")
    parser.add_argument("--epochs", type=int, default=100, help="Epoch sayısı")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")

    args = parser.parse_args()

    if args.train:
        print("\n" + "=" * 60)
        print("SPINE-AI")
        print("=" * 60 + "\n")

        model = SpineAI()
        print(f"Model parametreleri: {model.count_parameters():,}")
        print()

        history = train_model(
            model,
            args.train,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
        )

        save_model(model, args.model)

        if args.test:
            test_model(model, args.test)

    elif args.test:
        model = load_model(args.model)
        test_model(model, args.test)

    else:
        print("Kullanım:")
        print("  Eğitim: python spine_ai.py --train dataset.csv --epochs 100")
        print("  Test:   python spine_ai.py --test dataset.csv --model spine_ai_model.pth")
