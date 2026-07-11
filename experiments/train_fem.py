"""
SPINE-AI FEM Verisi ile Eğitim

FEM verisi kullanarak delikli plakaları öğrenme testi.
Analitik formülün olmadığı durumlar için AI'ın öğrenme kapasitesi.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import csv
import math
from typing import Dict, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm


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
class FEMSample:
    E: float
    nu: float
    h: float
    L: float
    W: float
    has_hole: int
    hole_radius: float
    N_cr_fem: float


def load_fem_dataset(csv_path: str) -> List[FEMSample]:
    samples = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(FEMSample(
                E=float(row['E_GPa']) * 1e9,
                nu=float(row['nu']),
                h=float(row['h_mm']) / 1000,
                L=float(row['L_m']),
                W=float(row['W_m']),
                has_hole=int(float(row['has_hole'])),
                hole_radius=float(row['hole_radius_mm']) / 1000,
                N_cr_fem=float(row['N_cr_fem']),
            ))
    return samples


class FEMDataset(Dataset):
    def __init__(self, samples: List[FEMSample], norm: dict = None):
        self.samples = samples
        self.norm = norm if norm else self._compute_norm()

    def _compute_norm(self):
        return {
            'E': {'min': min(s.E for s in self.samples),
                  'max': max(s.E for s in self.samples)},
            'nu': {'min': min(s.nu for s in self.samples),
                   'max': max(s.nu for s in self.samples)},
            'h': {'min': min(s.h for s in self.samples),
                  'max': max(s.h for s in self.samples)},
            'L': {'min': min(s.L for s in self.samples),
                  'max': max(s.L for s in self.samples)},
            'W': {'min': min(s.W for s in self.samples),
                  'max': max(s.W for s in self.samples)},
            'hole_radius': {'min': 0, 'max': max(s.hole_radius for s in self.samples) + 0.01},
            'log_N_cr': {
                'min': min(math.log10(max(s.N_cr_fem, 1e-6)) for s in self.samples),
                'max': max(math.log10(max(s.N_cr_fem, 1e-6)) for s in self.samples)
            }
        }

    def normalize(self, value: float, key: str) -> float:
        vmin, vmax = self.norm[key]['min'], self.norm[key]['max']
        if vmax - vmin < 1e-9:
            return 0.5
        return (value - vmin) / (vmax - vmin)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # 8 input: E, nu, h, L, W, has_hole, hole_radius, aspect_ratio
        inputs = torch.tensor([
            self.normalize(s.E, 'E'),
            self.normalize(s.nu, 'nu'),
            self.normalize(s.h, 'h'),
            self.normalize(s.L, 'L'),
            self.normalize(s.W, 'W'),
            float(s.has_hole),
            self.normalize(s.hole_radius, 'hole_radius'),
            s.L / s.W / 3.0,  # aspect ratio normalized
        ], dtype=torch.float32)

        log_ncr = math.log10(max(s.N_cr_fem, 1e-6))
        log_ncr_norm = (log_ncr - self.norm['log_N_cr']['min']) / \
                       (self.norm['log_N_cr']['max'] - self.norm['log_N_cr']['min'] + 1e-9)

        return {
            'inputs': inputs,
            'target': torch.tensor([log_ncr_norm], dtype=torch.float32),
            'raw_target': torch.tensor([s.N_cr_fem], dtype=torch.float32),
            'has_hole': s.has_hole,
            'E': s.E, 'nu': s.nu, 'h': s.h, 'L': s.L, 'W': s.W,
            'hole_radius': s.hole_radius,
        }


# =============================================================================
# MODEL - Hybrid Physics + Hole Correction
# =============================================================================

class SpineHybridHole(nn.Module):
    """
    SPINE Hybrid - Delik Düzeltmeli

    N_cr = N_cr_physics × (1 - hole_effect) × (1 + fine_tune)

    - N_cr_physics: Analitik formül (minimum mod)
    - hole_effect: Delik etkisi (MLP ile öğrenilir)
    - fine_tune: İnce ayar
    """

    def __init__(self):
        super().__init__()

        input_dim = 8  # E, nu, h, L, W, has_hole, hole_radius, aspect

        # Delik etkisi için MLP
        self.hole_net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 0-1 arası çıktı
        )

        # Fine-tuning için küçük MLP
        self.fine_tune = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Tanh()
        )

        self.max_hole_effect = 0.9  # max %90 azalma
        self.max_fine_tune = 0.05   # max %5 ince ayar

    def analytical_N_cr_min(self, E, nu, h, L, W):
        """Minimum kritik yük - tüm modları tarayarak."""
        D = E * h**3 / (12 * (1 - nu**2))

        min_N_cr = torch.full_like(E, float('inf'))

        for m in range(1, 8):
            for n in range(1, 4):
                alpha = m * math.pi / L
                beta = n * math.pi / W
                N_cr = D * (alpha**2 + beta**2)**2 / alpha**2

                min_N_cr = torch.minimum(min_N_cr, N_cr)

        return min_N_cr

    def forward(self, inputs, E, nu, h, L, W, has_hole, hole_radius):
        batch_size = inputs.size(0)

        # 1. Analitik fizik
        N_cr_physics = self.analytical_N_cr_min(E, nu, h, L, W)

        # 2. Delik etkisi
        hole_effect = self.hole_net(inputs).squeeze() * self.max_hole_effect
        # Delik yoksa etki sıfır
        hole_effect = hole_effect * has_hole.float()

        # 3. Fine-tuning
        fine_tune = self.fine_tune(inputs).squeeze() * self.max_fine_tune

        # 4. Final
        N_cr = N_cr_physics * (1 - hole_effect) * (1 + fine_tune)

        return N_cr, N_cr_physics, hole_effect

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PureMLP(nn.Module):
    """Karşılaştırma için saf MLP."""

    def __init__(self, hidden_dims: List[int] = [64, 128, 64]):
        super().__init__()

        layers = []
        prev_dim = 8
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)
        self.norm = None

    def forward(self, inputs, **kwargs):
        return self.network(inputs).squeeze(), None, None

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# TRAINING
# =============================================================================

def train_model(model, train_loader, val_loader, epochs, device, is_hybrid=True):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_err = float('inf')
    best_state = None

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            inputs = batch['inputs'].to(device)
            raw_target = batch['raw_target'].to(device).squeeze()

            optimizer.zero_grad()

            if is_hybrid:
                E = torch.as_tensor(batch['E'], dtype=torch.float32, device=device)
                nu = torch.as_tensor(batch['nu'], dtype=torch.float32, device=device)
                h = torch.as_tensor(batch['h'], dtype=torch.float32, device=device)
                L = torch.as_tensor(batch['L'], dtype=torch.float32, device=device)
                W = torch.as_tensor(batch['W'], dtype=torch.float32, device=device)
                has_hole = torch.as_tensor(batch['has_hole'], dtype=torch.float32, device=device)
                hole_radius = torch.as_tensor(batch['hole_radius'], dtype=torch.float32, device=device)

                pred, _, _ = model(inputs, E, nu, h, L, W, has_hole, hole_radius)
            else:
                pred, _, _ = model(inputs)
                # Denormalize
                log_min = model.norm['log_N_cr']['min']
                log_max = model.norm['log_N_cr']['max']
                pred = 10 ** (log_min + pred * (log_max - log_min))

            loss = F.mse_loss(torch.log(pred + 1), torch.log(raw_target + 1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_err_all = []
        val_err_hole = []
        val_err_no_hole = []

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['inputs'].to(device)
                raw_target = batch['raw_target'].to(device).squeeze()
                has_hole = batch['has_hole']

                if is_hybrid:
                    E = torch.as_tensor(batch['E'], dtype=torch.float32, device=device)
                    nu = torch.as_tensor(batch['nu'], dtype=torch.float32, device=device)
                    h = torch.as_tensor(batch['h'], dtype=torch.float32, device=device)
                    L = torch.as_tensor(batch['L'], dtype=torch.float32, device=device)
                    W = torch.as_tensor(batch['W'], dtype=torch.float32, device=device)
                    has_hole_t = torch.as_tensor(batch['has_hole'], dtype=torch.float32, device=device)
                    hole_radius = torch.as_tensor(batch['hole_radius'], dtype=torch.float32, device=device)

                    pred, _, _ = model(inputs, E, nu, h, L, W, has_hole_t, hole_radius)
                else:
                    pred, _, _ = model(inputs)
                    log_min = model.norm['log_N_cr']['min']
                    log_max = model.norm['log_N_cr']['max']
                    pred = 10 ** (log_min + pred * (log_max - log_min))

                rel_err = (torch.abs(pred - raw_target) / (raw_target + 1e-6)).cpu().numpy()

                for i, err in enumerate(rel_err):
                    val_err_all.append(err)
                    if has_hole[i]:
                        val_err_hole.append(err)
                    else:
                        val_err_no_hole.append(err)

        val_err = np.mean(val_err_all)

        if val_err < best_val_err:
            best_val_err = val_err
            best_state = model.state_dict().copy()

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            hole_err = np.mean(val_err_hole) if val_err_hole else 0
            no_hole_err = np.mean(val_err_no_hole) if val_err_no_hole else 0
            print(f"Epoch {epoch:03d} | All: {val_err*100:.2f}% | "
                  f"Hole: {hole_err*100:.2f}% | NoHole: {no_hole_err*100:.2f}%")

    model.load_state_dict(best_state)
    return best_val_err


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("SPINE-AI FEM Verisi ile Eğitim")
    print("Delikli plakaları öğrenebiliyor mu?")
    print("=" * 70)
    print()

    # Load data
    samples = load_fem_dataset("fem_dataset.csv")
    print(f"Toplam örnek: {len(samples)}")

    n_holes = sum(1 for s in samples if s.has_hole)
    print(f"Delikli: {n_holes}, Deliksiz: {len(samples) - n_holes}")
    print()

    # Split
    np.random.seed(42)
    np.random.shuffle(samples)

    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    train_dataset = FEMDataset(train_samples)
    val_dataset = FEMDataset(val_samples, norm=train_dataset.norm)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # =========================================================================
    # Model 1: Hybrid (Physics + Hole MLP)
    # =========================================================================
    print("=" * 70)
    print("Model 1: SpineHybridHole (Fizik + Delik MLP)")
    print("=" * 70)

    hybrid = SpineHybridHole()
    hybrid = hybrid.to(DEVICE)
    print(f"Parametre: {hybrid.count_parameters()}")
    print()

    print("Eğitim:")
    train_model(hybrid, train_loader, val_loader, epochs=100, device=DEVICE, is_hybrid=True)

    # =========================================================================
    # Model 2: Pure MLP
    # =========================================================================
    print("\n" + "=" * 70)
    print("Model 2: Pure MLP (Fizik yok)")
    print("=" * 70)

    mlp = PureMLP(hidden_dims=[64, 128, 64])
    mlp.norm = train_dataset.norm
    mlp = mlp.to(DEVICE)
    print(f"Parametre: {mlp.count_parameters()}")
    print()

    print("Eğitim:")
    train_model(mlp, train_loader, val_loader, epochs=100, device=DEVICE, is_hybrid=False)

    # =========================================================================
    # Final Comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("SONUÇ")
    print("=" * 70)

    # Detailed evaluation
    for name, model, is_hybrid in [("SpineHybrid", hybrid, True), ("PureMLP", mlp, False)]:
        model.eval()

        err_hole = []
        err_no_hole = []

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['inputs'].to(DEVICE)
                raw_target = batch['raw_target'].to(DEVICE).squeeze()
                has_hole = batch['has_hole']

                if is_hybrid:
                    E = torch.as_tensor(batch['E'], dtype=torch.float32, device=DEVICE)
                    nu = torch.as_tensor(batch['nu'], dtype=torch.float32, device=DEVICE)
                    h = torch.as_tensor(batch['h'], dtype=torch.float32, device=DEVICE)
                    L = torch.as_tensor(batch['L'], dtype=torch.float32, device=DEVICE)
                    W = torch.as_tensor(batch['W'], dtype=torch.float32, device=DEVICE)
                    has_hole_t = torch.as_tensor(batch['has_hole'], dtype=torch.float32, device=DEVICE)
                    hole_radius = torch.as_tensor(batch['hole_radius'], dtype=torch.float32, device=DEVICE)

                    pred, _, _ = model(inputs, E, nu, h, L, W, has_hole_t, hole_radius)
                else:
                    pred, _, _ = model(inputs)
                    log_min = model.norm['log_N_cr']['min']
                    log_max = model.norm['log_N_cr']['max']
                    pred = 10 ** (log_min + pred * (log_max - log_min))

                rel_err = (torch.abs(pred - raw_target) / (raw_target + 1e-6)).cpu().numpy()

                for i, err in enumerate(rel_err):
                    if has_hole[i]:
                        err_hole.append(err)
                    else:
                        err_no_hole.append(err)

        print(f"\n{name}:")
        print(f"  Deliksiz plaka hatası: {np.mean(err_no_hole)*100:.2f}%")
        print(f"  Delikli plaka hatası:  {np.mean(err_hole)*100:.2f}%")
        print(f"  Toplam:                {np.mean(err_hole + err_no_hole)*100:.2f}%")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
