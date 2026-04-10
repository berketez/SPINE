"""
SPINE Genelleme Testi

Amaç: Model hiç görmediği malzemelere genelleyebiliyor mu?

Strateji:
- Veriyi malzeme parametrelerine göre böl (rastgele değil!)
- %80 malzeme kombinasyonu ile eğit
- %20 HİÇ GÖRMEDİĞİ malzeme kombinasyonu ile test
- Gerçek dünya senaryosu: Yeni bir malzeme geldi, model tahmin edebilecek mi?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import csv
import numpy as np
from typing import Dict, List, Tuple
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
# DATASET - Malzeme bazlı bölme
# =============================================================================

@dataclass
class Sample:
    E: float
    nu: float
    h: float
    L: float
    W: float
    m: int
    n: int
    N_cr: float
    theory: str


def load_dataset(csv_path: str) -> List[Sample]:
    """CSV'den veri yükle."""
    samples = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(Sample(
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
    return samples


def split_by_material(samples: List[Sample], test_ratio: float = 0.2, seed: int = 42):
    """
    Malzeme bazlı bölme.

    Aynı (E, nu, h) kombinasyonuna sahip örnekler ya hep train ya hep test'te.
    Bu şekilde model gerçekten YENİ malzemelere genelleme yapıyor mu görürüz.
    """
    np.random.seed(seed)

    # Benzersiz malzeme kombinasyonları bul
    material_keys = set()
    for s in samples:
        # E ve h'yi grupla (küçük farkları yok say)
        E_bucket = round(s.E / 1e9, 0)  # GPa cinsinden yuvarla
        h_bucket = round(s.h * 1000, 0)  # mm cinsinden yuvarla
        nu_bucket = round(s.nu, 2)
        material_keys.add((E_bucket, nu_bucket, h_bucket))

    material_keys = list(material_keys)
    np.random.shuffle(material_keys)

    # Test için ayır
    n_test = max(1, int(len(material_keys) * test_ratio))
    test_keys = set(material_keys[:n_test])
    train_keys = set(material_keys[n_test:])

    # Örnekleri ayır
    train_samples = []
    test_samples = []

    for s in samples:
        E_bucket = round(s.E / 1e9, 0)
        h_bucket = round(s.h * 1000, 0)
        nu_bucket = round(s.nu, 2)
        key = (E_bucket, nu_bucket, h_bucket)

        if key in test_keys:
            test_samples.append(s)
        else:
            train_samples.append(s)

    return train_samples, test_samples, len(train_keys), len(test_keys)


class MaterialDataset(Dataset):
    """PyTorch dataset."""

    def __init__(self, samples: List[Sample], norm: dict = None):
        self.samples = samples

        if norm is None:
            self.norm = self._compute_norm()
        else:
            self.norm = norm

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
            'log_N_cr': {
                'min': min(math.log10(max(s.N_cr, 1e-6)) for s in self.samples),
                'max': max(math.log10(max(s.N_cr, 1e-6)) for s in self.samples)
            }
        }

    def normalize(self, value: float, key: str) -> float:
        vmin = self.norm[key]['min']
        vmax = self.norm[key]['max']
        if vmax - vmin < 1e-9:
            return 0.5
        return (value - vmin) / (vmax - vmin)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        inputs = torch.tensor([
            self.normalize(s.E, 'E'),
            self.normalize(s.nu, 'nu'),
            self.normalize(s.h, 'h'),
            self.normalize(s.L, 'L'),
            self.normalize(s.W, 'W'),
            float(s.m) / 5.0,
            float(s.n) / 5.0,
        ], dtype=torch.float32)

        log_ncr = math.log10(max(s.N_cr, 1e-6))
        log_ncr_norm = (log_ncr - self.norm['log_N_cr']['min']) / \
                       (self.norm['log_N_cr']['max'] - self.norm['log_N_cr']['min'] + 1e-9)

        return {
            'inputs': inputs,
            'target': torch.tensor([log_ncr_norm], dtype=torch.float32),
            'raw_target': torch.tensor([s.N_cr], dtype=torch.float32),
            'E': s.E,
            'nu': s.nu,
            'h': s.h,
        }


# =============================================================================
# MODEL - Hybrid Physics + MLP
# =============================================================================

class SpineHybrid(nn.Module):
    """
    Hybrid SPINE: Fizik Formülü + Küçük MLP Düzeltmesi

    N_cr = N_cr_physics × (1 + correction)

    - N_cr_physics: Kirchhoff-Love + Mindlin formülü (eğitim YOK)
    - correction: Küçük MLP, [-0.1, +0.1] arası (eğitim VAR)

    Bu şekilde:
    - Fizik %95'i veriyor
    - MLP sadece ince ayar
    - Çok daha az parametre
    - Daha iyi genelleme
    """

    def __init__(self, correction_hidden: List[int] = [32, 16],
                 max_correction: float = 0.1):
        super().__init__()

        self.max_correction = max_correction

        # Küçük düzeltme MLP'si
        layers = []
        prev_dim = 7  # E, nu, h, L, W, m, n
        for hidden in correction_hidden:
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(nn.Tanh())
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Tanh())  # [-1, 1] çıktı

        self.correction_net = nn.Sequential(*layers)

        # Normalizasyon parametreleri (eğitimde set edilecek)
        self.norm = None

    def physics_N_cr(self, E: torch.Tensor, nu: torch.Tensor, h: torch.Tensor,
                     L: torch.Tensor, W: torch.Tensor, m: torch.Tensor, n: torch.Tensor,
                     use_mindlin: bool = True) -> torch.Tensor:
        """
        Fizik formülü ile N_cr hesapla.

        Kirchhoff-Love: N_cr = D × (p² + q²)² / p²
        Mindlin düzeltmesi: N_cr = N_cr_KL × F(η, ν)
        """
        # Plaka rijitliği
        D = E * h**3 / (12 * (1 - nu**2))

        # Dalga sayıları
        p = m * math.pi / L
        q = n * math.pi / W
        T = p**2 + q**2

        # Kirchhoff-Love kritik yük
        N_cr_kl = D * (T**2) / (p**2)

        if use_mindlin:
            # Mindlin düzeltmesi
            kappa = 5.0 / 6.0
            G = E / (2 * (1 + nu))
            shear_rigidity = kappa * G * h
            eta = (D * T) / shear_rigidity

            # F(η, ν)
            a = (1.0 - nu) / 2.0
            b = (3.0 - nu) / 2.0
            F = (1.0 + a * eta) / (1.0 + b * eta + a * eta**2)

            return N_cr_kl * F

        return N_cr_kl

    def forward(self, inputs: torch.Tensor,
                E: torch.Tensor, nu: torch.Tensor, h: torch.Tensor,
                L: torch.Tensor, W: torch.Tensor,
                m: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        inputs: [batch, 7] normalized
        E, nu, h, L, W, m, n: gerçek değerler (denormalize)
        """
        # 1. Fizik hesabı
        N_cr_physics = self.physics_N_cr(E, nu, h, L, W, m, n)

        # 2. Küçük düzeltme
        correction = self.correction_net(inputs)  # [-1, 1]
        correction = correction * self.max_correction  # [-0.1, 0.1]

        # 3. Final
        N_cr = N_cr_physics * (1 + correction.squeeze())

        return N_cr, N_cr_physics, correction.squeeze()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SpinePureMLP(nn.Module):
    """
    Karşılaştırma için: Saf MLP (fizik yok).
    """

    def __init__(self, hidden_dims: List[int] = [64, 128, 64, 32]):
        super().__init__()

        layers = []
        prev_dim = 7
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)
        self.norm = None

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.network(inputs), None, None

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, loader, optimizer, device, is_hybrid=True):
    model.train()
    total_loss = 0
    total_rel_err = 0
    count = 0

    for batch in loader:
        inputs = batch['inputs'].to(device)
        raw_target = batch['raw_target'].to(device).squeeze()

        optimizer.zero_grad()

        if is_hybrid:
            # Gerçek değerleri de geçir
            E = torch.tensor([b for b in batch['E']], dtype=torch.float32, device=device)
            nu = torch.tensor([b for b in batch['nu']], dtype=torch.float32, device=device)
            h = torch.tensor([b for b in batch['h']], dtype=torch.float32, device=device)

            # L, W, m, n'i inputs'tan denormalize et (yaklaşık)
            L = inputs[:, 3] * 1.7 + 0.3  # 0.3-2.0 aralığı
            W = inputs[:, 4] * 1.7 + 0.3
            m = (inputs[:, 5] * 5).clamp(1, 5)
            n = (inputs[:, 6] * 5).clamp(1, 5)

            pred, physics_pred, correction = model(inputs, E, nu, h, L, W, m, n)

            # Log-space loss
            loss = F.mse_loss(torch.log(pred + 1), torch.log(raw_target + 1))
        else:
            pred, _, _ = model(inputs)
            # Denormalize
            log_min = model.norm['log_N_cr']['min']
            log_max = model.norm['log_N_cr']['max']
            log_pred = log_min + pred.squeeze() * (log_max - log_min)
            pred = 10 ** log_pred

            loss = F.mse_loss(torch.log(pred + 1), torch.log(raw_target + 1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Metrics
        rel_err = (torch.abs(pred - raw_target) / (raw_target + 1e-6)).mean()

        total_loss += loss.item() * inputs.size(0)
        total_rel_err += rel_err.item() * inputs.size(0)
        count += inputs.size(0)

    return total_loss / count, total_rel_err / count


@torch.no_grad()
def evaluate(model, loader, device, is_hybrid=True):
    model.eval()
    total_rel_err = 0
    count = 0

    all_preds = []
    all_targets = []
    all_physics = []

    for batch in loader:
        inputs = batch['inputs'].to(device)
        raw_target = batch['raw_target'].to(device).squeeze()

        if is_hybrid:
            E = torch.tensor([b for b in batch['E']], dtype=torch.float32, device=device)
            nu = torch.tensor([b for b in batch['nu']], dtype=torch.float32, device=device)
            h = torch.tensor([b for b in batch['h']], dtype=torch.float32, device=device)

            L = inputs[:, 3] * 1.7 + 0.3
            W = inputs[:, 4] * 1.7 + 0.3
            m = (inputs[:, 5] * 5).clamp(1, 5)
            n = (inputs[:, 6] * 5).clamp(1, 5)

            pred, physics_pred, _ = model(inputs, E, nu, h, L, W, m, n)
            all_physics.extend(physics_pred.cpu().tolist())
        else:
            pred, _, _ = model(inputs)
            log_min = model.norm['log_N_cr']['min']
            log_max = model.norm['log_N_cr']['max']
            log_pred = log_min + pred.squeeze() * (log_max - log_min)
            pred = 10 ** log_pred

        rel_err = (torch.abs(pred - raw_target) / (raw_target + 1e-6)).mean()
        total_rel_err += rel_err.item() * inputs.size(0)
        count += inputs.size(0)

        all_preds.extend(pred.cpu().tolist())
        all_targets.extend(raw_target.cpu().tolist())

    return total_rel_err / count, all_preds, all_targets, all_physics


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("SPINE Genelleme Testi")
    print("Hiç görmediği malzemelere genelleyebiliyor mu?")
    print("=" * 70)
    print()

    # Load data
    csv_path = "/Users/apple/Desktop/dosyalar/okul/güz26/bitirmeproje1/kodlar/bucklingpinn/genelleme/lasttraining/reference_dataset.csv"
    samples = load_dataset(csv_path)
    print(f"Toplam örnek: {len(samples)}")

    # Split by material
    train_samples, test_samples, n_train_mat, n_test_mat = split_by_material(samples, test_ratio=0.2)

    print(f"\nMalzeme bazlı bölme:")
    print(f"  Eğitim malzemeleri: {n_train_mat}")
    print(f"  Test malzemeleri: {n_test_mat} (HİÇ GÖRMEDİĞİ)")
    print(f"  Eğitim örnekleri: {len(train_samples)}")
    print(f"  Test örnekleri: {len(test_samples)}")

    # Datasets
    train_dataset = MaterialDataset(train_samples)
    test_dataset = MaterialDataset(test_samples, norm=train_dataset.norm)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # =========================================================================
    # Model 1: Hybrid (Fizik + küçük MLP)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Model 1: SpineHybrid (Fizik + Küçük MLP Düzeltmesi)")
    print("=" * 70)

    hybrid_model = SpineHybrid(correction_hidden=[32, 16], max_correction=0.1)
    hybrid_model.norm = train_dataset.norm
    hybrid_model = hybrid_model.to(DEVICE)

    print(f"Parametre sayısı: {hybrid_model.count_parameters()}")

    optimizer = torch.optim.AdamW(hybrid_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    print("\nEğitim başlıyor...")
    for epoch in range(1, 101):
        train_loss, train_err = train_epoch(hybrid_model, train_loader, optimizer, DEVICE, is_hybrid=True)
        scheduler.step()

        if epoch == 1 or epoch % 20 == 0:
            test_err, _, _, _ = evaluate(hybrid_model, test_loader, DEVICE, is_hybrid=True)
            print(f"Epoch {epoch:03d} | Train RelErr: {train_err:.4f} | Test RelErr: {test_err:.4f}")

    # Final evaluation
    test_err_hybrid, preds_h, targets_h, physics_h = evaluate(hybrid_model, test_loader, DEVICE, is_hybrid=True)

    print(f"\n*** Hybrid Model - Hiç Görmediği Malzemeler ***")
    print(f"Ortalama Bağıl Hata: {test_err_hybrid*100:.2f}%")

    # Physics alone hatası
    physics_err = np.mean([abs(p - t) / (t + 1e-6) for p, t in zip(physics_h, targets_h)])
    print(f"Sadece Fizik Hatası: {physics_err*100:.2f}%")
    print(f"MLP Düzeltmesi Katkısı: {(physics_err - test_err_hybrid)*100:.2f}%")

    # =========================================================================
    # Model 2: Pure MLP (karşılaştırma için)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Model 2: Pure MLP (Fizik yok, karşılaştırma için)")
    print("=" * 70)

    mlp_model = SpinePureMLP(hidden_dims=[64, 128, 64, 32])
    mlp_model.norm = train_dataset.norm
    mlp_model = mlp_model.to(DEVICE)

    print(f"Parametre sayısı: {mlp_model.count_parameters()}")

    optimizer = torch.optim.AdamW(mlp_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    print("\nEğitim başlıyor...")
    for epoch in range(1, 101):
        train_loss, train_err = train_epoch(mlp_model, train_loader, optimizer, DEVICE, is_hybrid=False)
        scheduler.step()

        if epoch == 1 or epoch % 20 == 0:
            test_err, _, _, _ = evaluate(mlp_model, test_loader, DEVICE, is_hybrid=False)
            print(f"Epoch {epoch:03d} | Train RelErr: {train_err:.4f} | Test RelErr: {test_err:.4f}")

    test_err_mlp, _, _, _ = evaluate(mlp_model, test_loader, DEVICE, is_hybrid=False)

    print(f"\n*** Pure MLP - Hiç Görmediği Malzemeler ***")
    print(f"Ortalama Bağıl Hata: {test_err_mlp*100:.2f}%")

    # =========================================================================
    # Karşılaştırma
    # =========================================================================
    print("\n" + "=" * 70)
    print("SONUÇ KARŞILAŞTIRMASI")
    print("=" * 70)
    print(f"\n{'Model':<30} {'Parametre':<15} {'Test Hatası':<15}")
    print("-" * 60)
    print(f"{'Sadece Fizik (formül)':<30} {'0':<15} {physics_err*100:.2f}%")
    print(f"{'SpineHybrid (fizik+MLP)':<30} {hybrid_model.count_parameters():<15} {test_err_hybrid*100:.2f}%")
    print(f"{'Pure MLP (fizik yok)':<30} {mlp_model.count_parameters():<15} {test_err_mlp*100:.2f}%")
    print("-" * 60)

    print("\n*** YORUM ***")
    if test_err_hybrid < test_err_mlp:
        improvement = (test_err_mlp - test_err_hybrid) / test_err_mlp * 100
        print(f"SpineHybrid, Pure MLP'den %{improvement:.1f} daha iyi genelleme yapıyor!")
        print("Fizik gömülü yaklaşım, yeni malzemelere daha iyi genelliyor.")
    else:
        print("Pure MLP daha iyi performans gösterdi.")
        print("Bu veri seti için fizik düzeltmesi çok katkı sağlamadı.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
