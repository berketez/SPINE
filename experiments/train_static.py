"""
StaticNeuron Eğitim Testi
- Statik eğilme analizi: D∇⁴w = q
- SpineHybridStatic vs Pure MLP karşılaştırması
- Hedef: Görülmemiş parametrelerde <%5 hata
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")


# =============================================================================
# VERİ ÜRETİMİ
# =============================================================================

def plate_rigidity(E, nu, h):
    """Plaka rijitliği D = Eh³/12(1-ν²)"""
    return E * h**3 / (12 * (1 - nu**2))


def navier_coefficient(aspect):
    """Navier katsayısını hesapla."""
    alpha = 0.0
    for m in range(1, 20, 2):
        for n in range(1, 20, 2):
            denom = m * n * (m**2 + n**2 * aspect**2)**2
            alpha += 16 / (math.pi**6 * denom)
    return alpha


def generate_static_dataset(n_samples=1000):
    """
    Statik eğilme veri seti üret.

    Girdiler: E, ν, h, Lx, Ly, q
    Çıktı: w_max (maksimum sehim)
    """
    np.random.seed(42)

    # Malzeme parametreleri
    E_range = (50e9, 300e9)      # 50-300 GPa
    nu_range = (0.2, 0.4)        # Poisson oranı
    h_range = (0.002, 0.02)      # 2-20 mm kalınlık

    # Geometri parametreleri
    Lx_range = (0.3, 2.0)        # 0.3-2 m
    aspect_range = (0.5, 2.0)    # Lx/Ly oranı

    # Yük parametresi
    q_range = (1000, 50000)      # 1-50 kN/m²

    data = []

    for _ in range(n_samples):
        E = np.random.uniform(*E_range)
        nu = np.random.uniform(*nu_range)
        h = np.random.uniform(*h_range)
        Lx = np.random.uniform(*Lx_range)
        aspect = np.random.uniform(*aspect_range)
        Ly = Lx / aspect
        q = np.random.uniform(*q_range)

        # Analitik çözüm
        D = plate_rigidity(E, nu, h)
        alpha = navier_coefficient(aspect)
        w_max = alpha * q * Lx**4 / D

        data.append([E, nu, h, Lx, Ly, q, w_max])

    return np.array(data)


def split_by_material(data, test_ratio=0.2):
    """Malzeme bazlı ayrım - görülmemiş E değerleri test'te."""
    E_values = data[:, 0]
    E_threshold = np.percentile(E_values, (1 - test_ratio) * 100)

    train_mask = E_values < E_threshold
    test_mask = ~train_mask

    return data[train_mask], data[test_mask]


# =============================================================================
# MODELLER
# =============================================================================

class SpineHybridStatic(nn.Module):
    """
    Fizik-gömülü statik eğilme modeli.

    w_max = physics_formula × (1 + fine_tune)

    Fizik: w = α × q × L⁴ / D (Navier çözümü)
    Fine-tune: Küçük MLP düzeltmesi
    """

    def __init__(self, hidden_size=64):
        super().__init__()

        # Input scaling (büyük sayıları normalize et)
        self.register_buffer('E_scale', torch.tensor(1e11))
        self.register_buffer('h_scale', torch.tensor(0.01))
        self.register_buffer('L_scale', torch.tensor(1.0))
        self.register_buffer('q_scale', torch.tensor(1e4))

        # Fine-tune MLP (scaled inputs)
        self.fine_tune = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.output_scale = nn.Parameter(torch.tensor(0.05))

        # Pre-compute Navier coefficients
        aspects = torch.linspace(0.5, 2.0, 50)
        alphas = torch.tensor([navier_coefficient(a.item()) for a in aspects])
        self.register_buffer('aspect_points', aspects)
        self.register_buffer('alpha_values', alphas)

    def _interpolate_alpha(self, aspect):
        """Aspect ratio için alpha değerini interpolasyon ile bul."""
        aspect = torch.clamp(aspect, 0.5, 2.0)
        idx_float = (aspect - 0.5) / (2.0 - 0.5) * (len(self.aspect_points) - 1)
        idx_low = torch.floor(idx_float).long()
        idx_high = torch.clamp(torch.ceil(idx_float).long(), max=len(self.aspect_points) - 1)
        weight = idx_float - idx_low.float()

        alpha_low = self.alpha_values[idx_low]
        alpha_high = self.alpha_values[idx_high]

        return alpha_low + weight * (alpha_high - alpha_low)

    def forward(self, x):
        """x: [E, ν, h, Lx, Ly, q] - orijinal değerler"""
        E = x[:, 0]
        nu = x[:, 1]
        h = x[:, 2]
        Lx = x[:, 3]
        Ly = x[:, 4]
        q = x[:, 5]

        # Fizik: Plaka rijitliği
        D = E * h**3 / (12 * (1 - nu**2))

        # Fizik: Navier katsayısı
        aspect = Lx / Ly
        alpha = self._interpolate_alpha(aspect)

        # Fizik: Sehim
        w_physics = alpha * q * Lx**4 / D

        # MLP için scaled input
        x_scaled = torch.stack([
            E / self.E_scale,
            nu,
            h / self.h_scale,
            Lx / self.L_scale,
            Ly / self.L_scale,
            q / self.q_scale
        ], dim=1)

        # Fine-tune
        correction = self.fine_tune(x_scaled).squeeze(-1) * self.output_scale

        # Hibrit çıktı
        w_max = w_physics * (1 + correction)

        return w_max.unsqueeze(-1)


class PureMLP(nn.Module):
    """Saf MLP - fizik bilgisi yok."""

    def __init__(self, hidden_size=64):
        super().__init__()

        # Input scaling
        self.register_buffer('E_scale', torch.tensor(1e11))
        self.register_buffer('h_scale', torch.tensor(0.01))
        self.register_buffer('L_scale', torch.tensor(1.0))
        self.register_buffer('q_scale', torch.tensor(1e4))
        self.register_buffer('w_scale', torch.tensor(1e-3))

        self.net = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # Scale inputs
        x_scaled = torch.stack([
            x[:, 0] / self.E_scale,
            x[:, 1],
            x[:, 2] / self.h_scale,
            x[:, 3] / self.L_scale,
            x[:, 4] / self.L_scale,
            x[:, 5] / self.q_scale
        ], dim=1)

        # Scale output back
        return self.net(x_scaled) * self.w_scale


# =============================================================================
# EĞİTİM
# =============================================================================

def train_model(model, train_loader, test_loader, y_test, epochs=500, lr=1e-3):
    """Model eğitimi."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)

    best_test_error = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            # Relative loss for better scaling
            loss = torch.mean(((pred - y_batch) / (y_batch + 1e-10))**2)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Test
        model.eval()
        with torch.no_grad():
            test_preds = []
            for X_batch, _ in test_loader:
                pred = model(X_batch)
                test_preds.append(pred)
            test_preds = torch.cat(test_preds).cpu().numpy()

        # MAPE (Mean Absolute Percentage Error)
        test_error = np.mean(np.abs(test_preds.flatten() - y_test) / (np.abs(y_test) + 1e-10)) * 100

        scheduler.step(test_error)

        if test_error < best_test_error:
            best_test_error = test_error

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.6f}, "
                  f"Test Error: {test_error:.2f}%")

    return best_test_error


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# ANA TEST
# =============================================================================

def main():
    print("=" * 60)
    print("STATIC NEURON EĞİTİM TESTİ")
    print("=" * 60)

    # 1. Veri üret
    print("\n[1] Veri üretiliyor...")
    data = generate_static_dataset(n_samples=1000)
    print(f"    Toplam örnek: {len(data)}")

    # 2. Train/test ayrımı (malzeme bazlı)
    train_data, test_data = split_by_material(data, test_ratio=0.2)
    print(f"    Eğitim: {len(train_data)}, Test: {len(test_data)}")

    # 3. Girdiler ve çıktılar (normalizasyon YOK)
    X_train = train_data[:, :6]
    y_train = train_data[:, 6:7]
    X_test = test_data[:, :6]
    y_test = test_data[:, 6:7]

    # 4. Tensor'lara dönüştür
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=DEVICE)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=32)

    # 5. SpineHybridStatic eğit
    print("\n[2] SpineHybridStatic eğitiliyor...")
    spine_model = SpineHybridStatic(hidden_size=64).to(DEVICE)
    print(f"    Parametre sayısı: {count_parameters(spine_model)}")

    spine_error = train_model(spine_model, train_loader, test_loader,
                              y_test.flatten(), epochs=500, lr=1e-3)

    # 6. Pure MLP eğit
    print("\n[3] Pure MLP eğitiliyor...")
    mlp_model = PureMLP(hidden_size=64).to(DEVICE)
    print(f"    Parametre sayısı: {count_parameters(mlp_model)}")

    mlp_error = train_model(mlp_model, train_loader, test_loader,
                            y_test.flatten(), epochs=500, lr=1e-3)

    # 7. Sonuçlar
    print("\n" + "=" * 60)
    print("SONUÇLAR")
    print("=" * 60)
    print(f"\n{'Model':<25} {'Parametre':<12} {'Test Hatası':<12}")
    print("-" * 50)
    print(f"{'SpineHybridStatic':<25} {count_parameters(spine_model):<12} {spine_error:.2f}%")
    print(f"{'Pure MLP':<25} {count_parameters(mlp_model):<12} {mlp_error:.2f}%")
    print("-" * 50)

    if spine_error < mlp_error:
        improvement = (mlp_error - spine_error) / mlp_error * 100
        print(f"\n✅ SpineHybridStatic {improvement:.1f}% daha iyi!")
    else:
        print(f"\n⚠️  MLP daha iyi performans gösterdi")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
