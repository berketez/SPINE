"""
ModalNeuron Eğitim Testi
- Titreşim analizi: doğal frekanslar
- SpineHybridModal vs Pure MLP karşılaştırması
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

def natural_frequency(E, nu, h, rho, Lx, Ly, m=1, n=1):
    """Simply supported dikdörtgen plaka doğal frekansı."""
    D = E * h**3 / (12 * (1 - nu**2))
    omega = math.pi**2 * math.sqrt(D / (rho * h)) * ((m / Lx)**2 + (n / Ly)**2)
    return omega / (2 * math.pi)


def generate_modal_dataset(n_samples=1000):
    np.random.seed(42)

    E_range = (50e9, 300e9)
    nu_range = (0.2, 0.4)
    h_range = (0.002, 0.02)
    rho_range = (2500, 9000)
    Lx_range = (0.3, 2.0)
    Ly_range = (0.3, 2.0)

    data = []
    for _ in range(n_samples):
        E = np.random.uniform(*E_range)
        nu = np.random.uniform(*nu_range)
        h = np.random.uniform(*h_range)
        rho = np.random.uniform(*rho_range)
        Lx = np.random.uniform(*Lx_range)
        Ly = np.random.uniform(*Ly_range)
        f1 = natural_frequency(E, nu, h, rho, Lx, Ly, m=1, n=1)
        data.append([E, nu, h, rho, Lx, Ly, f1])

    return np.array(data)


def split_by_material(data, test_ratio=0.2):
    rho_values = data[:, 3]
    rho_threshold = np.percentile(rho_values, (1 - test_ratio) * 100)
    train_mask = rho_values < rho_threshold
    return data[train_mask], data[~train_mask]


# =============================================================================
# MODELLER
# =============================================================================

class SpineHybridModal(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()

        self.register_buffer('E_scale', torch.tensor(1e11))
        self.register_buffer('h_scale', torch.tensor(0.01))
        self.register_buffer('rho_scale', torch.tensor(5000.0))
        self.register_buffer('L_scale', torch.tensor(1.0))

        self.fine_tune = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.output_scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        E = x[:, 0]
        nu = x[:, 1]
        h = x[:, 2]
        rho = x[:, 3]
        Lx = x[:, 4]
        Ly = x[:, 5]

        # Fizik
        D = E * h**3 / (12 * (1 - nu**2))
        omega = math.pi**2 * torch.sqrt(D / (rho * h)) * ((1 / Lx)**2 + (1 / Ly)**2)
        f_physics = omega / (2 * math.pi)

        # Scaled input for MLP
        x_scaled = torch.stack([
            E / self.E_scale, nu, h / self.h_scale,
            rho / self.rho_scale, Lx / self.L_scale, Ly / self.L_scale
        ], dim=1)

        correction = self.fine_tune(x_scaled).squeeze(-1) * self.output_scale
        return (f_physics * (1 + correction)).unsqueeze(-1)


class PureMLP(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.register_buffer('E_scale', torch.tensor(1e11))
        self.register_buffer('h_scale', torch.tensor(0.01))
        self.register_buffer('rho_scale', torch.tensor(5000.0))
        self.register_buffer('L_scale', torch.tensor(1.0))
        self.register_buffer('f_scale', torch.tensor(100.0))

        self.net = nn.Sequential(
            nn.Linear(6, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        x_scaled = torch.stack([
            x[:, 0] / self.E_scale, x[:, 1], x[:, 2] / self.h_scale,
            x[:, 3] / self.rho_scale, x[:, 4] / self.L_scale, x[:, 5] / self.L_scale
        ], dim=1)
        return self.net(x_scaled) * self.f_scale


# =============================================================================
# EĞİTİM
# =============================================================================

def train_model(model, train_loader, test_loader, y_test, epochs=500, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
    best_test_error = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = torch.mean(((pred - y_batch) / (y_batch + 1e-10))**2)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            test_preds = torch.cat([model(X) for X, _ in test_loader]).cpu().numpy()

        test_error = np.mean(np.abs(test_preds.flatten() - y_test) / (np.abs(y_test) + 1e-10)) * 100
        scheduler.step(test_error)
        if test_error < best_test_error:
            best_test_error = test_error

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.6f}, Error: {test_error:.2f}%")

    return best_test_error


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print("=" * 60)
    print("MODAL NEURON EĞİTİM TESTİ")
    print("=" * 60)

    print("\n[1] Veri üretiliyor...")
    data = generate_modal_dataset(n_samples=1000)
    train_data, test_data = split_by_material(data, test_ratio=0.2)
    print(f"    Eğitim: {len(train_data)}, Test: {len(test_data)}")

    X_train, y_train = train_data[:, :6], train_data[:, 6:7]
    X_test, y_test = test_data[:, :6], test_data[:, 6:7]

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=DEVICE)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=32)

    print("\n[2] SpineHybridModal eğitiliyor...")
    spine_model = SpineHybridModal(hidden_size=64).to(DEVICE)
    print(f"    Parametre: {count_parameters(spine_model)}")
    spine_error = train_model(spine_model, train_loader, test_loader, y_test.flatten(), epochs=500)

    print("\n[3] Pure MLP eğitiliyor...")
    mlp_model = PureMLP(hidden_size=64).to(DEVICE)
    print(f"    Parametre: {count_parameters(mlp_model)}")
    mlp_error = train_model(mlp_model, train_loader, test_loader, y_test.flatten(), epochs=500)

    print("\n" + "=" * 60)
    print("SONUÇLAR")
    print("=" * 60)
    print(f"{'Model':<25} {'Parametre':<12} {'Test Hatası':<12}")
    print("-" * 50)
    print(f"{'SpineHybridModal':<25} {count_parameters(spine_model):<12} {spine_error:.2f}%")
    print(f"{'Pure MLP':<25} {count_parameters(mlp_model):<12} {mlp_error:.2f}%")

    if spine_error < mlp_error:
        print(f"\n✅ SpineHybridModal {(mlp_error - spine_error) / mlp_error * 100:.1f}% daha iyi!")
    print("=" * 60)


if __name__ == "__main__":
    main()
