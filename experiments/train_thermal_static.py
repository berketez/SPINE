"""
Termal + Statik Entegrasyon Testi
- Gerçek dünya senaryosu: Isınan plaka + mekanik yük
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")


def generate_thermal_static_dataset(n_samples=1000):
    np.random.seed(42)
    data = []

    for _ in range(n_samples):
        E = np.random.uniform(50e9, 300e9)
        nu = np.random.uniform(0.2, 0.4)
        h = np.random.uniform(0.005, 0.02)
        Lx = np.random.uniform(0.5, 1.5)
        Ly = np.random.uniform(0.5, 1.5)
        q = np.random.uniform(5000, 30000)
        alpha = np.random.uniform(10e-6, 25e-6)
        delta_T = np.random.uniform(20, 150)

        sigma_th = E * alpha * delta_T / (1 - nu)
        L = min(Lx, Ly)
        sigma_st = 0.287 * q * (L / h)**2
        sigma_total = sigma_th + sigma_st

        data.append([E, nu, h, Lx, Ly, q, alpha, delta_T, sigma_total])

    return np.array(data)


def split_data(data, test_ratio=0.2):
    delta_T = data[:, 7]
    threshold = np.percentile(delta_T, (1 - test_ratio) * 100)
    train_mask = delta_T < threshold
    return data[train_mask], data[~train_mask]


class SpineHybridThermalStatic(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.register_buffer('E_scale', torch.tensor(1e11))
        self.register_buffer('h_scale', torch.tensor(0.01))
        self.register_buffer('L_scale', torch.tensor(1.0))
        self.register_buffer('q_scale', torch.tensor(1e4))
        self.register_buffer('alpha_scale', torch.tensor(1e-5))
        self.register_buffer('T_scale', torch.tensor(100.0))

        self.fine_tune = nn.Sequential(
            nn.Linear(8, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        E, nu, h = x[:, 0], x[:, 1], x[:, 2]
        Lx, Ly, q = x[:, 3], x[:, 4], x[:, 5]
        alpha, delta_T = x[:, 6], x[:, 7]

        sigma_thermal = E * alpha * delta_T / (1 - nu)
        L = torch.minimum(Lx, Ly)
        sigma_static = 0.287 * q * (L / h)**2
        sigma_physics = sigma_thermal + sigma_static

        x_scaled = torch.stack([
            E / self.E_scale, nu, h / self.h_scale, Lx / self.L_scale,
            Ly / self.L_scale, q / self.q_scale, alpha / self.alpha_scale, delta_T / self.T_scale
        ], dim=1)

        correction = self.fine_tune(x_scaled).squeeze(-1) * self.scale
        return (sigma_physics * (1 + correction)).unsqueeze(-1)


class PureMLP(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.register_buffer('E_scale', torch.tensor(1e11))
        self.register_buffer('sigma_scale', torch.tensor(1e8))

        self.net = nn.Sequential(
            nn.Linear(8, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        x_scaled = x.clone()
        x_scaled[:, 0] = x[:, 0] / 1e11
        x_scaled[:, 2] = x[:, 2] / 0.01
        x_scaled[:, 5] = x[:, 5] / 1e4
        x_scaled[:, 6] = x[:, 6] / 1e-5
        x_scaled[:, 7] = x[:, 7] / 100
        return self.net(x_scaled) * self.sigma_scale


def train_model(model, train_loader, test_loader, y_test, epochs=500, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
    best_error = float('inf')

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = torch.mean(((model(X_batch) - y_batch) / (y_batch + 1e-10))**2)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = torch.cat([model(X) for X, _ in test_loader]).cpu().numpy()
        error = np.mean(np.abs(preds.flatten() - y_test) / (np.abs(y_test) + 1e-10)) * 100
        scheduler.step(error)
        if error < best_error:
            best_error = error
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}, Error: {error:.2f}%")

    return best_error


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print("=" * 60)
    print("TERMAL + STATİK ENTEGRASYON TESTİ")
    print("=" * 60)

    data = generate_thermal_static_dataset(1000)
    train_data, test_data = split_data(data)
    print(f"Eğitim: {len(train_data)}, Test: {len(test_data)}")

    X_train, y_train = train_data[:, :8], train_data[:, 8:9]
    X_test, y_test = test_data[:, :8], test_data[:, 8:9]

    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_train, dtype=torch.float32, device=DEVICE),
        torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    ), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(
        torch.tensor(X_test, dtype=torch.float32, device=DEVICE),
        torch.tensor(y_test, dtype=torch.float32, device=DEVICE)
    ), batch_size=32)

    print("\nSpineHybridThermalStatic eğitiliyor...")
    spine = SpineHybridThermalStatic(64).to(DEVICE)
    print(f"  Parametre: {count_parameters(spine)}")
    spine_err = train_model(spine, train_loader, test_loader, y_test.flatten())

    print("\nPure MLP eğitiliyor...")
    mlp = PureMLP(64).to(DEVICE)
    print(f"  Parametre: {count_parameters(mlp)}")
    mlp_err = train_model(mlp, train_loader, test_loader, y_test.flatten())

    print("\n" + "=" * 60)
    print(f"{'Model':<30} {'Parametre':<12} {'Hata':<10}")
    print("-" * 55)
    print(f"{'SpineHybridThermalStatic':<30} {count_parameters(spine):<12} {spine_err:.2f}%")
    print(f"{'Pure MLP':<30} {count_parameters(mlp):<12} {mlp_err:.2f}%")
    if spine_err < mlp_err:
        print(f"\n✅ SpineHybrid {(mlp_err - spine_err) / mlp_err * 100:.1f}% daha iyi!")
    print("=" * 60)


if __name__ == "__main__":
    main()
