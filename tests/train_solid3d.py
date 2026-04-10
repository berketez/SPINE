"""
SolidNeuron3D Eğitim Testi
- 3D Elastisite: Von Mises gerilme
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")


def compute_stress_3d(E, nu, eps_xx, eps_yy, eps_zz, gamma_yz, gamma_xz, gamma_xy):
    C11 = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
    C12 = E * nu / ((1 + nu) * (1 - 2 * nu))
    C44 = E / (2 * (1 + nu))

    sigma_xx = C11 * eps_xx + C12 * eps_yy + C12 * eps_zz
    sigma_yy = C12 * eps_xx + C11 * eps_yy + C12 * eps_zz
    sigma_zz = C12 * eps_xx + C12 * eps_yy + C11 * eps_zz
    tau_yz = C44 * gamma_yz
    tau_xz = C44 * gamma_xz
    tau_xy = C44 * gamma_xy

    return sigma_xx, sigma_yy, sigma_zz, tau_yz, tau_xz, tau_xy


def von_mises(sigma_xx, sigma_yy, sigma_zz, tau_yz, tau_xz, tau_xy):
    term1 = (sigma_xx - sigma_yy)**2 + (sigma_yy - sigma_zz)**2 + (sigma_zz - sigma_xx)**2
    term2 = 6 * (tau_xy**2 + tau_yz**2 + tau_xz**2)
    return math.sqrt(0.5 * (term1 + term2))


def generate_solid3d_dataset(n_samples=1000):
    np.random.seed(42)
    data = []

    for _ in range(n_samples):
        E = np.random.uniform(50e9, 300e9)
        nu = np.random.uniform(0.2, 0.45)
        eps_xx = np.random.uniform(-0.005, 0.005)
        eps_yy = np.random.uniform(-0.005, 0.005)
        eps_zz = np.random.uniform(-0.005, 0.005)
        gamma_yz = np.random.uniform(-0.003, 0.003)
        gamma_xz = np.random.uniform(-0.003, 0.003)
        gamma_xy = np.random.uniform(-0.003, 0.003)

        sigma_xx, sigma_yy, sigma_zz, tau_yz, tau_xz, tau_xy = \
            compute_stress_3d(E, nu, eps_xx, eps_yy, eps_zz, gamma_yz, gamma_xz, gamma_xy)
        sigma_vm = von_mises(sigma_xx, sigma_yy, sigma_zz, tau_yz, tau_xz, tau_xy)

        data.append([E, nu, eps_xx, eps_yy, eps_zz, gamma_yz, gamma_xz, gamma_xy, sigma_vm])

    return np.array(data)


def split_data(data, test_ratio=0.2):
    nu = data[:, 1]
    threshold = np.percentile(nu, (1 - test_ratio) * 100)
    train_mask = nu < threshold
    return data[train_mask], data[~train_mask]


class SpineHybridSolid3D(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()

        self.fine_tune = nn.Sequential(
            nn.Linear(8, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        E, nu = x[:, 0], x[:, 1]
        eps_xx, eps_yy, eps_zz = x[:, 2], x[:, 3], x[:, 4]
        gamma_yz, gamma_xz, gamma_xy = x[:, 5], x[:, 6], x[:, 7]

        # Fizik: Lamé sabitleri
        C11 = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
        C12 = E * nu / ((1 + nu) * (1 - 2 * nu))
        C44 = E / (2 * (1 + nu))

        # Fizik: Gerilmeler
        sigma_xx = C11 * eps_xx + C12 * eps_yy + C12 * eps_zz
        sigma_yy = C12 * eps_xx + C11 * eps_yy + C12 * eps_zz
        sigma_zz = C12 * eps_xx + C12 * eps_yy + C11 * eps_zz
        tau_yz = C44 * gamma_yz
        tau_xz = C44 * gamma_xz
        tau_xy = C44 * gamma_xy

        # Fizik: Von Mises
        term1 = (sigma_xx - sigma_yy)**2 + (sigma_yy - sigma_zz)**2 + (sigma_zz - sigma_xx)**2
        term2 = 6 * (tau_xy**2 + tau_yz**2 + tau_xz**2)
        sigma_vm_physics = torch.sqrt(0.5 * (term1 + term2) + 1e-10)

        # Scaled input
        x_scaled = torch.stack([
            E / 1e11, nu,
            eps_xx / 0.005, eps_yy / 0.005, eps_zz / 0.005,
            gamma_yz / 0.003, gamma_xz / 0.003, gamma_xy / 0.003
        ], dim=1)

        correction = self.fine_tune(x_scaled).squeeze(-1) * self.scale
        return (sigma_vm_physics * (1 + correction)).unsqueeze(-1)


class PureMLP(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        x_scaled = torch.stack([
            x[:, 0] / 1e11, x[:, 1],
            x[:, 2] / 0.005, x[:, 3] / 0.005, x[:, 4] / 0.005,
            x[:, 5] / 0.003, x[:, 6] / 0.003, x[:, 7] / 0.003
        ], dim=1)
        return self.net(x_scaled) * 1e9


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
    print("3D SOLID NEURON EĞİTİM TESTİ")
    print("=" * 60)

    data = generate_solid3d_dataset(1000)
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

    print("\nSpineHybridSolid3D eğitiliyor...")
    spine = SpineHybridSolid3D(64).to(DEVICE)
    print(f"  Parametre: {count_parameters(spine)}")
    spine_err = train_model(spine, train_loader, test_loader, y_test.flatten())

    print("\nPure MLP eğitiliyor...")
    mlp = PureMLP(64).to(DEVICE)
    print(f"  Parametre: {count_parameters(mlp)}")
    mlp_err = train_model(mlp, train_loader, test_loader, y_test.flatten())

    print("\n" + "=" * 60)
    print(f"{'Model':<25} {'Parametre':<12} {'Hata':<10}")
    print("-" * 50)
    print(f"{'SpineHybridSolid3D':<25} {count_parameters(spine):<12} {spine_err:.2f}%")
    print(f"{'Pure MLP':<25} {count_parameters(mlp):<12} {mlp_err:.2f}%")
    if spine_err < mlp_err:
        print(f"\n✅ SpineHybrid {(mlp_err - spine_err) / mlp_err * 100:.1f}% daha iyi!")
    print("=" * 60)


if __name__ == "__main__":
    main()
