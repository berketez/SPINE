"""
Thermal + Buckling Entegrasyon Testi
- Termal gerilme ile burkulma analizi
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")


def generate_thermal_buckling_dataset(n_samples=1000):
    np.random.seed(42)
    data = []

    for _ in range(n_samples):
        E = np.random.uniform(70e9, 210e9)
        nu = np.random.uniform(0.25, 0.35)
        alpha = np.random.uniform(10e-6, 25e-6)
        h = np.random.uniform(0.005, 0.02)
        L = np.random.uniform(0.5, 1.5)
        W = np.random.uniform(0.3, 0.8)
        delta_T = np.random.uniform(50, 200)

        # Plaka rijitligi
        D = E * h**3 / (12 * (1 - nu**2))

        # Termal gerilme (serbest genlesme engellendiginde)
        sigma_thermal = E * alpha * delta_T / (1 - nu)

        # Termal membran kuvveti
        N_thermal = sigma_thermal * h

        # Kritik burkulma yuku (basit mesnetli plaka, m=n=1)
        aspect = L / W
        m, n = 1, 1
        N_cr = D * math.pi**2 / L**2 * (m + n**2 * aspect**2 / m)**2

        # Burkulma guvenlik katsayisi
        safety = N_cr / N_thermal

        # Kritik sicaklik farki (burkulma icin)
        delta_T_cr = N_cr * (1 - nu) / (E * alpha * h)

        data.append([E, nu, alpha, h, L, W, delta_T,
                     sigma_thermal, N_thermal, N_cr, safety, delta_T_cr])

    return np.array(data)


def split_data(data, test_ratio=0.2):
    delta_T = data[:, 6]
    threshold = np.percentile(delta_T, (1 - test_ratio) * 100)
    train_mask = delta_T < threshold
    return data[train_mask], data[~train_mask]


class SpineHybridThermalBuckling(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(7, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh()
        )
        self.heads = nn.Linear(hidden_size, 5)  # sigma, N_th, N_cr, safety, dT_cr
        self.scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        E, nu, alpha, h, L, W, delta_T = [x[:, i] for i in range(7)]

        # Fizik: Rijitlik
        D = E * h**3 / (12 * (1 - nu**2))

        # Fizik: Termal gerilme
        sigma_physics = E * alpha * delta_T / (1 - nu)
        N_thermal_physics = sigma_physics * h

        # Fizik: Kritik burkulma yuku
        aspect = L / W
        m, n = 1, 1
        N_cr_physics = D * math.pi**2 / L**2 * (m + n**2 * aspect**2 / m)**2

        # Fizik: Guvenlik
        safety_physics = N_cr_physics / (N_thermal_physics + 1e-10)

        # Fizik: Kritik sicaklik
        delta_T_cr_physics = N_cr_physics * (1 - nu) / (E * alpha * h)

        # Scaled input
        x_scaled = torch.stack([
            E / 1e11, nu, alpha / 1e-5, h / 0.01, L, W, delta_T / 100
        ], dim=1)

        features = self.encoder(x_scaled)
        corrections = self.heads(features) * self.scale

        sigma = sigma_physics * (1 + corrections[:, 0])
        N_thermal = N_thermal_physics * (1 + corrections[:, 1])
        N_cr = N_cr_physics * (1 + corrections[:, 2])
        safety = safety_physics * (1 + corrections[:, 3])
        delta_T_cr = delta_T_cr_physics * (1 + corrections[:, 4])

        return torch.stack([sigma, N_thermal, N_cr, safety, delta_T_cr], dim=1)


class PureMLP(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 5)
        )
        self.register_buffer('out_scale', torch.tensor([1e8, 1e6, 1e5, 1.0, 100.0]))

    def forward(self, x):
        x_scaled = torch.stack([
            x[:, 0] / 1e11, x[:, 1], x[:, 2] / 1e-5, x[:, 3] / 0.01,
            x[:, 4], x[:, 5], x[:, 6] / 100
        ], dim=1)
        return self.net(x_scaled) * self.out_scale


def train_model(model, train_loader, test_loader, y_test, epochs=500, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
    best_error = float('inf')

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = torch.mean(((pred - y_batch) / (y_batch + 1e-10))**2)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = torch.cat([model(X) for X, _ in test_loader]).cpu().numpy()

        errors = []
        for i in range(y_test.shape[1]):
            err = np.mean(np.abs(preds[:, i] - y_test[:, i]) / (np.abs(y_test[:, i]) + 1e-10)) * 100
            errors.append(err)
        error = np.mean(errors)

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
    print("THERMAL + BUCKLING ENTEGRASYON TESTI")
    print("=" * 60)

    data = generate_thermal_buckling_dataset(1000)
    train_data, test_data = split_data(data)
    print(f"Egitim: {len(train_data)}, Test: {len(test_data)}")

    X_train, y_train = train_data[:, :7], train_data[:, 7:]
    X_test, y_test = test_data[:, :7], test_data[:, 7:]

    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_train, dtype=torch.float32, device=DEVICE),
        torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    ), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(
        torch.tensor(X_test, dtype=torch.float32, device=DEVICE),
        torch.tensor(y_test, dtype=torch.float32, device=DEVICE)
    ), batch_size=32)

    print("\nSpineHybridThermalBuckling egitiliyor...")
    spine = SpineHybridThermalBuckling(64).to(DEVICE)
    print(f"  Parametre: {count_parameters(spine)}")
    spine_err = train_model(spine, train_loader, test_loader, y_test)

    print("\nPure MLP egitiliyor...")
    mlp = PureMLP(64).to(DEVICE)
    print(f"  Parametre: {count_parameters(mlp)}")
    mlp_err = train_model(mlp, train_loader, test_loader, y_test)

    print("\n" + "=" * 60)
    print(f"{'Model':<35} {'Parametre':<12} {'Hata':<10}")
    print("-" * 60)
    print(f"{'SpineHybridThermalBuckling':<35} {count_parameters(spine):<12} {spine_err:.2f}%")
    print(f"{'Pure MLP':<35} {count_parameters(mlp):<12} {mlp_err:.2f}%")
    if spine_err < mlp_err:
        print(f"\n SPINE {(mlp_err - spine_err) / mlp_err * 100:.1f}% daha iyi!")
    print("=" * 60)


if __name__ == "__main__":
    main()
