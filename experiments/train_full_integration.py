"""
TAM ENTEGRASYON TESTİ
- Static + Thermal + Crack + Modal birlikte
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")


def geometry_factor(a, W):
    r = a / W
    return 1.12 - 0.231*r + 10.55*r**2 - 21.72*r**3 + 30.39*r**4


def generate_full_dataset(n_samples=1000):
    np.random.seed(42)
    data = []

    for _ in range(n_samples):
        E = np.random.uniform(70e9, 210e9)
        nu = np.random.uniform(0.25, 0.35)
        rho = np.random.uniform(2700, 8000)
        alpha = np.random.uniform(10e-6, 25e-6)
        L = np.random.uniform(0.5, 1.5)
        h = np.random.uniform(0.005, 0.02)
        W = np.random.uniform(0.1, 0.3)
        q = np.random.uniform(5000, 30000)
        delta_T = np.random.uniform(20, 100)
        a_over_W = np.random.uniform(0.05, 0.25)
        a = a_over_W * W
        K_IC = np.random.uniform(30e6, 80e6)

        D = E * h**3 / (12 * (1 - nu**2))
        sigma_thermal = E * alpha * delta_T / (1 - nu)
        sigma_static = 0.287 * q * (L / h)**2
        sigma_total = sigma_thermal + sigma_static
        w_max = 0.00406 * q * L**4 / D
        omega = math.pi**2 * math.sqrt(D / (rho * h)) * (2 / L**2)
        f1 = omega / (2 * math.pi)
        Y = geometry_factor(a, W)
        K_I = sigma_total * math.sqrt(math.pi * a) * Y
        safety = K_IC / K_I

        data.append([E, nu, rho, alpha, L, h, W, q, delta_T, a, K_IC,
                     sigma_total, w_max, f1, K_I, safety])

    return np.array(data)


def split_data(data, test_ratio=0.2):
    rho = data[:, 2]
    threshold = np.percentile(rho, (1 - test_ratio) * 100)
    train_mask = rho < threshold
    return data[train_mask], data[~train_mask]


class SpineFullIntegration(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(11, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh()
        )
        self.heads = nn.Linear(hidden_size, 4)  # sigma, w, f1, K_I corrections
        self.scale = nn.Parameter(torch.tensor(0.05))

    def geometry_factor(self, a_over_W):
        r = a_over_W
        return 1.12 - 0.231*r + 10.55*r**2 - 21.72*r**3 + 30.39*r**4

    def forward(self, x):
        E, nu, rho, alpha = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        L, h, W, q, delta_T = x[:, 4], x[:, 5], x[:, 6], x[:, 7], x[:, 8]
        a, K_IC = x[:, 9], x[:, 10]

        # Fizik
        D = E * h**3 / (12 * (1 - nu**2))
        sigma_thermal = E * alpha * delta_T / (1 - nu)
        sigma_static = 0.287 * q * (L / h)**2
        sigma_physics = sigma_thermal + sigma_static

        w_physics = 0.00406 * q * L**4 / D

        omega = math.pi**2 * torch.sqrt(D / (rho * h)) * (2 / L**2)
        f1_physics = omega / (2 * math.pi)

        a_over_W = a / W
        Y = self.geometry_factor(a_over_W)
        K_I_physics = sigma_physics * torch.sqrt(math.pi * a) * Y

        safety_physics = K_IC / (K_I_physics + 1e-10)

        # Scaled input
        x_scaled = torch.stack([
            E / 1e11, nu, rho / 5000, alpha / 1e-5,
            L, h / 0.01, W / 0.1, q / 1e4, delta_T / 100,
            a / 0.01, K_IC / 1e7
        ], dim=1)

        features = self.encoder(x_scaled)
        corrections = self.heads(features) * self.scale

        sigma = sigma_physics * (1 + corrections[:, 0])
        w = w_physics * (1 + corrections[:, 1])
        f1 = f1_physics * (1 + corrections[:, 2])
        K_I = K_I_physics * (1 + corrections[:, 3])
        safety = K_IC / (K_I + 1e-10)

        return torch.stack([sigma, w, f1, K_I, safety], dim=1)


class PureMLP(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 5)
        )
        self.register_buffer('out_scale', torch.tensor([1e8, 1e-3, 100.0, 1e7, 1.0]))

    def forward(self, x):
        x_scaled = torch.stack([
            x[:, 0] / 1e11, x[:, 1], x[:, 2] / 5000, x[:, 3] / 1e-5,
            x[:, 4], x[:, 5] / 0.01, x[:, 6] / 0.1, x[:, 7] / 1e4,
            x[:, 8] / 100, x[:, 9] / 0.01, x[:, 10] / 1e7
        ], dim=1)
        return self.net(x_scaled) * self.out_scale


def train_model(model, train_loader, test_loader, y_test, epochs=500, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
    best_error = float('inf')
    best_errors = {}

    output_names = ['σ_total', 'w_max', 'f1', 'K_I', 'safety']

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

        errors = {}
        for i, name in enumerate(output_names):
            err = np.mean(np.abs(preds[:, i] - y_test[:, i]) / (np.abs(y_test[:, i]) + 1e-10)) * 100
            errors[name] = err

        error = np.mean(list(errors.values()))
        scheduler.step(error)

        if error < best_error:
            best_error = error
            best_errors = errors.copy()

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}, Error: {error:.2f}%")

    return best_error, best_errors


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print("=" * 60)
    print("TAM ENTEGRASYON TESTİ")
    print("Static + Thermal + Modal + Crack")
    print("=" * 60)

    data = generate_full_dataset(1000)
    train_data, test_data = split_data(data)
    print(f"Eğitim: {len(train_data)}, Test: {len(test_data)}")

    X_train, y_train = train_data[:, :11], train_data[:, 11:]
    X_test, y_test = test_data[:, :11], test_data[:, 11:]

    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_train, dtype=torch.float32, device=DEVICE),
        torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    ), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(
        torch.tensor(X_test, dtype=torch.float32, device=DEVICE),
        torch.tensor(y_test, dtype=torch.float32, device=DEVICE)
    ), batch_size=32)

    print("\nSpineFullIntegration eğitiliyor...")
    spine = SpineFullIntegration(64).to(DEVICE)
    print(f"  Parametre: {count_parameters(spine)}")
    spine_err, spine_errors = train_model(spine, train_loader, test_loader, y_test)

    print("\nPure MLP eğitiliyor...")
    mlp = PureMLP(64).to(DEVICE)
    print(f"  Parametre: {count_parameters(mlp)}")
    mlp_err, mlp_errors = train_model(mlp, train_loader, test_loader, y_test)

    print("\n" + "=" * 60)
    print("SONUÇLAR: TAM ENTEGRASYON")
    print("=" * 60)
    print(f"\n{'Model':<25} {'Parametre':<12} {'Ortalama Hata':<15}")
    print("-" * 55)
    print(f"{'SpineFullIntegration':<25} {count_parameters(spine):<12} {spine_err:.2f}%")
    print(f"{'Pure MLP':<25} {count_parameters(mlp):<12} {mlp_err:.2f}%")

    print("\n📊 Çıktı bazlı hatalar:")
    print(f"{'Çıktı':<12} {'SPINE':<12} {'MLP':<12}")
    print("-" * 40)
    for name in spine_errors.keys():
        print(f"{name:<12} {spine_errors[name]:.2f}%{'':<5} {mlp_errors[name]:.2f}%")

    if spine_err < mlp_err:
        print(f"\n✅ SpineFullIntegration {(mlp_err - spine_err) / mlp_err * 100:.1f}% daha iyi!")
        print("   4 fizik nöronu entegre çalışıyor!")
    print("=" * 60)


if __name__ == "__main__":
    main()
