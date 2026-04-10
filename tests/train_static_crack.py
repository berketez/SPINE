"""
Statik + Çatlak Entegrasyon Testi
- Yüklü plakada çatlak analizi
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


def generate_static_crack_dataset(n_samples=1000):
    np.random.seed(42)
    data = []

    for _ in range(n_samples):
        E = np.random.uniform(70e9, 210e9)
        h = np.random.uniform(0.005, 0.02)
        L = np.random.uniform(0.3, 1.0)
        W = np.random.uniform(0.1, 0.3)
        q = np.random.uniform(5000, 30000)
        a_over_W = np.random.uniform(0.05, 0.3)
        a = a_over_W * W
        K_IC = np.random.uniform(30e6, 80e6)

        # Statik gerilme
        sigma = 0.287 * q * (L / h)**2

        # K_I
        Y = geometry_factor(a, W)
        K_I = sigma * math.sqrt(math.pi * a) * Y

        # Güvenlik
        safety = K_IC / K_I

        data.append([E, h, L, W, q, a, K_IC, sigma, K_I, safety])

    return np.array(data)


def split_data(data, test_ratio=0.2):
    q = data[:, 4]
    threshold = np.percentile(q, (1 - test_ratio) * 100)
    train_mask = q < threshold
    return data[train_mask], data[~train_mask]


class SpineHybridStaticCrack(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()

        self.fine_tune = nn.Sequential(
            nn.Linear(7, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 3)  # sigma, K_I, safety corrections
        )
        self.scale = nn.Parameter(torch.tensor(0.05))

    def geometry_factor(self, a_over_W):
        r = a_over_W
        return 1.12 - 0.231*r + 10.55*r**2 - 21.72*r**3 + 30.39*r**4

    def forward(self, x):
        E, h, L, W, q, a, K_IC = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5], x[:, 6]

        # Fizik: Statik gerilme
        sigma_physics = 0.287 * q * (L / h)**2

        # Fizik: K_I
        a_over_W = a / W
        Y = self.geometry_factor(a_over_W)
        K_I_physics = sigma_physics * torch.sqrt(math.pi * a) * Y

        # Fizik: Safety
        safety_physics = K_IC / (K_I_physics + 1e-10)

        # Scaled input
        x_scaled = torch.stack([
            E / 1e11, h / 0.01, L, W / 0.1, q / 1e4, a / 0.01, K_IC / 1e7
        ], dim=1)

        corrections = self.fine_tune(x_scaled) * self.scale

        sigma = sigma_physics * (1 + corrections[:, 0])
        K_I = K_I_physics * (1 + corrections[:, 1])
        safety = safety_physics * (1 + corrections[:, 2])

        return torch.stack([sigma, K_I, safety], dim=1)


class PureMLP(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 3)
        )
        self.register_buffer('out_scale', torch.tensor([1e8, 1e7, 1.0]))

    def forward(self, x):
        x_scaled = torch.stack([
            x[:, 0] / 1e11, x[:, 1] / 0.01, x[:, 2], x[:, 3] / 0.1,
            x[:, 4] / 1e4, x[:, 5] / 0.01, x[:, 6] / 1e7
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
    print("STATİK + ÇATLAK ENTEGRASYON TESTİ")
    print("=" * 60)

    data = generate_static_crack_dataset(1000)
    train_data, test_data = split_data(data)
    print(f"Eğitim: {len(train_data)}, Test: {len(test_data)}")

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

    print("\nSpineHybridStaticCrack eğitiliyor...")
    spine = SpineHybridStaticCrack(64).to(DEVICE)
    print(f"  Parametre: {count_parameters(spine)}")
    spine_err = train_model(spine, train_loader, test_loader, y_test)

    print("\nPure MLP eğitiliyor...")
    mlp = PureMLP(64).to(DEVICE)
    print(f"  Parametre: {count_parameters(mlp)}")
    mlp_err = train_model(mlp, train_loader, test_loader, y_test)

    print("\n" + "=" * 60)
    print(f"{'Model':<30} {'Parametre':<12} {'Hata':<10}")
    print("-" * 55)
    print(f"{'SpineHybridStaticCrack':<30} {count_parameters(spine):<12} {spine_err:.2f}%")
    print(f"{'Pure MLP':<30} {count_parameters(mlp):<12} {mlp_err:.2f}%")
    if spine_err < mlp_err:
        print(f"\n✅ SpineHybrid {(mlp_err - spine_err) / mlp_err * 100:.1f}% daha iyi!")
    print("=" * 60)


if __name__ == "__main__":
    main()
