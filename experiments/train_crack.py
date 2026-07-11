"""
CrackNeuron Eğitim Testi
- Kırılma mekaniği: K_I = σ√(πa) × Y(a/W)
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


def generate_crack_dataset(n_samples=1000):
    np.random.seed(42)
    data = []

    for _ in range(n_samples):
        sigma = np.random.uniform(50e6, 300e6)
        W = np.random.uniform(0.05, 0.2)
        a_over_W = np.random.uniform(0.05, 0.4)
        a = a_over_W * W
        K_IC = np.random.uniform(20e6, 100e6)

        Y = geometry_factor(a, W)
        K_I = sigma * math.sqrt(math.pi * a) * Y

        data.append([sigma, a, W, K_IC, K_I])

    return np.array(data)


def split_data(data, test_ratio=0.2):
    K_IC = data[:, 3]
    threshold = np.percentile(K_IC, (1 - test_ratio) * 100)
    train_mask = K_IC < threshold
    return data[train_mask], data[~train_mask]


class SpineHybridCrack(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.register_buffer('sigma_scale', torch.tensor(1e8))
        self.register_buffer('a_scale', torch.tensor(0.01))
        self.register_buffer('W_scale', torch.tensor(0.1))
        self.register_buffer('K_scale', torch.tensor(1e7))

        self.fine_tune = nn.Sequential(
            nn.Linear(4, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.scale = nn.Parameter(torch.tensor(0.05))

    def geometry_factor(self, a_over_W):
        r = a_over_W
        return 1.12 - 0.231*r + 10.55*r**2 - 21.72*r**3 + 30.39*r**4

    def forward(self, x):
        sigma, a, W, K_IC = x[:, 0], x[:, 1], x[:, 2], x[:, 3]

        # Fizik
        a_over_W = a / W
        Y = self.geometry_factor(a_over_W)
        K_I_physics = sigma * torch.sqrt(math.pi * a) * Y

        # Scaled input for MLP
        x_scaled = torch.stack([
            sigma / self.sigma_scale, a / self.a_scale,
            W / self.W_scale, K_IC / self.K_scale
        ], dim=1)

        correction = self.fine_tune(x_scaled).squeeze(-1) * self.scale
        return (K_I_physics * (1 + correction)).unsqueeze(-1)


class PureMLP(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.register_buffer('K_scale', torch.tensor(1e7))

        self.net = nn.Sequential(
            nn.Linear(4, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        x_scaled = torch.stack([
            x[:, 0] / 1e8, x[:, 1] / 0.01, x[:, 2] / 0.1, x[:, 3] / 1e7
        ], dim=1)
        return self.net(x_scaled) * self.K_scale


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
    print("CRACK NEURON EĞİTİM TESTİ")
    print("=" * 60)

    data = generate_crack_dataset(1000)
    train_data, test_data = split_data(data)
    print(f"Eğitim: {len(train_data)}, Test: {len(test_data)}")

    X_train, y_train = train_data[:, :4], train_data[:, 4:5]
    X_test, y_test = test_data[:, :4], test_data[:, 4:5]

    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_train, dtype=torch.float32, device=DEVICE),
        torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    ), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(
        torch.tensor(X_test, dtype=torch.float32, device=DEVICE),
        torch.tensor(y_test, dtype=torch.float32, device=DEVICE)
    ), batch_size=32)

    print("\nSpineHybridCrack eğitiliyor...")
    spine = SpineHybridCrack(64).to(DEVICE)
    print(f"  Parametre: {count_parameters(spine)}")
    spine_err = train_model(spine, train_loader, test_loader, y_test.flatten())

    print("\nPure MLP eğitiliyor...")
    mlp = PureMLP(64).to(DEVICE)
    print(f"  Parametre: {count_parameters(mlp)}")
    mlp_err = train_model(mlp, train_loader, test_loader, y_test.flatten())

    print("\n" + "=" * 60)
    print(f"{'Model':<25} {'Parametre':<12} {'Hata':<10}")
    print("-" * 50)
    print(f"{'SpineHybridCrack':<25} {count_parameters(spine):<12} {spine_err:.2f}%")
    print(f"{'Pure MLP':<25} {count_parameters(mlp):<12} {mlp_err:.2f}%")
    if spine_err < mlp_err:
        print(f"\n✅ SpineHybrid {(mlp_err - spine_err) / mlp_err * 100:.1f}% daha iyi!")
    print("=" * 60)


if __name__ == "__main__":
    main()
