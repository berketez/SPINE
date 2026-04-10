"""
ThermalNeuron Egitim Testi
- Termal gerilme: sigma_thermal = E * alpha * delta_T / (1 - nu)
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")


def generate_thermal_dataset(n_samples=1000):
    np.random.seed(42)
    data = []

    for _ in range(n_samples):
        E = np.random.uniform(70e9, 210e9)
        nu = np.random.uniform(0.25, 0.35)
        alpha = np.random.uniform(10e-6, 25e-6)
        delta_T = np.random.uniform(20, 150)

        # Termal gerilme
        sigma_thermal = E * alpha * delta_T / (1 - nu)

        data.append([E, nu, alpha, delta_T, sigma_thermal])

    return np.array(data)


def split_data(data, test_ratio=0.2):
    delta_T = data[:, 3]
    threshold = np.percentile(delta_T, (1 - test_ratio) * 100)
    train_mask = delta_T < threshold
    return data[train_mask], data[~train_mask]


class SpineHybridThermal(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()

        self.fine_tune = nn.Sequential(
            nn.Linear(4, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        E, nu, alpha, delta_T = x[:, 0], x[:, 1], x[:, 2], x[:, 3]

        # Fizik: Termal gerilme
        sigma_physics = E * alpha * delta_T / (1 - nu)

        # Scaled input
        x_scaled = torch.stack([
            E / 1e11, nu, alpha / 1e-5, delta_T / 100
        ], dim=1)

        correction = self.fine_tune(x_scaled).squeeze(-1) * self.scale
        return (sigma_physics * (1 + correction)).unsqueeze(-1)


class PureMLP(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        x_scaled = torch.stack([
            x[:, 0] / 1e11, x[:, 1], x[:, 2] / 1e-5, x[:, 3] / 100
        ], dim=1)
        return self.net(x_scaled) * 1e8


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
    print("THERMAL NEURON EGITIM TESTI")
    print("=" * 60)

    data = generate_thermal_dataset(1000)
    train_data, test_data = split_data(data)
    print(f"Egitim: {len(train_data)}, Test: {len(test_data)}")

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

    print("\nSpineHybridThermal egitiliyor...")
    spine = SpineHybridThermal(64).to(DEVICE)
    print(f"  Parametre: {count_parameters(spine)}")
    spine_err = train_model(spine, train_loader, test_loader, y_test.flatten())

    print("\nPure MLP egitiliyor...")
    mlp = PureMLP(64).to(DEVICE)
    print(f"  Parametre: {count_parameters(mlp)}")
    mlp_err = train_model(mlp, train_loader, test_loader, y_test.flatten())

    print("\n" + "=" * 60)
    print(f"{'Model':<25} {'Parametre':<12} {'Hata':<10}")
    print("-" * 50)
    print(f"{'SpineHybridThermal':<25} {count_parameters(spine):<12} {spine_err:.2f}%")
    print(f"{'Pure MLP':<25} {count_parameters(mlp):<12} {mlp_err:.2f}%")
    if spine_err < mlp_err:
        print(f"\n SPINE {(mlp_err - spine_err) / mlp_err * 100:.1f}% daha iyi!")
    print("=" * 60)


if __name__ == "__main__":
    main()
