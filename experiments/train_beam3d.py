"""
BeamNeuron3D Egitim Testi
- 3D Kiris: Euler burkulma, egilme gerilmesi
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")


def generate_beam3d_dataset(n_samples=1000):
    np.random.seed(42)
    data = []

    for _ in range(n_samples):
        E = np.random.uniform(70e9, 210e9)
        L = np.random.uniform(1.0, 5.0)
        b = np.random.uniform(0.05, 0.2)  # Genislik
        h = np.random.uniform(0.1, 0.4)   # Yukseklik
        P = np.random.uniform(10e3, 100e3)  # Noktasal yuk (ucta)

        # Kesit ozellikleri (dikdortgen kesit)
        I = b * h**3 / 12  # Atalet momenti
        A = b * h          # Alan
        c = h / 2          # Notr eksenden uzaklik

        # Euler burkulma yuku (ankastre-serbest: K=2)
        K = 2.0
        P_cr = math.pi**2 * E * I / (K * L)**2

        # Max egilme gerilmesi (konsol kiris, ucta yuk)
        M_max = P * L
        sigma_max = M_max * c / I

        # Max sehim (konsol kiris)
        w_max = P * L**3 / (3 * E * I)

        data.append([E, L, b, h, P, P_cr, sigma_max, w_max])

    return np.array(data)


def split_data(data, test_ratio=0.2):
    L = data[:, 1]
    threshold = np.percentile(L, (1 - test_ratio) * 100)
    train_mask = L < threshold
    return data[train_mask], data[~train_mask]


class SpineHybridBeam3D(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()

        self.fine_tune = nn.Sequential(
            nn.Linear(5, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 3)  # P_cr, sigma, w corrections
        )
        self.scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        E, L, b, h, P = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]

        # Fizik: Kesit ozellikleri
        I = b * h**3 / 12
        c = h / 2

        # Fizik: Euler burkulma
        K = 2.0
        P_cr_physics = math.pi**2 * E * I / (K * L)**2

        # Fizik: Max gerilme
        M_max = P * L
        sigma_physics = M_max * c / I

        # Fizik: Max sehim
        w_physics = P * L**3 / (3 * E * I)

        # Scaled input
        x_scaled = torch.stack([
            E / 1e11, L, b / 0.1, h / 0.2, P / 1e5
        ], dim=1)

        corrections = self.fine_tune(x_scaled) * self.scale

        P_cr = P_cr_physics * (1 + corrections[:, 0])
        sigma = sigma_physics * (1 + corrections[:, 1])
        w = w_physics * (1 + corrections[:, 2])

        return torch.stack([P_cr, sigma, w], dim=1)


class PureMLP(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 3)
        )
        self.register_buffer('out_scale', torch.tensor([1e6, 1e8, 0.01]))

    def forward(self, x):
        x_scaled = torch.stack([
            x[:, 0] / 1e11, x[:, 1], x[:, 2] / 0.1, x[:, 3] / 0.2, x[:, 4] / 1e5
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
    print("3D BEAM NEURON EGITIM TESTI")
    print("=" * 60)

    data = generate_beam3d_dataset(1000)
    train_data, test_data = split_data(data)
    print(f"Egitim: {len(train_data)}, Test: {len(test_data)}")

    X_train, y_train = train_data[:, :5], train_data[:, 5:]
    X_test, y_test = test_data[:, :5], test_data[:, 5:]

    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_train, dtype=torch.float32, device=DEVICE),
        torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    ), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(
        torch.tensor(X_test, dtype=torch.float32, device=DEVICE),
        torch.tensor(y_test, dtype=torch.float32, device=DEVICE)
    ), batch_size=32)

    print("\nSpineHybridBeam3D egitiliyor...")
    spine = SpineHybridBeam3D(64).to(DEVICE)
    print(f"  Parametre: {count_parameters(spine)}")
    spine_err = train_model(spine, train_loader, test_loader, y_test)

    print("\nPure MLP egitiliyor...")
    mlp = PureMLP(64).to(DEVICE)
    print(f"  Parametre: {count_parameters(mlp)}")
    mlp_err = train_model(mlp, train_loader, test_loader, y_test)

    print("\n" + "=" * 60)
    print(f"{'Model':<25} {'Parametre':<12} {'Hata':<10}")
    print("-" * 50)
    print(f"{'SpineHybridBeam3D':<25} {count_parameters(spine):<12} {spine_err:.2f}%")
    print(f"{'Pure MLP':<25} {count_parameters(mlp):<12} {mlp_err:.2f}%")
    if spine_err < mlp_err:
        print(f"\n SPINE {(mlp_err - spine_err) / mlp_err * 100:.1f}% daha iyi!")
    print("=" * 60)


if __name__ == "__main__":
    main()
