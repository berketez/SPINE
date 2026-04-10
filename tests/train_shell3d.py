"""
ShellNeuron3D Egitim Testi
- 3D Kabuk: Membran + Egilme gerilmeleri
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")


def generate_shell3d_dataset(n_samples=1000):
    np.random.seed(42)
    data = []

    for _ in range(n_samples):
        E = np.random.uniform(70e9, 210e9)
        nu = np.random.uniform(0.25, 0.35)
        h = np.random.uniform(0.002, 0.02)
        R = np.random.uniform(0.5, 2.0)  # Kabuk yaricapi
        p = np.random.uniform(1e5, 5e6)  # Ic basinc

        # Membran gerilmeleri (silindirik kabuk)
        sigma_hoop = p * R / h  # Cevre gerilmesi
        sigma_axial = p * R / (2 * h)  # Eksenel gerilme

        # Von Mises (membran)
        sigma_vm = np.sqrt(sigma_hoop**2 - sigma_hoop*sigma_axial + sigma_axial**2)

        data.append([E, nu, h, R, p, sigma_hoop, sigma_axial, sigma_vm])

    return np.array(data)


def split_data(data, test_ratio=0.2):
    R = data[:, 3]
    threshold = np.percentile(R, (1 - test_ratio) * 100)
    train_mask = R < threshold
    return data[train_mask], data[~train_mask]


class SpineHybridShell3D(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()

        self.fine_tune = nn.Sequential(
            nn.Linear(5, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 3)  # hoop, axial, vm corrections
        )
        self.scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        E, nu, h, R, p = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]

        # Fizik: Membran gerilmeleri
        sigma_hoop_physics = p * R / h
        sigma_axial_physics = p * R / (2 * h)
        sigma_vm_physics = torch.sqrt(
            sigma_hoop_physics**2 -
            sigma_hoop_physics * sigma_axial_physics +
            sigma_axial_physics**2 + 1e-10
        )

        # Scaled input
        x_scaled = torch.stack([
            E / 1e11, nu, h / 0.01, R, p / 1e6
        ], dim=1)

        corrections = self.fine_tune(x_scaled) * self.scale

        sigma_hoop = sigma_hoop_physics * (1 + corrections[:, 0])
        sigma_axial = sigma_axial_physics * (1 + corrections[:, 1])
        sigma_vm = sigma_vm_physics * (1 + corrections[:, 2])

        return torch.stack([sigma_hoop, sigma_axial, sigma_vm], dim=1)


class PureMLP(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 3)
        )
        self.register_buffer('out_scale', torch.tensor([1e8, 1e8, 1e8]))

    def forward(self, x):
        x_scaled = torch.stack([
            x[:, 0] / 1e11, x[:, 1], x[:, 2] / 0.01, x[:, 3], x[:, 4] / 1e6
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
    print("3D SHELL NEURON EGITIM TESTI")
    print("=" * 60)

    data = generate_shell3d_dataset(1000)
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

    print("\nSpineHybridShell3D egitiliyor...")
    spine = SpineHybridShell3D(64).to(DEVICE)
    print(f"  Parametre: {count_parameters(spine)}")
    spine_err = train_model(spine, train_loader, test_loader, y_test)

    print("\nPure MLP egitiliyor...")
    mlp = PureMLP(64).to(DEVICE)
    print(f"  Parametre: {count_parameters(mlp)}")
    mlp_err = train_model(mlp, train_loader, test_loader, y_test)

    print("\n" + "=" * 60)
    print(f"{'Model':<25} {'Parametre':<12} {'Hata':<10}")
    print("-" * 50)
    print(f"{'SpineHybridShell3D':<25} {count_parameters(spine):<12} {spine_err:.2f}%")
    print(f"{'Pure MLP':<25} {count_parameters(mlp):<12} {mlp_err:.2f}%")
    if spine_err < mlp_err:
        print(f"\n SPINE {(mlp_err - spine_err) / mlp_err * 100:.1f}% daha iyi!")
    print("=" * 60)


if __name__ == "__main__":
    main()
