"""
Shell + Beam Entegrasyon Testi
- 3D Kabuk ve Kiris birlesimleri
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")


def generate_shell_beam_dataset(n_samples=1000):
    np.random.seed(42)
    data = []

    for _ in range(n_samples):
        E = np.random.uniform(70e9, 210e9)
        nu = np.random.uniform(0.25, 0.35)

        # Kabuk parametreleri
        h_shell = np.random.uniform(0.003, 0.015)
        R = np.random.uniform(0.3, 1.5)
        p = np.random.uniform(5e4, 2e6)

        # Kiris parametreleri (kabuk uzerindeki takviye)
        b_beam = np.random.uniform(0.02, 0.08)
        h_beam = np.random.uniform(0.05, 0.15)
        L_beam = np.random.uniform(0.5, 2.0)

        # Kabuk gerilmeleri
        sigma_hoop = p * R / h_shell
        sigma_axial = p * R / (2 * h_shell)

        # Kiris ozellikleri
        I_beam = b_beam * h_beam**3 / 12
        A_beam = b_beam * h_beam

        # Kiris uzerindeki yuk (kabuktan aktarilan)
        q_beam = sigma_hoop * h_shell  # N/m olarak

        # Kiris max momenti ve gerilmesi
        M_max = q_beam * L_beam**2 / 8  # Basit mesnetli
        sigma_beam = M_max * (h_beam / 2) / I_beam

        # Toplam Von Mises (kabuk + kiris etkilesimi)
        sigma_vm_total = np.sqrt(sigma_hoop**2 + sigma_beam**2 - sigma_hoop * sigma_beam)

        # Kiris sehimi
        w_beam = 5 * q_beam * L_beam**4 / (384 * E * I_beam)

        data.append([E, nu, h_shell, R, p, b_beam, h_beam, L_beam,
                     sigma_hoop, sigma_beam, sigma_vm_total, w_beam])

    return np.array(data)


def split_data(data, test_ratio=0.2):
    R = data[:, 3]
    threshold = np.percentile(R, (1 - test_ratio) * 100)
    train_mask = R < threshold
    return data[train_mask], data[~train_mask]


class SpineHybridShellBeam(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(8, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh()
        )
        self.heads = nn.Linear(hidden_size, 4)  # sigma_hoop, sigma_beam, sigma_vm, w
        self.scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        E, nu, h_shell, R, p, b_beam, h_beam, L_beam = [x[:, i] for i in range(8)]

        # Fizik: Kabuk gerilmesi
        sigma_hoop_physics = p * R / h_shell

        # Fizik: Kiris ozellikleri
        I_beam = b_beam * h_beam**3 / 12

        # Fizik: Kiris uzerindeki yuk
        q_beam = sigma_hoop_physics * h_shell

        # Fizik: Kiris gerilmesi
        M_max = q_beam * L_beam**2 / 8
        sigma_beam_physics = M_max * (h_beam / 2) / I_beam

        # Fizik: Von Mises
        sigma_vm_physics = torch.sqrt(
            sigma_hoop_physics**2 + sigma_beam_physics**2 -
            sigma_hoop_physics * sigma_beam_physics + 1e-10
        )

        # Fizik: Sehim
        w_physics = 5 * q_beam * L_beam**4 / (384 * E * I_beam)

        # Scaled input
        x_scaled = torch.stack([
            E / 1e11, nu, h_shell / 0.01, R, p / 1e6,
            b_beam / 0.05, h_beam / 0.1, L_beam
        ], dim=1)

        features = self.encoder(x_scaled)
        corrections = self.heads(features) * self.scale

        sigma_hoop = sigma_hoop_physics * (1 + corrections[:, 0])
        sigma_beam = sigma_beam_physics * (1 + corrections[:, 1])
        sigma_vm = sigma_vm_physics * (1 + corrections[:, 2])
        w = w_physics * (1 + corrections[:, 3])

        return torch.stack([sigma_hoop, sigma_beam, sigma_vm, w], dim=1)


class PureMLP(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 4)
        )
        self.register_buffer('out_scale', torch.tensor([1e8, 1e8, 1e8, 0.01]))

    def forward(self, x):
        x_scaled = torch.stack([
            x[:, 0] / 1e11, x[:, 1], x[:, 2] / 0.01, x[:, 3], x[:, 4] / 1e6,
            x[:, 5] / 0.05, x[:, 6] / 0.1, x[:, 7]
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
    print("SHELL + BEAM ENTEGRASYON TESTI")
    print("=" * 60)

    data = generate_shell_beam_dataset(1000)
    train_data, test_data = split_data(data)
    print(f"Egitim: {len(train_data)}, Test: {len(test_data)}")

    X_train, y_train = train_data[:, :8], train_data[:, 8:]
    X_test, y_test = test_data[:, :8], test_data[:, 8:]

    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_train, dtype=torch.float32, device=DEVICE),
        torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    ), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(
        torch.tensor(X_test, dtype=torch.float32, device=DEVICE),
        torch.tensor(y_test, dtype=torch.float32, device=DEVICE)
    ), batch_size=32)

    print("\nSpineHybridShellBeam egitiliyor...")
    spine = SpineHybridShellBeam(64).to(DEVICE)
    print(f"  Parametre: {count_parameters(spine)}")
    spine_err = train_model(spine, train_loader, test_loader, y_test)

    print("\nPure MLP egitiliyor...")
    mlp = PureMLP(64).to(DEVICE)
    print(f"  Parametre: {count_parameters(mlp)}")
    mlp_err = train_model(mlp, train_loader, test_loader, y_test)

    print("\n" + "=" * 60)
    print(f"{'Model':<30} {'Parametre':<12} {'Hata':<10}")
    print("-" * 55)
    print(f"{'SpineHybridShellBeam':<30} {count_parameters(spine):<12} {spine_err:.2f}%")
    print(f"{'Pure MLP':<30} {count_parameters(mlp):<12} {mlp_err:.2f}%")
    if spine_err < mlp_err:
        print(f"\n SPINE {(mlp_err - spine_err) / mlp_err * 100:.1f}% daha iyi!")
    print("=" * 60)


if __name__ == "__main__":
    main()
