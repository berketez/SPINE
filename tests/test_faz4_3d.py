"""
FAZ 4 — 3D nöron testleri (güncel API): SpectralOps3D, SolidNeuron3D,
ShellNeuron3D, BeamNeuron3D.

Eski sürüm argümansız kurucular kullanıyordu (API drift, satır 20'de
TypeError); bu sürüm güncel imzalarla aynı fizik doğrulamalarını yapar.

Çalıştırma:
    PYTHONPATH=. python3 tests/test_faz4_3d.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from spine import (
    BeamNeuron3D,
    MaterialProperties,
    ShellNeuron3D,
    SolidNeuron3D,
    SpectralOps3D,
)

E_STEEL, NU_STEEL = 200e9, 0.3


def test_spectral_ops_3d_periodic():
    """3D spektral operatörler periyodik rejimde:
    ∇²[sin(x)sin(y)sin(z)] = -3·f, ∇·(∇f) = ∇²f."""
    N = 32
    L = 2 * math.pi
    ops = SpectralOps3D(resolution=N, Lx=L, Ly=L, Lz=L)

    x = torch.arange(N) * (L / N)
    X, Y, Z = torch.meshgrid(x, x, x, indexing="ij")
    f = torch.sin(X) * torch.sin(Y) * torch.sin(Z)

    lap = ops.laplacian(f)
    rel = (torch.norm(lap + 3.0 * f) / torch.norm(3.0 * f)).item()
    assert rel < 1e-4, f"3D Laplasyen bağıl hata {rel:.2e}"

    # Tutarlılık: div(grad f) = ∇²f
    gx, gy, gz = ops.gradient(f)
    div_grad = ops.divergence(gx, gy, gz)
    rel2 = (torch.norm(div_grad - lap) / torch.norm(lap)).item()
    assert rel2 < 1e-4, f"div∘grad ≠ laplacian: {rel2:.2e}"
    print(f"[3D Spektral] ∇² hata {rel:.1e}, div∘grad tutarlılık {rel2:.1e} — OK")


def test_solid_3d_hooke():
    """3D Hooke yasası: Lamé sabitleri, tek eksenli şekil değiştirme durumu
    (σ_xx = (λ+2μ)ε, σ_yy = σ_zz = λε) ve saf kaymada von Mises = √3·τ."""
    solid = SolidNeuron3D(resolution=8, Lx=1, Ly=1, Lz=1, E=E_STEEL, nu=NU_STEEL)

    lam, mu = solid.lame_parameters()
    lam_ref = E_STEEL * NU_STEEL / ((1 + NU_STEEL) * (1 - 2 * NU_STEEL))
    mu_ref = E_STEEL / (2 * (1 + NU_STEEL))
    assert abs(lam - lam_ref) / lam_ref < 1e-9
    assert abs(mu - mu_ref) / mu_ref < 1e-9

    shape = (8, 8, 8)
    zero = torch.zeros(shape)

    # Tek eksenli şekil değiştirme (constrained): ε_xx = 1e-3, diğerleri 0
    eps = 1e-3
    strain = {"eps_xx": torch.full(shape, eps), "eps_yy": zero, "eps_zz": zero,
              "eps_xy": zero, "eps_xz": zero, "eps_yz": zero}
    with torch.no_grad():
        stress = solid.stress_from_strain(strain)
    s_xx_ref = (lam_ref + 2 * mu_ref) * eps
    s_yy_ref = lam_ref * eps
    assert abs(stress["sigma_xx"][0, 0, 0].item() - s_xx_ref) / s_xx_ref < 1e-6
    assert abs(stress["sigma_yy"][0, 0, 0].item() - s_yy_ref) / s_yy_ref < 1e-6
    assert torch.allclose(stress["sigma_yy"], stress["sigma_zz"])

    # Saf kayma: ε_xy = γ/2 → τ = 2με_xy, von Mises = √3·τ
    gamma_half = 5e-4
    strain_shear = {"eps_xx": zero, "eps_yy": zero, "eps_zz": zero,
                    "eps_xy": torch.full(shape, gamma_half),
                    "eps_xz": zero, "eps_yz": zero}
    with torch.no_grad():
        stress_s = solid.stress_from_strain(strain_shear)
        vm = solid.von_mises_stress(stress_s)
    tau = 2 * mu_ref * gamma_half
    vm_ref = math.sqrt(3) * tau
    assert abs(vm[0, 0, 0].item() - vm_ref) / vm_ref < 1e-6, \
        f"von Mises {vm[0,0,0].item():.4e} vs √3·τ={vm_ref:.4e}"
    print("[Solid3D] Lamé + tek eksenli Hooke + von Mises(√3τ) birebir — OK")


def test_shell_3d_constitutive():
    """Kabuk bünye denklemleri (doğrudan κ/ε girdisiyle, FFT'siz — saf
    kapalı form): Mx = D(κx + ν·κy), Nx = A(εx + ν·εy);
    D = Eh³/[12(1-ν²)], A = Eh/(1-ν²)."""
    h = 0.005
    shell = ShellNeuron3D(resolution=16, Lx=1.0, Ly=1.0, h=h,
                          E=E_STEEL, nu=NU_STEEL)

    D_ref = E_STEEL * h**3 / (12 * (1 - NU_STEEL**2))
    A_ref = E_STEEL * h / (1 - NU_STEEL**2)
    assert abs(shell.D - D_ref) / D_ref < 1e-12
    assert abs(shell.A - A_ref) / A_ref < 1e-12

    shape = (16, 16)
    kx = torch.full(shape, 0.1)
    ky = torch.full(shape, 0.05)
    kxy = torch.full(shape, 0.02)
    with torch.no_grad():
        Mx, My, Mxy = shell.bending_moments(kx, ky, kxy)
    assert abs(Mx[0, 0].item() - D_ref * (0.1 + NU_STEEL * 0.05)) < 1e-6 * D_ref
    assert abs(My[0, 0].item() - D_ref * (0.05 + NU_STEEL * 0.1)) < 1e-6 * D_ref
    assert abs(Mxy[0, 0].item() - D_ref * (1 - NU_STEEL) / 2 * 0.02) < 1e-6 * D_ref

    ex = torch.full(shape, 1e-4)
    ey = torch.full(shape, 5e-5)
    gxy = torch.full(shape, 2e-5)
    with torch.no_grad():
        Nx, Ny, Nxy = shell.membrane_forces(ex, ey, gxy)
    assert abs(Nx[0, 0].item() - A_ref * (1e-4 + NU_STEEL * 5e-5)) < 1e-6 * A_ref * 1e-4
    assert abs(Nxy[0, 0].item() - A_ref * (1 - NU_STEEL) / 2 * 2e-5) < 1e-6 * A_ref * 1e-4
    print("[Shell3D] D, A, momentler ve membran kuvvetleri kapalı formla birebir — OK")


def test_beam_3d():
    """Kiriş: Euler burkulması P_cr = π²EI/(KL)², doğal frekans (Hz) ve
    uniform yüklü basit kirişte w_max = 5qL⁴/(384EI)."""
    L, I, A, rho = 2.0, 8.33e-6, 0.01, 7850.0  # 10cm kare kesit
    beam = BeamNeuron3D(length=L, resolution=256, E=E_STEEL, I=I, A=A,
                        rho=rho, nu=NU_STEEL)

    # Euler burkulması, K faktörü davranışıyla birlikte
    P_ref = math.pi**2 * E_STEEL * I / L**2
    assert abs(beam.critical_buckling_load(n=1, K=1.0) - P_ref) / P_ref < 1e-9
    assert abs(beam.critical_buckling_load(n=1, K=2.0) - P_ref / 4) / P_ref < 1e-9
    assert abs(beam.critical_buckling_load(n=1, K=0.5) - 4 * P_ref) / P_ref < 1e-9

    # Doğal frekans (Hz döndürür — bilinen API davranışı)
    f1 = beam.natural_frequency(n=1, theory="euler")
    f1_ref = (math.pi / L) ** 2 * math.sqrt(E_STEEL * I / (rho * A)) / (2 * math.pi)
    assert abs(f1 - f1_ref) / f1_ref < 1e-9, f"f₁={f1:.3f} Hz vs {f1_ref:.3f} Hz"

    # Uniform yük: Navier serisi w_max ↔ 5qL⁴/(384EI)
    q_val = 1000.0  # N/m
    q = torch.full((256,), q_val)
    with torch.no_grad():
        w = beam.euler_bernoulli_solve(q)
    w_max_ref = 5 * q_val * L**4 / (384 * E_STEEL * I)
    rel = abs(w.max().item() - w_max_ref) / w_max_ref
    assert rel < 0.02, f"w_max={w.max().item():.6e} vs {w_max_ref:.6e} ({rel:.2%})"

    # Mod şekli: birim genlik, uçlarda sıfır
    phi = beam.mode_shape(n=1)
    assert abs(phi.max().item() - 1.0) < 1e-3
    assert abs(phi[0].item()) < 1e-6
    print(f"[Beam3D] P_cr(K), f₁={f1:.2f} Hz, w_max sapma {rel:.3%} — OK")


if __name__ == "__main__":
    test_spectral_ops_3d_periodic()
    test_solid_3d_hooke()
    test_shell_3d_constitutive()
    test_beam_3d()
    print("\n=== TÜM 4 TEST GEÇTİ ===")
