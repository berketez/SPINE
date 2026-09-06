"""
FAZ 3 nöron testleri (güncel API): StaticNeuron, ModalNeuron, CrackNeuron,
ThermalNeuron.

Eski sürüm SPINE(resolution=64) + analyze_* metotlarına (silinmiş API)
dayanıyordu; bu sürüm nöron sınıflarını doğrudan kullanır ve aynı analitik
referansları assert'e bağlar.

Çalıştırma:
    PYTHONPATH=. python3 tests/test_faz3.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from spine import (
    CrackNeuron,
    MaterialProperties,
    ModalNeuron,
    StaticNeuron,
    ThermalNeuron,
)

STEEL = MaterialProperties(E=200e9, nu=0.3, h=0.01)


def test_static_square_plate():
    """Basit mesnetli kare plaka, uniform yük: w_max = 0.00406·qL⁴/D
    (Timoshenko & Woinowsky-Krieger, Tablo 8). Navier serisi %1 içinde
    yakınsamalı; çözüm simetrik olmalı ve maksimum merkezde çıkmalı."""
    Lx = Ly = 1.0
    q = 10_000.0  # 10 kN/m²
    neuron = StaticNeuron(resolution=64, Lx=Lx, Ly=Ly, material=STEEL)

    w_max = neuron.max_displacement(q)
    w_ref = 0.00406 * q * Lx**4 / STEEL.D
    rel = abs(w_max - w_ref) / w_ref
    assert rel < 0.01, f"w_max={w_max:.6e} vs 0.00406·qL⁴/D={w_ref:.6e} ({rel:.2%})"

    with torch.no_grad():
        w = neuron.solve(q)
    # Simetri: w(x,y) = w(Lx-x, y) = w(x, Ly-y)
    assert torch.allclose(w, torch.flip(w, dims=[0]), atol=w.max().item() * 1e-4)
    assert torch.allclose(w, torch.flip(w, dims=[1]), atol=w.max().item() * 1e-4)
    # Maksimum merkez civarında
    idx = torch.argmax(w)
    i, j = divmod(idx.item(), w.shape[1])
    c = w.shape[0] // 2
    assert abs(i - c) <= 1 and abs(j - c) <= 1, f"maksimum ({i},{j}), merkez ~({c},{c})"
    print(f"[Statik] w_max={w_max*1000:.4f} mm, analitik sapma {rel:.3%} — OK")


def test_modal_frequencies():
    """Basit mesnetli plaka doğal frekansı: ω_mn = π²[(m/L)²+(n/W)²]·√(D/ρh).
    Kapalı form olduğundan birebir eşleşmeli; find_modes artan sıralı olmalı."""
    Lx = Ly = 1.0
    rho = 7850.0
    neuron = ModalNeuron(resolution=64, Lx=Lx, Ly=Ly, material=STEEL, rho=rho)

    for m, n in [(1, 1), (2, 1), (2, 2)]:
        omega = neuron.natural_frequency(m, n)
        ref = math.pi**2 * ((m / Lx) ** 2 + (n / Ly) ** 2) * \
            math.sqrt(STEEL.D / (rho * STEEL.h))
        assert abs(omega - ref) / ref < 1e-9, f"ω({m},{n})={omega} vs {ref}"

    # Hz varyantı 2π ile tutarlı
    assert abs(neuron.natural_frequency_hz(1, 1) -
               neuron.natural_frequency(1, 1) / (2 * math.pi)) < 1e-9

    modes = neuron.find_modes(max_m=3, max_n=3)
    omegas = [mo["omega"] for mo in modes]
    assert omegas == sorted(omegas), "find_modes artan frekansla sıralı olmalı"
    assert (modes[0]["m"], modes[0]["n"]) == (1, 1), "temel mod (1,1) olmalı"

    # Modal kütle: ρhLW/4
    M_ref = rho * STEEL.h * Lx * Ly / 4
    assert abs(neuron.modal_mass() - M_ref) / M_ref < 1e-12
    print(f"[Modal] f₁₁={neuron.natural_frequency_hz(1,1):.2f} Hz analitikle birebir — OK")


def test_crack_sif():
    """Kırılma mekaniği: K_I = σ√(πa)·Y(a/W) (merkez çatlak, secant
    geometri faktörü) ve J = K²(1-ν²)/E (düzlem şekil değiştirme)."""
    a, W = 0.01, 0.1  # a/W = 0.1
    sigma = 100e6
    neuron = CrackNeuron(material=STEEL, crack_length=a, width=W)

    Y_ref = math.sqrt(1 / math.cos(math.pi * (a / W) / 2))
    K_ref = sigma * math.sqrt(math.pi * a) * Y_ref
    K = neuron.K_I(sigma, crack_type="center")
    assert abs(K - K_ref) / K_ref < 1e-9, f"K_I={K:.4e} vs {K_ref:.4e}"

    # Kenar çatlağı: polinom geometri faktörü
    Y_edge = 1.12 - 0.231 * 0.1 + 10.55 * 0.01 - 21.72 * 0.001 + 30.39 * 0.0001
    K_edge = neuron.K_I(sigma, crack_type="edge")
    K_edge_ref = sigma * math.sqrt(math.pi * a) * Y_edge
    assert abs(K_edge - K_edge_ref) / K_edge_ref < 1e-9

    # J-integrali (düzlem şekil değiştirme, saf Mod I)
    J = neuron.J_integral(K_I=K)
    J_ref = K**2 * (1 - STEEL.nu**2) / STEEL.E
    assert abs(J - J_ref) / J_ref < 1e-9, f"J={J:.4e} vs {J_ref:.4e}"

    # Güvenlik faktörü tanımı: SF = K_Ic / K_I
    K_Ic = 50e6  # Pa·√m
    assert abs(neuron.safety_factor(sigma, K_Ic) - K_Ic / K) < 1e-9
    print(f"[Çatlak] K_I={K/1e6:.2f} MPa·√m, J={J:.1f} J/m² analitikle birebir — OK")


def test_thermal():
    """Termal analiz kapalı formları: ε_T = αΔT, σ_T = -EαΔT/(1-ν),
    N_T = σ_T·h ve kritik termal burkulma ΔT_cr = N_cr(1-ν)/(Eαh)."""
    alpha = 12e-6  # çelik, 1/K
    Lx = Ly = 1.0
    dT = 50.0
    neuron = ThermalNeuron(material=STEEL, alpha=alpha,
                           resolution=64, Lx=Lx, Ly=Ly)

    eps = neuron.thermal_strain(dT)
    assert abs(eps - alpha * dT) / (alpha * dT) < 1e-9

    sigma = neuron.thermal_stress_constrained(dT)
    sigma_ref = -STEEL.E * alpha * dT / (1 - STEEL.nu)
    assert abs(sigma - sigma_ref) / abs(sigma_ref) < 1e-9
    assert sigma < 0, "ısınan kısıtlı plakada basma (negatif) gerilme oluşmalı"

    N_T = neuron.thermal_membrane_force(dT)
    assert abs(N_T - sigma * STEEL.h) / abs(sigma * STEEL.h) < 1e-9

    # Kritik termal burkulma: N_T(ΔT_cr) büyüklüğü N_cr'ye eşit olmalı.
    # VARSAYILAN biaxial'dir: düzlem içinde tam kısıtlı bir plaka ısındığında
    # her iki yönde de genleşemez → N_x = N_y = N_T. Kritik yükü uniaxial
    # formülüyle almak (eski davranış) ΔT_cr'yi kare plakada 2 kat fazla,
    # yani güvensiz yönde verir.
    tx = (math.pi / Lx) ** 2
    ty = (math.pi / Ly) ** 2

    dT_cr = neuron.critical_thermal_buckling(m=1, n=1)          # biaxial
    N_cr_bi = STEEL.D * (tx + ty)
    N_at_crit = abs(neuron.thermal_membrane_force(dT_cr))
    assert abs(N_at_crit - N_cr_bi) / N_cr_bi < 1e-9, \
        f"biaxial ΔT_cr'de |N_T|={N_at_crit:.4e} ≠ N_cr={N_cr_bi:.4e}"

    # Uniaxial seçeneği açıkça istendiğinde eski kapalı formu vermeli
    dT_cr_uni = neuron.critical_thermal_buckling(m=1, n=1, loading="uniaxial")
    N_cr_uni = STEEL.D * (tx + ty) ** 2 / tx
    N_at_crit_uni = abs(neuron.thermal_membrane_force(dT_cr_uni))
    assert abs(N_at_crit_uni - N_cr_uni) / N_cr_uni < 1e-9, \
        f"uniaxial ΔT_cr'de |N_T|={N_at_crit_uni:.4e} ≠ N_cr={N_cr_uni:.4e}"

    # Kare plakada uniaxial kritik yük biaxial'in tam 2 katıdır
    assert abs(dT_cr_uni / dT_cr - 2.0) < 1e-9, \
        f"kare plakada ΔT_cr(uni)/ΔT_cr(bi) = {dT_cr_uni/dT_cr:.6f}, 2.0 bekleniyordu"

    # forward() zinciri: ε_T → σ_T → N_T → N_cr → SF tutarlı olmalı
    res = neuron(dT, max_mode=3)
    assert abs(res["eps_T"] - alpha * dT) / (alpha * dT) < 1e-9
    assert abs(res["N_T"] - N_T) / abs(N_T) < 1e-9
    assert abs(res["SF_thermal"] - res["N_cr"] / abs(res["N_T"])) < 1e-9
    # Kare plakada en kritik mod (1,1) olmalı
    assert (res["m"], res["n"]) == (1, 1)

    # Termal eğrilik: κ = αΔT/h
    kappa = neuron.thermal_curvature(T_top=80.0, T_bottom=20.0)
    assert abs(kappa - alpha * 60.0 / STEEL.h) < 1e-12
    print(f"[Termal] σ_T={sigma/1e6:.1f} MPa, ΔT_cr={dT_cr:.2f} K analitikle birebir — OK")


if __name__ == "__main__":
    test_static_square_plate()
    test_modal_frequencies()
    test_crack_sif()
    test_thermal()
    print("\n=== TÜM 4 TEST GEÇTİ ===")
