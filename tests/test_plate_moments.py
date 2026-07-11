"""
SPINE.forward plaka momentleri için regresyon testleri.

Düzeltilen bug (2026-07-11 denetimi):
  spine.py SPINE.forward eski hali M = -D·κ kullanıyordu. StrainNeuron
  konvansiyonunda κ_ij = -w_ij olduğundan bu M_xx = +D·w_xx demekti —
  hem işaret ters hem ν çaprazlama terimi eksikti (analitik Kirchhoff'a
  oran merkezde -0.454 ölçülmüştü). Doğru form (ShellNeuron3D ile aynı):
      M_xx = D(κ_xx + ν·κ_yy)
      M_yy = D(κ_yy + ν·κ_xx)
      M_xy = D(1-ν)·κ_xy      (κ_xy = -w_xy konvansiyonuyla)

Not: FFT ikinci türevleri linspace(0, L, N) gridinde (N/(N-1))^2 ≈ %3
ölçek sapması taşır (grid konvansiyonu, ayrı bilinen sorun). Bu çarpan
yalnızca TEK m,n modlarının anti-node'unda (merkez) temiz geçerlidir;
çift modlarda merkez düğüme yakın olduğundan ve kenar şeritlerinde
(periyodik olmayan sinyal → Gibbs) genellenemez. Testler bu yüzden
m=n=1 merkez noktası + iç bölgeyle sınırlıdır; bünye denklemi testleri
ise κ üzerinden makine hassasiyetindedir.

Çalıştırma:
    PYTHONPATH=. python3 tests/test_plate_moments.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from spine import SPINE, MaterialProperties


RESOLUTION = 64
LX, LY = 1.0, 0.5
MAT = MaterialProperties(E=200e9, nu=0.3, h=0.005)


def _model_and_state(m=1, n=1):
    model = SPINE(resolution=RESOLUTION, Lx=LX, Ly=LY, material=MAT)
    w = model.get_mode_shape(m=m, n=n)
    return model, w, model.forward(w)


def test_constitutive_form_matches_kappa():
    """forward'ın momentleri, aynı κ alanlarından hesaplanan standart
    Kirchhoff bünye denklemine makine hassasiyetinde eşit olmalı."""
    model, w, state = _model_and_state()
    kxx, kyy, kxy = model.strain(w)
    D, nu = MAT.D, MAT.nu

    for got, ref, name in [
        (state.M_xx, D * (kxx + nu * kyy), "M_xx"),
        (state.M_yy, D * (kyy + nu * kxx), "M_yy"),
        (state.M_xy, D * (1 - nu) * kxy, "M_xy"),
    ]:
        rel = (torch.norm(got - ref) / torch.norm(ref)).item()
        assert rel < 1e-6, f"{name} bünye formu sapıyor: bağıl fark {rel:.3e}"
    print("[Bünye] M = D(κ + ν·κ_çapraz) formu κ'dan makine hassasiyetinde — OK")


def test_center_ratio_against_analytic_kirchhoff():
    """Navier modu için merkez noktada analitik momentle karşılaştırma.

    Analitik: w = sin(mπx/Lx)·sin(nπy/Ly) için
        M_xx = D(a_x + ν·a_y)·w,  a_x=(mπ/Lx)², a_y=(nπ/Ly)²
    FFT grid sapması (N/(N-1))² bilinen çarpan olarak düşülür; kalan
    sapma %1'in altında olmalı. Eski kod bu oranı -0.454 veriyordu —
    işaret ve büyüklük regresyonunu bu test yakalar.
    """
    m, n = 1, 1
    model, w, state = _model_and_state(m, n)
    D, nu = MAT.D, MAT.nu
    ax, ay = (m * math.pi / LX) ** 2, (n * math.pi / LY) ** 2

    c = RESOLUTION // 2
    w_c = w[c, c].item()
    grid_factor = (RESOLUTION / (RESOLUTION - 1)) ** 2

    ratio_xx = state.M_xx[c, c].item() / (D * (ax + nu * ay) * w_c) / grid_factor
    ratio_yy = state.M_yy[c, c].item() / (D * (ay + nu * ax) * w_c) / grid_factor
    assert abs(ratio_xx - 1.0) < 0.01, f"M_xx/analitik = {ratio_xx:.4f} (1.0 olmalı)"
    assert abs(ratio_yy - 1.0) < 0.01, f"M_yy/analitik = {ratio_yy:.4f} (1.0 olmalı)"
    assert ratio_xx > 0, "M_xx işareti ters (eski bug: -0.454)"
    print(f"[Analitik] merkez oranlar M_xx={ratio_xx:.4f}, M_yy={ratio_yy:.4f} — OK")


def test_twist_moment_sign_and_scale():
    """M_xy = -D(1-ν)·w_xy: çeyrek noktada (cos·cos maksimuma yakın)
    analitik burulma momentiyle işaret + ölçek karşılaştırması."""
    m, n = 1, 1
    model, w, state = _model_and_state(m, n)
    D, nu = MAT.D, MAT.nu

    N = RESOLUTION
    q = N // 4
    x = torch.linspace(0, LX, N)
    y = torch.linspace(0, LY, N)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    Mxy_ref = (-D * (1 - nu) * (m * math.pi / LX) * (n * math.pi / LY)
               * torch.cos(m * math.pi * X / LX) * torch.cos(n * math.pi * Y / LY))

    grid_factor = (N / (N - 1)) ** 2
    ratio = state.M_xy[q, q].item() / Mxy_ref[q, q].item() / grid_factor
    assert abs(ratio - 1.0) < 0.01, f"M_xy/analitik = {ratio:.4f} (1.0 olmalı)"
    print(f"[Burulma] çeyrek nokta M_xy oranı {ratio:.4f} — OK")


def test_strain_energy_reconstruction():
    """strain_energy'nin ∇²w geri kazanımı: (M_xx+M_yy)/(D(1+ν)) = -∇²w.
    Navier modunda -∇²w = (a_x+a_y)·w; merkez noktada doğrulanır."""
    m, n = 1, 1
    model, w, state = _model_and_state(m, n)
    D, nu = MAT.D, MAT.nu
    ax, ay = (m * math.pi / LX) ** 2, (n * math.pi / LY) ** 2

    c = RESOLUTION // 2
    grid_factor = (RESOLUTION / (RESOLUTION - 1)) ** 2
    lap_rec = (state.M_xx[c, c] + state.M_yy[c, c]).item() / (D * (1 + nu))
    lap_ref = (ax + ay) * w[c, c].item()
    ratio = lap_rec / lap_ref / grid_factor
    assert abs(ratio - 1.0) < 0.01, f"enerji geri kazanım oranı {ratio:.4f}"

    U = state.strain_energy(D, nu)
    assert U.item() > 0, "şekil değiştirme enerjisi pozitif olmalı"

    # strain_energy İÇ formülünü sentetik temiz momentlerle doğrula
    # (FFT'siz — M_xx = M_yy = c·w verilirse U = 0.5·D·mean((2c·w/(D(1+ν)))²)
    # olmalı; (1+ν) böleni bozulursa bu assert kırılır)
    from spine import PlateState
    c = 123.4
    M_syn = c * w
    zero = torch.zeros_like(w)
    state_syn = PlateState(w=w, theta_x=zero, theta_y=zero,
                           M_xx=M_syn, M_yy=M_syn, M_xy=zero)
    U_syn = state_syn.strain_energy(D, nu).item()
    U_ref = (0.5 * D * ((2 * c * w / (D * (1 + nu))) ** 2).mean()).item()
    assert abs(U_syn - U_ref) / U_ref < 1e-6, \
        f"strain_energy iç formülü sapıyor: {U_syn:.6e} vs {U_ref:.6e}"
    # Not: FFT'li gerçek alanda U'nun MUTLAK değeri kenar Gibbs artefaktları
    # nedeniyle güvenilir değildir (bilinen sınırlama) — burada test edilen
    # şey formülün kendisidir, alan integrali değil.
    print(f"[Enerji] ∇²w geri kazanım oranı {ratio:.4f}, iç formül doğrulandı — OK")


def test_shell_consistency():
    """SPINE.forward ile ShellNeuron3D.bending_moments aynı κ girdisinde
    aynı M_xx/M_yy vermeli (κ_xy konvansiyon farkı belgeli: Shell 2κ bekler)."""
    from spine import ShellNeuron3D

    model, w, state = _model_and_state()
    kxx, kyy, kxy = model.strain(w)

    shell = ShellNeuron3D(resolution=RESOLUTION, Lx=LX, Ly=LY,
                          E=MAT.E, nu=MAT.nu, h=MAT.h, R=1e12)
    with torch.no_grad():
        Mx_s, My_s, _ = shell.bending_moments(kxx, kyy, kxy)

    # Shell D'si modülatör içerebilir; oran alanının sabitliğini test et
    ratio = (state.M_xx / Mx_s)
    spread = (ratio.max() - ratio.min()).item() / ratio.mean().abs().item()
    assert spread < 1e-5, f"SPINE/Shell M_xx oranı alan boyunca sabit değil: {spread:.3e}"
    print(f"[Shell tutarlılık] M_xx oranı alan boyunca sabit (sapma {spread:.1e}) — OK")


if __name__ == "__main__":
    test_constitutive_form_matches_kappa()
    test_center_ratio_against_analytic_kirchhoff()
    test_twist_moment_sign_and_scale()
    test_strain_energy_reconstruction()
    test_shell_consistency()
    print("\n=== TÜM 5 TEST GEÇTİ ===")
