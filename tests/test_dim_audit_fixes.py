"""
Dim-audit düzeltmeleri için regresyon testleri.

Üç bug için (rapor: /tmp/dim_audit_spine.md):
  1. spine.py:1002-1003 — Kirchhoff plate stress'inde ν cross-coupling
     terimleri (eklendi).
  2. spine.py:2452-2456 — Timoshenko shear stiffness:
       - kappa kullanılmalı (κGA olarak),
       - Poisson oranı (ν) hardcoded 0.3 yerine kullanıcıdan alınmalı.
  3. spine.py:2541-2549 — BeamNeuron3D.critical_buckling_load K
     (effective length factor) parametresi kabul etmeli.

Çalıştırma:
    PYTHONPATH=. python3 tests/test_dim_audit_fixes.py
"""

import math
import os
import sys
import warnings

# Repo kökünü dinamik bul (tests/ dizininin bir üstü) — hardcoded yol yok.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: F401  (BeamNeuron3D buffer kayıt için ihtiyacı var)

from spine import BeamNeuron3D, MaterialProperties, StaticNeuron


# ---------------------------------------------------------------------------
# Bug #1 — Kirchhoff plate stress: ν·w_yy ve ν·w_xx terimleri
# ---------------------------------------------------------------------------

def test_kirchhoff_stress_includes_poisson_coupling():
    """σ_xx ve σ_yy formüllerinin tam Kirchhoff formuna sahip olduğunu
    doğrular: σ_xx = -E·z/(1-ν²)·(w_xx + ν·w_yy).

    Strateji: max_stress fonksiyonunu çağır, aynı w türevleriyle hem yeni
    (ν cross-coupling'li) hem de eski (atlanmış) formülü manuel hesapla,
    `max_stress` çıktısının yeni formülle eşleşip eskisiyle uyuşmadığını
    doğrula.
    """
    mat = MaterialProperties(E=200e9, nu=0.3, h=0.005)
    Lx, Ly = 1.0, 0.5
    neuron = StaticNeuron(resolution=64, Lx=Lx, Ly=Ly, material=mat)

    q = 1000.0
    sx_measured, sy_measured = neuron.max_stress(q)

    # Aynı w'yi al, türevleri hesapla, formülleri manuel uygula
    w = neuron.solve(q)
    w_xx, w_yy, _ = neuron.spectral_ops.second_derivatives(w)
    z = mat.h / 2.0
    nu = mat.nu
    pre = -mat.E * z / (1 - nu ** 2)

    # Yeni (doğru, Kirchhoff tam formu)
    sx_new = (pre * (w_xx + nu * w_yy)).abs().max().item()
    sy_new = (pre * (w_yy + nu * w_xx)).abs().max().item()

    # Eski (yanlış, ν cross-coupling YOK)
    sx_old = (pre * w_xx).abs().max().item()
    sy_old = (pre * w_yy).abs().max().item()

    # max_stress çıktısı YENİ formülle eşleşmeli
    assert math.isclose(sx_measured, sx_new, rel_tol=1e-9), (
        f"σ_xx max_stress yeni formülle eşleşmiyor: "
        f"ölçülen={sx_measured:.6e}, yeni={sx_new:.6e}"
    )
    assert math.isclose(sy_measured, sy_new, rel_tol=1e-9), (
        f"σ_yy max_stress yeni formülle eşleşmiyor: "
        f"ölçülen={sy_measured:.6e}, yeni={sy_new:.6e}"
    )

    # Pointwise (alan-bazlı) doğrulama: ν cross-coupling terimi gerçekten
    # uygulanıyor mu? `.max()` bazı geometrilerde aynı noktayı seçebilir
    # (w_xx baskınsa w_yy ≈ 0 olan yer), bu yüzden alan ortalaması kullan.
    sigma_xx_field_new = (pre * (w_xx + nu * w_yy)).abs()
    sigma_xx_field_old = (pre * w_xx).abs()
    pointwise_max_diff = (sigma_xx_field_new - sigma_xx_field_old).abs().max().item()
    relative_field_diff = pointwise_max_diff / sigma_xx_field_old.max().item()
    assert relative_field_diff > 0.05, (
        "Yeni ile eski formül arasında alan-bazlı pointwise fark olmalı "
        f"(ν·w_yy terimi). pointwise_max_diff={pointwise_max_diff:.3e}, "
        f"relative={relative_field_diff:.3f}"
    )

    print(
        f"[Bug#1] σ_xx max_stress={sx_measured:.3e} Pa (Kirchhoff tam formu "
        f"ile bire-bir eşleşti), pointwise sapma=%{relative_field_diff*100:.1f} — OK"
    )


# ---------------------------------------------------------------------------
# Bug #2 — Timoshenko: κGA kullanılmalı, ν artık hardcoded değil
# ---------------------------------------------------------------------------

def test_shear_stiffness_uses_kappa_and_material_nu():
    """`shear_stiffness` (κGA) hem kappa'yı hem material.nu'yu kullanmalı."""
    # Aluminyum: ν=0.33 (hardcoded 0.3'ten farklı)
    mat = MaterialProperties(E=70e9, nu=0.33, h=0.003)
    A = 1e-3
    kappa = 5 / 6

    beam = BeamNeuron3D(
        length=3.0, resolution=64, I=1e-6, A=A, kappa=kappa, material=mat
    )

    expected_G = mat.E / (2.0 * (1.0 + mat.nu))
    expected_kGA = kappa * expected_G * A

    assert math.isclose(beam.G, expected_G, rel_tol=1e-9), (
        f"G yanlış: {beam.G} vs beklenen {expected_G}"
    )
    assert math.isclose(beam.shear_stiffness, expected_kGA, rel_tol=1e-9), (
        f"κGA yanlış: {beam.shear_stiffness} vs beklenen {expected_kGA}"
    )
    # Alternatif erişim isimleri (geri uyum + yeni)
    assert beam.kGA == beam.shear_stiffness
    assert beam.GA == beam.shear_stiffness  # eski isim, artık κGA değerine sahip

    # nu farklı materyallerle değişmeli
    steel = MaterialProperties(E=200e9, nu=0.3, h=0.005)
    beam_steel = BeamNeuron3D(length=3.0, resolution=64, I=1e-6, A=A, material=steel)
    expected_steel_kGA = (5 / 6) * (steel.E / (2.0 * (1.0 + steel.nu))) * A
    assert math.isclose(beam_steel.shear_stiffness, expected_steel_kGA, rel_tol=1e-9)

    # ν farklı malzemelerde gerçekten farklı sonuç vermeli (hardcoded olsa eşit olurdu)
    # Alüminyum ile çelik aynı ν olsaydı G = E/(2(1+ν)) sadece E ile orantılı olurdu;
    # ν=0.33 vs ν=0.3 ile orantı bozulmalı.
    ratio_actual = beam_steel.shear_stiffness / beam.shear_stiffness
    ratio_E_only = (steel.E / mat.E)
    assert not math.isclose(ratio_actual, ratio_E_only, rel_tol=1e-3), (
        "ν hardcoded olsaydı oran tam E oranına eşit olurdu — düzeltme tutmamış."
    )
    print(
        f"[Bug#2] Aluminyum κGA={beam.shear_stiffness:.3e} (ν=0.33), "
        f"Çelik κGA={beam_steel.shear_stiffness:.3e} (ν=0.30) — OK"
    )


def test_kappa_is_actually_used():
    """kappa değiştiğinde shear_stiffness değişmeli (eski kodda saklanıp
    kullanılmıyordu)."""
    mat = MaterialProperties(E=200e9, nu=0.3, h=0.005)
    A = 1e-3

    beam_default = BeamNeuron3D(length=3.0, resolution=64, I=1e-6, A=A, material=mat)
    beam_circular = BeamNeuron3D(
        length=3.0, resolution=64, I=1e-6, A=A, kappa=0.9, material=mat
    )

    ratio = beam_circular.shear_stiffness / beam_default.shear_stiffness
    expected = 0.9 / (5 / 6)
    assert math.isclose(ratio, expected, rel_tol=1e-9), (
        f"kappa kullanılmıyor: oran {ratio} vs beklenen {expected}"
    )
    print(f"[Bug#2] kappa=5/6 → {beam_default.shear_stiffness:.3e}, "
          f"kappa=0.9 → {beam_circular.shear_stiffness:.3e} (oran {ratio:.4f}) — OK")


def test_legacy_api_still_works_with_warning():
    """Eski API (`E=..., I=..., A=...` ile, nu yok) hala çalışmalı ama uyarı vermeli."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        beam = BeamNeuron3D(length=3.0, resolution=64, E=200e9, I=1e-6, A=1e-3)

    nu_warnings = [m for m in w if "nu" in str(m.message)]
    assert nu_warnings, "Eski API çağrısı uyarı vermeli"
    assert beam.nu == 0.3, "Geri-uyumluluk için varsayılan ν=0.3 olmalı"
    expected_kGA = (5 / 6) * (200e9 / (2 * 1.3)) * 1e-3
    assert math.isclose(beam.shear_stiffness, expected_kGA, rel_tol=1e-9)
    print("[Bug#2] Eski API geri-uyumlu, uyarı veriliyor — OK")


# ---------------------------------------------------------------------------
# Bug #3 — critical_buckling_load K parametresi
# ---------------------------------------------------------------------------

def test_critical_buckling_load_K_factor():
    """K factor parametresi farklı sınır koşulları için doğru çarpan vermeli.

    P_cr(K) = π²EI / (KL)²  →  P_cr(K)/P_cr(1) = 1/K²
    """
    mat = MaterialProperties(E=200e9, nu=0.3, h=0.005)
    beam = BeamNeuron3D(length=3.0, resolution=64, I=1e-6, A=1e-3, material=mat)

    P_default = beam.critical_buckling_load(n=1)             # K=1.0 default
    P_pinned = beam.critical_buckling_load(n=1, K=1.0)
    P_fixed_fixed = beam.critical_buckling_load(n=1, K=0.5)
    P_cantilever = beam.critical_buckling_load(n=1, K=2.0)
    P_pinned_fixed = beam.critical_buckling_load(n=1, K=0.7)

    # Default = pinned-pinned
    assert P_default == P_pinned

    # Beklenen oranlar
    assert math.isclose(P_fixed_fixed / P_pinned, 4.0, rel_tol=1e-9)
    assert math.isclose(P_pinned / P_cantilever, 4.0, rel_tol=1e-9)
    assert math.isclose(P_pinned_fixed / P_pinned, 1.0 / 0.7**2, rel_tol=1e-9)

    # Mutlak değer (analitik)
    expected_pinned = math.pi**2 * 200e9 * 1e-6 / 3.0**2
    assert math.isclose(P_pinned, expected_pinned, rel_tol=1e-9)

    # n=2 modu (pinned-pinned)
    P_n2 = beam.critical_buckling_load(n=2, K=1.0)
    assert math.isclose(P_n2 / P_pinned, 4.0, rel_tol=1e-9)

    print(
        f"[Bug#3] K=1.0→{P_pinned:.0f}N, K=0.5→{P_fixed_fixed:.0f}N (4×), "
        f"K=2.0→{P_cantilever:.0f}N (¼×), K=0.7→{P_pinned_fixed:.0f}N — OK"
    )


def test_critical_buckling_load_invalid_K():
    """K ≤ 0 ValueError vermeli."""
    mat = MaterialProperties(E=200e9, nu=0.3, h=0.005)
    beam = BeamNeuron3D(length=3.0, resolution=64, I=1e-6, A=1e-3, material=mat)

    for bad_K in (0.0, -1.0, -0.5):
        try:
            beam.critical_buckling_load(K=bad_K)
        except ValueError:
            continue
        raise AssertionError(f"K={bad_K} ValueError vermeli")
    print("[Bug#3] Geçersiz K (≤0) → ValueError — OK")


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_all():
    tests = [
        test_kirchhoff_stress_includes_poisson_coupling,
        test_shear_stiffness_uses_kappa_and_material_nu,
        test_kappa_is_actually_used,
        test_legacy_api_still_works_with_warning,
        test_critical_buckling_load_K_factor,
        test_critical_buckling_load_invalid_K,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {t.__name__}: {exc}")
    print()
    if failed:
        print(f"=== {failed}/{len(tests)} test KALDI ===")
        sys.exit(1)
    print(f"=== TÜM {len(tests)} TEST GEÇTİ ===")


if __name__ == "__main__":
    run_all()
