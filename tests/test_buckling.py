"""
SPINE — Burkulma ve spektral çekirdek testleri (güncel API).

Eski sürüm silinen bir API'ye (BoundaryCondition, SpectralOps, MaterialNeuron,
GeometryNeuron) karşı yazılmıştı ve pytest koleksiyonunu komple kırıyordu.
Bu sürüm aynı analitik doğrulamaları güncel sınıflarla yapar:
  - Navier kritik yükü ↔ analitik formül
  - Klasik k=4 sonucu (a/b=2 basit mesnetli plaka, kritik mod m=2)
  - Spektral biharmonik/Laplasyen (periyodik domainde, FFT'nin geçerli rejimi)
  - Çok malzemeli tutarlılık

Çalıştırma:
    PYTHONPATH=. python3 tests/test_buckling.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from spine import SPINE, MaterialProperties, SpectralOps2DStruct


def _analytic_navier(D: float, Lx: float, Ly: float, m: int, n: int) -> float:
    """N_cr = D·[(mπ/Lx)² + (nπ/Ly)²]² / (mπ/Lx)²"""
    tx = (m * math.pi / Lx) ** 2
    ty = (n * math.pi / Ly) ** 2
    return D * (tx + ty) ** 2 / tx


def test_critical_load_matches_analytic():
    """get_critical_load, Navier formülüyle birebir eşleşmeli (birkaç modda)."""
    mat = MaterialProperties(E=200e9, nu=0.3, h=0.005)
    Lx, Ly = 1.0, 0.5
    model = SPINE(resolution=64, Lx=Lx, Ly=Ly, material=mat)

    for m, n in [(1, 1), (2, 1), (3, 2), (1, 3)]:
        got = model.get_critical_load(m=m, n=n)
        ref = _analytic_navier(mat.D, Lx, Ly, m, n)
        rel = abs(got - ref) / ref
        assert rel < 1e-9, f"mod({m},{n}): N_cr={got:.6e} vs analitik {ref:.6e}"
    print("[Navier] 4 modda kritik yük analitikle birebir — OK")


def test_classic_k4_aspect_ratio_two():
    """Klasik plaka burkulma sonucu: a/b=2 basit mesnetli plakada kritik mod
    m=2, n=1 ve burkulma katsayısı k = N_cr·b²/(π²D) = 4.0 olmalı
    (Timoshenko & Gere, Theory of Elastic Stability)."""
    mat = MaterialProperties(E=200e9, nu=0.3, h=0.005)
    Lx, Ly = 1.0, 0.5  # a/b = 2
    model = SPINE(resolution=64, Lx=Lx, Ly=Ly, material=mat)

    best = min(
        ((m, n, model.get_critical_load(m=m, n=n))
         for m in range(1, 6) for n in range(1, 4)),
        key=lambda t: t[2],
    )
    m_cr, n_cr, N_cr = best
    k = N_cr * Ly**2 / (math.pi**2 * mat.D)

    assert (m_cr, n_cr) == (2, 1), f"kritik mod ({m_cr},{n_cr}), (2,1) olmalı"
    assert abs(k - 4.0) < 1e-6, f"burkulma katsayısı k={k:.6f}, 4.0 olmalı"
    print(f"[k=4] kritik mod (2,1), k={k:.6f} — OK")


def test_spectral_biharmonic_periodic_domain():
    """FFT operatörleri kendi geçerli rejiminde (tam periyodik sinyal,
    uç-nokta-hariç grid) makine hassasiyetinde olmalı:
    ∇⁴[sin(x)·sin(y)] = 4·sin(x)·sin(y), ∇²[...] = -2·[...] on [0,2π)².

    float64 kullanılır: float32'de FFT taban gürültüsü (~1e-7) k⁴ ile
    ~10⁶ kat amplifiye olup %2'ye çıkar — bu hassasiyet sınırıdır,
    operatör hatası değil.
    """
    N = 64
    L = 2 * math.pi
    ops = SpectralOps2DStruct(resolution=N, Lx=L, Ly=L).double()

    # Periyodik grid: uç nokta HARİÇ (FFT konvansiyonu)
    x = torch.arange(N, dtype=torch.float64) * (L / N)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    w = torch.sin(X) * torch.sin(Y)

    biharm = ops.biharmonic(w)
    lap = ops.laplacian(w)

    rel_b = (torch.norm(biharm - 4.0 * w) / torch.norm(4.0 * w)).item()
    rel_l = (torch.norm(lap + 2.0 * w) / torch.norm(2.0 * w)).item()
    assert rel_b < 1e-9, f"biharmonik bağıl hata {rel_b:.2e}"
    assert rel_l < 1e-9, f"Laplasyen bağıl hata {rel_l:.2e}"
    print(f"[Spektral] periyodik domain (float64): ∇⁴ hata {rel_b:.1e}, ∇² hata {rel_l:.1e} — OK")


def test_multi_material_pipeline():
    """Farklı malzemelerde uçtan uca akış: D formülü, kritik yük ve
    forward geçişi tutarlı olmalı."""
    materials = {
        "çelik": MaterialProperties(E=200e9, nu=0.3, h=0.005),
        "alüminyum": MaterialProperties(E=70e9, nu=0.33, h=0.003),
        "titanyum": MaterialProperties(E=116e9, nu=0.34, h=0.004),
    }
    Lx, Ly = 1.0, 0.5

    for name, mat in materials.items():
        D_ref = mat.E * mat.h**3 / (12 * (1 - mat.nu**2))
        assert abs(mat.D - D_ref) / D_ref < 1e-12, f"{name}: D formülü sapıyor"

        model = SPINE(resolution=64, Lx=Lx, Ly=Ly, material=mat)
        N_cr = model.get_critical_load(m=1, n=1)
        ref = _analytic_navier(mat.D, Lx, Ly, 1, 1)
        assert abs(N_cr - ref) / ref < 1e-9, f"{name}: N_cr sapıyor"
        assert N_cr > 0

        state = model.forward(model.get_mode_shape(1, 1))
        assert torch.isfinite(state.M_xx).all(), f"{name}: M_xx sonlu değil"
        assert abs(state.max_displacement().item() - 1.0) < 0.01, \
            f"{name}: normalize mod şekli max ~1 olmalı"
    print(f"[Malzeme] {len(materials)} malzemede D + N_cr + forward tutarlı — OK")


if __name__ == "__main__":
    test_critical_load_matches_analytic()
    test_classic_k4_aspect_ratio_two()
    test_spectral_biharmonic_periodic_domain()
    test_multi_material_pipeline()
    print("\n=== TÜM 4 TEST GEÇTİ ===")
