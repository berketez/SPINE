"""
SPINE Kutuphanesi - Burkulma Testi

Simply supported dikdortgen plaka icin kritik yuk analizi.
Analitik cozumle karsilastirma.
"""

import sys
sys.path.insert(0, '/Users/apple/Desktop/buckneuron')

import torch
import math
from spine import (
    SPINE,
    BoundaryCondition,
    SpectralOps,
    MaterialNeuron,
    GeometryNeuron,
    BucklingNeuron,
    DEVICE
)


def test_critical_load():
    """Kritik yuku analitik formulle karsilastir."""

    print("=" * 60)
    print("SPINE - Burkulma Testi")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print()

    # Malzeme: Celik
    E = 200e9   # Pa (200 GPa)
    nu = 0.3
    h = 0.005   # m (5 mm)

    # Geometri
    Lx = 1.0    # m
    Ly = 0.5    # m

    # Plaka rijitligi (analitik)
    D_analytical = (E * h**3) / (12 * (1 - nu**2))

    print("Malzeme Ozellikleri:")
    print(f"  E  = {E/1e9:.0f} GPa")
    print(f"  nu = {nu}")
    print(f"  h  = {h*1000:.1f} mm")
    print(f"  D  = {D_analytical:.4e} N.m")
    print()

    print("Geometri:")
    print(f"  Lx = {Lx*1000:.0f} mm")
    print(f"  Ly = {Ly*1000:.0f} mm")
    print(f"  Aspect ratio = {Lx/Ly:.2f}")
    print()

    # SPINE modeli olustur
    model = SPINE(
        resolution=64,
        bc_type=BoundaryCondition.SIMPLY_SUPPORTED
    )

    # Model mimarisini yazdir
    model.print_architecture()
    print()

    # Tam analiz yap
    result = model(E=E, nu=nu, h=h, Lx=Lx, Ly=Ly, m=1, n=1)

    print("SPINE Sonuclari:")
    print("-" * 50)
    print(f"  Kritik yuk (N_cr): {result.N_cr.item()/1000:.2f} kN/m")
    print(f"  Plaka rijitligi (D): {result.D.item():.4e} N.m")
    print()

    # Analitik kritik yuk (m=1, n=1)
    m, n = 1, 1
    term_x = (m * math.pi / Lx) ** 2
    term_y = (n * math.pi / Ly) ** 2
    N_cr_analytical = D_analytical * ((term_x + term_y) ** 2) / term_x

    print("Analitik Sonuclar:")
    print("-" * 50)
    print(f"  N_cr (analitik): {N_cr_analytical/1000:.2f} kN/m")
    print()

    # Hata
    error = abs(result.N_cr.item() - N_cr_analytical) / N_cr_analytical * 100
    print(f"Bagil hata: {error:.4f}%")
    print()

    # Kritik modlari bul
    print("Kritik Burkulma Yukleri:")
    print("-" * 50)
    print(f"{'Mod':<6} {'m':<4} {'n':<4} {'N_cr (kN/m)':<15}")
    print("-" * 50)

    modes = model.analyze_buckling(E=E, nu=nu, h=h, Lx=Lx, Ly=Ly, max_modes=5)

    for i, mode in enumerate(modes[:6], 1):
        print(f"{i:<6} {mode['m']:<4} {mode['n']:<4} {mode['N_cr']/1000:<15.2f}")

    print("-" * 50)

    # Mod sekli istatistikleri
    print()
    print("Mod sekli istatistikleri (m=1, n=1):")
    print(f"  max(w)  = {result.w.max().item():.4f}")
    print(f"  min(w)  = {result.w.min().item():.4f}")
    print(f"  mean(w) = {result.w.mean().item():.6f}")

    # Gerilme istatistikleri
    print()
    print("Gerilme istatistikleri:")
    print(f"  max(sigma_xx) = {result.sigma_xx.max().item():.4e} Pa")
    print(f"  max(sigma_yy) = {result.sigma_yy.max().item():.4e} Pa")
    print(f"  max(sigma_xy) = {result.sigma_xy.max().item():.4e} Pa")

    print()
    print("=" * 60)
    if error < 1.0:
        print("TEST BASARILI! Hata < 1%")
    else:
        print(f"TEST UYARISI: Hata = {error:.2f}%")
    print("=" * 60)

    return result.N_cr.item(), N_cr_analytical


def test_spectral_biharmonic():
    """Spektral biharmonik operatoru test et (periyodik domain)."""

    print()
    print("=" * 60)
    print("Spektral Biharmonik Testi (Periyodik Domain)")
    print("=" * 60)

    resolution = 64
    Lx = 2 * math.pi
    Ly = 2 * math.pi

    # SpectralOps
    ops = SpectralOps()

    # Dalga sayilari
    kx = torch.fft.fftfreq(resolution, d=Lx/resolution) * 2 * math.pi
    ky = torch.fft.fftfreq(resolution, d=Ly/resolution) * 2 * math.pi
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    k_squared = KX**2 + KY**2
    k_fourth = k_squared**2

    # Grid (periyodik icin endpoint dahil degil)
    dx = Lx / resolution
    dy = Ly / resolution
    x = torch.arange(0, Lx, dx)
    y = torch.arange(0, Ly, dy)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    # Test fonksiyonu: sin(x) * sin(y)
    # Bu periyodik: f(0) = f(2pi) ve turevleri de oyle
    w = torch.sin(X) * torch.sin(Y)

    # Spektral biharmonik
    biharm_spectral = ops.biharmonic(w, k_fourth)

    # Analitik: nabla4[sin(x)sin(y)] = (1+1)^2 * sin(x)sin(y) = 4 * sin(x)sin(y)
    biharm_analytical = 4.0 * w

    # Hata
    error = torch.abs(biharm_spectral - biharm_analytical).max()
    rel_error = error / torch.abs(biharm_analytical).max()

    print(f"Test fonksiyonu: sin(x) * sin(y)")
    print(f"Domain: [0, 2pi] x [0, 2pi]")
    print(f"Resolution: {resolution} x {resolution}")
    print()
    print(f"Analitik nabla4(w) = 4 * sin(x)sin(y)")
    print()
    print(f"Maksimum mutlak hata: {error:.2e}")
    print(f"Bagil hata: {rel_error:.2e}")
    print()

    if rel_error < 1e-5:
        print("PASSED: Spektral biharmonik dogru calisiyor!")
    else:
        print("FAILED: Hata cok yuksek!")

    print("=" * 60)

    return rel_error < 1e-5


def test_material_neuron():
    """MaterialNeuron'u test et."""

    print()
    print("=" * 60)
    print("MaterialNeuron Testi")
    print("=" * 60)

    neuron = MaterialNeuron()

    # Test degerleri
    E = torch.tensor([200e9])   # Celik
    nu = torch.tensor([0.3])
    h = torch.tensor([0.005])

    # Hesapla
    D = neuron(E, nu, h)

    # Analitik
    D_analytical = (E * h**3) / (12 * (1 - nu**2))

    print(f"E = {E.item()/1e9:.0f} GPa")
    print(f"nu = {nu.item()}")
    print(f"h = {h.item()*1000:.1f} mm")
    print()
    print(f"D (neuron): {D.item():.4e} N.m")
    print(f"D (analitik): {D_analytical.item():.4e} N.m")
    print()

    error = abs(D.item() - D_analytical.item()) / D_analytical.item() * 100
    print(f"Bagil hata: {error:.6f}%")

    if error < 0.01:
        print("PASSED: MaterialNeuron dogru calisiyor!")
    else:
        print("FAILED!")

    print("=" * 60)

    return error < 0.01


def test_full_pipeline():
    """Tam SPINE pipeline'ini test et."""

    print()
    print("=" * 60)
    print("Tam SPINE Pipeline Testi")
    print("=" * 60)

    # Farkli malzemeler
    materials = [
        {"name": "Celik", "E": 200e9, "nu": 0.3, "h": 0.005},
        {"name": "Aluminyum", "E": 70e9, "nu": 0.33, "h": 0.003},
        {"name": "Titanyum", "E": 116e9, "nu": 0.34, "h": 0.004},
    ]

    Lx, Ly = 1.0, 0.5

    model = SPINE(resolution=64)

    print(f"Geometri: {Lx*1000:.0f} mm x {Ly*1000:.0f} mm")
    print()
    print(f"{'Malzeme':<12} {'E (GPa)':<10} {'h (mm)':<8} {'N_cr (kN/m)':<15} {'D (N.m)':<12}")
    print("-" * 65)

    for mat in materials:
        result = model(
            E=mat["E"],
            nu=mat["nu"],
            h=mat["h"],
            Lx=Lx,
            Ly=Ly
        )

        print(f"{mat['name']:<12} {mat['E']/1e9:<10.0f} {mat['h']*1000:<8.1f} "
              f"{result.N_cr.item()/1000:<15.2f} {result.D.item():<12.4e}")

    print("-" * 65)
    print()
    print("PASSED: Tum malzemeler icin hesaplama basarili!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SPINE KUTUPHANE TESTLERI")
    print("=" * 60 + "\n")

    # Test 1: Material Neuron
    test_material_neuron()

    # Test 2: Spektral operator
    test_spectral_biharmonic()

    # Test 3: Burkulma analizi
    test_critical_load()

    # Test 4: Tam pipeline
    test_full_pipeline()

    print("\n" + "=" * 60)
    print("TUM TESTLER TAMAMLANDI!")
    print("=" * 60)
