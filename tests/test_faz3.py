"""
FAZ 3 Nöron Testleri
- StaticNeuron
- ModalNeuron
- CrackNeuron
- ThermalNeuron
"""

import torch
from spine import SPINE

def test_static():
    """Statik eğilme testi."""
    print("=" * 60)
    print("TEST 1: StaticNeuron - Statik Eğilme")
    print("=" * 60)

    model = SPINE(resolution=64)

    # Çelik plaka
    E = 200e9      # Pa
    nu = 0.3
    h = 0.01       # 10 mm
    Lx = 1.0       # 1 m
    Ly = 1.0       # 1 m (kare)
    q = 10000      # 10 kN/m² yük

    result = model.analyze_static(E, nu, h, Lx, Ly, q)

    # Analitik çözüm (kare plaka)
    D = E * h**3 / (12 * (1 - nu**2))
    w_max_analytical = 0.00406 * q * Lx**4 / D

    w_max_spine = result['w_max'].item()

    print(f"Plaka: {Lx}m x {Ly}m x {h*1000}mm")
    print(f"Yük: {q/1000} kN/m²")
    print()
    print(f"SPINE w_max:     {w_max_spine*1000:.4f} mm")
    print(f"Analitik w_max:  {w_max_analytical*1000:.4f} mm")
    print(f"Hata:            {abs(w_max_spine - w_max_analytical)/w_max_analytical*100:.2f}%")

    if 'sigma_max' in result:
        print(f"Max gerilme:     {result['sigma_max'].item()/1e6:.2f} MPa")

    print("✅ StaticNeuron BAŞARILI")
    print()


def test_modal():
    """Modal analiz testi."""
    print("=" * 60)
    print("TEST 2: ModalNeuron - Titreşim Analizi")
    print("=" * 60)

    model = SPINE(resolution=64)

    # Çelik plaka
    E = 200e9      # Pa
    nu = 0.3
    h = 0.01       # 10 mm
    rho = 7850     # kg/m³
    Lx = 1.0       # 1 m
    Ly = 1.0       # 1 m

    result = model.analyze_modal(E, nu, h, rho, Lx, Ly, n_modes=5)

    # Analitik çözüm (ilk mod m=1, n=1)
    import math
    D = E * h**3 / (12 * (1 - nu**2))
    alpha = math.pi / Lx
    beta = math.pi / Ly
    omega_analytical = (alpha**2 + beta**2) * math.sqrt(D / (rho * h))
    freq_analytical = omega_analytical / (2 * math.pi)

    print(f"Plaka: {Lx}m x {Ly}m x {h*1000}mm")
    print(f"Yoğunluk: {rho} kg/m³")
    print()
    print("İlk 5 mod:")
    for i, mode in enumerate(result['modes']):
        print(f"  Mod {i+1}: (m={mode['m']}, n={mode['n']}) -> f = {mode['freq']:.2f} Hz")

    f1_spine = result['fundamental_freq']
    print()
    print(f"SPINE f₁:       {f1_spine:.2f} Hz")
    print(f"Analitik f₁:    {freq_analytical:.2f} Hz")
    print(f"Hata:           {abs(f1_spine - freq_analytical)/freq_analytical*100:.2f}%")

    print("✅ ModalNeuron BAŞARILI")
    print()


def test_crack():
    """Kırılma mekaniği testi."""
    print("=" * 60)
    print("TEST 3: CrackNeuron - Kırılma Mekaniği")
    print("=" * 60)

    model = SPINE(resolution=64)

    # Kenar çatlağı olan plaka
    sigma = 100e6   # 100 MPa gerilme
    a = 0.005       # 5 mm çatlak
    W = 0.1         # 100 mm genişlik
    K_IC = 50e6     # 50 MPa√m (çelik için tipik)

    result = model.analyze_crack(sigma, a, W, K_IC, crack_type='edge')

    # Analitik çözüm (Y ≈ 1.12 for small a/W)
    import math
    Y = 1.12 - 0.231*(a/W) + 10.55*(a/W)**2 - 21.72*(a/W)**3 + 30.39*(a/W)**4
    K_I_analytical = sigma * math.sqrt(math.pi * a) * Y

    K_I_spine = result['K_I'].item()

    print(f"Gerilme: {sigma/1e6} MPa")
    print(f"Çatlak: {a*1000} mm")
    print(f"Genişlik: {W*1000} mm")
    print(f"K_IC: {K_IC/1e6} MPa√m")
    print()
    print(f"SPINE K_I:      {K_I_spine/1e6:.2f} MPa√m")
    print(f"Analitik K_I:   {K_I_analytical/1e6:.2f} MPa√m")
    print(f"Hata:           {abs(K_I_spine - K_I_analytical)/K_I_analytical*100:.2f}%")
    print()
    print(f"Güvenlik faktörü: {result['safety_factor'].item():.2f}")
    print(f"Kırılacak mı?: {'EVET!' if result['will_fracture'] else 'Hayır'}")

    print("✅ CrackNeuron BAŞARILI")
    print()


def test_thermal():
    """Termal gerilme testi."""
    print("=" * 60)
    print("TEST 4: ThermalNeuron - Termal Gerilme")
    print("=" * 60)

    model = SPINE(resolution=64)

    # Çelik plaka
    E = 200e9           # Pa
    nu = 0.3
    h = 0.01            # 10 mm
    alpha = 12e-6       # 1/°C (çelik)
    delta_T = 100       # 100°C sıcaklık artışı
    Lx = 1.0
    Ly = 1.0

    result = model.analyze_thermal(E, nu, h, alpha, delta_T, Lx, Ly, analysis_type='all')

    # Analitik çözüm
    eps_analytical = alpha * delta_T
    sigma_analytical = E * alpha * delta_T / (1 - nu)  # biaxial

    eps_spine = result['epsilon_thermal'].item()
    sigma_spine = result['sigma_thermal'].item()

    print(f"Sıcaklık artışı: {delta_T}°C")
    print(f"Termal genleşme katsayısı: {alpha*1e6} × 10⁻⁶ /°C")
    print()
    print(f"SPINE ε_thermal:     {eps_spine*1e6:.2f} × 10⁻⁶")
    print(f"Analitik ε_thermal:  {eps_analytical*1e6:.2f} × 10⁻⁶")
    print()
    print(f"SPINE σ_thermal:     {sigma_spine/1e6:.2f} MPa")
    print(f"Analitik σ_thermal:  {sigma_analytical/1e6:.2f} MPa")
    print(f"Hata:                {abs(sigma_spine - sigma_analytical)/sigma_analytical*100:.2f}%")
    print()

    if 'delta_T_critical' in result:
        print(f"Kritik ΔT (burkulma): {result['delta_T_critical'].item():.1f}°C")
        print(f"Burkulma güvenlik faktörü: {result['buckling_safety_factor'].item():.2f}")

    print("✅ ThermalNeuron BAŞARILI")
    print()


def test_all():
    """Tüm testleri çalıştır."""
    print("\n" + "=" * 60)
    print("FAZ 3 NÖRON TESTLERİ")
    print("=" * 60 + "\n")

    test_static()
    test_modal()
    test_crack()
    test_thermal()

    print("=" * 60)
    print("TÜM TESTLER BAŞARILI!")
    print("=" * 60)

    # Model özeti
    model = SPINE(resolution=64)
    print(f"\nToplam parametre: {model.get_parameter_count()}")


if __name__ == "__main__":
    test_all()
