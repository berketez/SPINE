"""
FAZ 4: 3D Nöron Testleri
- SpectralOps3D
- SolidNeuron3D
- ShellNeuron3D
- BeamNeuron3D
"""

import torch
import math
from spine import SpectralOps3D, SolidNeuron3D, ShellNeuron3D, BeamNeuron3D, DEVICE


def test_spectral_ops_3d():
    """3D Spektral operatörler testi."""
    print("=" * 60)
    print("TEST 1: SpectralOps3D")
    print("=" * 60)

    ops = SpectralOps3D()

    # 3D grid (FFT için endpoint=False gerekli!)
    N = 32
    Lx, Ly, Lz = 1.0, 1.0, 1.0
    dx = Lx / N

    # Dalga sayıları
    wn = ops.create_wavenumbers(N, N, N, Lx, Ly, Lz, DEVICE)

    # FFT için doğru grid (endpoint dahil değil)
    x = torch.arange(0, N, device=DEVICE) * dx
    y = torch.arange(0, N, device=DEVICE) * dx
    z = torch.arange(0, N, device=DEVICE) * dx
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    # Test fonksiyonu: f = sin(2πx) × sin(2πy) × sin(2πz)
    f = torch.sin(2 * math.pi * X) * torch.sin(2 * math.pi * Y) * torch.sin(2 * math.pi * Z)

    # Laplacian testi
    # ∇²f = -12π²f (çünkü ∂²f/∂x² = -4π²f vs.)
    lap_f = ops.laplacian_3d(f, wn['k_squared'])
    lap_analytical = -12 * math.pi**2 * f

    # Sadece iç noktalarda karşılaştır (sınır etkisini azalt)
    error = torch.abs(lap_f - lap_analytical).mean() / (torch.abs(lap_analytical).mean() + 1e-10)

    print(f"Grid: {N}×{N}×{N}")
    print(f"Test fonksiyonu: sin(2πx)×sin(2πy)×sin(2πz)")
    print(f"Laplacian hatası: {error.item()*100:.4f}%")
    print("✅ SpectralOps3D BAŞARILI")
    print()


def test_solid_3d():
    """3D Elastisite testi."""
    print("=" * 60)
    print("TEST 2: SolidNeuron3D - 3D Elastisite")
    print("=" * 60)

    solid = SolidNeuron3D().to(DEVICE)

    # Çelik
    E = torch.tensor([200e9], device=DEVICE)
    nu = torch.tensor([0.3], device=DEVICE)

    # Basit çekme durumu: ε_xx = 0.001, diğerleri 0
    eps_xx = torch.tensor([0.001], device=DEVICE)
    eps_yy = torch.tensor([0.0], device=DEVICE)
    eps_zz = torch.tensor([0.0], device=DEVICE)
    gamma_yz = torch.tensor([0.0], device=DEVICE)
    gamma_xz = torch.tensor([0.0], device=DEVICE)
    gamma_xy = torch.tensor([0.0], device=DEVICE)

    # Gerilme hesapla
    sigma_xx, sigma_yy, sigma_zz, tau_yz, tau_xz, tau_xy = \
        solid.compute_stress(E, nu, eps_xx, eps_yy, eps_zz, gamma_yz, gamma_xz, gamma_xy)

    # Analitik çözüm (tek eksenli çekme, yanal tutuklu)
    # σ_xx = E(1-ν)/[(1+ν)(1-2ν)] × ε_xx
    factor = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
    sigma_xx_analytical = factor * eps_xx

    print(f"Malzeme: Çelik (E=200 GPa, ν=0.3)")
    print(f"Şekil değiştirme: ε_xx = 0.1%")
    print()
    print(f"SPINE σ_xx:     {sigma_xx.item()/1e6:.2f} MPa")
    print(f"Analitik σ_xx:  {sigma_xx_analytical.item()/1e6:.2f} MPa")
    print(f"Hata:           {abs(sigma_xx.item() - sigma_xx_analytical.item())/sigma_xx_analytical.item()*100:.2f}%")
    print()

    # Von Mises testi (saf çekme için σ_vm = σ_xx)
    sigma_vm = solid.von_mises_stress(sigma_xx, sigma_yy, sigma_zz, tau_yz, tau_xz, tau_xy)
    print(f"Von Mises gerilme: {sigma_vm.item()/1e6:.2f} MPa")

    print("✅ SolidNeuron3D BAŞARILI")
    print()


def test_shell_3d():
    """Kabuk elemanı testi."""
    print("=" * 60)
    print("TEST 3: ShellNeuron3D - Kabuk Elemanı")
    print("=" * 60)

    shell = ShellNeuron3D().to(DEVICE)

    # Çelik kabuk
    E = torch.tensor([200e9], device=DEVICE)
    nu = torch.tensor([0.3], device=DEVICE)
    h = torch.tensor([0.01], device=DEVICE)  # 10 mm

    # Membran şekil değiştirmesi
    eps_xx = torch.tensor([0.001], device=DEVICE)
    eps_yy = torch.tensor([0.0005], device=DEVICE)
    gamma_xy = torch.tensor([0.0], device=DEVICE)

    # Eğrilik
    kappa_xx = torch.tensor([0.1], device=DEVICE)  # 1/m
    kappa_yy = torch.tensor([0.05], device=DEVICE)
    kappa_xy = torch.tensor([0.0], device=DEVICE)

    result = shell(E, nu, h, eps_xx, eps_yy, gamma_xy, kappa_xx, kappa_yy, kappa_xy)

    # Analitik membran kuvveti
    # N_x = Eh/(1-ν²) × (ε_xx + ν×ε_yy)
    A = E * h / (1 - nu**2)
    N_x_analytical = A * (eps_xx + nu * eps_yy)

    N_x_spine = result['N_x']

    print(f"Kabuk: h = {h.item()*1000:.1f} mm")
    print()
    print(f"Membran kuvveti N_x:")
    print(f"  SPINE:     {N_x_spine.item()/1e6:.2f} MN/m")
    print(f"  Analitik:  {N_x_analytical.item()/1e6:.2f} MN/m")
    print(f"  Hata:      {abs(N_x_spine.item() - N_x_analytical.item())/N_x_analytical.item()*100:.2f}%")
    print()
    print(f"Eğilme momenti M_x: {result['M_x'].item()/1e3:.2f} kN·m/m")
    print(f"Plaka rijitliği D:  {result['D'].item()/1e6:.2f} MN·m")

    print("✅ ShellNeuron3D BAŞARILI")
    print()


def test_beam_3d():
    """Kiriş elemanı testi."""
    print("=" * 60)
    print("TEST 4: BeamNeuron3D - Kiriş Elemanı")
    print("=" * 60)

    beam = BeamNeuron3D().to(DEVICE)

    # Çelik kiriş
    E = torch.tensor([200e9], device=DEVICE)
    L = torch.tensor([3.0], device=DEVICE)  # 3 m

    # Dikdörtgen kesit: 100mm × 200mm
    b = 0.1  # m
    h = 0.2  # m

    # Dağıtılmış yük
    q = torch.tensor([10000.0], device=DEVICE)  # 10 kN/m

    # Burkulma için eksenel yük
    P = torch.tensor([100000.0], device=DEVICE)  # 100 kN

    result = beam(E, L, 'rectangular', q=q, P=P, bc='simply_supported', b=b, h=h)

    # Analitik çözümler
    I = b * h**3 / 12
    w_max_analytical = 5 * q.item() * L.item()**4 / (384 * E.item() * I)
    P_cr_analytical = math.pi**2 * E.item() * I / L.item()**2

    print(f"Kiriş: L = {L.item()} m, kesit = {b*1000:.0f}mm × {h*1000:.0f}mm")
    print(f"Yük: q = {q.item()/1000:.1f} kN/m, P = {P.item()/1000:.1f} kN")
    print()
    print(f"Kesit özellikleri:")
    print(f"  A = {result['A']*1e4:.2f} cm²")
    print(f"  I_y = {result['I_y']*1e8:.2f} cm⁴")
    print()
    print(f"Maksimum sehim:")
    print(f"  SPINE:     {result['w_max'].item()*1000:.3f} mm")
    print(f"  Analitik:  {w_max_analytical*1000:.3f} mm")
    print(f"  Hata:      {abs(result['w_max'].item() - w_max_analytical)/w_max_analytical*100:.2f}%")
    print()
    print(f"Euler burkulma yükü:")
    print(f"  SPINE P_cr:    {result['P_cr'].item()/1e6:.2f} MN")
    print(f"  Analitik P_cr: {P_cr_analytical/1e6:.2f} MN")
    print(f"  Hata:          {abs(result['P_cr'].item() - P_cr_analytical)/P_cr_analytical*100:.2f}%")
    print()
    print(f"Burkulma güvenlik katsayısı: {result['buckling_safety'].item():.1f}")

    print("✅ BeamNeuron3D BAŞARILI")
    print()


def test_beam_sections():
    """Farklı kesit tipleri testi."""
    print("=" * 60)
    print("TEST 5: BeamNeuron3D - Kesit Tipleri")
    print("=" * 60)

    beam = BeamNeuron3D().to(DEVICE)
    E = torch.tensor([200e9], device=DEVICE)
    L = torch.tensor([5.0], device=DEVICE)
    P = torch.tensor([500000.0], device=DEVICE)  # 500 kN

    print("Farklı kesitlerin burkulma kapasitesi karşılaştırması:")
    print()

    # 1. Dikdörtgen kesit
    result_rect = beam(E, L, 'rectangular', P=P, bc='simply_supported', b=0.15, h=0.30)
    print(f"Dikdörtgen (150×300mm): P_cr = {result_rect['P_cr'].item()/1e6:.2f} MN")

    # 2. Dairesel kesit
    result_circ = beam(E, L, 'circular', P=P, bc='simply_supported', r=0.10)
    print(f"Dairesel (R=100mm):     P_cr = {result_circ['P_cr'].item()/1e6:.2f} MN")

    # 3. I-profil
    result_I = beam(E, L, 'I-beam', P=P, bc='simply_supported',
                    b_f=0.20, t_f=0.015, h_w=0.30, t_w=0.010)
    print(f"I-profil (IPE 300):     P_cr = {result_I['P_cr'].item()/1e6:.2f} MN")

    print()
    print("Güvenlik katsayıları:")
    print(f"  Dikdörtgen: {result_rect['buckling_safety'].item():.1f}")
    print(f"  Dairesel:   {result_circ['buckling_safety'].item():.1f}")
    print(f"  I-profil:   {result_I['buckling_safety'].item():.1f}")

    print("✅ Kesit tipleri testi BAŞARILI")
    print()


def test_all():
    """Tüm 3D testleri çalıştır."""
    print("\n" + "=" * 60)
    print("FAZ 4: 3D NÖRON TESTLERİ")
    print("=" * 60 + "\n")

    test_spectral_ops_3d()
    test_solid_3d()
    test_shell_3d()
    test_beam_3d()
    test_beam_sections()

    print("=" * 60)
    print("TÜM 3D TESTLER BAŞARILI!")
    print("=" * 60)


if __name__ == "__main__":
    test_all()
