"""
Kirsch delik alanı, J2 plastisite ve çekme-basma enerji ayrışımı testleri.

Bu üç blok örnek 3'ün (delikli plaka kırılma + 45 malzemelik tersine
mühendislik) fizik çekirdeğidir ve betikten SPINE'a taşınmıştır. Süit iki
şeyi birden sınar:

1. KAPALI FORM: Kirsch çözümü klasik SCF = 3 değerini veriyor mu, J2 geri
   dönüşü akma yüzeyine oturuyor mu, enerji ayrışımı basmada sıfır mı.
2. TÜREVLENEBİLİRLİK: bu blokların hepsi ters problemde kullanıldığı için
   parametrelere göre gradyan gerçekten akmalı. Gradyanı koparan bir
   değişiklik burada yakalanır.
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spine import (  # noqa: E402
    PlasticityNeuron,
    StrainNeuron,
    kirsch_hole_stress,
)

DT = torch.float64


def _grid(n=401, L=0.4):
    x = torch.linspace(-L / 2, L / 2, n, dtype=DT)
    return torch.meshgrid(x, x, indexing="ij")


# =============================================================================
# 1) KIRSCH — dairesel delik etrafındaki gerilme alanı
# =============================================================================

def test_kirsch_scf_ucdur():
    """Delik kenarında gerilme yoğunlaşma katsayısı tam 3 olmalı.

    x yönünde çekilen sonsuz plakada, delik kenarının θ = ±90° noktasında
    teğetsel gerilme 3σ₀'dır. Teğetsel yön orada x eksenine denk geldiği
    için bu, kartezyen σ_xx bileşeninde görünür.
    """
    X, Y = _grid(n=801)
    a = 0.02
    r = kirsch_hole_stress(1.0e6, X, Y, a_hole=a)

    halka = (r["r"] >= a) & (r["r"] <= a * 1.02)
    scf = r["sigma_xx"][halka].max().item() / 1.0e6
    assert abs(scf - 3.0) < 1e-3, f"SCF = {scf:.6f}, 3.0 bekleniyordu"


def test_kirsch_theta_sifirda_basma():
    """θ = 0 (yükleme ekseni üzerinde) delik kenarında σ_θθ = -σ₀.

    Çekme yönünde delik kenarı BASMA görür; bu Kirsch çözümünün klasik
    ve sezgiye aykırı sonucudur.
    """
    X, Y = _grid(n=801)
    a = 0.02
    r = kirsch_hole_stress(1.0e6, X, Y, a_hole=a)
    halka = (r["r"] >= a) & (r["r"] <= a * 1.02)
    en_kucuk = r["sigma_yy"][halka].min().item() / 1.0e6
    assert abs(en_kucuk + 1.0) < 1e-3, f"σ_yy,min = {en_kucuk:.6f}, -1.0 bekleniyordu"


def test_kirsch_uzak_alanda_uygulanan_gerilmeye_doner():
    """Delikten uzakta alan, uygulanan tek eksenli çekmeye yakınsamalı."""
    X, Y = _grid(n=401, L=2.0)
    a = 0.01                       # delik, domaine göre çok küçük
    r = kirsch_hole_stress(1.0e6, X, Y, a_hole=a)

    kose = r["r"] > 0.8            # delikten çok uzak bölge
    assert abs(r["sigma_xx"][kose].mean().item() / 1e6 - 1.0) < 1e-2
    assert abs(r["sigma_yy"][kose].mean().item() / 1e6) < 1e-2


def test_kirsch_delik_ici_maskelenir():
    X, Y = _grid(n=201)
    a = 0.03
    r = kirsch_hole_stress(1.0e6, X, Y, a_hole=a, mask_hole=True)
    assert r["sigma_vm"][r["hole_mask"]].abs().max().item() == 0.0

    ham = kirsch_hole_stress(1.0e6, X, Y, a_hole=a, mask_hole=False)
    assert ham["sigma_vm"][ham["hole_mask"]].abs().max().item() > 0.0


def test_kirsch_gradyan_gecirgen():
    """Uzak alan gerilmesine göre gradyan akmalı (ters problem şartı)."""
    X, Y = _grid(n=101)
    sigma_0 = torch.tensor(1.0e6, dtype=DT, requires_grad=True)
    r = kirsch_hole_stress(sigma_0, X, Y, a_hole=0.02)
    r["sigma_vm"].max().backward()

    assert sigma_0.grad is not None
    # Alan σ₀ ile doğrusal ölçeklenir → türev pozitif ve sonlu
    assert sigma_0.grad.item() > 0
    assert math.isfinite(sigma_0.grad.item())


# =============================================================================
# 2) J2 PLASTİSİTE — radyal geri dönüş
# =============================================================================

def test_j2_akma_yuzeyine_oturur():
    """Akma aşıldığında geri dönüş sonrası σ_vm tam σ_y olmalı."""
    sy = 300e6
    pn = PlasticityNeuron(sigma_y=sy).to(DT)

    # Saf çekme: σ_vm = σ_xx
    sxx = torch.tensor([[500e6]], dtype=DT)
    syy = torch.zeros_like(sxx)
    txy = torch.zeros_like(sxx)

    r = pn.return_map(sxx, syy, txy)
    assert abs(r["sigma_vm"].item() - sy) / sy < 1e-9
    assert abs(r["sigma_vm_trial"].item() - 500e6) / 500e6 < 1e-12
    assert abs(r["yield_ratio"].item() - 500 / 300) < 1e-6


def test_j2_elastik_bolgede_dokunmaz():
    """Akma altındaki gerilme değiştirilmemeli."""
    pn = PlasticityNeuron(sigma_y=300e6).to(DT)
    sxx = torch.tensor([[100e6]], dtype=DT)
    syy = torch.tensor([[50e6]], dtype=DT)
    txy = torch.tensor([[20e6]], dtype=DT)

    r = pn.return_map(sxx, syy, txy)
    assert torch.allclose(r["sigma_xx"], sxx)
    assert torch.allclose(r["sigma_yy"], syy)
    assert torch.allclose(r["tau_xy"], txy)
    assert r["plastic_fraction"].item() == 0.0


def test_j2_deviatorik_mod_yuzeyi_asar():
    """'deviatoric' modu düzlem gerilmede akma yüzeyinin DIŞINDA kalır.

    Bu, örnek 3'ün betiğinde kullanılan formdur ve bilinen bir sapmadır:
    hidrostatik terim (σ_xx+σ_yy)/3 ile alınırken σ_zz = 0 kısıtı gözardı
    edilir. Test, sapmanın büyüklüğünü sabitler ki sessizce değişmesin.
    """
    sy = 300e6
    pn = PlasticityNeuron(sigma_y=sy).to(DT)
    sxx = torch.tensor([[500e6]], dtype=DT)
    z = torch.zeros_like(sxx)

    r_dev = pn.return_map(sxx, z, z, mode="deviatoric")
    r_rad = pn.return_map(sxx, z, z, mode="radial")

    assert r_rad["sigma_vm"].item() == pytest.approx(sy, rel=1e-12), \
        "radial mod akma yüzeyine tam oturmalı"
    asim = r_dev["sigma_vm"].item() / sy - 1
    assert 0.12 < asim < 0.14, f"deviatoric aşımı %{asim*100:.2f}, ~%12,8 bekleniyordu"

    # Aşımın kaynağı: düzlem gerilmede (σ_xx+σ_yy)/3 ile ayrılan "deviatörik"
    # kısmın izi sıfır DEĞİLDİR (σ_zz = 0 kısıtı hesaba katılmadığı için),
    # dolayısıyla ölçekleme von Mises ölçüsünü olduğu gibi küçültmez.
    sigma_h = ((sxx + z) / 3.0).item()
    s_iz = (sxx.item() - sigma_h) + (z.item() - sigma_h)
    assert abs(s_iz) > 1e6, \
        "düzlem gerilmede deviatörik izin sıfırdan farklı olması beklenir"

    # Her iki mod da elastik bölgede aynı sonucu vermeli
    kucuk = torch.tensor([[100e6]], dtype=DT)
    e_dev = pn.return_map(kucuk, z, z, mode="deviatoric")
    e_rad = pn.return_map(kucuk, z, z, mode="radial")
    assert torch.allclose(e_dev["sigma_xx"], e_rad["sigma_xx"])


def test_j2_sigma_y_gradyani_akar():
    """Akma gerilmesine göre gradyan akmalı — ters problemde σ_y kestirimi."""
    sy = torch.tensor(300e6, dtype=DT, requires_grad=True)
    pn = PlasticityNeuron(sigma_y=300e6).to(DT)

    sxx = torch.full((8, 8), 500e6, dtype=DT)
    syy = torch.zeros_like(sxx)
    txy = torch.zeros_like(sxx)

    r = pn.return_map(sxx, syy, txy, sigma_y=sy)
    r["sigma_vm"].mean().backward()

    assert sy.grad is not None, "σ_y'ye gradyan ulaşmadı"
    # Akma gerilmesi artarsa geri dönüş sonrası gerilme de artar
    assert sy.grad.item() > 0


def test_j2_plastik_oran_dogru_sayar():
    pn = PlasticityNeuron(sigma_y=100e6).to(DT)
    sxx = torch.tensor([[50e6, 200e6], [300e6, 10e6]], dtype=DT)
    sifir = torch.zeros_like(sxx)
    r = pn.return_map(sxx, sifir, sifir)
    assert abs(r["plastic_fraction"].item() - 0.5) < 1e-12


# =============================================================================
# 3) ÇEKME–BASMA ENERJİ AYRIŞIMI (faz-alanı sürücüsü)
# =============================================================================

def test_enerji_basmada_sifir():
    """Saf basma altında ψ⁺ sıfır olmalı — basma çatlak sürmez."""
    E, nu = 70e9, 0.3
    sxx = torch.tensor([[-100e6]], dtype=DT)
    syy = torch.tensor([[-100e6]], dtype=DT)
    txy = torch.zeros_like(sxx)

    psi = StrainNeuron.energy_density_split(sxx, syy, txy, E, nu)
    assert psi.item() < 1e-6, f"basmada ψ⁺ = {psi.item():.3e}, ~0 bekleniyordu"


def test_enerji_cekmede_pozitif_ve_kapali_formla_uyumlu():
    """Tek eksenli çekmede ψ⁺ elastik enerjiyle uyumlu olmalı."""
    E, nu = 70e9, 0.3
    s = 100e6
    sxx = torch.tensor([[s]], dtype=DT)
    syy = torch.zeros_like(sxx)
    txy = torch.zeros_like(sxx)

    psi = StrainNeuron.energy_density_split(sxx, syy, txy, E, nu).item()
    assert psi > 0

    # Kapalı form: ε_1 = s/E, ε_2 = -ν s/E (negatif → pozitif kısma girmez)
    lam_2d = E * nu / (1 - nu**2)
    mu = E / (2 * (1 + nu))
    e1 = s / E
    e2 = -nu * s / E
    beklenen = (lam_2d / 2) * max(e1 + e2, 0.0) ** 2 + mu * (max(e1, 0.0) ** 2
                                                             + max(e2, 0.0) ** 2)
    assert abs(psi - beklenen) / beklenen < 1e-9


def test_enerji_cekme_basmadan_buyuk():
    """Aynı büyüklükte çekme, basmadan belirgin biçimde fazla enerji üretmeli.

    Tek eksenli BASMADA ψ⁺ tam sıfır değildir: Poisson etkisiyle enine yönde
    çekme gerinimi doğar (ε₂ = -ν·σ/E > 0). Sıfır beklemek yanlış olur;
    doğru beklenti, çekmenin basmayı bir mertebe aşmasıdır. İki eksenli saf
    basmada ise ψ⁺ gerçekten sıfırdır (bkz. test_enerji_basmada_sifir).
    """
    E, nu = 70e9, 0.3
    txy = torch.zeros(1, 1, dtype=DT)
    sifir = torch.zeros(1, 1, dtype=DT)
    cekme = StrainNeuron.energy_density_split(
        torch.tensor([[200e6]], dtype=DT), sifir, txy, E, nu).item()
    basma = StrainNeuron.energy_density_split(
        torch.tensor([[-200e6]], dtype=DT), sifir, txy, E, nu).item()

    assert cekme > 10 * basma, f"oran {cekme/basma:.2f}, en az 10 bekleniyordu"
    # Basmadaki artık enerji yalnız Poisson genleşmesinden gelmeli:
    #   ε₂ = -ν σ/E → ψ⁺ = µ ε₂²
    mu = E / (2 * (1 + nu))
    beklenen_basma = mu * (nu * 200e6 / E) ** 2
    assert abs(basma - beklenen_basma) / beklenen_basma < 1e-9


def test_enerji_E_ve_nu_gradyani_akar():
    """E ve ν'ye göre gradyan akmalı — 45 malzemelik ters problemin şartı."""
    E = torch.tensor(70e9, dtype=DT, requires_grad=True)
    nu = torch.tensor(0.3, dtype=DT, requires_grad=True)

    sxx = torch.full((4, 4), 150e6, dtype=DT)
    syy = torch.full((4, 4), 40e6, dtype=DT)
    txy = torch.full((4, 4), 20e6, dtype=DT)

    psi = StrainNeuron.energy_density_split(sxx, syy, txy, E, nu)
    psi.sum().backward()

    assert E.grad is not None and nu.grad is not None
    assert math.isfinite(E.grad.item()) and math.isfinite(nu.grad.item())
    # Rijitlik artarken aynı gerilme daha az gerinim → enerji azalır
    assert E.grad.item() < 0


# =============================================================================
# 4) ZİNCİR: Kirsch → J2 → enerji, uçtan uca gradyan
# =============================================================================

def test_uctan_uca_gradyan_zinciri():
    """σ₀, σ_y, E, ν → ψ⁺ zincirinin tamamı türevlenebilir olmalı.

    Örnek 3'ün tersine mühendisliği tam olarak bu zinciri geriye doğru
    çözer; herhangi bir halkada gradyanın kopması ters problemi bozar.
    """
    X, Y = _grid(n=81, L=0.2)

    sigma_0 = torch.tensor(300e6, dtype=DT, requires_grad=True)
    sigma_y = torch.tensor(250e6, dtype=DT, requires_grad=True)
    E = torch.tensor(70e9, dtype=DT, requires_grad=True)
    nu = torch.tensor(0.3, dtype=DT, requires_grad=True)

    alan = kirsch_hole_stress(sigma_0, X, Y, a_hole=0.02)
    pn = PlasticityNeuron(sigma_y=250e6).to(DT)
    plastik = pn.return_map(alan["sigma_xx"], alan["sigma_yy"],
                            alan["tau_xy"], sigma_y=sigma_y)
    psi = StrainNeuron.energy_density_split(
        plastik["sigma_xx"], plastik["sigma_yy"], plastik["tau_xy"], E, nu)

    psi.mean().backward()

    for ad, p in (("σ₀", sigma_0), ("σ_y", sigma_y), ("E", E), ("ν", nu)):
        assert p.grad is not None, f"{ad} gradyanı yok"
        assert math.isfinite(p.grad.item()), f"{ad} gradyanı sonsuz/NaN"
        assert p.grad.item() != 0.0, f"{ad} gradyanı sıfır — zincir kopuk"
