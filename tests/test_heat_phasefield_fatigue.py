"""
Isı iletimi, faz-alanı hasar ve yorulma nöronlarının testleri.

Bu üç blok örnek 4'ün (4b ısı iletimi, 4c faz-alanı çatlak, 4d türbin bıçağı
yorulması) fizik çekirdeğidir ve betiklerden SPINE'a taşınmıştır.

Testler kapalı-form sağlamalarına dayanır: türevi bilinen bir alan, tam
sağlanan bir Poisson denklemi, elle çözülebilen bir S-N noktası. Böylece
"kod çalışıyor" değil, "fizik doğru" sınanmış olur.
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spine import (  # noqa: E402
    FatigueNeuron,
    HeatConductionNeuron,
    MaterialProperties,
    PhaseFieldNeuron,
)

DT = torch.float64


# =============================================================================
# 1) ISI İLETİMİ
# =============================================================================

def _kubik_sicaklik(N=161, L=2.0):
    r"""Kitap Eq. 6.49 alanı: T = -(5/6)(x³+y³) + 3x²y + 3xy².

    Bu alanın Laplasyeni tam olarak x+y'dir; hem gradyanın hem Laplasyenin
    analitik karşılığı bilindiği için ideal bir sağlama zeminidir.
    """
    x = torch.linspace(0, L, N, dtype=DT)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    T = -(5.0 / 6.0) * (X**3 + Y**3) + 3 * X**2 * Y + 3 * X * Y**2
    return T, X, Y, L, N


def test_isi_gradyani_analitikle_uyusur():
    T, X, Y, L, N = _kubik_sicaklik()
    hn = HeatConductionNeuron(resolution=N, Lx=L, Ly=L, k=1.0).to(DT)

    gx, gy = hn.gradient_fd(T)
    ax = -2.5 * X**2 + 6 * X * Y + 3 * Y**2      # ∂T/∂x
    ay = -2.5 * Y**2 + 3 * X**2 + 6 * X * Y      # ∂T/∂y

    for sayisal, analitik, ad in ((gx, ax, "x"), (gy, ay, "y")):
        hata = (sayisal - analitik).abs().max().item() / analitik.abs().max().item()
        assert hata < 1e-4, f"{ad} gradyanı sapıyor: {hata:.2e}"


def test_isi_gradyani_ikinci_mertebeden_yakinsar():
    """Sınırlarda tek taraflı fark da ikinci mertebeden olmalı."""
    hatalar = []
    for N in (41, 81, 161):
        T, X, Y, L, _ = _kubik_sicaklik(N=N)
        hn = HeatConductionNeuron(resolution=N, Lx=L, Ly=L).to(DT)
        gx, _ = hn.gradient_fd(T)
        ax = -2.5 * X**2 + 6 * X * Y + 3 * Y**2
        hatalar.append((gx - ax).abs().max().item())

    for kaba, ince in zip(hatalar[:-1], hatalar[1:]):
        mertebe = math.log2(kaba / ince)
        assert 1.7 < mertebe < 2.3, f"yakınsama mertebesi {mertebe:.2f}"


def test_isi_laplasyeni_poisson_denklemini_saglar():
    """∇²T = x + y tam sağlanmalı (analitik alanın tanımı gereği)."""
    T, X, Y, L, N = _kubik_sicaklik()
    hn = HeatConductionNeuron(resolution=N, Lx=L, Ly=L).to(DT)

    rez = hn.poisson_residual(T, X + Y, method="fd")
    ic = rez[1:-1, 1:-1]
    assert ic.abs().max().item() < 1e-8, \
        f"iç bölgede rezidüel {ic.abs().max().item():.2e}"


def test_isi_akisi_isaret_ve_olcek():
    """q = -k∇T: akı sıcaklığın azaldığı yöne akar."""
    N, L = 81, 1.0
    x = torch.linspace(0, L, N, dtype=DT)
    X, _ = torch.meshgrid(x, x, indexing="ij")
    T = 100.0 * X                       # x yönünde artan sıcaklık

    hn = HeatConductionNeuron(resolution=N, Lx=L, Ly=L, k=2.0).to(DT)
    qx, qy = hn.flux(T, method="fd")

    assert (qx < 0).all(), "sıcaklık artarken akı ters yöne olmalı"
    assert abs(qx.mean().item() - (-2.0 * 100.0)) < 1e-6
    assert qy.abs().max().item() < 1e-9


def test_isi_spektral_poisson_cozumu():
    """Periyodik alanda spektral çözüm kaynağı geri üretmeli."""
    N, L = 64, 2 * math.pi
    x = torch.linspace(0, L, N + 1, dtype=DT)[:-1]      # periyodik grid
    X, Y = torch.meshgrid(x, x, indexing="ij")
    T_gercek = torch.sin(X) * torch.cos(Y)
    f = -2.0 * T_gercek                                  # ∇²T = -2T

    hn = HeatConductionNeuron(resolution=N, Lx=L, Ly=L).to(DT)
    T_cozum = hn.solve_spectral(f)

    hata = (T_cozum - T_gercek).abs().max().item()
    assert hata < 1e-10, f"spektral Poisson hatası {hata:.2e}"


def test_isi_gecersiz_yontem():
    hn = HeatConductionNeuron(resolution=16, Lx=1.0, Ly=1.0)
    with pytest.raises(ValueError):
        hn.flux(torch.zeros(16, 16), method="sihirli")


# =============================================================================
# 2) FAZ-ALANI HASAR
# =============================================================================

def _faz_alani(N=64, L=0.05, ell=None, G_c=2700.0):
    ell = ell if ell is not None else 2.0 * L / N
    return PhaseFieldNeuron(resolution=N, Lx=L, Ly=L, ell=ell, G_c=G_c).to(DT)


def test_bozunum_fonksiyonu():
    """g(0) = 1+η (sağlam), g(1) = η (tamamen çatlak)."""
    pf = _faz_alani()
    d0 = torch.zeros(4, 4, dtype=DT)
    d1 = torch.ones(4, 4, dtype=DT)

    assert abs(pf.degradation(d0)[0, 0].item() - (1.0 + pf.eta)) < 1e-15
    assert abs(pf.degradation(d1)[0, 0].item() - pf.eta) < 1e-15
    # Monotonluk: hasar arttıkça rijitlik azalır
    d = torch.linspace(0, 1, 11, dtype=DT)
    g = pf.degradation(d)
    assert (g[1:] < g[:-1]).all()


def test_faz_enerjisi_basmada_sifir():
    """Saf basma gerinimi altında ψ⁺ = 0 — basma çatlak sürmez."""
    pf = _faz_alani()
    e = -1e-4 * torch.ones(4, 4, dtype=DT)
    sifir = torch.zeros(4, 4, dtype=DT)
    psi = pf.energy_density_strain(e, e, sifir, lam=1e10, mu=1e10)
    assert psi.abs().max().item() < 1e-20


def test_faz_enerjisi_cekmede_kapali_formla_uyusur():
    """Eşit iki eksenli çekmede ψ⁺ elle hesaplanabilir."""
    pf = _faz_alani()
    e = 1e-4
    exx = e * torch.ones(2, 2, dtype=DT)
    eyy = exx.clone()
    exy = torch.zeros(2, 2, dtype=DT)
    lam, mu = 1e10, 1e10

    psi = pf.energy_density_strain(exx, eyy, exy, lam, mu)
    beklenen = 0.5 * lam * (2 * e) ** 2 + mu * (e**2 + e**2)
    assert abs(psi[0, 0].item() - beklenen) / beklenen < 1e-12


def test_hasar_geri_donmez():
    """Yük kalksa bile hasar azalamaz (irreversibility)."""
    pf = _faz_alani()
    d_onceki = 0.4 * torch.ones(64, 64, dtype=DT)
    psi_sifir = torch.zeros(64, 64, dtype=DT)

    d_yeni, _ = pf.solve_damage(psi_sifir, d_onceki, mode="at2", n_iter=4)
    assert (d_yeni >= d_onceki - 1e-12).all(), "hasar geri döndü"


def test_hasar_enerjiyle_artar():
    """Daha yüksek sürücü enerji daha çok hasar üretmeli."""
    pf = _faz_alani()
    d0 = torch.zeros(64, 64, dtype=DT)
    kucuk = 1e3 * torch.ones(64, 64, dtype=DT)
    buyuk = 1e5 * torch.ones(64, 64, dtype=DT)

    d_k, _ = pf.solve_damage(kucuk, d0, mode="at2", n_iter=6)
    d_b, _ = pf.solve_damage(buyuk, d0, mode="at2", n_iter=6)
    assert d_b.mean().item() > d_k.mean().item()


def test_hasar_tarihce_maksimumu_tutar():
    """H = max(H_önceki, ψ⁺) — tarihçe düşmez."""
    N = 32
    pf = _faz_alani(N=N)
    d0 = torch.zeros(N, N, dtype=DT)
    yuksek = 5e4 * torch.ones(N, N, dtype=DT)
    dusuk = 1e2 * torch.ones(N, N, dtype=DT)

    _, H1 = pf.solve_damage(yuksek, d0, mode="linear", n_iter=2)
    _, H2 = pf.solve_damage(dusuk, d0, history_H=H1, mode="linear", n_iter=2)
    assert torch.allclose(H2, H1), "tarihçe azaldı"


def test_hasar_Gc_gradyani_akar():
    """G_c'ye göre gradyan akmalı — kırılma tokluğu ters problemi."""
    N = 32
    pf = _faz_alani(N=N)
    G_c = torch.tensor(2700.0, dtype=DT, requires_grad=True)
    psi = 1e4 * torch.ones(N, N, dtype=DT)
    d0 = torch.zeros(N, N, dtype=DT)

    d, _ = pf.solve_damage(psi, d0, G_c=G_c, mode="linear", n_iter=3)
    d.mean().backward()

    assert G_c.grad is not None, "G_c'ye gradyan ulaşmadı"
    assert math.isfinite(G_c.grad.item())
    # Tokluk arttıkça hasar azalır
    assert G_c.grad.item() < 0


def test_hasar_gecersiz_mod():
    pf = _faz_alani()
    with pytest.raises(ValueError):
        pf.solve_damage(torch.zeros(64, 64, dtype=DT),
                        torch.zeros(64, 64, dtype=DT), mode="at1")


# =============================================================================
# 3) YORULMA — Basquin ve Goodman
# =============================================================================

def _yorulma():
    mat = MaterialProperties(E=200e9, nu=0.3, h=0.004)
    return FatigueNeuron(material=mat, S_ut=1240e6, S_e=450e6)


def test_sn_egrisi_uc_noktalari():
    """S-N eğrisi kuruluş noktalarından geçmeli: (10³, 0,9 S_ut) ve (10⁶, S_e)."""
    fn = _yorulma()
    N1 = fn.cycles_to_failure(0.9 * fn.S_ut)
    assert abs(N1 / 1e3 - 1.0) < 1e-9, f"10³ noktası: {N1:.4e}"

    N2 = fn.cycles_to_failure(fn.S_e)
    assert abs(N2 / 1e6 - 1.0) < 1e-6, f"10⁶ noktası: {N2:.4e}"


def test_dayanim_siniri_altinda_sonsuz_omur():
    fn = _yorulma()
    assert math.isinf(fn.cycles_to_failure(fn.S_e * 0.99))
    assert math.isinf(fn.basquin_life(fn.S_e * 0.99, 1500e6, -0.075))


def test_basquin_kapali_formla_uyusur():
    """N_f = ½(σ_a/σ_f')^(1/b) elle doğrulanabilir."""
    fn = _yorulma()
    sigma_f, b = 1500e6, -0.075
    sa = 700e6

    N = fn.basquin_life(sa, sigma_f, b)
    beklenen = ((sa / sigma_f) ** (1.0 / b)) / 2.0
    assert abs(N - beklenen) / beklenen < 1e-12


def test_goodman_cekme_omru_kisaltir_basi_uzatir():
    """Ortalama gerilmenin işareti eşdeğer genliği doğru yönde değiştirmeli."""
    fn = _yorulma()
    sa = 400e6

    duzeltilmis_cekme = fn.goodman_corrected_amplitude(sa, +200e6)
    duzeltilmis_basi = fn.goodman_corrected_amplitude(sa, -200e6)

    assert duzeltilmis_cekme > sa, "çeki ortalama genliği artırmalı"
    assert duzeltilmis_basi < sa, "bası ortalama genliği azaltmalı"

    # Kapalı form: σ_a/(1 - σ_m/S_ut)
    assert abs(duzeltilmis_cekme - sa / (1 - 200e6 / fn.S_ut)) < 1.0
    # Ömür de buna göre kısalmalı
    assert fn.cycles_to_failure(duzeltilmis_cekme) < fn.cycles_to_failure(sa)


def test_goodman_kopma_sinirinda_sonsuz():
    fn = _yorulma()
    assert math.isinf(fn.goodman_corrected_amplitude(100e6, fn.S_ut))
    assert math.isinf(fn.goodman_corrected_amplitude(100e6, fn.S_ut * 1.1))


def test_goodman_kriteri_ile_duzeltme_farkli_sorular():
    """`goodman_criterion` güvenlik katsayısı, diğeri eşdeğer genlik verir.

    İkisinin karıştırılmaması için ayrı ayrı sabitlenir.
    """
    fn = _yorulma()
    sa, sm = 300e6, 150e6

    sf = fn.goodman_criterion(sa, sm)
    genlik = fn.goodman_corrected_amplitude(sa, sm)

    assert abs(sf - 1.0 / (sa / fn.S_e + sm / fn.S_ut)) < 1e-9
    assert abs(genlik - sa / (1 - sm / fn.S_ut)) < 1.0
    assert sf < 10 and genlik > sa
