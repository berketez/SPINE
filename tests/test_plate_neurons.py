"""
Plaka nöronlarının kapalı-form doğrulaması ve örnek betiklerle eşdeğerliği.

Bu süitin amacı iki yönlü:

1. FİZİK: `plate_critical_load`, `BucklingNeuron`, `ThermalNeuron` ve
   `StaticNeuron.solve_navier` kapalı formları doğru üretiyor mu.

2. EŞDEĞERLİK: "halit hoca" örnek betiklerinde elle yazılmış olan yerel
   fizik fonksiyonları SPINE nöronlarıyla değiştirildiğinde sonuç DEĞİŞMEMELİ.
   Aşağıdaki referans fonksiyonlar, betiklerin SPINE'a taşınmadan önceki
   hâlinin birebir kopyasıdır; nöron çıktısı bunlarla karşılaştırılır.
   Böylece "SPINE'a taşıdık ama sayılar kaydı" sessiz regresyonu imkânsız olur.
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spine import (  # noqa: E402
    BucklingNeuron,
    MaterialProperties,
    StaticNeuron,
    ThermalNeuron,
    plate_critical_load,
)

TOL = 1e-12


# =============================================================================
# 1) KAPALI FORM: kritik burkulma yükü
# =============================================================================

def test_critical_load_kapali_form():
    """N_cr kapalı formu: uniaxial (iki yön) ve biaxial."""
    D, Lx, Ly = 1234.5, 0.6, 0.3
    m, n = 2, 3
    ax2 = (m * math.pi / Lx) ** 2
    an2 = (n * math.pi / Ly) ** 2
    s = ax2 + an2

    N_x, _ = plate_critical_load(D, Lx, Ly, m, n, loading="uniaxial", direction="x")
    N_y, _ = plate_critical_load(D, Lx, Ly, m, n, loading="uniaxial", direction="y")
    N_b, _ = plate_critical_load(D, Lx, Ly, m, n, loading="biaxial")

    assert abs(N_x - D * s**2 / ax2) / N_x < TOL
    assert abs(N_y - D * s**2 / an2) / N_y < TOL
    assert abs(N_b - D * s) / N_b < TOL


def test_biaxial_kare_plakada_uniaxialin_yarisi():
    """Kare plaka, (1,1): biaxial kritik yük uniaxial'in tam yarısıdır.

    Fiziksel anlamı: iki yönde birden bastırılan plaka, tek yönde
    bastırılana göre yarı yükte burkulur.
    """
    D = 1.0
    N_uni, _ = plate_critical_load(D, 1.0, 1.0, 1, 1, loading="uniaxial")
    N_bi, _ = plate_critical_load(D, 1.0, 1.0, 1, 1, loading="biaxial")
    assert abs(N_uni / N_bi - 2.0) < 1e-12


def test_mindlin_kritik_yuku_dusurur():
    """Enine kayma esnekliği kritik yükü daima düşürür (asla artırmaz)."""
    E, nu, h = 73.1e9, 0.33, 0.01
    D = E * h**3 / (12 * (1 - nu**2))
    N_k, _ = plate_critical_load(D, 0.6, 0.3, 1, 1, loading="biaxial")
    N_m, N_k2 = plate_critical_load(D, 0.6, 0.3, 1, 1, loading="biaxial",
                                    mindlin=True, E=E, nu=nu, h=h)
    assert abs(N_k - N_k2) / N_k < TOL, "Mindlin öncesi değer korunmalı"
    assert N_m < N_k, "Mindlin düzeltmesi kritik yükü düşürmeli"

    G = E / (2 * (1 + nu))
    beklenen = N_k / (1 + N_k / ((5.0 / 6.0) * G * h))
    assert abs(N_m - beklenen) / beklenen < TOL


def test_mindlin_eksik_parametre_hatasi():
    with pytest.raises(ValueError):
        plate_critical_load(1.0, 1.0, 1.0, mindlin=True)


def test_gecersiz_yukleme_tipi():
    with pytest.raises(ValueError):
        plate_critical_load(1.0, 1.0, 1.0, loading="triaxial")
    with pytest.raises(ValueError):
        plate_critical_load(1.0, 1.0, 1.0, direction="z")


# =============================================================================
# 2) BucklingNeuron
# =============================================================================

def test_buckling_neuron_geriye_uyum():
    """Parametresiz eski çağrı, eski (uniaxial-x) sonucu vermeye devam etmeli."""
    mat = MaterialProperties(E=70e9, nu=0.33, h=0.002)
    bn = BucklingNeuron(resolution=16, Lx=1.0, Ly=0.5, material=mat)

    tx = (math.pi / 1.0) ** 2
    ty = (math.pi / 0.5) ** 2
    beklenen = mat.D * (tx + ty) ** 2 / tx
    assert abs(bn.analytical_critical_load(1, 1) - beklenen) / beklenen < TOL


def test_buckling_neuron_mod_taramasi_sirali():
    """find_critical_modes N_cr'ye göre artan sırada dönmeli; ilk eleman kritik."""
    mat = MaterialProperties(E=70e9, nu=0.33, h=0.002)
    bn = BucklingNeuron(resolution=16, Lx=1.0, Ly=0.3, material=mat)

    modes = bn.find_critical_modes(max_m=4, max_n=4)
    assert len(modes) == 16
    N_list = [md["N_cr"] for md in modes]
    assert N_list == sorted(N_list)

    en_kucuk = min(
        bn.analytical_critical_load(m, n)
        for m in range(1, 5) for n in range(1, 5)
    )
    assert abs(modes[0]["N_cr"] - en_kucuk) / en_kucuk < TOL


def test_buckling_neuron_dikdortgen_plakada_kritik_mod_11_degil():
    """Uzun dikdörtgen plakada kritik mod (1,1) değildir — tarama şart.

    Bu test, mod taraması yapmadan (1,1) varsaymanın hatalı olduğunu sabitler.
    """
    mat = MaterialProperties(E=70e9, nu=0.33, h=0.002)
    bn = BucklingNeuron(resolution=16, Lx=3.0, Ly=1.0, material=mat)
    modes = bn.find_critical_modes(max_m=6, max_n=6)
    assert (modes[0]["m"], modes[0]["n"]) != (1, 1)
    assert modes[0]["N_cr"] < bn.analytical_critical_load(1, 1)


# =============================================================================
# 3) ThermalNeuron — biaxial varsayılanı (güvenlik yönünde bug düzeltmesi)
# =============================================================================

def _termal_kurulum(Lx=1.0, Ly=1.0):
    mat = MaterialProperties(E=73.1e9, nu=0.33, h=0.01)
    return mat, ThermalNeuron(material=mat, alpha=23.2e-6,
                              resolution=16, Lx=Lx, Ly=Ly)


def test_termal_kritik_yuk_varsayilani_biaxial():
    """Varsayılan ΔT_cr biaxial olmalı: tam kısıtlı plakada N_x = N_y = N_T.

    Uniaxial varsayım kritik yükü kare plakada 2 kat fazla, yani GÜVENSİZ
    yönde tahmin ediyordu; bu test o davranışın geri gelmesini engeller.
    """
    mat, tn = _termal_kurulum()
    dT_bi = tn.critical_thermal_buckling(1, 1)
    dT_uni = tn.critical_thermal_buckling(1, 1, loading="uniaxial")

    assert abs(dT_uni / dT_bi - 2.0) < 1e-12
    assert dT_bi < dT_uni, "biaxial kritik sıcaklık daha düşük (güvenli yön) olmalı"

    # ΔT_cr'de membran kuvvetin büyüklüğü kritik yüke eşit olmalı
    tx = ty = (math.pi / 1.0) ** 2
    N_cr_bi = mat.D * (tx + ty)
    assert abs(abs(tn.thermal_membrane_force(dT_bi)) - N_cr_bi) / N_cr_bi < 1e-9


def test_termal_forward_zinciri_tutarli():
    """ε_T → σ_T → N_T → N_cr → SF zinciri kendi içinde tutarlı olmalı."""
    mat, tn = _termal_kurulum(Lx=0.6, Ly=0.3)
    dT = 24.0
    r = tn(dT, max_mode=4)

    assert abs(r["eps_T"] - 23.2e-6 * dT) / (23.2e-6 * dT) < 1e-12
    assert abs(r["sigma_T"] - (-mat.E * r["eps_T"] / (1 - mat.nu))) < 1e-3
    assert abs(r["N_T"] - r["sigma_T"] * mat.h) < 1e-6
    assert r["sigma_T"] < 0, "ısınan kısıtlı plakada bası (negatif) olmalı"
    assert abs(r["SF_thermal"] - r["N_cr"] / abs(r["N_T"])) < 1e-9

    # ΔT_cr'de güvenlik katsayısı tam 1 olmalı
    r_cr = tn(r["delta_T_cr"], max_mode=4)
    assert abs(r_cr["SF_thermal"] - 1.0) < 1e-9


def test_termal_sifir_dt_sonsuz_guvenlik():
    _, tn = _termal_kurulum()
    r = tn(0.0, max_mode=2)
    assert math.isinf(r["SF_thermal"])


# =============================================================================
# 4) StaticNeuron.solve_navier — analitik türev ve P-Δ
# =============================================================================

def _sonlu_fark_hatasi(res, n_modes=5):
    """Analitik w_xx ile merkezi-fark w_xx arasındaki bağıl hata.

    Çift hassasiyet şart: ikinci merkezi fark dx² ile bölündüğü için,
    float32 gridin ~1e-7'lik konum hatası bu karşılaştırmada büyütülür ve
    gerçek kesme hatasını gölgeler.
    """
    mat = MaterialProperties(E=70e9, nu=0.3, h=0.01)
    sn = StaticNeuron(resolution=res, Lx=1.0, Ly=1.0, material=mat,
                      n_modes=n_modes, dtype=torch.float64)
    out = sn.solve_navier(q=1000.0, n_modes=n_modes)
    w = out["w"]
    dx = 1.0 / (res - 1)
    fd_xx = (w[2:, 1:-1] - 2 * w[1:-1, 1:-1] + w[:-2, 1:-1]) / dx**2
    an_xx = out["w_xx"][1:-1, 1:-1]
    return (fd_xx - an_xx).abs().max().item() / an_xx.abs().max().item()


def test_navier_turevleri_analitik_dogru():
    """w_xx gerçekten ∂²w/∂x² mi — sonlu farkla ikinci mertebeden yakınsamalı.

    Analitik türev doğruysa, merkezi farkın ondan sapması O(dx²) olmalıdır:
    grid iki katına çıkınca hata ~4 kat azalır. Sabit bir toleransa bakmak
    yerine bu yakınsama mertebesini ölçmek, türevin doğruluğunun asıl kanıtı.
    """
    h1 = _sonlu_fark_hatasi(101)
    h2 = _sonlu_fark_hatasi(201)
    h3 = _sonlu_fark_hatasi(401)

    assert h3 < h2 < h1, f"hata azalmıyor: {h1:.2e} → {h2:.2e} → {h3:.2e}"
    for kaba, ince in ((h1, h2), (h2, h3)):
        mertebe = math.log2(kaba / ince)
        assert 1.8 < mertebe < 2.2, \
            f"yakınsama mertebesi {mertebe:.2f}, 2'ye yakın olmalı"
    assert h3 < 1e-3, f"en ince gridde hata hâlâ büyük: {h3:.2e}"


def test_navier_kappa_isaret_konvansiyonu():
    """κ_ij = -w_ij konvansiyonu StrainNeuron ile aynı olmalı."""
    mat = MaterialProperties(E=70e9, nu=0.3, h=0.01)
    sn = StaticNeuron(resolution=32, Lx=1.0, Ly=1.0, material=mat, n_modes=3)
    out = sn.solve_navier(q=1000.0, n_modes=3)
    assert torch.allclose(out["kappa_xx"], -out["w_xx"])
    assert torch.allclose(out["kappa_yy"], -out["w_yy"])
    assert torch.allclose(out["kappa_xy"], -out["w_xy"])


def test_navier_pdelta_basi_deplasmani_buyutur():
    """Düzlem içi bası arttıkça deplasman büyümeli; N=0'da büyütme yok."""
    mat = MaterialProperties(E=70e9, nu=0.3, h=0.01)
    sn = StaticNeuron(resolution=32, Lx=1.0, Ly=1.0, material=mat, n_modes=5)

    w0 = sn.solve_navier(q=1000.0, n_modes=5)
    N_cr, _ = plate_critical_load(mat.D, 1.0, 1.0, 1, 1, loading="biaxial")
    w_half = sn.solve_navier(q=1000.0, N_x=0.5 * N_cr, N_y=0.5 * N_cr, n_modes=5)

    assert abs(w0["r_max"]) < 1e-15
    assert not w0["buckled"]
    assert w_half["w_max"].item() > w0["w_max"].item()
    # (1,1) modunda r = 0.5 → o modda büyütme 2×; toplam alan da büyümeli.
    # r_max float32 rijitlik tensöründen türediği için ~1e-7 sapabilir.
    assert abs(w_half["r_max"] - 0.5) < 1e-6


def test_navier_kritik_yukte_burkulma_bayragi():
    """N ≥ N_cr olduğunda çözüm ıraksamamalı, 'buckled' bayrağı kalkmalı."""
    mat = MaterialProperties(E=70e9, nu=0.3, h=0.01)
    sn = StaticNeuron(resolution=32, Lx=1.0, Ly=1.0, material=mat, n_modes=3)
    N_cr, _ = plate_critical_load(mat.D, 1.0, 1.0, 1, 1, loading="biaxial")

    out = sn.solve_navier(q=1000.0, N_x=1.2 * N_cr, N_y=1.2 * N_cr, n_modes=3)
    assert out["buckled"]
    assert torch.isfinite(out["w"]).all(), "burkulma sonrası çözüm sonlu kalmalı"


# =============================================================================
# 5) EŞDEĞERLİK: örnek betiklerinin taşınmadan önceki fizik fonksiyonları
# =============================================================================

def _ornek1_navier_pdelta_referans(F_compression, E_val, nu_val, h_val,
                                   Lx, Ly, RES, rho, g, n_modes=15,
                                   dtype=torch.float32):
    """örnek1/fracture_pla_compression.py içindeki özgün fonksiyonun kopyası."""
    D = E_val * h_val**3 / (12 * (1 - nu_val**2))
    q = rho * g * h_val
    G = E_val / (2 * (1 + nu_val))
    kappa_s = 5.0 / 6.0

    x = torch.linspace(0, Lx, RES, dtype=dtype)
    y = torch.linspace(0, Ly, RES, dtype=dtype)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    w = torch.zeros(RES, RES, dtype=dtype)
    w_xx = torch.zeros_like(w)
    w_yy = torch.zeros_like(w)
    w_xy = torch.zeros_like(w)

    for m in range(1, n_modes + 1, 2):
        for n in range(1, n_modes + 1, 2):
            am = m * math.pi / Lx
            an = n * math.pi / Ly
            q_mn = 16 * q / (math.pi**2 * m * n)
            W0 = q_mn / (D * (am**2 + an**2) ** 2)
            N_cr_mn = D * (am**2 + an**2) ** 2 / an**2
            ratio = F_compression / N_cr_mn
            amp = 1.0 / (1.0 - min(ratio, 0.99)) if ratio < 1.0 else 50.0
            mindlin = 1.0 + (am**2 + an**2) * D / (kappa_s * G * h_val)
            W = W0 * amp * mindlin
            sm, sn_ = torch.sin(am * X), torch.sin(an * Y)
            cm, cn_ = torch.cos(am * X), torch.cos(an * Y)
            w = w + W * sm * sn_
            w_xx = w_xx + (-am**2) * W * sm * sn_
            w_yy = w_yy + (-an**2) * W * sm * sn_
            w_xy = w_xy + (am * an) * W * cm * cn_
    return w, w_xx, w_yy, w_xy


def _ornek2_thermal_navier_referans(delta_T, E_val, nu_val, alpha_val, h_val,
                                    a_val, b_val, RES, rho, g, n_modes=15,
                                    dtype=torch.float32):
    """örnek2/thermal_buckling_panel.py içindeki özgün fonksiyonun kopyası."""
    D = E_val * h_val**3 / (12 * (1 - nu_val**2))
    G = E_val / (2 * (1 + nu_val))
    kappa_s = 5.0 / 6.0
    N_T = E_val * alpha_val * delta_T * h_val / (1 - nu_val)
    q = rho * g * h_val

    x = torch.linspace(0, a_val, RES, dtype=dtype)
    y = torch.linspace(0, b_val, RES, dtype=dtype)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    w = torch.zeros(RES, RES, dtype=dtype)
    w_xx = torch.zeros_like(w)
    w_yy = torch.zeros_like(w)
    w_xy = torch.zeros_like(w)

    for m_i in range(1, n_modes + 1, 2):
        for n_i in range(1, n_modes + 1, 2):
            am = m_i * math.pi / a_val
            an = n_i * math.pi / b_val
            am2, an2 = am**2, an**2
            q_mn = 16 * q / (math.pi**2 * m_i * n_i)
            W0 = q_mn / (D * (am2 + an2) ** 2)
            N_cr_mn = D * (am2 + an2)
            N_cr_mn_mindlin = N_cr_mn / (1 + N_cr_mn / (kappa_s * G * h_val))
            ratio = N_T / N_cr_mn_mindlin
            amp = 1.0 / (1.0 - ratio) if ratio < 1.0 else 50.0
            mindlin = 1.0 + (am2 + an2) * D / (kappa_s * G * h_val)
            W = W0 * amp * mindlin
            sm, sn_ = torch.sin(am * X), torch.sin(an * Y)
            cm, cn_ = torch.cos(am * X), torch.cos(an * Y)
            w = w + W * sm * sn_
            w_xx = w_xx + (-am2) * W * sm * sn_
            w_yy = w_yy + (-an2) * W * sm * sn_
            w_xy = w_xy + (am * an) * W * cm * cn_
    return w, w_xx, w_yy, w_xy


def _en_buyuk_bagil_fark(A, B):
    olcek = B.abs().max().item()
    return (A - B).abs().max().item() / olcek if olcek > 0 else 0.0


# Eşdeğerlik iki hassasiyette sınanır:
#   float64 → BİT-BİT aynı olmalı (fizik ve işlem sırası birebir aynı)
#   float32 → yalnızca yuvarlama mertebesinde sapabilir
# Nöron artık gradyan-geçirgen tensör aritmetiği kullandığı için (ters problem
# şart koşuyor) float32'de son bitler referansın Python-float aritmetiğinden
# ayrışır; fizik aynıdır, bunu float64 katmanı kanıtlar.
# float32 toleransı 1e-5'tir çünkü burkulma sınırına yakın hâllerde
# (ΔT = 36 K → SF ≈ 1,02) P-Δ büyütmesi 1/(1-r) ile ~40 kata çıkar ve
# yuvarlama hatasını aynı oranda büyütür. Bu, çözümün o bölgede kötü
# koşullanmasının doğal sonucudur; fizik eşitliğini float64 katmanı kanıtlar.
ESDEGERLIK_HASSASIYET = [
    (torch.float64, 1e-12),
    (torch.float32, 1e-5),
]


@pytest.mark.parametrize("dtype,tol", ESDEGERLIK_HASSASIYET)
@pytest.mark.parametrize("F", [500e3, 1500e3, 3000e3])
def test_esdegerlik_ornek1_pla_plaka(F, dtype, tol):
    """SPINE nöronu, örnek1'in yerel Navier fonksiyonuyla aynı sonucu vermeli.

    Yükleme: y yönünde tek eksenli bası (N_x=0, N_y=F), Mindlin çarpanı açık,
    P-Δ paydası Kirchhoff (örnek1'in özgün seçimi).
    """
    Lx, Ly, h, RES = 0.5, 1.0, 0.05, 64
    E, nu, rho, g = 3.5e9, 0.36, 1250.0, 9.81

    ref = _ornek1_navier_pdelta_referans(F, E, nu, h, Lx, Ly, RES, rho, g,
                                         dtype=dtype)
    mat = MaterialProperties(E=E, nu=nu, h=h)
    sn = StaticNeuron(resolution=RES, Lx=Lx, Ly=Ly, material=mat,
                      n_modes=15, dtype=dtype)
    out = sn.solve_navier(q=rho * g * h, N_x=0.0, N_y=F,
                          mindlin=True, pdelta_mindlin=False, n_modes=15)

    for anahtar, referans in zip(["w", "w_xx", "w_yy", "w_xy"], ref):
        fark = _en_buyuk_bagil_fark(out[anahtar], referans)
        assert fark < tol, \
            f"{anahtar} örnek1 referansından saptı ({dtype}): {fark:.2e}"


@pytest.mark.parametrize("dtype,tol", ESDEGERLIK_HASSASIYET)
@pytest.mark.parametrize("delta_T", [0.0, 12.0, 24.0, 36.0])
def test_esdegerlik_ornek2_termal_panel(delta_T, dtype, tol):
    """SPINE nöronu, örnek2'nin yerel termal Navier fonksiyonuyla aynı sonuç.

    Yükleme: iki eksenli termal bası (N_x=N_y=N_T), Mindlin çarpanı açık,
    P-Δ paydası da Mindlin düzeltmeli (örnek2'nin özgün seçimi).
    """
    a, b, t, RES = 0.6, 0.3, 0.01, 64
    E, nu, alpha, rho, g = 73.1e9, 0.33, 23.2e-6, 2780.0, 9.81

    ref = _ornek2_thermal_navier_referans(delta_T, E, nu, alpha, t,
                                          a, b, RES, rho, g, dtype=dtype)
    mat = MaterialProperties(E=E, nu=nu, h=t)
    sn = StaticNeuron(resolution=RES, Lx=a, Ly=b, material=mat,
                      n_modes=15, dtype=dtype)
    N_T = E * alpha * delta_T * t / (1 - nu)
    out = sn.solve_navier(q=rho * g * t, N_x=N_T, N_y=N_T,
                          mindlin=True, pdelta_mindlin=True, n_modes=15)

    for anahtar, referans in zip(["w", "w_xx", "w_yy", "w_xy"], ref):
        fark = _en_buyuk_bagil_fark(out[anahtar], referans)
        assert fark < tol, \
            f"{anahtar} örnek2 referansından saptı ({dtype}): {fark:.2e}"


def test_gradyan_gecirgenligi():
    """Ters problem için rijitlik üzerinden gradyan akmalı.

    `stiffness_modulator` ölçeklenebilir bir rijitlik çarpanıdır; ölçülen bir
    alandan E'yi geri çözmek autograd'ın buradan akmasına bağlıdır. Rijitliğin
    skalara çevrilmesi (eski davranış) bu zinciri sessizce koparıyordu.
    """
    mat = MaterialProperties(E=70e9, nu=0.3, h=0.01)
    sn = StaticNeuron(resolution=32, Lx=1.0, Ly=1.0, material=mat, n_modes=5)

    out = sn.solve_navier(q=1000.0, N_x=1e4, N_y=1e4, n_modes=5)
    assert out["w"].requires_grad, "w gradyan taşımıyor"

    out["w"].abs().max().backward()
    grad = sn.stiffness_modulator.grad
    assert grad is not None, "stiffness_modulator'a gradyan ulaşmadı"
    assert torch.isfinite(grad).all()
    # Rijitlik artarsa deplasman azalır → türev negatif olmalı
    assert grad.item() < 0, f"beklenen işaret negatif, bulunan {grad.item():.3e}"


def test_stiffness_scale_disaridan_gradyan_tasir():
    """Dışarıdan verilen rijitlik çarpanı gradyan zincirini korumalı.

    Ters problemde optimize edilen değişken çözücüye bu parametreyle
    geçirilir. Alternatif yol — `stiffness_modulator`'ı yeni bir
    nn.Parameter ile değiştirmek — zinciri sessizce koparır ve eğilme
    yolundan gelen bilgi kaybolur; bu test o tuzağı kapatır.
    """
    mat = MaterialProperties(E=70e9, nu=0.3, h=0.01)
    sn = StaticNeuron(resolution=32, Lx=1.0, Ly=1.0, material=mat,
                      n_modes=5, dtype=torch.float64)

    olcek = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    out = sn.solve_navier(q=1000.0, n_modes=5, stiffness_scale=olcek)
    out["w"].abs().max().backward()

    assert olcek.grad is not None, "dışarıdan verilen çarpana gradyan ulaşmadı"
    assert olcek.grad.item() < 0, "rijitlik artarken deplasman azalmalı"

    # Verilen çarpan, modülatörün yerini almalı (değeri gerçekten kullanılmalı)
    with torch.no_grad():
        varsayilan = sn.solve_navier(q=1000.0, n_modes=5)["w"].abs().max()
        olcekli = sn.solve_navier(q=1000.0, n_modes=5,
                                  stiffness_scale=torch.tensor(
                                      2.0, dtype=torch.float64))["w"].abs().max()
    # Yükleme yokken w ∝ 1/D: çarpan 2 ise deplasman yarıya iner
    assert abs(olcekli.item() / varsayilan.item() - 0.5) < 1e-12


def test_esdegerlik_ornek2_kritik_yuk():
    """örnek2'nin biaxial_critical_load taraması ThermalNeuron ile aynı olmalı."""
    a, b, t = 0.6, 0.3, 0.01
    E, nu, alpha = 73.1e9, 0.33, 23.2e-6
    D = E * t**3 / (12 * (1 - nu**2))
    G = E / (2 * (1 + nu))
    kappa_s = 5.0 / 6.0

    # örnek2'nin özgün taraması
    en_iyi, bm, bn_ = float("inf"), 1, 1
    for m in range(1, 7):
        for n in range(1, 7):
            am2 = (m * math.pi / a) ** 2
            an2 = (n * math.pi / b) ** 2
            N_k = D * (am2 + an2)
            N_m = N_k / (1 + N_k / (kappa_s * G * t))
            if N_m < en_iyi:
                en_iyi, bm, bn_ = N_m, m, n

    mat = MaterialProperties(E=E, nu=nu, h=t)
    tn = ThermalNeuron(material=mat, alpha=alpha, resolution=16, Lx=a, Ly=b)
    tarama = tn.critical_mode_scan(6, 6, loading="biaxial", mindlin=True)

    assert (tarama["m"], tarama["n"]) == (bm, bn_)
    assert abs(tarama["N_cr"] - en_iyi) / en_iyi < TOL


# =============================================================================
# 6) Şeffaflık (explain) modu — çıktı üretmeli ve sonucu değiştirmemeli
# =============================================================================

def test_explain_sonucu_degistirmez(capsys):
    """explain=True yalnızca yazdırmalı; döndürülen sayılar aynı kalmalı."""
    mat = MaterialProperties(E=73.1e9, nu=0.33, h=0.01)
    bn = BucklingNeuron(resolution=16, Lx=0.6, Ly=0.3, material=mat)
    tn = ThermalNeuron(material=mat, alpha=23.2e-6, resolution=16, Lx=0.6, Ly=0.3)
    sn = StaticNeuron(resolution=32, Lx=0.6, Ly=0.3, material=mat, n_modes=5)

    sessiz_b = bn.analytical_critical_load(1, 1, loading="biaxial")
    sessiz_t = tn(24.0, max_mode=3)
    sessiz_s = sn.solve_navier(q=273.0, N_x=1e4, N_y=1e4, n_modes=5)
    capsys.readouterr()

    konusan_b = bn.analytical_critical_load(1, 1, loading="biaxial",
                                            explain=True, label="test")
    konusan_t = tn(24.0, max_mode=3, explain=True, label="test")
    konusan_s = sn.solve_navier(q=273.0, N_x=1e4, N_y=1e4, n_modes=5,
                                explain=True, label="test")
    cikti = capsys.readouterr().out

    assert konusan_b == sessiz_b
    assert konusan_t["SF_thermal"] == sessiz_t["SF_thermal"]
    assert torch.equal(konusan_s["w"], sessiz_s["w"])

    # Çıktı gerçekten adımları içermeli
    for beklenen in ["N_cr", "KURULUM", "TERMAL", "MEMBRAN", "Navier"]:
        assert beklenen in cikti, f"explain çıktısında '{beklenen}' yok"


def test_mod_taramasi_explain_kritik_modu_bildirir(capsys):
    mat = MaterialProperties(E=70e9, nu=0.33, h=0.002)
    bn = BucklingNeuron(resolution=16, Lx=3.0, Ly=1.0, material=mat)
    modes = bn.find_critical_modes(max_m=4, max_n=4, explain=True, top_k=3)
    cikti = capsys.readouterr().out
    assert "kritik mod" in cikti
    assert f"({modes[0]['m']},{modes[0]['n']})" in cikti
