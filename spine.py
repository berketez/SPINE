"""
SPINE: Structural Physics-Inherent Neural Engine

Yapısal mekanik için fizik-bilen nöron kütüphanesi.

Temel Felsefe:
    PINN: Fizik → Loss fonksiyonu (soft constraint, hata birikir)
    SPINE: Fizik → Nöronun kendisi (hard constraint, spektral doğruluk)

Kullanım:
    from spine import BiharmonicNeuron, BucklingNeuron, PlateState

    # Tek tek nöron kullanımı
    biharmonic = BiharmonicNeuron(resolution=64, Lx=1.0, Ly=0.5)

    # Burkulma analizi
    buckling = BucklingNeuron(resolution=64, Lx=1.0, Ly=0.5, D=1000.0)
    N_cr = buckling.critical_load(mode_m=1, mode_n=1)

Nöronlar:
    - BiharmonicNeuron  : ∇⁴w biharmonik operatör (plaka eğilmesi)
    - BucklingNeuron    : Kritik yük ve mod şekli tahmini
    - StressNeuron      : σ = C:ε gerilme tensörü
    - StrainNeuron      : ε = ½(∇u + ∇uᵀ) şekil değiştirme
    - BoundaryNeuron    : Sınır koşulları (simply supported, clamped, free)

Yardımcı:
    - SpectralOps2DStruct : Spektral türev operatörleri (yapısal için)
    - SineBasis           : Non-periodic BC için sine serisi
    - PlateState          : Plaka durumu veri yapısı

Yazar: Bitirme Projesi
Lisans: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2, ifft2, fftfreq
import math
import os
import warnings
from typing import Optional, Tuple, Dict, List, Union, Callable
from dataclasses import dataclass
from enum import Enum

# =============================================================================
# MALZEME KÜTÜPHANESİ ENTEGRASYONU
# =============================================================================

try:
    from materials import (
        MaterialLibrary, 
        IsotropicMaterial, 
        OrthotropicMaterial,
        CompositeLaminate,
        Ply,
        MaterialType
    )
    MATERIALS_AVAILABLE = True
except ImportError:
    MATERIALS_AVAILABLE = False
    # materials.py yoksa uyarı ver ama çalışmaya devam et
    warnings.warn(
        "materials.py bulunamadı. MaterialLibrary kullanılamayacak. "
        "Sayısal değerlerle (E, nu, h) çalışmaya devam edebilirsiniz."
    )


# =============================================================================
# CİHAZ SEÇİMİ VE OPTİMİZASYON
# =============================================================================

def get_device() -> torch.device:
    """
    Otomatik cihaz seçimi: CUDA → MPS → CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def check_fft_support(device: torch.device) -> bool:
    """FFT desteğini kontrol et."""
    try:
        test_tensor = torch.randn(4, 4, device=device)
        _ = torch.fft.fft2(test_tensor)
        return True
    except Exception:
        return False


def setup_device_optimizations(device: torch.device) -> None:
    """Cihaza özel optimizasyonları uygula."""
    torch.set_default_dtype(torch.float32)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        if not check_fft_support(device):
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


def safe_fft2(x: torch.Tensor) -> torch.Tensor:
    """Güvenli FFT2 - MPS fallback ile."""
    original_device = x.device
    try:
        return fft2(x)
    except Exception:
        result = fft2(x.cpu())
        return result.to(original_device)


def safe_ifft2(x: torch.Tensor) -> torch.Tensor:
    """Güvenli IFFT2 - MPS fallback ile."""
    original_device = x.device
    try:
        return ifft2(x)
    except Exception:
        result = ifft2(x.cpu())
        return result.to(original_device)


# Global device
DEVICE = get_device()
setup_device_optimizations(DEVICE)


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Veri Yapıları
    'PlateState',
    'MaterialProperties',
    'BoundaryConditionType',

    # Spektral Operatörler (2D)
    'SpectralOps2DStruct',
    'SineBasis',
    
    # Spektral Operatörler (3D)
    'SpectralOps3D',

    # Ortak Plaka Kapalı Formları
    'plate_critical_load',
    'kirsch_hole_stress',

    # Fizik-Bilen Nöronlar (2D - Temel)
    'BiharmonicNeuron',
    'BucklingNeuron',
    'StressNeuron',
    'StrainNeuron',
    'BoundaryNeuron',
    
    # Fizik-Bilen Nöronlar (2D - Faz 3)
    'StaticNeuron',
    'ModalNeuron',
    'CrackNeuron',
    'ThermalNeuron',
    'HeatConductionNeuron',
    'PhaseFieldNeuron',
    'PlasticityNeuron',
    'FatigueNeuron',
    'DynamicNeuron',
    'NonlinearNeuron',
    
    # Fizik-Bilen Nöronlar (3D - Faz 4)
    'SolidNeuron3D',
    'ShellNeuron3D',
    'BeamNeuron3D',

    # Eksenel Çubuk Nöronu (1D - Faz 5b)
    'RodNeuron',

    # Ana Model Container
    'SPINE',

    # Sinir ağları (SPINENet, PhysicsBlock, SpectralEncoder, MultiScaleSpectral)
    # bu modülde tanımlı değildir — benchmarkVelocity.py'den import edilir.

    # Yardımcılar
    'DEVICE',
    'get_device',
    'safe_fft2',
    'safe_ifft2',

    # Karmaşık Geometri (Faz 5)
    'GEOMETRY_AVAILABLE',
    'DEFORMATION_AVAILABLE',
]

# =============================================================================
# KARMAŞIK GEOMETRİ ENTEGRASYONU (Faz 5)
# =============================================================================

try:
    from spine_geometry import GeometryProcessor, MeshData, GridData, MeshGridInterpolator
    GEOMETRY_AVAILABLE = True
except ImportError:
    GEOMETRY_AVAILABLE = False

try:
    from spine_deformation import (
        DeformationNet, DeformedSpectralOps2D, GeometryEncoder,
        ComplexGeometrySolver, MeshToGridLayer, GridToMeshLayer
    )
    DEFORMATION_AVAILABLE = True
except ImportError:
    DEFORMATION_AVAILABLE = False


# =============================================================================
# VERİ YAPILARI
# =============================================================================

class BoundaryConditionType(Enum):
    """Sınır koşulu tipleri"""
    SIMPLY_SUPPORTED = "simply_supported"  # w=0, M=0 (moment sıfır)
    CLAMPED = "clamped"                    # w=0, ∂w/∂n=0 (eğim sıfır)
    FREE = "free"                          # M=0, V=0 (moment ve kesme sıfır)


@dataclass
class MaterialProperties:
    """
    Malzeme özellikleri.

    Attributes:
        E: Young modülü (Pa)
        nu: Poisson oranı (boyutsuz)
        h: Plaka kalınlığı (m)
    """
    E: float      # Young modülü (Pa)
    nu: float     # Poisson oranı
    h: float      # Kalınlık (m)

    @property
    def D(self) -> float:
        """Plaka rijitliği: D = Eh³/[12(1-ν²)]"""
        return (self.E * self.h**3) / (12 * (1 - self.nu**2))

    def __repr__(self):
        return f"MaterialProperties(E={self.E:.2e}, nu={self.nu}, h={self.h}, D={self.D:.2e})"


@dataclass
class PlateState:
    """
    2D Plaka durumu.

    Tensör boyutları: [B, Ny, Nx] (Batch, Y, X)

    Attributes:
        w: Düşey deplasman (çökme)
        theta_x: x yönünde dönme (∂w/∂x)
        theta_y: y yönünde dönme (∂w/∂y)
        M_xx: x yönünde eğilme momenti
        M_yy: y yönünde eğilme momenti
        M_xy: Burulma momenti
    """
    w: torch.Tensor           # Deplasman [B, Ny, Nx]
    theta_x: torch.Tensor     # ∂w/∂x [B, Ny, Nx]
    theta_y: torch.Tensor     # ∂w/∂y [B, Ny, Nx]
    M_xx: torch.Tensor        # Moment-xx [B, Ny, Nx]
    M_yy: torch.Tensor        # Moment-yy [B, Ny, Nx]
    M_xy: torch.Tensor        # Moment-xy [B, Ny, Nx]

    def max_displacement(self) -> torch.Tensor:
        """Maksimum deplasman"""
        return torch.max(torch.abs(self.w))

    def strain_energy(self, D: float, nu: float) -> torch.Tensor:
        """
        Şekil değiştirme enerjisi:
        U = (D/2) ∫∫ [(∇²w)² - 2(1-ν)(w_xx·w_yy - w_xy²)] dA
        """
        # Basitleştirilmiş: U ≈ (D/2) ∫∫ (∇²w)² dA
        # Kirchhoff momentlerinden geri kazanım: M_xx + M_yy = -D(1+ν)·∇²w
        lap_w_sq = ((self.M_xx + self.M_yy) / (D * (1 + nu))) ** 2
        return 0.5 * D * lap_w_sq.mean(dim=(-2, -1))


# =============================================================================
# SPEKTRAL OPERATÖRLER (Yapısal Mekanik İçin)
# =============================================================================

class SpectralOps2DStruct(nn.Module):
    """
    Yapısal mekanik için spektral türev operatörleri.

    - Anizotropik domain (Lx ≠ Ly) desteği
    - Biharmonik (∇⁴) operatörü built-in
    - Yapısal mekanik için optimize

    Args:
        resolution: Grid çözünürlüğü (Nx = Ny = resolution)
        Lx: x yönünde domain uzunluğu (m)
        Ly: y yönünde domain uzunluğu (m)
    """

    def __init__(self, resolution: int, Lx: float = 1.0, Ly: float = 1.0):
        super().__init__()
        self.resolution = resolution
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx / resolution
        self.dy = Ly / resolution

        # Dalga sayıları (anizotropik domain için)
        kx = fftfreq(resolution, d=Lx/resolution) * 2 * math.pi
        ky = fftfreq(resolution, d=Ly/resolution) * 2 * math.pi
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')

        self.register_buffer('kx', KX)
        self.register_buffer('ky', KY)
        self.register_buffer('k_squared', KX**2 + KY**2)
        self.register_buffer('k_fourth', (KX**2 + KY**2)**2)  # Biharmonik için

        # Ayrık bileşenler (anizotropik biharmonik için)
        self.register_buffer('kx2', KX**2)
        self.register_buffer('ky2', KY**2)
        self.register_buffer('kx4', KX**4)
        self.register_buffer('ky4', KY**4)
        self.register_buffer('kx2_ky2', KX**2 * KY**2)

        # Dealiasing filtresi (2/3 kuralı)
        k_max = resolution // 3
        dealias_mask = (torch.abs(KX) < k_max * 2*math.pi/Lx) & \
                       (torch.abs(KY) < k_max * 2*math.pi/Ly)
        self.register_buffer('dealias_mask', dealias_mask.float())

    def gradient(self, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ∇f = (∂f/∂x, ∂f/∂y)

        Spektral: ∂f/∂x → i·kx·f̂
        """
        f_hat = safe_fft2(f)
        df_dx = safe_ifft2(1j * self.kx * f_hat).real
        df_dy = safe_ifft2(1j * self.ky * f_hat).real
        return df_dx, df_dy

    def laplacian(self, f: torch.Tensor) -> torch.Tensor:
        """
        ∇²f = ∂²f/∂x² + ∂²f/∂y²

        Spektral: ∇²f → -k²·f̂
        """
        f_hat = safe_fft2(f)
        lap_f = safe_ifft2(-self.k_squared * f_hat).real
        return lap_f

    def biharmonic(self, f: torch.Tensor) -> torch.Tensor:
        """
        ∇⁴f = ∂⁴f/∂x⁴ + 2∂⁴f/∂x²∂y² + ∂⁴f/∂y⁴

        Spektral: ∇⁴f → k⁴·f̂ (TEK ADIM!)

        Bu, PINN'deki 4x autograd zincirinin yerine geçer.
        """
        f_hat = safe_fft2(f)
        biharm_f = safe_ifft2(self.k_fourth * f_hat).real
        return biharm_f

    def second_derivatives(self, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        İkinci türevler: (∂²f/∂x², ∂²f/∂y², ∂²f/∂x∂y)

        Moment hesaplamaları için gerekli.
        """
        f_hat = safe_fft2(f)
        f_xx = safe_ifft2(-self.kx2 * f_hat).real
        f_yy = safe_ifft2(-self.ky2 * f_hat).real
        f_xy = safe_ifft2(-self.kx * self.ky * f_hat).real
        return f_xx, f_yy, f_xy

    def fourth_derivatives(self, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dördüncü türevler: (∂⁴f/∂x⁴, ∂⁴f/∂y⁴, ∂⁴f/∂x²∂y²)

        Biharmonik operatörün bileşenleri.
        """
        f_hat = safe_fft2(f)
        f_xxxx = safe_ifft2(self.kx4 * f_hat).real
        f_yyyy = safe_ifft2(self.ky4 * f_hat).real
        f_xxyy = safe_ifft2(self.kx2_ky2 * f_hat).real
        return f_xxxx, f_yyyy, f_xxyy

    def dealias(self, f: torch.Tensor) -> torch.Tensor:
        """Aliasing önleme (nonlinear terimler için)"""
        f_hat = safe_fft2(f)
        return safe_ifft2(f_hat * self.dealias_mask).real

    def extra_repr(self) -> str:
        return f"resolution={self.resolution}, Lx={self.Lx}, Ly={self.Ly}"


class SineBasis(nn.Module):
    """
    Sine serisi basis fonksiyonları.

    Simply supported sınır koşulları için ideal:
    w(x,y) = Σ Wmn · sin(mπx/L) · sin(nπy/W)

    Bu basis, sınırda w=0 koşulunu DOĞAL OLARAK sağlar.

    Args:
        resolution: Grid çözünürlüğü
        Lx, Ly: Domain boyutları
        n_modes_x, n_modes_y: Her yönde mod sayısı
    """

    def __init__(self, resolution: int, Lx: float, Ly: float,
                 n_modes_x: int = 10, n_modes_y: int = 10):
        super().__init__()
        self.resolution = resolution
        self.Lx = Lx
        self.Ly = Ly
        self.n_modes_x = n_modes_x
        self.n_modes_y = n_modes_y

        # Grid noktaları
        x = torch.linspace(0, Lx, resolution)
        y = torch.linspace(0, Ly, resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        self.register_buffer('X', X)
        self.register_buffer('Y', Y)

        # Basis fonksiyonlarını önceden hesapla
        # sin(mπx/Lx) · sin(nπy/Ly) for m=1..n_modes_x, n=1..n_modes_y
        basis = []
        for m in range(1, n_modes_x + 1):
            for n in range(1, n_modes_y + 1):
                phi_mn = torch.sin(m * math.pi * X / Lx) * \
                         torch.sin(n * math.pi * Y / Ly)
                basis.append(phi_mn)

        # [n_modes, Nx, Ny]
        basis_tensor = torch.stack(basis, dim=0)
        self.register_buffer('basis', basis_tensor)

        # Mod indeksleri
        modes = []
        for m in range(1, n_modes_x + 1):
            for n in range(1, n_modes_y + 1):
                modes.append((m, n))
        self.modes = modes

    def forward(self, coefficients: torch.Tensor) -> torch.Tensor:
        """
        Katsayılardan alan oluştur.

        Args:
            coefficients: [B, n_modes] veya [n_modes]

        Returns:
            w: [B, Nx, Ny] veya [Nx, Ny]
        """
        if coefficients.dim() == 1:
            # [n_modes] → [Nx, Ny]
            return torch.einsum('m,mxy->xy', coefficients, self.basis)
        else:
            # [B, n_modes] → [B, Nx, Ny]
            return torch.einsum('bm,mxy->bxy', coefficients, self.basis)

    def project(self, w: torch.Tensor) -> torch.Tensor:
        """
        Alanı sine basis'e project et.

        Args:
            w: [B, Nx, Ny] veya [Nx, Ny]

        Returns:
            coefficients: [B, n_modes] veya [n_modes]
        """
        # Basit projeksiyon (ortogonallik kullanarak)
        # <w, φ_mn> / <φ_mn, φ_mn>
        if w.dim() == 2:
            # [Nx, Ny]
            coeffs = []
            for i, phi in enumerate(self.basis):
                c = (w * phi).sum() / (phi * phi).sum()
                coeffs.append(c)
            return torch.stack(coeffs)
        else:
            # [B, Nx, Ny]
            B = w.shape[0]
            coeffs = []
            for i, phi in enumerate(self.basis):
                c = (w * phi).sum(dim=(-2, -1)) / (phi * phi).sum()
                coeffs.append(c)
            return torch.stack(coeffs, dim=1)  # [B, n_modes]

    def get_mode(self, m: int, n: int) -> torch.Tensor:
        """Belirli bir modu getir."""
        idx = (m - 1) * self.n_modes_y + (n - 1)
        return self.basis[idx]


# =============================================================================
# ORTAK PLAKA KAPALI FORMLARI
# =============================================================================

def plate_critical_load(D: float, Lx: float, Ly: float,
                        m: int = 1, n: int = 1,
                        loading: str = "uniaxial",
                        direction: str = "x",
                        mindlin: bool = False,
                        E: Optional[float] = None,
                        nu: Optional[float] = None,
                        h: Optional[float] = None) -> Tuple[float, float]:
    r"""
    Dört kenarı basit mesnetli plakanın kritik burkulma yükü (kapalı form).

    w = sin(mπx/Lx)·sin(nπy/Ly) burkulma denklemine konursa
    (α_m = mπ/Lx, α_n = nπ/Ly):

        D(α_m² + α_n²)² = N_x·α_m² + N_y·α_n²

    - uniaxial, x yönü (N_y=0):  N_cr = D(α_m²+α_n²)²/α_m²
    - uniaxial, y yönü (N_x=0):  N_cr = D(α_m²+α_n²)²/α_n²
    - biaxial (N_x=N_y=N):       N_cr = D(α_m²+α_n²)

    Bu fonksiyon `BucklingNeuron` ve `ThermalNeuron` tarafından ORTAK
    kullanılır; formülün ikinci bir kopyası tutulmaz.

    Returns:
        (N_cr, N_cr_kirchhoff): Mindlin kapalıysa ikisi eşittir.
    """
    if loading not in ("uniaxial", "biaxial"):
        raise ValueError("loading 'uniaxial' veya 'biaxial' olmalı.")
    if direction not in ("x", "y"):
        raise ValueError("direction 'x' veya 'y' olmalı.")

    term_x = (m * math.pi / Lx) ** 2
    term_y = (n * math.pi / Ly) ** 2
    s = term_x + term_y

    if loading == "uniaxial":
        denom = term_x if direction == "x" else term_y
        N_cr_kirchhoff = D * (s ** 2) / denom
    else:
        N_cr_kirchhoff = D * s

    N_cr = N_cr_kirchhoff
    if mindlin:
        if E is None or nu is None or h is None:
            raise ValueError("mindlin=True için E, nu ve h verilmeli.")
        G = E / (2 * (1 + nu))
        kappa_s = 5.0 / 6.0
        N_cr = N_cr_kirchhoff / (1 + N_cr_kirchhoff / (kappa_s * G * h))

    return N_cr, N_cr_kirchhoff


def kirsch_hole_stress(sigma_0, X: torch.Tensor, Y: torch.Tensor,
                       a_hole: float,
                       cx: float = 0.0, cy: float = 0.0,
                       mask_hole: bool = True,
                       vm_eps: float = 1e-20
                       ) -> Dict[str, torch.Tensor]:
    r"""
    Kirsch çözümü — sonsuz plakada dairesel delik etrafındaki gerilme alanı.

    Tek eksenli (x yönü) σ₀ çekmesi altında, polar koordinatlarda:

        σ_rr = (σ₀/2)[(1 - a²/r²) + (1 - 4a²/r² + 3a⁴/r⁴)·cos2θ]
        σ_θθ = (σ₀/2)[(1 + a²/r²) - (1 + 3a⁴/r⁴)·cos2θ]
        τ_rθ = -(σ₀/2)[(1 + 2a²/r² - 3a⁴/r⁴)·sin2θ]

    Delik kenarında (r = a, θ = ±π/2) σ_θθ = 3σ₀ olur: klasik gerilme
    yoğunlaşma katsayısı SCF = 3. Bu, çözümün kapalı-form sağlamasıdır.

    Bütün girdiler tensör olabilir; ifade autograd-geçirgendir, dolayısıyla
    ölçülen bir gerilme alanından σ₀ veya malzeme parametrelerini geri
    çözmek mümkündür.

    Args:
        sigma_0: Uzak alan çekme gerilmesi [Pa] — skaler veya tensör.
        X, Y: Koordinat gridleri [N, N].
        a_hole: Delik yarıçapı [m].
        cx, cy: Delik merkezi.
        mask_hole: True ise delik içi sıfırlanır.
        vm_eps: von Mises karekökü altına eklenen küçük sabit.

    Returns:
        dict: sigma_xx, sigma_yy, tau_xy, sigma_vm, hole_mask, r, theta
    """
    dx_g = X - cx
    dy_g = Y - cy

    r = torch.sqrt(dx_g ** 2 + dy_g ** 2)
    theta = torch.atan2(dy_g, dx_g)

    # Delik merkezinde r→0 tekilliği: yarıçapın %1'inde tabanlanır
    r_safe = torch.clamp(r, min=a_hole * 0.01)
    hole_mask = (r < a_hole)

    ar2 = (a_hole / r_safe) ** 2
    ar4 = ar2 ** 2
    cos2t = torch.cos(2 * theta)
    sin2t = torch.sin(2 * theta)

    sigma_rr = (sigma_0 / 2) * ((1 - ar2) + (1 - 4 * ar2 + 3 * ar4) * cos2t)
    sigma_tt = (sigma_0 / 2) * ((1 + ar2) - (1 + 3 * ar4) * cos2t)
    tau_rt = -(sigma_0 / 2) * ((1 + 2 * ar2 - 3 * ar4) * sin2t)

    # Polar → Kartezyen
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    cos2, sin2 = cos_t ** 2, sin_t ** 2
    sincos = sin_t * cos_t

    sigma_xx = sigma_rr * cos2 + sigma_tt * sin2 - 2 * tau_rt * sincos
    sigma_yy = sigma_rr * sin2 + sigma_tt * cos2 + 2 * tau_rt * sincos
    tau_xy = (sigma_rr - sigma_tt) * sincos + tau_rt * (cos2 - sin2)

    sigma_vm = torch.sqrt(sigma_xx ** 2 - sigma_xx * sigma_yy
                          + sigma_yy ** 2 + 3 * tau_xy ** 2 + vm_eps)

    if mask_hole:
        disi = (~hole_mask).to(sigma_xx.dtype)
        sigma_xx = sigma_xx * disi
        sigma_yy = sigma_yy * disi
        tau_xy = tau_xy * disi
        sigma_vm = sigma_vm * disi

    return {
        'sigma_xx': sigma_xx, 'sigma_yy': sigma_yy, 'tau_xy': tau_xy,
        'sigma_vm': sigma_vm, 'hole_mask': hole_mask, 'r': r, 'theta': theta,
    }


# =============================================================================
# FİZİK-BİLEN NÖRONLAR
# =============================================================================

class BiharmonicNeuron(nn.Module):
    """
    Biharmonik Operatör Nöronu: ∇⁴w

    Kirchhoff-Love plaka teorisinin temel operatörü.

    PINN'de 4x autograd zinciri → gradient vanishing/exploding
    SPINE'da spektral k⁴ çarpımı → TEK ADIM, TAM DOĞRU

    Öğrenilen: rigidity_modulator (1 parametre)

    Args:
        resolution: Grid çözünürlüğü
        Lx, Ly: Domain boyutları
        spectral_ops: Paylaşılan SpectralOps2DStruct (opsiyonel)
    """

    def __init__(self, resolution: int, Lx: float = 1.0, Ly: float = 1.0,
                 spectral_ops: Optional[SpectralOps2DStruct] = None):
        super().__init__()

        self.spectral_ops = spectral_ops if spectral_ops else \
                           SpectralOps2DStruct(resolution, Lx, Ly)

        # TEK ÖĞRENİLEBİLİR PARAMETRE
        self.rigidity_modulator = nn.Parameter(torch.ones(1))

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """
        ∇⁴w hesapla.

        Args:
            w: Deplasman alanı [B, Ny, Nx]

        Returns:
            ∇⁴w: Biharmonik [B, Ny, Nx]
        """
        biharm = self.spectral_ops.biharmonic(w)
        return self.rigidity_modulator * biharm

    def extra_repr(self) -> str:
        return f"rigidity_modulator={self.rigidity_modulator.item():.4f}"


class BucklingNeuron(nn.Module):
    """
    Burkulma Analizi Nöronu.

    Kirchhoff-Love plaka burkulma denklemi:
    D∇⁴w + Nx·∂²w/∂x² + Ny·∂²w/∂y² = 0

    Kritik yük formülü (basit mesnetli):
    Ncr = D·π²/L² · (m + n²L²/W²m)²

    Rayleigh Quotient yaklaşımı:
    λ = bending_energy / membrane_work
    Ncr = λ · D

    Args:
        resolution: Grid çözünürlüğü
        Lx, Ly: Domain boyutları (m)
        material: Malzeme özellikleri
    """

    def __init__(self, resolution: int, Lx: float, Ly: float,
                 material: MaterialProperties,
                 spectral_ops: Optional[SpectralOps2DStruct] = None):
        super().__init__()

        self.resolution = resolution
        self.Lx = Lx
        self.Ly = Ly
        self.material = material
        self.D = material.D

        self.spectral_ops = spectral_ops if spectral_ops else \
                           SpectralOps2DStruct(resolution, Lx, Ly)

        # Öğrenilebilir parametreler
        self.load_scale = nn.Parameter(torch.ones(1))

    def analytical_critical_load(self, m: int = 1, n: int = 1,
                                 loading: str = "uniaxial",
                                 direction: str = "x",
                                 mindlin: bool = False,
                                 explain: bool = False,
                                 label: str = "") -> float:
        r"""
        Analitik kritik burkulma yükü (dört kenarı basit mesnetli plaka).

        Kirchhoff-Love burkulma denkleminden, w = sin(mπx/Lx)·sin(nπy/Ly)
        yerine konarak (α_m = mπ/Lx, α_n = nπ/Ly):

            D(α_m² + α_n²)² = N_x·α_m² + N_y·α_n²

        Yükleme durumuna göre:
          - uniaxial (N_y = 0, yük x yönünde):  N_cr = D(α_m²+α_n²)² / α_m²
          - uniaxial (N_x = 0, yük y yönünde):  N_cr = D(α_m²+α_n²)² / α_n²
          - biaxial  (N_x = N_y = N):          N_cr = D(α_m²+α_n²)

        Biaxial hal, tam kısıtlı bir plakada TERMAL bası için doğru olandır:
        ısınan plaka her iki yönde de genleşmeye zorlanır, dolayısıyla
        N_x = N_y = N_T oluşur. Kare plakada biaxial kritik yük uniaxial'in
        yarısıdır; uniaxial formülü termal probleme uygulamak kritik yükü
        (ve dolayısıyla ΔT_cr'yi) güvensiz yönde ~2 kat fazla tahmin eder.

        Args:
            m, n: Burkulma mod numaraları (x ve y yönünde yarım dalga sayısı).
            loading: 'uniaxial' (tek eksenli) veya 'biaxial' (iki eksenli).
            direction: uniaxial halde yükün yönü — 'x' veya 'y'.
            mindlin: True ise enine kayma esnekliği (Mindlin-Reissner)
                düzeltmesi uygulanır: N_cr ← N_cr/(1 + N_cr/(κ_s·G·h)).
                Kalın plakalarda kritik yükü düşürür.
            explain: True ise her ara adım sayısal değerleriyle yazdırılır.
            label: explain çıktısında problemi etiketler.

        Returns:
            N_cr: Kritik burkulma yükü [N/m] (birim genişlik başına kuvvet).
        """
        term_x = (m * math.pi / self.Lx) ** 2   # α_m²
        term_y = (n * math.pi / self.Ly) ** 2   # α_n²

        N_cr, N_cr_kirchhoff = plate_critical_load(
            self.D, self.Lx, self.Ly, m=m, n=n,
            loading=loading, direction=direction, mindlin=mindlin,
            E=self.material.E, nu=self.material.nu, h=self.material.h)

        if explain:
            self._explain_critical_load(label, m, n, loading, direction, mindlin,
                                        term_x, term_y, N_cr_kirchhoff, N_cr)
        return N_cr

    def _explain_critical_load(self, label, m, n, loading, direction, mindlin,
                               term_x, term_y, N_cr_kirchhoff, N_cr):
        """Kritik yük hesabını adım adım yazdır (şeffaf mod)."""
        sep = "-" * 68
        head = "  BucklingNeuron — kritik yük"
        if label:
            head += f" [{label}]"
        print(f"{sep}\n{head}\n{sep}")
        print(f"    Plaka: Lx = {self.Lx:.4g} m, Ly = {self.Ly:.4g} m, "
              f"h = {self.material.h:.4g} m")
        print(f"    Eğilme rijitliği  D = E·h³/[12(1-ν²)] = {self.D:.6g} N·m")
        print(f"    Mod (m,n) = ({m},{n})")
        print(f"      α_m² = (mπ/Lx)² = {term_x:.6g} 1/m²")
        print(f"      α_n² = (nπ/Ly)² = {term_y:.6g} 1/m²")
        if loading == "uniaxial":
            d = "α_m²" if direction == "x" else "α_n²"
            print(f"    Yükleme: tek eksenli ({direction} yönünde)")
            print(f"      N_cr = D(α_m²+α_n²)²/{d} = {N_cr_kirchhoff:.6g} N/m")
        else:
            print(f"    Yükleme: iki eksenli (N_x = N_y)")
            print(f"      N_cr = D(α_m²+α_n²) = {N_cr_kirchhoff:.6g} N/m")
        if mindlin:
            G = self.material.E / (2 * (1 + self.material.nu))
            print(f"    Mindlin kayma düzeltmesi (κ_s=5/6, G={G:.4g} Pa):")
            print(f"      N_cr ← N_cr/(1+N_cr/(κ_s·G·h)) = {N_cr:.6g} N/m"
                  f"   (düşüş %{100*(1-N_cr/N_cr_kirchhoff):.3f})")
        print(f"    → N_cr = {N_cr:.6g} N/m  ({N_cr/1e3:.4g} kN/m)")
        print(sep)

    def mode_shape(self, m: int = 1, n: int = 1) -> torch.Tensor:
        """
        Burkulma mod şekli (analitik).

        w(x,y) = sin(mπx/L) · sin(nπy/W)
        """
        # Grid, modülün kendi device'ını takip eder (global DEVICE değil):
        # model taşınmadan MPS'e tensor üretmek buffer'larla device çakışması yaratıyordu
        device = self.load_scale.device
        x = torch.linspace(0, self.Lx, self.resolution, device=device)
        y = torch.linspace(0, self.Ly, self.resolution, device=device)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        w = torch.sin(m * math.pi * X / self.Lx) * \
            torch.sin(n * math.pi * Y / self.Ly)
        return w

    def rayleigh_quotient(self, w: torch.Tensor) -> torch.Tensor:
        """
        Rayleigh Quotient hesapla.

        λ = ∫(∇²w)² dA / ∫(∂w/∂x)² dA

        Bu oran, kritik yük parametresini verir: Ncr = λ · D
        """
        # Laplacian
        lap_w = self.spectral_ops.laplacian(w)

        # Gradient
        dw_dx, dw_dy = self.spectral_ops.gradient(w)

        # Enerji integralleri
        bending_energy = (lap_w ** 2).mean()
        membrane_work = (dw_dx ** 2).mean()

        # Rayleigh quotient
        lambda_r = bending_energy / (membrane_work + 1e-10)

        return lambda_r

    def forward(self, w: torch.Tensor, Nx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Burkulma denkleminin rezidüelini hesapla.

        Rezidüel: D∇⁴w + Nx·∂²w/∂x² = 0

        Args:
            w: Deplasman alanı [B, Ny, Nx]
            Nx: Eksenel yük (opsiyonel, None ise Rayleigh'den tahmin)

        Returns:
            residual: PDE rezidüeli [B, Ny, Nx]
        """
        # Biharmonik
        biharm_w = self.spectral_ops.biharmonic(w)

        # İkinci türev
        w_xx, w_yy, _ = self.spectral_ops.second_derivatives(w)

        if Nx is None:
            # Rayleigh quotient'tan Nx tahmin et
            lambda_r = self.rayleigh_quotient(w)
            Nx = lambda_r * self.D * self.load_scale

        # Rezidüel
        residual = self.D * biharm_w + Nx * w_xx

        return residual

    def find_critical_modes(self, max_m: int = 5, max_n: int = 5,
                            loading: str = "uniaxial",
                            direction: str = "x",
                            mindlin: bool = False,
                            explain: bool = False,
                            label: str = "",
                            top_k: int = 5) -> List[Dict]:
        """
        (m,n) mod uzayını tara, kritik yükleri küçükten büyüğe sırala.

        Burkulma en düşük N_cr'yi veren modda başlar; bu yüzden tarama
        yapılmadan (m,n)=(1,1) varsaymak dikdörtgen plakalarda yanıltıcıdır.

        Args:
            max_m, max_n: Taranacak en büyük mod numaraları.
            loading, direction, mindlin: `analytical_critical_load` ile aynı.
            explain: True ise en düşük `top_k` mod tablo hâlinde yazdırılır.
            top_k: explain çıktısında gösterilecek mod sayısı.

        Returns:
            [{'m':…, 'n':…, 'N_cr':…}, …] — N_cr'ye göre artan sırada.
        """
        modes = []
        for m in range(1, max_m + 1):
            for n in range(1, max_n + 1):
                N_cr = self.analytical_critical_load(
                    m, n, loading=loading, direction=direction, mindlin=mindlin)
                modes.append({'m': m, 'n': n, 'N_cr': N_cr})

        modes.sort(key=lambda x: x['N_cr'])

        if explain:
            sep = "-" * 68
            head = "  BucklingNeuron — mod taraması"
            if label:
                head += f" [{label}]"
            kind = (f"tek eksenli ({direction})" if loading == "uniaxial"
                    else "iki eksenli")
            print(f"{sep}\n{head}\n{sep}")
            print(f"    Yükleme: {kind}"
                  f"{' + Mindlin düzeltmesi' if mindlin else ''}"
                  f"   |   taranan mod: {max_m}×{max_n}")
            print(f"    {'sıra':>4}  {'m':>2} {'n':>2}   {'N_cr [kN/m]':>14}   oran")
            best = modes[0]['N_cr']
            for i, md in enumerate(modes[:top_k], start=1):
                mark = "  ← kritik mod" if i == 1 else ""
                print(f"    {i:>4}  {md['m']:>2} {md['n']:>2}   "
                      f"{md['N_cr']/1e3:>14.4f}   {md['N_cr']/best:>5.3f}×{mark}")
            print(f"    → burkulma modu (m,n) = ({modes[0]['m']},{modes[0]['n']}), "
                  f"N_cr = {best:.6g} N/m")
            print(sep)

        return modes

    def extra_repr(self) -> str:
        return f"Lx={self.Lx}, Ly={self.Ly}, D={self.D:.2e}"


class StressNeuron(nn.Module):
    """
    Gerilme Tensörü Nöronu: σ = C : ε

    Hooke yasası built-in.

    2D düzlem gerilme için:
    σ_xx = E/(1-ν²) · (ε_xx + ν·ε_yy)
    σ_yy = E/(1-ν²) · (ε_yy + ν·ε_xx)
    σ_xy = E/(2(1+ν)) · γ_xy

    Öğrenilen: E_modulator, nu_modulator (2 parametre)
    """

    def __init__(self, material: MaterialProperties):
        super().__init__()

        self.E = material.E
        self.nu = material.nu

        # Öğrenilebilir modulatörler
        self.E_modulator = nn.Parameter(torch.ones(1))
        self.nu_modulator = nn.Parameter(torch.ones(1))

    def forward(self, eps_xx: torch.Tensor, eps_yy: torch.Tensor,
                gamma_xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Strain → Stress dönüşümü.

        Args:
            eps_xx, eps_yy: Normal strainler
            gamma_xy: Kayma straini

        Returns:
            (sigma_xx, sigma_yy, sigma_xy): Gerilme bileşenleri
        """
        E = self.E * self.E_modulator
        nu = self.nu * self.nu_modulator

        # Düzlem gerilme katsayıları
        C = E / (1 - nu**2)
        G = E / (2 * (1 + nu))

        # Hooke yasası
        sigma_xx = C * (eps_xx + nu * eps_yy)
        sigma_yy = C * (eps_yy + nu * eps_xx)
        sigma_xy = G * gamma_xy

        return sigma_xx, sigma_yy, sigma_xy

    @staticmethod
    def surface_strain(kappa_xx: torch.Tensor, kappa_yy: torch.Tensor,
                       kappa_xy: torch.Tensor, z: float
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Kirchhoff varsayımıyla eğrilikten z kotundaki yüzey şekil değiştirmesi.

            ε_xx = z·κ_xx,   ε_yy = z·κ_yy,   γ_xy = 2z·κ_xy

        γ_xy'deki 2 çarpanı mühendislik kayma şekil değiştirmesi tanımından
        gelir (γ = 2ε_xy) ve κ_xy = -∂²w/∂x∂y konvansiyonuyla uyumludur.

        Args:
            kappa_xx, kappa_yy, kappa_xy: Eğrilikler.
            z: Kalınlık boyunca kot; üst yüzey için +h/2, alt yüzey için -h/2.
        """
        return z * kappa_xx, z * kappa_yy, 2 * z * kappa_xy

    @staticmethod
    def von_mises_2d(sigma_xx: torch.Tensor, sigma_yy: torch.Tensor,
                     sigma_xy: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
        r"""
        Düzlem gerilme hâli için von Mises eşdeğer gerilmesi.

            σ_vm = √(σ_xx² - σ_xx·σ_yy + σ_yy² + 3σ_xy²)

        Args:
            eps: Karekök altına eklenen küçük sabit (gradyan güvenliği için;
                varsayılan 0 — tam analitik değer).
        """
        return torch.sqrt(sigma_xx**2 - sigma_xx * sigma_yy
                          + sigma_yy**2 + 3 * sigma_xy**2 + eps)

    def plate_surface_stress(self, kappa_xx: torch.Tensor,
                             kappa_yy: torch.Tensor,
                             kappa_xy: torch.Tensor,
                             z: float,
                             membrane_xx: float = 0.0,
                             membrane_yy: float = 0.0,
                             vm_eps: float = 0.0,
                             explain: bool = False,
                             label: str = "") -> Dict[str, torch.Tensor]:
        r"""
        Plaka yüzeyindeki toplam gerilme: eğilme + düzlem içi (membran).

            ε = z·κ  →  σ_eğilme = C(ε + ν·ε_çapraz)  →  σ = σ_eğilme + σ_membran

        Membran katkısı doğrudan gerilme olarak verilir (Pa). Kuvvet/birim
        uzunluk cinsinden bir N biliniyorsa membran gerilmesi N/h'dir; BASI
        için işaret negatiftir.

        Args:
            kappa_xx, kappa_yy, kappa_xy: Eğrilik alanları.
            z: Yüzey kotu (üst yüzey +h/2).
            membrane_xx, membrane_yy: Düzlem içi gerilme katkıları [Pa].
            vm_eps: von Mises karekökü için küçük sabit.
            explain: True ise adımlar sayısal değerleriyle yazdırılır.
            label: explain çıktısı için etiket.

        Returns:
            dict: eps_xx, eps_yy, gamma_xy, sigma_xx, sigma_yy, sigma_xy,
                  sigma_vm
        """
        eps_xx, eps_yy, gamma_xy = self.surface_strain(
            kappa_xx, kappa_yy, kappa_xy, z)
        s_xx_b, s_yy_b, s_xy = self.forward(eps_xx, eps_yy, gamma_xy)

        sigma_xx = s_xx_b + membrane_xx
        sigma_yy = s_yy_b + membrane_yy
        sigma_vm = self.von_mises_2d(sigma_xx, sigma_yy, s_xy, eps=vm_eps)

        out = {
            'eps_xx': eps_xx, 'eps_yy': eps_yy, 'gamma_xy': gamma_xy,
            'sigma_xx': sigma_xx, 'sigma_yy': sigma_yy, 'sigma_xy': s_xy,
            'sigma_bend_xx': s_xx_b, 'sigma_bend_yy': s_yy_b,
            'sigma_vm': sigma_vm,
        }

        if explain:
            sep = "-" * 68
            head = "  StressNeuron — yüzey gerilmesi (Hooke, düzlem gerilme)"
            if label:
                head += f" [{label}]"
            print(f"{sep}\n{head}\n{sep}")
            print(f"    Kot z = {z:.4g} m   |   E = {self.E:.4g} Pa, ν = {self.nu:.4g}")
            print(f"    ε = z·κ  →  max|ε_xx| = {eps_xx.abs().max().item():.6g}, "
                  f"max|ε_yy| = {eps_yy.abs().max().item():.6g}")
            print(f"    Eğilme: σ = C(ε + ν·ε_çapraz), C = E/(1-ν²) = "
                  f"{self.E/(1-self.nu**2):.6g} Pa")
            print(f"      max|σ_eğilme,xx| = {s_xx_b.abs().max().item()/1e6:.6g} MPa, "
                  f"max|σ_eğilme,yy| = {s_yy_b.abs().max().item()/1e6:.6g} MPa")
            print(f"    Membran katkısı: σ_xx += {membrane_xx/1e6:.6g} MPa, "
                  f"σ_yy += {membrane_yy/1e6:.6g} MPa")
            print(f"    von Mises: σ_vm = √(σ_xx²-σ_xx σ_yy+σ_yy²+3σ_xy²)")
            print(f"      → max σ_vm = {sigma_vm.max().item()/1e6:.6g} MPa")
            print(sep)

        return out

    def extra_repr(self) -> str:
        return f"E={self.E:.2e}, nu={self.nu}"


class StrainNeuron(nn.Module):
    """
    Şekil Değiştirme Nöronu: ε = ½(∇u + ∇uᵀ)

    Kinematik bağıntılar - PARAMETRESİZ (saf geometri).

    Plaka teorisinde:
    κ_xx = -∂²w/∂x² (eğrilik)
    κ_yy = -∂²w/∂y²
    κ_xy = -∂²w/∂x∂y
    """

    def __init__(self, resolution: int, Lx: float, Ly: float,
                 spectral_ops: Optional[SpectralOps2DStruct] = None):
        super().__init__()

        self.spectral_ops = spectral_ops if spectral_ops else \
                           SpectralOps2DStruct(resolution, Lx, Ly)

    def curvatures(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Plaka eğriliklerini hesapla.

        κ_xx = -∂²w/∂x²
        κ_yy = -∂²w/∂y²
        κ_xy = -∂²w/∂x∂y
        """
        w_xx, w_yy, w_xy = self.spectral_ops.second_derivatives(w)

        kappa_xx = -w_xx
        kappa_yy = -w_yy
        kappa_xy = -w_xy

        return kappa_xx, kappa_yy, kappa_xy

    def forward(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Eğrilikleri hesapla."""
        return self.curvatures(w)

    @staticmethod
    def energy_density_split(sigma_xx: torch.Tensor, sigma_yy: torch.Tensor,
                             tau_xy: torch.Tensor,
                             E, nu, eps: float = 1e-20) -> torch.Tensor:
        r"""
        Çekme–basma ayrışımlı gerinim enerji yoğunluğu ψ⁺ (düzlem gerilme).

        Faz-alanı hasar modellerinde çatlağı yalnız ÇEKME enerjisi sürer;
        basma altında malzeme hasar almamalıdır. Bu yüzden asal gerinimlerin
        pozitif kısmı alınır (spektral ayrışım):

            σ_1,2 = (σ_xx+σ_yy)/2 ± √[((σ_xx-σ_yy)/2)² + τ_xy²]
            ε_1 = (σ_1 - ν σ_2)/E ,   ε_2 = (σ_2 - ν σ_1)/E
            ψ⁺ = (λ_2d/2)⟨ε_1+ε_2⟩₊² + µ(⟨ε_1⟩₊² + ⟨ε_2⟩₊²)

        burada ⟨·⟩₊ = max(·, 0), λ_2d = Eν/(1-ν²), µ = E/[2(1+ν)].

        E ve nu tensör olabilir; ifade autograd-geçirgendir.
        """
        lam_2d = E * nu / (1 - nu ** 2)
        mu_val = E / (2 * (1 + nu))

        sigma_avg = (sigma_xx + sigma_yy) / 2
        R = torch.sqrt(((sigma_xx - sigma_yy) / 2) ** 2 + tau_xy ** 2 + eps)
        sigma_1 = sigma_avg + R
        sigma_2 = sigma_avg - R

        eps_1 = (sigma_1 - nu * sigma_2) / E
        eps_2 = (sigma_2 - nu * sigma_1) / E

        eps_1_plus = torch.clamp(eps_1, min=0)
        eps_2_plus = torch.clamp(eps_2, min=0)
        eps_vol_plus = torch.clamp(eps_1 + eps_2, min=0)

        return (lam_2d / 2) * eps_vol_plus ** 2 \
            + mu_val * (eps_1_plus ** 2 + eps_2_plus ** 2)


class BoundaryNeuron(nn.Module):
    """
    Sınır Koşulları Nöronu.

    Türler:
    - Simply Supported: w=0, M=0 (moment sıfır)
    - Clamped: w=0, ∂w/∂n=0 (eğim sıfır)
    - Free: M=0, V=0 (moment ve kesme sıfır)

    Sine basis kullanıldığında simply supported doğal olarak sağlanır.
    """

    def __init__(self, resolution: int, Lx: float, Ly: float,
                 bc_type: BoundaryConditionType = BoundaryConditionType.SIMPLY_SUPPORTED):
        super().__init__()

        self.resolution = resolution
        self.Lx = Lx
        self.Ly = Ly
        self.bc_type = bc_type

        # Sınır maskeleri oluştur
        mask = torch.zeros(resolution, resolution)
        mask[0, :] = 1.0   # Alt
        mask[-1, :] = 1.0  # Üst
        mask[:, 0] = 1.0   # Sol
        mask[:, -1] = 1.0  # Sağ

        self.register_buffer('boundary_mask', mask.bool())

        # BC kuvvet parametresi
        self.bc_strength = nn.Parameter(torch.ones(1))

    def apply_bc(self, w: torch.Tensor) -> torch.Tensor:
        """
        Sınır koşullarını uygula.

        Simply supported için: w = 0 at boundaries
        """
        if self.bc_type == BoundaryConditionType.SIMPLY_SUPPORTED:
            # Sınırda w = 0
            w = w.clone()
            if w.dim() == 2:
                w[self.boundary_mask] = 0
            else:
                w[:, self.boundary_mask] = 0

        return w

    def bc_loss(self, w: torch.Tensor, w_xx: Optional[torch.Tensor] = None,
                w_yy: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sınır koşulu kaybını hesapla.
        """
        if self.bc_type == BoundaryConditionType.SIMPLY_SUPPORTED:
            # w = 0 at boundaries
            if w.dim() == 2:
                bc_w = w[self.boundary_mask]
            else:
                bc_w = w[:, self.boundary_mask]

            loss = (bc_w ** 2).mean()

            # M = 0 at boundaries (w_xx = 0, w_yy = 0)
            if w_xx is not None and w_yy is not None:
                if w_xx.dim() == 2:
                    loss += (w_xx[self.boundary_mask] ** 2).mean()
                    loss += (w_yy[self.boundary_mask] ** 2).mean()
                else:
                    loss += (w_xx[:, self.boundary_mask] ** 2).mean()
                    loss += (w_yy[:, self.boundary_mask] ** 2).mean()

            return self.bc_strength * loss

        return torch.tensor(0.0, device=w.device)

    def extra_repr(self) -> str:
        return f"bc_type={self.bc_type.value}"


# =============================================================================
# FAZ 3 NÖRONLARı (2D)
# =============================================================================

class StaticNeuron(nn.Module):
    """
    Statik Plaka Eğilme Nöronu: D∇⁴w = q(x,y)
    
    Navier serisi çözümü (basit mesnetli):
    w(x,y) = Σ Wmn · sin(mπx/L) · sin(nπy/W)
    Wmn = qmn / [D·π⁴·((m/L)² + (n/W)²)²]
    
    Args:
        resolution: Grid çözünürlüğü
        Lx, Ly: Domain boyutları (m)
        material: Malzeme özellikleri
        n_modes: Navier serisi mod sayısı
    """
    
    def __init__(self, resolution: int, Lx: float, Ly: float,
                 material: MaterialProperties,
                 spectral_ops: Optional[SpectralOps2DStruct] = None,
                 n_modes: int = 20,
                 dtype: torch.dtype = torch.float32):
        super().__init__()

        self.resolution = resolution
        self.Lx = Lx
        self.Ly = Ly
        self.material = material
        self.D = material.D
        self.n_modes = n_modes
        self.dtype = dtype

        self.spectral_ops = spectral_ops if spectral_ops else \
                           SpectralOps2DStruct(resolution, Lx, Ly)

        # Grid — dtype burada belirlenir. Sonradan .double() çağırmak grid'i
        # yükseltir ama KAYBOLAN HASSASİYETİ GERİ GETİRMEZ: float32'de üretilmiş
        # bir konum float64'e çevrilince ~1e-7 göreli hatasını korur. Bu hata
        # ikinci türev karşılaştırmalarında dx² ile bölündüğü için büyür.
        # Hassas iş için modülü doğrudan dtype=torch.float64 ile kurun.
        x = torch.linspace(0, Lx, resolution, dtype=dtype)
        y = torch.linspace(0, Ly, resolution, dtype=dtype)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        self.register_buffer('X', X)
        self.register_buffer('Y', Y)

        # Öğrenilebilir parametre
        self.stiffness_modulator = nn.Parameter(torch.ones(1, dtype=dtype))
    
    def solve_navier(self, q: float,
                     N_x: float = 0.0, N_y: float = 0.0,
                     mindlin: bool = False,
                     pdelta_mindlin: bool = False,
                     n_modes: Optional[int] = None,
                     max_amplification: float = 50.0,
                     ratio_cap: float = 0.99,
                     stiffness_scale: Optional[torch.Tensor] = None,
                     explain: bool = False,
                     label: str = "") -> Dict[str, torch.Tensor]:
        r"""
        Membran basısı altındaki basit mesnetli plakanın Navier çözümü —
        deplasman VE eğrilikler analitik türevle (FFT yok, Gibbs yok).

        Uniform q yükü için (m,n tek):

            q_mn = 16q/(π²mn),    W₀ = q_mn / [D(α_m²+α_n²)²]

        Düzlem içi bası, eğilme deplasmanını büyütür (P-Δ etkisi). Mod
        bazında büyütme çarpanı:

            r_mn = (N_x·α_m² + N_y·α_n²) / [D(α_m²+α_n²)²]
            amp  = 1/(1 - r_mn)        (r_mn → 1 iken burkulma)

        r_mn'in paydası ilgili modun kritik yüküdür; tek eksenli yükte
        r_mn = N/N_cr,mn, iki eksenli yükte r_mn = N/(D(α_m²+α_n²)) olur —
        yani aynı ifade her iki yükleme tipini de kapsar.

        `mindlin=True` ise enine kayma esnekliğinden gelen ek deplasman
        çarpanı uygulanır: 1 + (α_m²+α_n²)·D/(κ_s·G·h).

        Args:
            q: Uniform dağıtılmış yük [Pa] (örn. öz-ağırlık ρgh).
            N_x, N_y: Düzlem içi BASI kuvvetleri [N/m] (pozitif = bası).
            mindlin: Deplasman büyütmesinde kayma esnekliği çarpanı
                uygulansın mı.
            pdelta_mindlin: P-Δ oranının paydasındaki kritik yüke de Mindlin
                düzeltmesi uygulansın mı. Kalın plakada kayma esnekliği
                kritik yükü düşürür, dolayısıyla r = N/N_cr'yi büyütür;
                False ise payda ince-plaka (Kirchhoff) kritik yüküdür.
            n_modes: Seri mod sayısı (None → self.n_modes).
            max_amplification: r_mn ≥ 1 olduğunda (burkulmuş mod) kullanılan
                sonlu büyütme tavanı — çözüm ıraksamasın diye.
            ratio_cap: r_mn için üst kırpma (varsayılan 0.99).
            stiffness_scale: Rijitliği çarpan opsiyonel tensör. Verilmezse
                modülün kendi `stiffness_modulator` parametresi kullanılır.
                Ters problemde dışarıdan optimize edilen bir değişkeni
                buradan geçirin — `stiffness_modulator`'ı yeni bir
                nn.Parameter ile değiştirmek gradyan zincirini koparır.
            explain: True ise adımlar sayısal değerleriyle yazdırılır.
            label: explain çıktısı için etiket.

        Returns:
            dict: w, w_xx, w_yy, w_xy, kappa_xx, kappa_yy, kappa_xy,
                  w_max, r_max (en büyük mod bası oranı), buckled (bool)
        """
        n_modes = self.n_modes if n_modes is None else n_modes
        # D bir TENSÖRDÜR: rijitlik çarpanı üzerinden gradyan akar, böylece
        # ölçülen bir alandan rijitliği (dolayısıyla E'yi) geri çözmek için
        # autograd kullanılabilir. Skalara çevirmek ters problemi imkânsız kılar.
        # `stiffness_scale` verilirse modülatör yerine o kullanılır; dışarıdan
        # optimize edilen bir tensör geçirmenin doğru yolu budur (modülatörü
        # nn.Parameter ile değiştirmek gradyan geçmişini KOPARIR).
        olcek = (self.stiffness_modulator.squeeze() if stiffness_scale is None
                 else stiffness_scale)
        D = self.D * olcek
        E, nu, h = self.material.E, self.material.nu, self.material.h
        G = E / (2 * (1 + nu))
        kappa_s = 5.0 / 6.0

        X, Y = self.X, self.Y
        w = torch.zeros_like(X)
        w_xx = torch.zeros_like(X)
        w_yy = torch.zeros_like(X)
        w_xy = torch.zeros_like(X)

        r_max = 0.0
        n_terms = 0
        for m in range(1, n_modes + 1, 2):        # uniform yükte yalnız tek modlar
            for n in range(1, n_modes + 1, 2):
                alpha_m = m * math.pi / self.Lx
                alpha_n = n * math.pi / self.Ly
                s = alpha_m ** 2 + alpha_n ** 2

                q_mn = 16.0 * q / (math.pi ** 2 * m * n)
                W0 = q_mn / (D * s ** 2)

                # P-Δ: modun kritik yüküne oranı (D tensör olduğu için r de
                # tensördür; karşılaştırmalar .item() ile yapılır ama büyütme
                # çarpanı torch işlemleriyle kurulur, böylece gradyan korunur)
                r_mn = (N_x * alpha_m ** 2 + N_y * alpha_n ** 2) / (D * s ** 2)
                if pdelta_mindlin and r_mn.detach().item() > 0.0:
                    # Yük vektörü yönündeki kritik büyüklük ve onun Mindlin
                    # düzeltmesi:  N_cr ← N_cr/(1+N_cr/(κ_s·G·h))
                    #   ⇒  r ← r·(1 + N_cr/(κ_s·G·h))
                    N_mag = max(abs(N_x), abs(N_y))
                    N_cr_mn = N_mag / r_mn
                    r_mn = r_mn * (1.0 + N_cr_mn / (kappa_s * G * h))
                r_mn_val = r_mn.detach().item()
                r_max = max(r_max, r_mn_val)
                if r_mn_val < 1.0:
                    amp = 1.0 / (1.0 - torch.clamp(r_mn, max=ratio_cap))
                else:
                    amp = max_amplification

                shear = 1.0 + s * D / (kappa_s * G * h) if mindlin else 1.0
                W = W0 * amp * shear

                sm = torch.sin(alpha_m * X)
                sn = torch.sin(alpha_n * Y)
                cm = torch.cos(alpha_m * X)
                cn = torch.cos(alpha_n * Y)

                w = w + W * sm * sn
                w_xx = w_xx + (-alpha_m ** 2) * W * sm * sn
                w_yy = w_yy + (-alpha_n ** 2) * W * sm * sn
                w_xy = w_xy + (alpha_m * alpha_n) * W * cm * cn
                n_terms += 1

        out = {
            'w': w, 'w_xx': w_xx, 'w_yy': w_yy, 'w_xy': w_xy,
            # StrainNeuron konvansiyonu: κ_ij = -w_ij
            'kappa_xx': -w_xx, 'kappa_yy': -w_yy, 'kappa_xy': -w_xy,
            'w_max': w.abs().max(),
            'r_max': r_max,
            'buckled': r_max >= 1.0,
        }

        if explain:
            self._explain_navier(label, q, N_x, N_y, mindlin, n_modes,
                                 n_terms, D, out)
        return out

    def _explain_navier(self, label, q, N_x, N_y, mindlin, n_modes,
                        n_terms, D, out):
        """Navier çözümünü adım adım yazdır (şeffaf mod)."""
        sep = "-" * 68
        head = "  StaticNeuron — Navier serisi çözümü"
        if label:
            head += f" [{label}]"
        print(f"{sep}\n{head}\n{sep}")
        print(f"    Plaka: Lx = {self.Lx:.4g} m, Ly = {self.Ly:.4g} m, "
              f"h = {self.material.h:.4g} m   |   D = {D:.6g} N·m")
        print(f"    Uniform yük q = {q:.6g} Pa")
        print(f"    Düzlem içi bası: N_x = {N_x/1e3:.4g} kN/m, "
              f"N_y = {N_y/1e3:.4g} kN/m")
        print(f"    Seri: m,n ≤ {n_modes} (tek modlar) → {n_terms} terim, "
              f"türevler analitik (FFT yok)")
        print(f"    w_mn = q_mn/[D(α_m²+α_n²)²],  q_mn = 16q/(π²mn)")
        if mindlin:
            print("    Mindlin kayma esnekliği çarpanı uygulandı")
        amp1 = (1.0 / (1.0 - min(out['r_max'], 0.99)) if out['r_max'] < 1.0
                else float('inf'))
        print(f"    P-Δ: en büyük mod oranı r = N/N_cr,mn = {out['r_max']:.6f}"
              f"  → büyütme ≈ {amp1:.4f}×")
        print(f"    → w_max = {out['w_max'].item()*1e3:.6g} mm"
              f"{'   [MOD BURKULDU — r ≥ 1]' if out['buckled'] else ''}")
        print(sep)

    def navier_coefficients(self, q: torch.Tensor) -> torch.Tensor:
        """
        Yük alanından Navier katsayılarını hesapla.

        qmn = (4/LW) ∫∫ q(x,y)·sin(mπx/L)·sin(nπy/W) dxdy
        """
        coeffs = []
        for m in range(1, self.n_modes + 1):
            for n in range(1, self.n_modes + 1):
                phi_mn = torch.sin(m * math.pi * self.X / self.Lx) * \
                         torch.sin(n * math.pi * self.Y / self.Ly)
                
                # Numerik integrasyon (trapez kuralı)
                q_mn = (4 / (self.Lx * self.Ly)) * (q * phi_mn).mean() * self.Lx * self.Ly
                coeffs.append(q_mn)
        
        return torch.stack(coeffs)
    
    def solve(self, q: Union[torch.Tensor, float]) -> torch.Tensor:
        """
        Statik eğilme problemini çöz.
        
        Args:
            q: Dağıtılmış yük [Nx, Ny] veya uniform yük (skalar)
        
        Returns:
            w: Deplasman alanı [Nx, Ny]
        """
        D = self.D * self.stiffness_modulator
        
        if isinstance(q, (int, float)):
            # Uniform yük için analitik Navier serisi
            q_val = float(q)
            w = torch.zeros_like(self.X)
            
            for m in range(1, self.n_modes + 1):
                for n in range(1, self.n_modes + 1):
                    # Uniform yük için qmn
                    if m % 2 == 1 and n % 2 == 1:  # Sadece tek modlar
                        q_mn = 16 * q_val / (math.pi**2 * m * n)
                    else:
                        continue
                    
                    # Deplasman katsayısı
                    alpha_m = m * math.pi / self.Lx
                    alpha_n = n * math.pi / self.Ly
                    W_mn = q_mn / (D * (alpha_m**2 + alpha_n**2)**2)
                    
                    # Mod şekli
                    phi_mn = torch.sin(m * math.pi * self.X / self.Lx) * \
                             torch.sin(n * math.pi * self.Y / self.Ly)
                    
                    w = w + W_mn * phi_mn
            
            return w
        else:
            # Genel yük dağılımı
            q_coeffs = self.navier_coefficients(q)
            w = torch.zeros_like(self.X)
            
            idx = 0
            for m in range(1, self.n_modes + 1):
                for n in range(1, self.n_modes + 1):
                    q_mn = q_coeffs[idx]
                    idx += 1
                    
                    alpha_m = m * math.pi / self.Lx
                    alpha_n = n * math.pi / self.Ly
                    W_mn = q_mn / (D * (alpha_m**2 + alpha_n**2)**2)
                    
                    phi_mn = torch.sin(m * math.pi * self.X / self.Lx) * \
                             torch.sin(n * math.pi * self.Y / self.Ly)
                    
                    w = w + W_mn * phi_mn
            
            return w
    
    def max_displacement(self, q: float) -> float:
        """
        Uniform yük altında maksimum deplasman (plaka merkezinde).
        
        Basit mesnetli kare plaka için: w_max ≈ 0.00416 qL⁴/D
        """
        w = self.solve(q)
        return w.max().item()
    
    def max_stress(self, q: float) -> Tuple[float, float]:
        """
        Maksimum eğilme gerilmeleri (Kirchhoff plate teorisi, tam form).

        σ_xx = -E·z/(1-ν²) · (∂²w/∂x² + ν·∂²w/∂y²)
        σ_yy = -E·z/(1-ν²) · (∂²w/∂y² + ν·∂²w/∂x²)

        Not: Eğrilik gösteriminde κ_x = -∂²w/∂x², κ_y = -∂²w/∂y² alınır;
        bu nedenle σ = E·z/(1-ν²) · (κ_x + ν·κ_y) ile aynı formdur ve
        `bending_moments` Mx = D(κ_x + ν·κ_y) formülüyle tutarlıdır.
        """
        w = self.solve(q)
        w_xx, w_yy, _ = self.spectral_ops.second_derivatives(w)

        z_max = self.material.h / 2
        E = self.material.E
        nu = self.material.nu

        # Kirchhoff plate stress — ν cross-coupling terimleri dahil
        prefactor = -E * z_max / (1 - nu**2)
        sigma_xx = prefactor * (w_xx + nu * w_yy)
        sigma_yy = prefactor * (w_yy + nu * w_xx)

        return sigma_xx.abs().max().item(), sigma_yy.abs().max().item()
    
    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """Statik çözüm."""
        return self.solve(q)
    
    def extra_repr(self) -> str:
        return f"Lx={self.Lx}, Ly={self.Ly}, D={self.D:.2e}, n_modes={self.n_modes}"


class ModalNeuron(nn.Module):
    """
    Modal Analiz Nöronu: Doğal Frekanslar ve Mod Şekilleri.
    
    Titreşim denklemi:
    D∇⁴w = ρh·ω²·w
    
    Doğal frekanslar (basit mesnetli):
    ωmn = π²·[(m/L)² + (n/W)²]·√(D/ρh)
    
    Args:
        resolution: Grid çözünürlüğü
        Lx, Ly: Domain boyutları (m)
        material: Malzeme özellikleri
        rho: Yoğunluk (kg/m³)
    """
    
    def __init__(self, resolution: int, Lx: float, Ly: float,
                 material: MaterialProperties, rho: float,
                 spectral_ops: Optional[SpectralOps2DStruct] = None):
        super().__init__()
        
        self.resolution = resolution
        self.Lx = Lx
        self.Ly = Ly
        self.material = material
        self.D = material.D
        self.h = material.h
        self.rho = rho  # Yoğunluk
        
        self.spectral_ops = spectral_ops if spectral_ops else \
                           SpectralOps2DStruct(resolution, Lx, Ly)
        
        # Grid
        x = torch.linspace(0, Lx, resolution)
        y = torch.linspace(0, Ly, resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        self.register_buffer('X', X)
        self.register_buffer('Y', Y)
        
        # Öğrenilebilir parametreler
        self.mass_modulator = nn.Parameter(torch.ones(1))
        self.stiffness_modulator = nn.Parameter(torch.ones(1))
    
    def natural_frequency(self, m: int = 1, n: int = 1) -> float:
        """
        Doğal frekans (rad/s).
        
        ωmn = π²·[(m/L)² + (n/W)²]·√(D/ρh)
        """
        D = self.D * self.stiffness_modulator.item()
        rho_h = self.rho * self.h * self.mass_modulator.item()
        
        term = (m / self.Lx)**2 + (n / self.Ly)**2
        omega = math.pi**2 * term * math.sqrt(D / rho_h)
        
        return omega
    
    def natural_frequency_hz(self, m: int = 1, n: int = 1) -> float:
        """Doğal frekans (Hz)."""
        return self.natural_frequency(m, n) / (2 * math.pi)
    
    def mode_shape(self, m: int = 1, n: int = 1) -> torch.Tensor:
        """
        Mod şekli.
        
        φmn(x,y) = sin(mπx/L)·sin(nπy/W)
        """
        phi = torch.sin(m * math.pi * self.X / self.Lx) * \
              torch.sin(n * math.pi * self.Y / self.Ly)
        return phi
    
    def find_modes(self, max_m: int = 5, max_n: int = 5) -> List[Dict]:
        """
        Tüm modları bul ve frekansa göre sırala.
        
        Returns:
            List[Dict]: [{m, n, omega, f_hz}, ...]
        """
        modes = []
        for m in range(1, max_m + 1):
            for n in range(1, max_n + 1):
                omega = self.natural_frequency(m, n)
                f_hz = omega / (2 * math.pi)
                modes.append({
                    'm': m,
                    'n': n,
                    'omega': omega,
                    'f_hz': f_hz
                })
        
        modes.sort(key=lambda x: x['omega'])
        return modes
    
    def modal_mass(self, m: int = 1, n: int = 1) -> float:
        """
        Modal kütle.
        
        Mmn = ∫∫ ρh·φmn² dA = ρh·L·W/4
        """
        return self.rho * self.h * self.Lx * self.Ly / 4
    
    def modal_stiffness(self, m: int = 1, n: int = 1) -> float:
        """
        Modal rijitlik.
        
        Kmn = Mmn·ωmn²
        """
        M_mn = self.modal_mass(m, n)
        omega = self.natural_frequency(m, n)
        return M_mn * omega**2
    
    def rayleigh_quotient(self, w: torch.Tensor) -> torch.Tensor:
        """
        Rayleigh quotient ile frekans tahmini.
        
        ω² = ∫D(∇²w)² dA / ∫ρh·w² dA
        """
        lap_w = self.spectral_ops.laplacian(w)
        
        bending_energy = self.D * (lap_w ** 2).mean()
        kinetic_ref = self.rho * self.h * (w ** 2).mean()
        
        omega_sq = bending_energy / (kinetic_ref + 1e-10)
        return torch.sqrt(omega_sq)
    
    def forward(self, m: int = 1, n: int = 1) -> Tuple[float, torch.Tensor]:
        """Mod frekansı ve şekli döndür."""
        omega = self.natural_frequency(m, n)
        phi = self.mode_shape(m, n)
        return omega, phi
    
    def extra_repr(self) -> str:
        return f"Lx={self.Lx}, Ly={self.Ly}, D={self.D:.2e}, rho={self.rho}"


class CrackNeuron(nn.Module):
    """
    Kırılma Mekaniği Nöronu: Stress Intensity Factors.
    
    Williams asimptotik alanı:
    σij = K_I / √(2πr) · f_I(θ) + K_II / √(2πr) · f_II(θ)
    
    Stress Intensity Factors:
    - K_I: Mode I (açılma)
    - K_II: Mode II (kayma, düzlem içi)
    - K_III: Mode III (kayma, düzlem dışı)
    
    Args:
        material: Malzeme özellikleri
        crack_length: Çatlak uzunluğu (m)
        width: Numune genişliği (m)
    """
    
    def __init__(self, material: MaterialProperties,
                 crack_length: float, width: float):
        super().__init__()
        
        self.E = material.E
        self.nu = material.nu
        self.a = crack_length  # Yarım çatlak uzunluğu
        self.W = width
        
        # Düzlem gerilme/şekil değiştirme
        self.plane_strain = True
        
        # Öğrenilebilir parametre
        self.K_scale = nn.Parameter(torch.ones(1))
    
    def geometry_factor_center(self, a_W: float) -> float:
        """
        Merkezi çatlak için geometri faktörü Y(a/W).
        
        Sonsuz plaka: Y = 1
        Sonlu genişlik: Y = √(sec(πa/2W))
        """
        if a_W < 0.7:
            # Secant yaklaşımı
            Y = math.sqrt(1 / math.cos(math.pi * a_W / 2))
        else:
            # Feddersen formülü
            Y = math.sqrt(1 / math.cos(math.pi * a_W / 2)) * \
                (1 - 0.025 * a_W**2 + 0.06 * a_W**4)
        return Y
    
    def geometry_factor_edge(self, a_W: float) -> float:
        """
        Kenar çatlağı için geometri faktörü.
        
        Y = 1.12 - 0.231(a/W) + 10.55(a/W)² - 21.72(a/W)³ + 30.39(a/W)⁴
        """
        Y = 1.12 - 0.231 * a_W + 10.55 * a_W**2 - \
            21.72 * a_W**3 + 30.39 * a_W**4
        return Y
    
    def K_I(self, sigma: float, crack_type: str = 'center') -> float:
        """
        Mode I Stress Intensity Factor.
        
        K_I = σ·√(πa)·Y(a/W)
        
        Args:
            sigma: Uzak alan gerilmesi (Pa)
            crack_type: 'center' veya 'edge'
        """
        a_W = self.a / self.W
        
        if crack_type == 'center':
            Y = self.geometry_factor_center(a_W)
        else:
            Y = self.geometry_factor_edge(a_W)
        
        K = sigma * math.sqrt(math.pi * self.a) * Y * self.K_scale.item()
        return K
    
    def K_II(self, tau: float) -> float:
        """
        Mode II Stress Intensity Factor.
        
        K_II = τ·√(πa)·Y
        """
        a_W = self.a / self.W
        Y = self.geometry_factor_center(a_W)
        
        K = tau * math.sqrt(math.pi * self.a) * Y * self.K_scale.item()
        return K
    
    def K_III(self, tau_zx: float) -> float:
        """
        Mode III Stress Intensity Factor.
        
        K_III = τ_zx·√(πa)
        """
        K = tau_zx * math.sqrt(math.pi * self.a) * self.K_scale.item()
        return K
    
    def equivalent_K(self, K_I: float, K_II: float, K_III: float = 0) -> float:
        """
        Eşdeğer (kombine) SIF.
        
        K_eq = √(K_I² + K_II² + K_III²/(1-ν))
        """
        K_eq = math.sqrt(K_I**2 + K_II**2 + K_III**2 / (1 - self.nu))
        return K_eq
    
    def J_integral(self, K_I: float, K_II: float = 0, K_III: float = 0) -> float:
        """
        J-integral (enerji salınım oranı).
        
        Düzlem gerilme: J = K²/E
        Düzlem şekil değiştirme: J = K²(1-ν²)/E
        """
        K_eq = self.equivalent_K(K_I, K_II, K_III)
        
        if self.plane_strain:
            J = K_eq**2 * (1 - self.nu**2) / self.E
        else:
            J = K_eq**2 / self.E
        
        return J
    
    def critical_crack_length(self, sigma: float, K_Ic: float) -> float:
        """
        Kritik çatlak uzunluğu.
        
        a_c = (K_Ic / σY)² / π
        """
        a_W = self.a / self.W
        Y = self.geometry_factor_center(a_W)
        
        a_c = (K_Ic / (sigma * Y))**2 / math.pi
        return a_c
    
    def safety_factor(self, sigma: float, K_Ic: float, 
                      crack_type: str = 'center') -> float:
        """
        Güvenlik faktörü.
        
        SF = K_Ic / K_I
        """
        K_I = self.K_I(sigma, crack_type)
        return K_Ic / K_I
    
    def forward(self, sigma: float, tau: float = 0) -> Dict[str, float]:
        """
        Tüm SIF'leri hesapla.
        """
        K_I = self.K_I(sigma)
        K_II = self.K_II(tau) if tau != 0 else 0
        K_eq = self.equivalent_K(K_I, K_II)
        J = self.J_integral(K_I, K_II)
        
        return {
            'K_I': K_I,
            'K_II': K_II,
            'K_eq': K_eq,
            'J': J
        }
    
    def extra_repr(self) -> str:
        return f"a={self.a}, W={self.W}, E={self.E:.2e}"


class ThermalNeuron(nn.Module):
    """
    Termal Analiz Nöronu.
    
    Termal strain: εT = α·ΔT
    Termal stress: σT = -Eα(T-T0)/(1-ν) (tam kısıtlı)
    Termal burkulma: NT,cr
    
    Args:
        material: Malzeme özellikleri
        alpha: Termal genleşme katsayısı (1/K)
    """
    
    def __init__(self, material: MaterialProperties, alpha: float,
                 resolution: int = 64, Lx: float = 1.0, Ly: float = 1.0):
        super().__init__()
        
        self.E = material.E
        self.nu = material.nu
        self.h = material.h
        self.D = material.D
        self.alpha = alpha  # Termal genleşme katsayısı
        
        self.resolution = resolution
        self.Lx = Lx
        self.Ly = Ly
        
        # Grid
        x = torch.linspace(0, Lx, resolution)
        y = torch.linspace(0, Ly, resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        self.register_buffer('X', X)
        self.register_buffer('Y', Y)
        
        # Öğrenilebilir parametre
        self.thermal_modulator = nn.Parameter(torch.ones(1))
    
    def thermal_strain(self, delta_T: float) -> float:
        """
        Termal strain.
        
        εT = α·ΔT
        """
        return self.alpha * delta_T * self.thermal_modulator.item()
    
    def thermal_stress_constrained(self, delta_T: float) -> float:
        """
        Tam kısıtlı termal gerilme.
        
        σT = -Eα·ΔT/(1-ν)
        """
        eps_T = self.thermal_strain(delta_T)
        sigma = -self.E * eps_T / (1 - self.nu)
        return sigma
    
    def thermal_membrane_force(self, delta_T: float) -> float:
        """
        Termal membran kuvveti (plaka için).
        
        NT = -Eα·ΔT·h/(1-ν)
        """
        sigma = self.thermal_stress_constrained(delta_T)
        NT = sigma * self.h
        return NT
    
    def critical_thermal_buckling(self, m: int = 1, n: int = 1,
                                  loading: str = "biaxial",
                                  direction: str = "x",
                                  mindlin: bool = False,
                                  explain: bool = False,
                                  label: str = "") -> float:
        r"""
        Kritik termal burkulma sıcaklık farkı ΔT_cr.

        |N_T(ΔT_cr)| = N_cr koşulundan:

            E·α·ΔT_cr·h/(1-ν) = N_cr   →   ΔT_cr = N_cr(1-ν)/(E·α·h)

        VARSAYILAN 'biaxial'dir ve bu bilinçli bir seçimdir: düzlem içinde
        tam kısıtlı bir plaka ısındığında her iki yönde de genleşemez, yani
        N_x = N_y = N_T oluşur. Kritik yükün uniaxial formülüyle hesaplanması
        (bu metodun eski davranışı) kare plakada N_cr'yi — dolayısıyla
        ΔT_cr'yi — 2 kat FAZLA, yani güvensiz yönde tahmin ediyordu.
        Tek eksenli kısıtlama için açıkça loading='uniaxial' verilmelidir.

        Args:
            m, n: Burkulma mod numaraları.
            loading: 'biaxial' (varsayılan, tam kısıtlı plaka) veya 'uniaxial'.
            direction: uniaxial halde kısıtlama yönü ('x' veya 'y').
            mindlin: Enine kayma düzeltmesi uygulansın mı.
            explain: True ise adımlar sayısal değerleriyle yazdırılır.
            label: explain çıktısı için etiket.

        Returns:
            ΔT_cr: Kritik sıcaklık farkı [K] (pozitif büyüklük).
        """
        N_cr, N_cr_kirchhoff = plate_critical_load(
            self.D, self.Lx, self.Ly, m=m, n=n,
            loading=loading, direction=direction, mindlin=mindlin,
            E=self.E, nu=self.nu, h=self.h)

        # NT = N_cr → E·α·ΔT·h/(1-ν) = N_cr
        delta_T_cr = abs(N_cr * (1 - self.nu) / (self.E * self.alpha * self.h))

        if explain:
            sep = "-" * 68
            head = "  ThermalNeuron — kritik termal burkulma"
            if label:
                head += f" [{label}]"
            kind = (f"tek eksenli ({direction})" if loading == "uniaxial"
                    else "iki eksenli (tam kısıtlı plaka)")
            print(f"{sep}\n{head}\n{sep}")
            print(f"    Kısıtlama: {kind}   |   mod (m,n) = ({m},{n})")
            print(f"    D = {self.D:.6g} N·m,  α = {self.alpha:.4g} 1/K,  "
                  f"h = {self.h:.4g} m")
            print(f"    N_cr = {N_cr:.6g} N/m"
                  + (f"  (Mindlin öncesi {N_cr_kirchhoff:.6g})" if mindlin else ""))
            print(f"    ΔT_cr = N_cr(1-ν)/(E·α·h) = {delta_T_cr:.6g} K")
            print(sep)

        return delta_T_cr
    
    def temperature_field(self, T_top: float, T_bottom: float) -> torch.Tensor:
        """
        Doğrusal sıcaklık dağılımı (kalınlık boyunca).
        
        ΔT = T_top - T_bottom
        """
        T_avg = (T_top + T_bottom) / 2
        delta_T = T_top - T_bottom
        
        # Sıcaklık alanı (düzlemde uniform varsayım)
        T_field = torch.ones_like(self.X) * T_avg
        return T_field
    
    def thermal_moment(self, T_top: float, T_bottom: float) -> float:
        """
        Termal moment (sıcaklık gradyanından).
        
        MT = Eαh²·ΔT / [12(1-ν)]
        """
        delta_T = T_top - T_bottom
        MT = self.E * self.alpha * self.h**2 * delta_T / (12 * (1 - self.nu))
        return MT
    
    def thermal_curvature(self, T_top: float, T_bottom: float) -> float:
        """
        Termal eğrilik.
        
        κT = α·ΔT/h
        """
        delta_T = T_top - T_bottom
        kappa = self.alpha * delta_T / self.h
        return kappa
    
    def critical_mode_scan(self, max_m: int = 6, max_n: int = 6,
                           loading: str = "biaxial",
                           direction: str = "x",
                           mindlin: bool = False) -> Dict[str, float]:
        """
        (m,n) uzayını tarayıp en düşük kritik yükü veren termal burkulma
        modunu bul. Dikdörtgen panelde kritik mod (1,1) olmayabilir.

        Returns:
            {'N_cr':…, 'm':…, 'n':…, 'delta_T_cr':…}
        """
        best = None
        for m in range(1, max_m + 1):
            for n in range(1, max_n + 1):
                N_cr, _ = plate_critical_load(
                    self.D, self.Lx, self.Ly, m=m, n=n,
                    loading=loading, direction=direction, mindlin=mindlin,
                    E=self.E, nu=self.nu, h=self.h)
                if best is None or N_cr < best['N_cr']:
                    best = {'N_cr': N_cr, 'm': m, 'n': n}

        best['delta_T_cr'] = abs(
            best['N_cr'] * (1 - self.nu) / (self.E * self.alpha * self.h))
        return best

    def forward(self, delta_T: float,
                loading: str = "biaxial",
                direction: str = "x",
                mindlin: bool = False,
                max_mode: int = 6,
                explain: bool = False,
                label: str = "") -> Dict[str, float]:
        r"""
        Tam termal burkulma zinciri:

            ΔT → ε_T = αΔT → σ_T = -EαΔT/(1-ν) → N_T = σ_T·h
               → N_cr (mod taraması) → SF = N_cr/|N_T| , ΔT_cr

        Args:
            delta_T: Sıcaklık farkı [K].
            loading, direction, mindlin: Kritik yük hesabının ayarları.
            max_mode: Mod taramasının üst sınırı (m,n ≤ max_mode).
            explain: True ise her adım sayısal değeriyle yazdırılır.
            label: explain çıktısı için etiket.

        Returns:
            eps_T, sigma_T, N_T, N_cr, m, n, delta_T_cr, SF_thermal
        """
        eps_T = self.thermal_strain(delta_T)
        sigma_T = self.thermal_stress_constrained(delta_T)
        N_T = self.thermal_membrane_force(delta_T)
        best = self.critical_mode_scan(max_mode, max_mode,
                                       loading=loading, direction=direction,
                                       mindlin=mindlin)
        SF = (best['N_cr'] / abs(N_T)) if abs(N_T) > 0 else float('inf')

        out = {
            'eps_T': eps_T,
            'sigma_T': sigma_T,
            'N_T': N_T,
            'N_cr': best['N_cr'],
            'm': best['m'],
            'n': best['n'],
            'delta_T_cr': best['delta_T_cr'],
            'SF_thermal': SF,
        }

        if explain:
            self._explain(label, delta_T, loading, direction, mindlin,
                          max_mode, out)
        return out

    def _explain(self, label, delta_T, loading, direction, mindlin,
                 max_mode, out):
        """Termal burkulma zincirini adım adım yazdır (şeffaf mod)."""
        sep = "=" * 68
        title = " ThermalNeuron — Adım Adım Termal Burkulma "
        if label:
            title += f"[{label}] "
        print(f"\n{sep}\n{title}\n{sep}")

        kind = (f"tek eksenli ({direction})" if loading == "uniaxial"
                else "iki eksenli (düzlem içi tam kısıtlı)")
        print("[0] KURULUM")
        print(f"    Panel: Lx = {self.Lx:.4g} m, Ly = {self.Ly:.4g} m, "
              f"h = {self.h:.4g} m")
        print(f"    E = {self.E:.4g} Pa, ν = {self.nu:.4g}, "
              f"α = {self.alpha:.4g} 1/K")
        print(f"    Sıcaklık farkı ΔT = {delta_T:.4g} K   |   kısıtlama: {kind}")
        print(f"    D = E·h³/[12(1-ν²)] = {self.D:.6g} N·m")

        print("[1] TERMAL GENLEŞME — serbest olsaydı")
        print(f"    ε_T = α·ΔT = {out['eps_T']:.6g}"
              f"   ({out['eps_T']*1e6:.4g} µstrain)")

        print("[2] KISITLAMA — genleşme engellendiği için gerilme doğar")
        isaret = ("bası" if out['sigma_T'] < 0 else
                  "çeki" if out['sigma_T'] > 0 else "yok")
        print(f"    σ_T = -E·α·ΔT/(1-ν) = {out['sigma_T']/1e6:.6g} MPa"
              f"   ({isaret})")

        print("[3] MEMBRAN KUVVETİ — kalınlık boyunca integral")
        print(f"    N_T = σ_T·h = {out['N_T']/1e3:.6g} kN/m"
              f"   (büyüklük {abs(out['N_T'])/1e3:.6g} kN/m)")

        print(f"[4] KRİTİK YÜK — (m,n) taraması, m,n ≤ {max_mode}"
              f"{', Mindlin düzeltmeli' if mindlin else ''}")
        print(f"    N_cr = {out['N_cr']/1e3:.6g} kN/m"
              f"   en kritik mod (m,n) = ({out['m']},{out['n']})")

        print("[5] HÜKÜM")
        sf = out['SF_thermal']
        durum = ("BURKULDU" if sf < 1.0 else
                 "SINIRDA" if sf < 1.5 else "GÜVENLİ")
        print(f"    SF = N_cr/|N_T| = {sf:.4f}   → {durum}")
        print(f"    ΔT_cr = {out['delta_T_cr']:.6g} K"
              f"   (mevcut ΔT = {delta_T:.4g} K)")
        print(sep)
    
    def extra_repr(self) -> str:
        return f"alpha={self.alpha:.2e}, E={self.E:.2e}"


class HeatConductionNeuron(nn.Module):
    r"""
    Kararlı-Hal Isı İletimi Nöronu.

    Poisson denklemi:   ∇²T = -f/k   (kaynaklı),  akı:  q = -k·∇T

    İki türev yolu sunar ve ikisi de bilinçli seçimdir:

    • method='spectral' — FFT ile türev. Düzgün, süreksizliksiz alanlarda
      makine hassasiyetine yakın doğruluk verir. Ancak yalıtılmış bir çatlak
      gibi SÜREKSİZLİK varsa Gibbs salınımı üretir; orada kullanılmamalıdır.

    • method='fd' — ikinci mertebe sonlu fark, sınırlarda tek taraflı.
      Süreksizlik ve Dirichlet/Neumann sınırlarıyla uyumludur.

    Bu ayrım önemlidir: SPINE'ın spektral çekirdeği her probleme uygun
    değildir ve nöron bunu gizlemez.

    Args:
        resolution: Grid çözünürlüğü.
        Lx, Ly: Domain boyutları [m].
        k: Isı iletim katsayısı [W/(m·K)].
        spectral_ops: Hazır SpectralOps2DStruct (yoksa kurulur).
    """

    def __init__(self, resolution: int, Lx: float, Ly: float,
                 k: float = 1.0,
                 spectral_ops: Optional[SpectralOps2DStruct] = None):
        super().__init__()
        self.resolution = resolution
        self.Lx = Lx
        self.Ly = Ly
        self.k = k
        self.dx = Lx / (resolution - 1)
        self.dy = Ly / (resolution - 1)
        self.spectral_ops = spectral_ops if spectral_ops else \
            SpectralOps2DStruct(resolution, Lx, Ly)

    # --------------------------------------------------------------- türev
    def gradient_fd(self, T: torch.Tensor
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        İkinci mertebe sonlu farkla gradyan; sınırlarda tek taraflı.

            iç:     (T[i+1] - T[i-1]) / (2h)
            sol:    (-3T[0] + 4T[1] - T[2]) / (2h)
            sağ:    ( 3T[-1] - 4T[-2] + T[-3]) / (2h)

        Tek taraflı formüller de ikinci mertebedendir; kenarlarda doğruluk
        düşmez.
        """
        dTdx = torch.zeros_like(T)
        dTdy = torch.zeros_like(T)

        dTdx[1:-1, :] = (T[2:, :] - T[:-2, :]) / (2 * self.dx)
        dTdx[0, :] = (-3 * T[0, :] + 4 * T[1, :] - T[2, :]) / (2 * self.dx)
        dTdx[-1, :] = (3 * T[-1, :] - 4 * T[-2, :] + T[-3, :]) / (2 * self.dx)

        dTdy[:, 1:-1] = (T[:, 2:] - T[:, :-2]) / (2 * self.dy)
        dTdy[:, 0] = (-3 * T[:, 0] + 4 * T[:, 1] - T[:, 2]) / (2 * self.dy)
        dTdy[:, -1] = (3 * T[:, -1] - 4 * T[:, -2] + T[:, -3]) / (2 * self.dy)

        return dTdx, dTdy

    def flux(self, T: torch.Tensor, method: str = "fd"
             ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Isı akısı q = -k·∇T.

        Args:
            method: 'fd' (süreksizlik/sınır varsa) veya 'spectral'.
        """
        if method == "fd":
            dTdx, dTdy = self.gradient_fd(T)
        elif method == "spectral":
            dTdx, dTdy = self.spectral_ops.gradient(T)
        else:
            raise ValueError("method 'fd' veya 'spectral' olmalı.")
        return -self.k * dTdx, -self.k * dTdy

    def laplacian_fd(self, T: torch.Tensor) -> torch.Tensor:
        """5 noktalı sonlu fark Laplasyeni (kenarlar sıfır bırakılır)."""
        lap = torch.zeros_like(T)
        lap[1:-1, 1:-1] = (
            (T[2:, 1:-1] + T[:-2, 1:-1] - 2 * T[1:-1, 1:-1]) / self.dx ** 2
            + (T[1:-1, 2:] + T[1:-1, :-2] - 2 * T[1:-1, 1:-1]) / self.dy ** 2
        )
        return lap

    def poisson_residual(self, T: torch.Tensor, f: torch.Tensor,
                         method: str = "fd") -> torch.Tensor:
        r"""
        ∇²T - f rezidüeli. Sıfıra ne kadar yakınsa çözüm o kadar tutarlıdır.

        Not: f burada doğrudan ∇²T'nin karşılığıdır (k ile bölünmüş hâli
        bekleniyorsa çağıran taraf ölçekler).
        """
        if method == "fd":
            lap = self.laplacian_fd(T)
        elif method == "spectral":
            lap = self.spectral_ops.laplacian(T)
        else:
            raise ValueError("method 'fd' veya 'spectral' olmalı.")
        return lap - f

    def solve_spectral(self, f: torch.Tensor,
                       mean_value: float = 0.0) -> torch.Tensor:
        r"""
        Periyodik domainde ∇²T = f çözümü (FFT).

            T̂ = -f̂ / k²,   k=0 modu `mean_value` ile sabitlenir

        Yalnızca süreksizliksiz ve periyodik hâller için uygundur;
        Dirichlet sınırı veya çatlak varsa sonuç yanıltıcıdır.
        """
        f_hat = safe_fft2(f)
        k2 = self.spectral_ops.k_squared
        k2_safe = k2.clone()
        k2_safe[0, 0] = 1.0

        T_hat = -f_hat / k2_safe
        T_hat[0, 0] = 0.0
        T = safe_ifft2(T_hat).real
        return T - T.mean() + mean_value

    def forward(self, T: torch.Tensor, method: str = "fd"):
        """nn.Module arayüzü — akı hesabı."""
        return self.flux(T, method=method)

    def extra_repr(self) -> str:
        return f"resolution={self.resolution}, Lx={self.Lx}, Ly={self.Ly}, k={self.k}"


class PhaseFieldNeuron(nn.Module):
    r"""
    AT2 Faz-Alanı Hasar Nöronu — çatlak, ayrı bir yüzey olarak değil,
    sürekli bir hasar alanı d ∈ [0,1] olarak temsil edilir (d=0 sağlam,
    d=1 tamamen çatlak).

    Denge denklemi (AT2, ℓ karakteristik çatlak bandı genişliği):

        (1 + 2ℓH/G_c)·d - ℓ²∇²d = 2ℓH/G_c

    Burada H, tarihçe alanıdır: H = max(H_önceki, ψ⁺). Tarihçe kullanmak
    çatlağın GERİ DÖNMEMESİNİ (irreversibility) sağlar — yük kalksa bile
    hasar kaybolmaz.

    Denklem Helmholtz tipindedir ve FFT ile çözülür: sabit katsayılı kısım
    frekans uzayında doğrudan bölünür, uzayda değişen kısım sabit-nokta
    yinelemesiyle taşınır.

    İki mod:
      • mode='at2'    : yukarıdaki tam form (sol tarafta H'ye bağlı katsayı)
      • mode='linear' : α ≡ 1 alan lineerleştirilmiş form,
                        d - ℓ²∇²d = 2ℓH/G_c
        İkincisi daha yumuşak ve gradyan açısından daha kararlıdır; ters
        problemde (G_c kestirimi) tercih edilir.

    G_c tensör olabilir ve ona göre türev alınabilir — 45 malzemelik
    tersine mühendislikte kırılma tokluğunun kestirilebilmesinin sebebi budur.

    Args:
        resolution: Grid çözünürlüğü.
        Lx, Ly: Domain boyutları [m].
        ell: Karakteristik uzunluk ℓ [m]; tipik olarak 2-3 grid aralığı.
        G_c: Kritik enerji salım oranı [J/m²].
        eta: Bozunum fonksiyonundaki artık rijitlik (sayısal tekilliği önler).
        spectral_ops: Hazır SpectralOps2DStruct (yoksa kurulur).
    """

    def __init__(self, resolution: int, Lx: float, Ly: float,
                 ell: float, G_c: float, eta: float = 1e-7,
                 spectral_ops: Optional[SpectralOps2DStruct] = None):
        super().__init__()
        self.resolution = resolution
        self.Lx = Lx
        self.Ly = Ly
        self.ell = ell
        self.eta = eta
        self.spectral_ops = spectral_ops if spectral_ops else \
            SpectralOps2DStruct(resolution, Lx, Ly)
        self.register_buffer('G_c', torch.tensor(float(G_c)))

    # ------------------------------------------------------------ bozunum
    def degradation(self, d: torch.Tensor) -> torch.Tensor:
        r"""Bozunum fonksiyonu g(d) = (1-d)² + η.

        Hasarlı bölgede rijitlik g(d) ile çarpılır; η, d→1 iken sistemin
        tekilleşmesini önleyen küçük artık rijitliktir.
        """
        return (1.0 - d) ** 2 + self.eta

    @staticmethod
    def energy_density_strain(eps_xx: torch.Tensor, eps_yy: torch.Tensor,
                              eps_xy: torch.Tensor,
                              lam, mu, eps: float = 1e-30) -> torch.Tensor:
        r"""
        Gerinimden çekme enerjisi ψ⁺ (spektral ayrışım).

            ψ⁺ = ½λ⟨tr ε⟩₊² + µ(⟨ε_1⟩₊² + ⟨ε_2⟩₊²)

        ⟨·⟩₊ Macaulay parantezidir. `StrainNeuron.energy_density_split`
        aynı ayrışımı GERİLMEDEN başlayarak yapar; hangisi elde varsa o
        kullanılır.

        Not: küçük gerinimlerde (~1e-5) softplus gibi yumuşak yaklaşımlar
        büyük bağıl hata verir; burada tam Macaulay parantezi (relu)
        kullanılır.
        """
        tr_eps = eps_xx + eps_yy
        ortalama = 0.5 * tr_eps
        R = torch.sqrt(0.25 * (eps_xx - eps_yy) ** 2 + eps_xy ** 2 + eps)

        e1_plus = torch.relu(ortalama + R)
        e2_plus = torch.relu(ortalama - R)
        tr_plus = torch.relu(tr_eps)

        return 0.5 * lam * tr_plus ** 2 + mu * (e1_plus ** 2 + e2_plus ** 2)

    # ------------------------------------------------------------- çözüm
    def solve_damage(self, psi_plus: torch.Tensor,
                     d_prev: torch.Tensor,
                     history_H: Optional[torch.Tensor] = None,
                     G_c: Optional[torch.Tensor] = None,
                     mode: str = "at2",
                     n_iter: int = 8,
                     relax: float = 1.0,
                     source_cap: Optional[float] = None,
                     soft_clamp: bool = False,
                     explain: bool = False,
                     label: str = "") -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Hasar alanını bir adım ilerlet.

        Args:
            psi_plus: Çekme enerji yoğunluğu alanı ψ⁺.
            d_prev: Önceki hasar alanı.
            history_H: Tarihçe alanı; None ise ψ⁺ ile başlatılır.
            G_c: Kırılma tokluğu; None ise modülün değeri (ters problemde
                dışarıdan optimize edilen tensör buradan geçirilir).
            mode: 'at2' (tam) veya 'linear' (α≡1).
            n_iter: Sabit-nokta yineleme sayısı.
            relax: Gevşetme katsayısı (0<relax≤1); 1 tam güncelleme.
            source_cap: Kaynak terimine üst sınır (G_c çok küçükken
                sayısal patlamayı önler).
            soft_clamp: True ise [0,1] kısıtı sigmoid ile yumuşak uygulanır
                (gradyan her yerde akar); False ise sert clamp.
            explain: True ise adımlar yazdırılır.

        Returns:
            (d_yeni, H): güncel hasar alanı ve tarihçe.
        """
        if mode not in ("at2", "linear"):
            raise ValueError("mode 'at2' veya 'linear' olmalı.")

        gc = self.G_c if G_c is None else G_c
        H_onceki = psi_plus if history_H is None else history_H
        H = torch.max(H_onceki, psi_plus)          # tarihçe: geri dönüş yok

        kaynak = 2.0 * self.ell * H / (gc + 1e-12)
        if source_cap is not None:
            kaynak = torch.clamp(kaynak, max=source_cap)

        k2_ell = self.ell ** 2 * self.spectral_ops.k_squared

        if mode == "at2":
            alpha = 1.0 + kaynak                    # α = 1 + 2ℓH/G_c
            alpha_ort = alpha.mean().detach()
            payda = alpha_ort + k2_ell
        else:
            alpha = None
            payda = 1.0 + k2_ell

        d = d_prev.clone()
        for _ in range(n_iter):
            if mode == "at2":
                # Sabit-nokta: uzayda değişen kısım sağ tarafa taşınır
                rhs = kaynak + (alpha_ort - alpha) * d
            else:
                rhs = kaynak
            d_cozum = safe_ifft2(safe_fft2(rhs) / payda).real

            if soft_clamp:
                # Sigmoid tabanlı yumuşak kısıt: gradyan her yerde akar
                d_cozum = torch.sigmoid(10.0 * (d_cozum - 0.5)) * 0.98 + 0.01
            else:
                d_cozum = d_cozum.clamp(0.0, 1.0)

            d = relax * d_cozum + (1.0 - relax) * d

        # Geri dönüşsüzlük: hasar azalamaz
        d = torch.max(d, d_prev)
        if not soft_clamp:
            d = d.clamp(0.0, 1.0)

        if explain:
            sep = "-" * 68
            head = "  PhaseFieldNeuron — AT2 hasar adımı"
            if label:
                head += f" [{label}]"
            gc_val = float(gc) if not torch.is_tensor(gc) else gc.item()
            print(f"{sep}\n{head}\n{sep}")
            print(f"    Karakteristik uzunluk ℓ = {self.ell:.4g} m, "
                  f"G_c = {gc_val:.6g} J/m²")
            print(f"    Form: {'tam AT2' if mode == 'at2' else 'lineerleştirilmiş'}"
                  f"   |   {n_iter} sabit-nokta yinelemesi")
            print(f"    Sürücü enerji: max ψ⁺ = {psi_plus.max().item():.6g} J/m³")
            print(f"    Tarihçe:       max H  = {H.max().item():.6g} J/m³")
            print(f"    Hasar: max d = {d.max().item():.6f}, "
                  f"ortalama d = {d.mean().item():.6f}")
            hasarli = (d > 0.5).to(d.dtype).mean().item()
            print(f"    d > 0,5 olan alan oranı: %{hasarli*100:.3f}")
            print(sep)

        return d, H

    def forward(self, psi_plus, d_prev, **kwargs):
        """nn.Module arayüzü — solve_damage() sarmalayıcısı."""
        return self.solve_damage(psi_plus, d_prev, **kwargs)

    def extra_repr(self) -> str:
        return (f"ell={self.ell:.3e}, G_c={float(self.G_c):.3e}, "
                f"eta={self.eta:.1e}")


class PlasticityNeuron(nn.Module):
    r"""
    J2 (von Mises) Plastisite Nöronu — radyal geri dönüş (return mapping).

    Elastik deneme (trial) gerilmesi akma yüzeyini aştığında, deviatörik
    kısım yüzeye geri çekilir; hidrostatik kısım değişmez:

        s   = σ - (tr σ/3)·I           (deviatörik)
        σ_vm = √(σ_xx² - σ_xx σ_yy + σ_yy² + 3τ_xy²)
        σ_vm > σ_y  ise   s ← s·(σ_y/σ_vm)     (radyal geri dönüş)

    Bu, düzlem gerilme hâli için gradyan-dostu bir formdur: bölme işlemi
    türevlenebilirdir, dolayısıyla akma gerilmesi σ_y ölçülen bir gerilme
    alanından geri çözülebilir (ters problem).

    Not: burada `sigma_y` bir TENSÖR olabilir ve ona göre türev alınabilir —
    plastisitenin ters problemde kullanılabilmesinin sebebi budur.

    Args:
        sigma_y: Akma gerilmesi [Pa] (başlangıç değeri).
        H_hard: Doğrusal pekleşme modülü [Pa]. Şu an akma yüzeyini
            genişletmede kullanılmıyor; ileride kinematik/izotropik
            pekleşme eklenmesi için saklanır.
        learnable: True ise akma gerilmesi öğrenilebilir parametre olur.
    """

    def __init__(self, sigma_y: float, H_hard: float = 500e6,
                 learnable: bool = False):
        super().__init__()
        self.H_hard = H_hard
        if learnable:
            self.sigma_y = nn.Parameter(torch.tensor(float(sigma_y)))
        else:
            self.register_buffer('sigma_y', torch.tensor(float(sigma_y)))

    @staticmethod
    def von_mises_2d(sigma_xx, sigma_yy, tau_xy, eps: float = 1e-20):
        """Düzlem gerilme von Mises eşdeğeri."""
        return torch.sqrt(sigma_xx ** 2 - sigma_xx * sigma_yy
                          + sigma_yy ** 2 + 3 * tau_xy ** 2 + eps)

    def return_map(self, sigma_xx: torch.Tensor, sigma_yy: torch.Tensor,
                   tau_xy: torch.Tensor,
                   sigma_y: Optional[torch.Tensor] = None,
                   mode: str = "radial",
                   explain: bool = False,
                   label: str = "") -> Dict[str, torch.Tensor]:
        r"""
        Elastik deneme gerilmesini akma yüzeyine geri çek.

        İki mod vardır ve farkları önemlidir:

        • mode='radial' (VARSAYILAN): gerilme tensörünün tamamı σ_y/σ_vm ile
          ölçeklenir. von Mises ölçüsü gerilmede 1. dereceden homojen
          olduğundan sonuç TAM olarak akma yüzeyine oturur: σ_vm = σ_y.

        • mode='deviatoric': hidrostatik kısım korunup yalnız deviatörik kısım
          ölçeklenir. Bu, üç boyutlu J2'nin standart adımıdır; ancak DÜZLEM
          GERİLME hâlinde (σ_zz = 0) hidrostatik terim (σ_xx+σ_yy)/3 ile
          alındığında σ_zz kısıtı gözardı edildiği için sonuç akma yüzeyinin
          DIŞINDA kalır. Örneğin saf çekmede σ_y = 300 MPa iken 500 MPa'lık
          deneme gerilmesi 338 MPa'ya iner — yüzeyi %12,8 aşar.

        Düzlem gerilmenin tam doğru geri dönüşü σ_zz = 0 kısıtı altında
        iteratif çözüm ister; 'radial' bunun akma yüzeyine oturan, kapalı
        formda ve türevlenebilir bir yaklaşımıdır. 'deviatoric' seçeneği,
        bu formu kullanan mevcut çalışmalarla birebir karşılaştırma
        yapılabilsin diye korunmuştur.

        Args:
            sigma_xx, sigma_yy, tau_xy: Elastik deneme gerilmeleri.
            sigma_y: Akma gerilmesi; None ise modülün kendi değeri kullanılır
                (ters problemde dışarıdan optimize edilen tensör buradan
                geçirilir).
            mode: 'radial' veya 'deviatoric'.
            explain: True ise adımlar sayısal değerleriyle yazdırılır.

        Returns:
            dict: sigma_xx, sigma_yy, tau_xy (düzeltilmiş), sigma_vm_trial,
                  sigma_vm, yield_ratio, plastic_fraction
        """
        if mode not in ("radial", "deviatoric"):
            raise ValueError("mode 'radial' veya 'deviatoric' olmalı.")

        sy = self.sigma_y if sigma_y is None else sigma_y

        vm_trial = self.von_mises_2d(sigma_xx, sigma_yy, tau_xy)
        yield_ratio = vm_trial / (sy + 1e-10)

        # Akan noktalarda ölçek σ_y/σ_vm, kalanlarda 1 — her iki dalda da
        # türevlenebilir
        olcek = torch.where(vm_trial > sy,
                            sy / (vm_trial + 1e-20),
                            torch.ones_like(vm_trial))

        if mode == "radial":
            sxx_c = sigma_xx * olcek
            syy_c = sigma_yy * olcek
            txy_c = tau_xy * olcek
        else:
            sigma_h = (sigma_xx + sigma_yy) / 3.0
            sxx_c = (sigma_xx - sigma_h) * olcek + sigma_h
            syy_c = (sigma_yy - sigma_h) * olcek + sigma_h
            txy_c = tau_xy * olcek

        vm_c = self.von_mises_2d(sxx_c, syy_c, txy_c)
        plastik_oran = (vm_trial > sy).to(vm_trial.dtype).mean()

        if explain:
            sep = "-" * 68
            head = "  PlasticityNeuron — J2 radyal geri dönüş"
            if label:
                head += f" [{label}]"
            print(f"{sep}\n{head}\n{sep}")
            sy_val = float(sy) if not torch.is_tensor(sy) else sy.item()
            print(f"    Akma gerilmesi σ_y = {sy_val/1e6:.4g} MPa")
            print(f"    Elastik deneme: max σ_vm = "
                  f"{vm_trial.max().item()/1e6:.6g} MPa")
            print(f"    Akma oranı σ_vm/σ_y: max = {yield_ratio.max().item():.4f}")
            print(f"    Akan düğüm oranı: %{plastik_oran.item()*100:.2f}")
            print(f"    Geri dönüş sonrası: max σ_vm = "
                  f"{vm_c.max().item()/1e6:.6g} MPa"
                  f"   (akma yüzeyine oturdu)")
            print(sep)

        return {
            'sigma_xx': sxx_c, 'sigma_yy': syy_c, 'tau_xy': txy_c,
            'sigma_vm_trial': vm_trial, 'sigma_vm': vm_c,
            'yield_ratio': yield_ratio, 'plastic_fraction': plastik_oran,
        }

    def forward(self, sigma_xx, sigma_yy, tau_xy, **kwargs):
        """nn.Module arayüzü — return_map() sarmalayıcısı."""
        return self.return_map(sigma_xx, sigma_yy, tau_xy, **kwargs)

    def extra_repr(self) -> str:
        sy = self.sigma_y
        return f"sigma_y={float(sy):.3e}, H={self.H_hard:.3e}"


class FatigueNeuron(nn.Module):
    """
    Yorulma Analizi Nöronu.
    
    S-N eğrisi (Basquin denklemi):
    N = C / (σa)^m
    
    Miner kuralı (doğrusal hasar birikimi):
    D = Σ(ni/Ni)
    D >= 1 → Kırılma
    
    Args:
        material: Malzeme özellikleri
        S_ut: Çekme mukavemeti (Pa)
        S_e: Dayanım sınırı (Pa, 10^6 çevrimde)
    """
    
    def __init__(self, material: MaterialProperties,
                 S_ut: float, S_e: float):
        super().__init__()
        
        self.E = material.E
        self.S_ut = S_ut  # Ultimate tensile strength
        self.S_e = S_e    # Endurance limit
        
        # S-N eğrisi parametreleri (Basquin)
        # N = C / σ^m
        # 10^3 çevrimde: 0.9*S_ut
        # 10^6 çevrimde: S_e
        self.N1 = 1e3
        self.N2 = 1e6
        self.S1 = 0.9 * S_ut
        self.S2 = S_e
        
        # m ve C hesapla
        self.m = math.log10(self.N2 / self.N1) / math.log10(self.S1 / self.S2)
        self.C = self.N1 * self.S1**self.m
        
        # Öğrenilebilir parametre
        self.damage_rate = nn.Parameter(torch.ones(1))
    
    def cycles_to_failure(self, sigma_a: float) -> float:
        """
        Belirli bir gerilme genliğinde kırılmaya kadar çevrim sayısı.
        
        N = C / σa^m
        
        Args:
            sigma_a: Gerilme genliği (Pa)
        """
        if sigma_a < self.S_e:
            return float('inf')  # Dayanım sınırı altında sonsuz ömür
        
        N = self.C / (sigma_a ** self.m)
        return N
    
    def stress_amplitude_from_cycles(self, N: float) -> float:
        """
        Belirli çevrim sayısında izin verilen gerilme genliği.
        """
        if N >= 1e6:
            return self.S_e
        
        sigma_a = (self.C / N) ** (1 / self.m)
        return sigma_a
    
    def miner_damage(self, stress_cycles: List[Tuple[float, int]]) -> float:
        """
        Miner doğrusal hasar birikimi.
        
        D = Σ(ni/Ni)
        
        Args:
            stress_cycles: [(σa1, n1), (σa2, n2), ...] 
                          (gerilme genliği, çevrim sayısı) çiftleri
        
        Returns:
            D: Toplam hasar (D >= 1 → kırılma)
        """
        D = 0.0
        for sigma_a, n in stress_cycles:
            N_i = self.cycles_to_failure(sigma_a)
            if N_i != float('inf'):
                D += n / N_i
        
        return D * self.damage_rate.item()
    
    def remaining_life(self, stress_cycles: List[Tuple[float, int]], 
                       future_sigma: float) -> float:
        """
        Kalan ömür tahmini.
        
        Mevcut hasar D varken, σa gerilmesiyle çalışmaya devam ederse
        kalan çevrim sayısı.
        """
        D_current = self.miner_damage(stress_cycles)
        
        if D_current >= 1:
            return 0  # Zaten hasar görmüş
        
        N_future = self.cycles_to_failure(future_sigma)
        
        if N_future == float('inf'):
            return float('inf')
        
        # Kalan ömür oranı
        remaining_fraction = 1 - D_current
        return N_future * remaining_fraction
    
    def safety_factor(self, sigma_a: float, n_actual: int, n_target: int) -> float:
        """
        Yorulma güvenlik faktörü.
        
        SF = N_allowable / n_target
        """
        N_allow = self.cycles_to_failure(sigma_a)
        
        if N_allow == float('inf'):
            return float('inf')
        
        return N_allow / n_target
    
    def goodman_criterion(self, sigma_a: float, sigma_m: float) -> float:
        """
        Goodman kriteri (ortalama gerilme etkisi).

        σa/S_e + σm/S_ut = 1/SF

        Returns:
            SF: Güvenlik faktörü
        """
        denominator = sigma_a / self.S_e + sigma_m / self.S_ut
        if denominator <= 0:
            return float('inf')

        return 1 / denominator

    def goodman_corrected_amplitude(self, sigma_a: float,
                                    sigma_m: float) -> float:
        r"""
        Goodman ortalama gerilme düzeltmesi — EŞDEĞER genlik.

            σ_a,düzeltilmiş = σ_a / (1 - σ_m/S_ut)

        `goodman_criterion` güvenlik katsayısı döndürür; bu metot ise
        ortalama gerilmenin etkisini içeren eşdeğer bir genlik verir ve
        doğrudan S-N eğrisine girilebilir. İkisi farklı sorulara cevaptır.

        Fiziksel yorum:
          σ_m > 0 (çeki)  → genlik artar, ömür kısalır
          σ_m < 0 (bası)  → genlik azalır, ömür uzar
          σ_m ≥ S_ut      → statik kopma; sonsuz genlik döndürülür
        """
        if sigma_m >= self.S_ut:
            return float('inf')
        payda = 1.0 - sigma_m / self.S_ut
        if payda <= 0:
            return float('inf')
        return sigma_a / payda

    def basquin_life(self, sigma_a: float,
                     sigma_f_prime: float, b: float,
                     min_cycles: float = 1.0) -> float:
        r"""
        Basquin bağıntısıyla ömür.

            σ_a = σ_f'·(2N_f)^b   ⇒   N_f = ½·(σ_a/σ_f')^(1/b)

        `cycles_to_failure` (N = C/σ_a^m) ile aynı fiziği farklı
        parametrelendirir ve genelde FARKLI sonuç verir; hangisinin
        kullanıldığı raporlanmalıdır.

        Dayanım sınırı S_e altında sonsuz ömür varsayılır.

        Args:
            sigma_a: Gerilme genliği [Pa].
            sigma_f_prime: Yorulma dayanım katsayısı σ_f' [Pa].
            b: Basquin üsteli (negatif).
            min_cycles: Döndürülecek en küçük çevrim sayısı.
        """
        if sigma_a <= 0 or sigma_a < self.S_e:
            return float('inf')
        oran = sigma_a / sigma_f_prime
        N_f = (oran ** (1.0 / b)) / 2.0
        return max(N_f, min_cycles)
    
    def forward(self, sigma_a: float, n_cycles: int = 1) -> Dict[str, float]:
        """
        Yorulma analizi.
        """
        N_f = self.cycles_to_failure(sigma_a)
        D = n_cycles / N_f if N_f != float('inf') else 0
        
        return {
            'N_f': N_f,
            'damage': D,
            'remaining_cycles': max(0, N_f - n_cycles) if N_f != float('inf') else float('inf')
        }
    
    def extra_repr(self) -> str:
        return f"S_ut={self.S_ut:.2e}, S_e={self.S_e:.2e}, m={self.m:.2f}"


class DynamicNeuron(nn.Module):
    """
    Dinamik Analiz Nöronu: Zamana Bağlı Tepki.
    
    Hareket denklemi:
    D∇⁴w + ρh·∂²w/∂t² + c·∂w/∂t = q(x,y,t)
    
    Çözüm yöntemi: Modal süperpozisyon
    w(x,y,t) = Σ φmn(x,y)·ηmn(t)
    
    Args:
        resolution: Grid çözünürlüğü
        Lx, Ly: Domain boyutları (m)
        material: Malzeme özellikleri
        rho: Yoğunluk (kg/m³)
        damping_ratio: Sönüm oranı (ζ)
    """
    
    def __init__(self, resolution: int, Lx: float, Ly: float,
                 material: MaterialProperties, rho: float,
                 damping_ratio: float = 0.02,
                 n_modes: int = 10):
        super().__init__()
        
        self.resolution = resolution
        self.Lx = Lx
        self.Ly = Ly
        self.material = material
        self.D = material.D
        self.h = material.h
        self.rho = rho
        self.zeta = damping_ratio
        self.n_modes = n_modes
        
        # Modal neuron kullan
        self.modal = ModalNeuron(resolution, Lx, Ly, material, rho)
        
        # Grid
        x = torch.linspace(0, Lx, resolution)
        y = torch.linspace(0, Ly, resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        self.register_buffer('X', X)
        self.register_buffer('Y', Y)
        
        # Öğrenilebilir parametre
        self.damping_modulator = nn.Parameter(torch.ones(1))
    
    def modal_force(self, q: torch.Tensor, m: int, n: int) -> torch.Tensor:
        """
        Modal kuvvet.
        
        Qmn(t) = ∫∫ q(x,y,t)·φmn(x,y) dxdy
        """
        phi = self.modal.mode_shape(m, n)
        Q = (q * phi).sum() * (self.Lx / self.resolution) * (self.Ly / self.resolution)
        return Q / self.modal.modal_mass(m, n)
    
    def impulse_response(self, m: int, n: int, t: torch.Tensor) -> torch.Tensor:
        """
        Birim impuls tepkisi (modal koordinatlar).
        
        h(t) = (1/ω_d)·exp(-ζωt)·sin(ω_d·t)
        ω_d = ω·√(1-ζ²)
        """
        omega = self.modal.natural_frequency(m, n)
        zeta = self.zeta * self.damping_modulator.item()
        
        omega_d = omega * math.sqrt(1 - zeta**2)
        
        h = (1 / omega_d) * torch.exp(-zeta * omega * t) * torch.sin(omega_d * t)
        h[t < 0] = 0  # Kausalite
        
        return h
    
    def step_response(self, m: int, n: int, t: torch.Tensor, F0: float = 1.0) -> torch.Tensor:
        """
        Birim basamak tepkisi.
        
        u(t) = (F0/k)[1 - exp(-ζωt)(cos(ω_d·t) + (ζ/√(1-ζ²))sin(ω_d·t))]
        """
        omega = self.modal.natural_frequency(m, n)
        zeta = self.zeta * self.damping_modulator.item()
        k = self.modal.modal_stiffness(m, n)
        
        omega_d = omega * math.sqrt(1 - zeta**2)
        
        term1 = torch.cos(omega_d * t)
        term2 = (zeta / math.sqrt(1 - zeta**2)) * torch.sin(omega_d * t)
        
        u = (F0 / k) * (1 - torch.exp(-zeta * omega * t) * (term1 + term2))
        u[t < 0] = 0
        
        return u
    
    def harmonic_response(self, m: int, n: int, omega_f: float, F0: float = 1.0) -> complex:
        """
        Harmonik tepki (frekans domain).
        
        H(ω) = 1 / (k - mω² + i·c·ω)
        
        Args:
            omega_f: Zorlama frekansı (rad/s)
            F0: Kuvvet genliği
        
        Returns:
            Karmaşık genlik
        """
        omega_n = self.modal.natural_frequency(m, n)
        zeta = self.zeta * self.damping_modulator.item()
        k = self.modal.modal_stiffness(m, n)
        M = self.modal.modal_mass(m, n)
        
        # Frekans oranı
        r = omega_f / omega_n
        
        # Dinamik büyütme faktörü
        H_real = (1 - r**2) / ((1 - r**2)**2 + (2 * zeta * r)**2)
        H_imag = -2 * zeta * r / ((1 - r**2)**2 + (2 * zeta * r)**2)
        
        amplitude = (F0 / k) * complex(H_real, H_imag)
        return amplitude
    
    def transmissibility(self, omega_f: float, m: int = 1, n: int = 1) -> float:
        """
        İletkenlik (taban hareketi için).
        
        TR = √[(1 + (2ζr)²) / ((1-r²)² + (2ζr)²)]
        """
        omega_n = self.modal.natural_frequency(m, n)
        zeta = self.zeta * self.damping_modulator.item()
        
        r = omega_f / omega_n
        
        num = 1 + (2 * zeta * r)**2
        den = (1 - r**2)**2 + (2 * zeta * r)**2
        
        return math.sqrt(num / den)
    
    def time_integration_newmark(self, q_history: torch.Tensor, dt: float,
                                  m: int = 1, n: int = 1) -> torch.Tensor:
        """
        Newmark-β yöntemiyle zaman entegrasyonu.
        
        β = 0.25, γ = 0.5 (sabit ortalama ivme)
        """
        beta = 0.25
        gamma = 0.5
        
        omega = self.modal.natural_frequency(m, n)
        zeta = self.zeta * self.damping_modulator.item()
        k = self.modal.modal_stiffness(m, n)
        M = self.modal.modal_mass(m, n)
        c = 2 * zeta * omega * M
        
        n_steps = len(q_history)
        u = torch.zeros(n_steps)
        v = torch.zeros(n_steps)
        a = torch.zeros(n_steps)
        
        # Başlangıç ivmesi
        a[0] = (q_history[0] - c * v[0] - k * u[0]) / M
        
        # Efektif rijitlik
        k_eff = k + gamma / (beta * dt) * c + 1 / (beta * dt**2) * M
        
        for i in range(1, n_steps):
            # Efektif yük
            p_eff = q_history[i] + \
                    M * (1 / (beta * dt**2) * u[i-1] + 1 / (beta * dt) * v[i-1] + (1 / (2*beta) - 1) * a[i-1]) + \
                    c * (gamma / (beta * dt) * u[i-1] + (gamma / beta - 1) * v[i-1] + dt * (gamma / (2*beta) - 1) * a[i-1])
            
            # Yeni deplasman
            u[i] = p_eff / k_eff
            
            # Yeni ivme ve hız
            a[i] = 1 / (beta * dt**2) * (u[i] - u[i-1]) - 1 / (beta * dt) * v[i-1] - (1 / (2*beta) - 1) * a[i-1]
            v[i] = v[i-1] + dt * ((1 - gamma) * a[i-1] + gamma * a[i])
        
        return u
    
    def forward(self, t: torch.Tensor, load_type: str = 'impulse',
                m: int = 1, n: int = 1) -> torch.Tensor:
        """
        Dinamik tepki hesapla.
        """
        if load_type == 'impulse':
            return self.impulse_response(m, n, t)
        elif load_type == 'step':
            return self.step_response(m, n, t)
        else:
            raise ValueError(f"Desteklenmeyen yük tipi: {load_type}")
    
    def extra_repr(self) -> str:
        return f"zeta={self.zeta}, n_modes={self.n_modes}"


class NonlinearNeuron(nn.Module):
    """
    Nonlineer Analiz Nöronu: Büyük Deformasyon (von Karman).
    
    von Karman plaka denklemleri:
    D∇⁴w = q + h[F,w]
    ∇⁴F = -Eh/2 [w,w]
    
    Burada:
    - [F,w] = F,xx·w,yy + F,yy·w,xx - 2F,xy·w,xy (Airy bracket)
    - F: Airy gerilme fonksiyonu
    
    Args:
        resolution: Grid çözünürlüğü
        Lx, Ly: Domain boyutları (m)
        material: Malzeme özellikleri
    """
    
    def __init__(self, resolution: int, Lx: float, Ly: float,
                 material: MaterialProperties,
                 spectral_ops: Optional[SpectralOps2DStruct] = None):
        super().__init__()
        
        self.resolution = resolution
        self.Lx = Lx
        self.Ly = Ly
        self.material = material
        self.D = material.D
        self.E = material.E
        self.h = material.h
        
        self.spectral_ops = spectral_ops if spectral_ops else \
                           SpectralOps2DStruct(resolution, Lx, Ly)
        
        # Grid
        x = torch.linspace(0, Lx, resolution)
        y = torch.linspace(0, Ly, resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        self.register_buffer('X', X)
        self.register_buffer('Y', Y)
        
        # Öğrenilebilir parametreler
        self.nonlinear_scale = nn.Parameter(torch.ones(1))
    
    def airy_bracket(self, F: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Airy bracket: [F, w] = F,xx·w,yy + F,yy·w,xx - 2F,xy·w,xy
        """
        F_xx, F_yy, F_xy = self.spectral_ops.second_derivatives(F)
        w_xx, w_yy, w_xy = self.spectral_ops.second_derivatives(w)
        
        bracket = F_xx * w_yy + F_yy * w_xx - 2 * F_xy * w_xy
        return bracket
    
    def self_bracket(self, w: torch.Tensor) -> torch.Tensor:
        """
        Self bracket: [w, w] = 2(w,xx·w,yy - w,xy²)
        
        Gaussian eğrilik ile ilgili.
        """
        w_xx, w_yy, w_xy = self.spectral_ops.second_derivatives(w)
        
        bracket = 2 * (w_xx * w_yy - w_xy**2)
        return bracket
    
    def solve_airy_stress_function(self, w: torch.Tensor) -> torch.Tensor:
        """
        Airy gerilme fonksiyonunu çöz.
        
        ∇⁴F = -Eh/2 [w,w]
        """
        # Kaynak terimi
        source = -self.E * self.h / 2 * self.self_bracket(w)
        
        # Spektral çözüm: F_hat = source_hat / k^4
        source_hat = safe_fft2(source)
        
        # k^4 = 0 olan yerde bölme yapma
        k4 = self.spectral_ops.k_fourth
        k4_safe = torch.where(k4 > 1e-10, k4, torch.ones_like(k4))
        
        F_hat = source_hat / k4_safe
        F_hat[k4 < 1e-10] = 0  # DC bileşeni sıfır
        
        F = safe_ifft2(F_hat).real
        return F
    
    def nonlinear_residual(self, w: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Nonlineer rezidüel hesapla.
        
        R = D∇⁴w - q - h[F,w]
        """
        # Airy fonksiyonu
        F = self.solve_airy_stress_function(w)
        
        # Biharmonik
        biharm_w = self.spectral_ops.biharmonic(w)
        
        # Nonlineer terim
        nonlinear = self.h * self.airy_bracket(F, w) * self.nonlinear_scale
        
        # Rezidüel
        residual = self.D * biharm_w - q - nonlinear
        
        return residual
    
    def membrane_forces(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Membran kuvvetlerini hesapla.
        
        Nx = F,yy, Ny = F,xx, Nxy = -F,xy
        """
        F = self.solve_airy_stress_function(w)
        F_xx, F_yy, F_xy = self.spectral_ops.second_derivatives(F)
        
        Nx = F_yy
        Ny = F_xx
        Nxy = -F_xy
        
        return Nx, Ny, Nxy
    
    def geometric_stiffness_ratio(self, w: torch.Tensor) -> float:
        """
        Geometrik rijitlik oranı (nonlineerlik ölçüsü).
        
        Yüksek değerler = güçlü nonlineerlik
        """
        Nx, Ny, Nxy = self.membrane_forces(w)
        
        # Membran enerjisi
        membrane_energy = (Nx**2 + Ny**2 + 2*Nxy**2).mean()
        
        # Eğilme enerjisi
        lap_w = self.spectral_ops.laplacian(w)
        bending_energy = self.D * (lap_w**2).mean()
        
        ratio = membrane_energy / (bending_energy + 1e-10)
        return ratio.item()
    
    def newton_iteration(self, w_init: torch.Tensor, q: torch.Tensor,
                         max_iter: int = 20, tol: float = 1e-6) -> torch.Tensor:
        """
        Newton-Raphson iterasyonu ile nonlineer çözüm.
        """
        w = w_init.clone()
        
        for i in range(max_iter):
            # Rezidüel
            R = self.nonlinear_residual(w, q)
            residual_norm = R.abs().max().item()
            
            if residual_norm < tol:
                break
            
            # Basitleştirilmiş güncelleme (damped Newton)
            # Gerçek Jacobian yerine D∇⁴ kullan
            R_hat = safe_fft2(R)
            k4 = self.spectral_ops.k_fourth
            k4_safe = torch.where(k4 > 1e-10, k4, torch.ones_like(k4))
            
            dw_hat = R_hat / (self.D * k4_safe)
            dw_hat[k4 < 1e-10] = 0
            
            dw = safe_ifft2(dw_hat).real
            
            # Damping
            alpha = 0.5
            w = w - alpha * dw
        
        return w
    
    def forward(self, q: torch.Tensor, w_init: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Nonlineer çözüm.
        """
        if w_init is None:
            w_init = torch.zeros_like(self.X)
        
        return self.newton_iteration(w_init, q)
    
    def extra_repr(self) -> str:
        return f"Lx={self.Lx}, Ly={self.Ly}, D={self.D:.2e}"


# =============================================================================
# FAZ 4 NÖRONLARı (3D)
# =============================================================================

class SpectralOps3D(nn.Module):
    """
    3D Spektral Türev Operatörleri.
    
    FFT tabanlı gradient, Laplacian ve ilgili türevler.
    
    Args:
        resolution: Grid çözünürlüğü (Nx = Ny = Nz = resolution)
        Lx, Ly, Lz: Domain boyutları (m)
    """
    
    def __init__(self, resolution: int, Lx: float = 1.0, Ly: float = 1.0, Lz: float = 1.0):
        super().__init__()
        
        self.resolution = resolution
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        
        # Dalga sayıları
        kx = fftfreq(resolution, d=Lx/resolution) * 2 * math.pi
        ky = fftfreq(resolution, d=Ly/resolution) * 2 * math.pi
        kz = fftfreq(resolution, d=Lz/resolution) * 2 * math.pi
        
        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
        
        self.register_buffer('kx', KX)
        self.register_buffer('ky', KY)
        self.register_buffer('kz', KZ)
        self.register_buffer('k_squared', KX**2 + KY**2 + KZ**2)
        
        # Ayrık bileşenler
        self.register_buffer('kx2', KX**2)
        self.register_buffer('ky2', KY**2)
        self.register_buffer('kz2', KZ**2)
    
    def fft3(self, f: torch.Tensor) -> torch.Tensor:
        """3D FFT."""
        return torch.fft.fftn(f, dim=(-3, -2, -1))
    
    def ifft3(self, f_hat: torch.Tensor) -> torch.Tensor:
        """3D IFFT."""
        return torch.fft.ifftn(f_hat, dim=(-3, -2, -1))
    
    def gradient(self, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z)
        """
        f_hat = self.fft3(f)
        
        df_dx = self.ifft3(1j * self.kx * f_hat).real
        df_dy = self.ifft3(1j * self.ky * f_hat).real
        df_dz = self.ifft3(1j * self.kz * f_hat).real
        
        return df_dx, df_dy, df_dz
    
    def divergence(self, fx: torch.Tensor, fy: torch.Tensor, fz: torch.Tensor) -> torch.Tensor:
        """
        ∇·F = ∂fx/∂x + ∂fy/∂y + ∂fz/∂z
        """
        fx_hat = self.fft3(fx)
        fy_hat = self.fft3(fy)
        fz_hat = self.fft3(fz)
        
        div = self.ifft3(1j * (self.kx * fx_hat + self.ky * fy_hat + self.kz * fz_hat)).real
        return div
    
    def laplacian(self, f: torch.Tensor) -> torch.Tensor:
        """
        ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²
        """
        f_hat = self.fft3(f)
        lap_f = self.ifft3(-self.k_squared * f_hat).real
        return lap_f
    
    def curl(self, fx: torch.Tensor, fy: torch.Tensor, fz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ∇×F = (∂fz/∂y - ∂fy/∂z, ∂fx/∂z - ∂fz/∂x, ∂fy/∂x - ∂fx/∂y)
        """
        fx_hat = self.fft3(fx)
        fy_hat = self.fft3(fy)
        fz_hat = self.fft3(fz)
        
        curl_x = self.ifft3(1j * (self.ky * fz_hat - self.kz * fy_hat)).real
        curl_y = self.ifft3(1j * (self.kz * fx_hat - self.kx * fz_hat)).real
        curl_z = self.ifft3(1j * (self.kx * fy_hat - self.ky * fx_hat)).real
        
        return curl_x, curl_y, curl_z
    
    def strain_tensor(self, ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Küçük şekil değiştirme tensörü.
        
        ε_ij = (1/2)(∂ui/∂xj + ∂uj/∂xi)
        """
        # Gradientler
        dux_dx, dux_dy, dux_dz = self.gradient(ux)
        duy_dx, duy_dy, duy_dz = self.gradient(uy)
        duz_dx, duz_dy, duz_dz = self.gradient(uz)
        
        # Normal strainler
        eps_xx = dux_dx
        eps_yy = duy_dy
        eps_zz = duz_dz
        
        # Kayma strainleri
        eps_xy = 0.5 * (dux_dy + duy_dx)
        eps_xz = 0.5 * (dux_dz + duz_dx)
        eps_yz = 0.5 * (duy_dz + duz_dy)
        
        return {
            'eps_xx': eps_xx, 'eps_yy': eps_yy, 'eps_zz': eps_zz,
            'eps_xy': eps_xy, 'eps_xz': eps_xz, 'eps_yz': eps_yz
        }
    
    def extra_repr(self) -> str:
        return f"resolution={self.resolution}, Lx={self.Lx}, Ly={self.Ly}, Lz={self.Lz}"


class SolidNeuron3D(nn.Module):
    """
    3D Elastisite Nöronu: Cauchy Stress ve Strain.
    
    Hooke yasası (3D):
    σ_ij = λ·δ_ij·ε_kk + 2μ·ε_ij
    
    Lamé sabitleri:
    λ = Eν/[(1+ν)(1-2ν)]
    μ = E/[2(1+ν)]
    
    Args:
        resolution: Grid çözünürlüğü
        Lx, Ly, Lz: Domain boyutları (m)
        E: Young modülü (Pa)
        nu: Poisson oranı
    """
    
    def __init__(self, resolution: int, Lx: float, Ly: float, Lz: float,
                 E: float, nu: float):
        super().__init__()
        
        self.resolution = resolution
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.E = E
        self.nu = nu
        
        # Lamé sabitleri
        self.lam = E * nu / ((1 + nu) * (1 - 2 * nu))  # λ
        self.mu = E / (2 * (1 + nu))                    # μ (shear modulus)
        
        self.spectral_ops = SpectralOps3D(resolution, Lx, Ly, Lz)
        
        # Öğrenilebilir parametreler
        self.E_modulator = nn.Parameter(torch.ones(1))
        self.nu_modulator = nn.Parameter(torch.ones(1))
    
    def lame_parameters(self) -> Tuple[float, float]:
        """Güncel Lamé sabitleri (modulatörlerle)."""
        E = self.E * self.E_modulator.item()
        nu = self.nu * self.nu_modulator.item()
        
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        
        return lam, mu
    
    def stress_from_strain(self, strain: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Strain → Stress dönüşümü (3D Hooke).
        
        σ_ij = λ·δ_ij·ε_kk + 2μ·ε_ij
        """
        lam, mu = self.lame_parameters()
        
        # Volumetrik strain
        eps_vol = strain['eps_xx'] + strain['eps_yy'] + strain['eps_zz']
        
        # Normal stressler
        sigma_xx = lam * eps_vol + 2 * mu * strain['eps_xx']
        sigma_yy = lam * eps_vol + 2 * mu * strain['eps_yy']
        sigma_zz = lam * eps_vol + 2 * mu * strain['eps_zz']
        
        # Kayma stressleri
        sigma_xy = 2 * mu * strain['eps_xy']
        sigma_xz = 2 * mu * strain['eps_xz']
        sigma_yz = 2 * mu * strain['eps_yz']
        
        return {
            'sigma_xx': sigma_xx, 'sigma_yy': sigma_yy, 'sigma_zz': sigma_zz,
            'sigma_xy': sigma_xy, 'sigma_xz': sigma_xz, 'sigma_yz': sigma_yz
        }
    
    def von_mises_stress(self, stress: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        von Mises eşdeğer gerilme.
        
        σ_vm = √[((σ_xx-σ_yy)² + (σ_yy-σ_zz)² + (σ_zz-σ_xx)²)/2 + 3(σ_xy² + σ_yz² + σ_xz²)]
        """
        s_xx = stress['sigma_xx']
        s_yy = stress['sigma_yy']
        s_zz = stress['sigma_zz']
        s_xy = stress['sigma_xy']
        s_xz = stress['sigma_xz']
        s_yz = stress['sigma_yz']
        
        term1 = (s_xx - s_yy)**2 + (s_yy - s_zz)**2 + (s_zz - s_xx)**2
        term2 = 6 * (s_xy**2 + s_yz**2 + s_xz**2)
        
        sigma_vm = torch.sqrt((term1 + term2) / 2)
        return sigma_vm
    
    def principal_stresses(self, stress: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Asal gerilmeler (yaklaşık - köşe noktalarında).
        
        3x3 tensörün özdeğerleri.
        """
        # Her noktada stress tensor oluştur ve özdeğer hesapla
        # Bu pahalı işlem, basitleştirilmiş versiyon kullanıyoruz
        
        s_xx = stress['sigma_xx']
        s_yy = stress['sigma_yy']
        s_zz = stress['sigma_zz']
        
        # Basitleştirilmiş: sadece köşegen elemanları kullan
        sigma_1 = torch.maximum(torch.maximum(s_xx, s_yy), s_zz)
        sigma_3 = torch.minimum(torch.minimum(s_xx, s_yy), s_zz)
        sigma_2 = s_xx + s_yy + s_zz - sigma_1 - sigma_3
        
        return sigma_1, sigma_2, sigma_3
    
    def strain_energy_density(self, strain: Dict[str, torch.Tensor], 
                               stress: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Şekil değiştirme enerji yoğunluğu.
        
        U = (1/2)·σ_ij·ε_ij
        """
        U = 0.5 * (
            stress['sigma_xx'] * strain['eps_xx'] +
            stress['sigma_yy'] * strain['eps_yy'] +
            stress['sigma_zz'] * strain['eps_zz'] +
            2 * stress['sigma_xy'] * strain['eps_xy'] +
            2 * stress['sigma_xz'] * strain['eps_xz'] +
            2 * stress['sigma_yz'] * strain['eps_yz']
        )
        return U
    
    def forward(self, ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Deplasmandan stress/strain hesapla.
        """
        strain = self.spectral_ops.strain_tensor(ux, uy, uz)
        stress = self.stress_from_strain(strain)
        
        return {
            'strain': strain,
            'stress': stress,
            'von_mises': self.von_mises_stress(stress)
        }
    
    def extra_repr(self) -> str:
        return f"E={self.E:.2e}, nu={self.nu}"


class ShellNeuron3D(nn.Module):
    """
    Kabuk (Shell) Nöronu: Membran + Eğilme.
    
    Kabuk teorisi: İnce kabuklarda membran ve eğilme etkilerinin birleşimi.
    
    Membran kuvvetleri: Nx, Ny, Nxy (N/m)
    Eğilme momentleri: Mx, My, Mxy (Nm/m)
    Kesme kuvvetleri: Qx, Qy (N/m)
    
    Args:
        resolution: Grid çözünürlüğü
        Lx, Ly: Kabuk boyutları (m)
        h: Kabuk kalınlığı (m)
        E: Young modülü (Pa)
        nu: Poisson oranı
        R: Eğrilik yarıçapı (m), None ise düzlem
    """
    
    def __init__(self, resolution: int, Lx: float, Ly: float,
                 h: float, E: float, nu: float, R: Optional[float] = None):
        super().__init__()
        
        self.resolution = resolution
        self.Lx = Lx
        self.Ly = Ly
        self.h = h
        self.E = E
        self.nu = nu
        self.R = R  # Eğrilik yarıçapı (None = düz)
        
        # Rijitlikler
        self.A = E * h / (1 - nu**2)           # Membran rijitliği
        self.D = E * h**3 / (12 * (1 - nu**2)) # Eğilme rijitliği
        
        self.spectral_ops = SpectralOps2DStruct(resolution, Lx, Ly)
        
        # Grid
        x = torch.linspace(0, Lx, resolution)
        y = torch.linspace(0, Ly, resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        self.register_buffer('X', X)
        self.register_buffer('Y', Y)
        
        # Öğrenilebilir parametreler
        self.membrane_modulator = nn.Parameter(torch.ones(1))
        self.bending_modulator = nn.Parameter(torch.ones(1))
    
    def membrane_strains(self, u: torch.Tensor, v: torch.Tensor, 
                          w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Membran strainleri.
        
        εx = ∂u/∂x + w/R (eğri kabuk için)
        εy = ∂v/∂y
        γxy = ∂u/∂y + ∂v/∂x
        """
        du_dx, du_dy = self.spectral_ops.gradient(u)
        dv_dx, dv_dy = self.spectral_ops.gradient(v)
        
        eps_x = du_dx
        eps_y = dv_dy
        gamma_xy = du_dy + dv_dx
        
        # Eğri kabuk etkisi
        if self.R is not None:
            eps_x = eps_x + w / self.R
        
        return eps_x, eps_y, gamma_xy
    
    def curvatures(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Eğilme eğrilikleri.
        
        κx = -∂²w/∂x²
        κy = -∂²w/∂y²
        κxy = -2∂²w/∂x∂y
        """
        w_xx, w_yy, w_xy = self.spectral_ops.second_derivatives(w)
        
        kappa_x = -w_xx
        kappa_y = -w_yy
        kappa_xy = -2 * w_xy
        
        return kappa_x, kappa_y, kappa_xy
    
    def membrane_forces(self, eps_x: torch.Tensor, eps_y: torch.Tensor, 
                         gamma_xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Membran kuvvetleri.
        
        Nx = A(εx + ν·εy)
        Ny = A(εy + ν·εx)
        Nxy = A(1-ν)/2 · γxy
        """
        A = self.A * self.membrane_modulator
        nu = self.nu
        
        Nx = A * (eps_x + nu * eps_y)
        Ny = A * (eps_y + nu * eps_x)
        Nxy = A * (1 - nu) / 2 * gamma_xy
        
        return Nx, Ny, Nxy
    
    def bending_moments(self, kappa_x: torch.Tensor, kappa_y: torch.Tensor,
                        kappa_xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Eğilme momentleri.
        
        Mx = D(κx + ν·κy)
        My = D(κy + ν·κx)
        Mxy = D(1-ν)/2 · κxy
        """
        D = self.D * self.bending_modulator
        nu = self.nu
        
        Mx = D * (kappa_x + nu * kappa_y)
        My = D * (kappa_y + nu * kappa_x)
        Mxy = D * (1 - nu) / 2 * kappa_xy
        
        return Mx, My, Mxy
    
    def strain_energy(self, u: torch.Tensor, v: torch.Tensor, 
                      w: torch.Tensor) -> torch.Tensor:
        """
        Toplam şekil değiştirme enerjisi.
        
        U = U_membrane + U_bending
        """
        # Membran strainleri ve kuvvetleri
        eps_x, eps_y, gamma_xy = self.membrane_strains(u, v, w)
        Nx, Ny, Nxy = self.membrane_forces(eps_x, eps_y, gamma_xy)
        
        # Eğilme eğrilikleri ve momentleri
        kappa_x, kappa_y, kappa_xy = self.curvatures(w)
        Mx, My, Mxy = self.bending_moments(kappa_x, kappa_y, kappa_xy)
        
        # Enerji
        U_membrane = 0.5 * (Nx * eps_x + Ny * eps_y + Nxy * gamma_xy)
        U_bending = 0.5 * (Mx * kappa_x + My * kappa_y + Mxy * kappa_xy)
        
        return U_membrane.mean() + U_bending.mean()
    
    def forward(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Kabuk analizi.
        """
        # Membran
        eps_x, eps_y, gamma_xy = self.membrane_strains(u, v, w)
        Nx, Ny, Nxy = self.membrane_forces(eps_x, eps_y, gamma_xy)
        
        # Eğilme
        kappa_x, kappa_y, kappa_xy = self.curvatures(w)
        Mx, My, Mxy = self.bending_moments(kappa_x, kappa_y, kappa_xy)
        
        return {
            'membrane': {'Nx': Nx, 'Ny': Ny, 'Nxy': Nxy},
            'bending': {'Mx': Mx, 'My': My, 'Mxy': Mxy},
            'strain_energy': self.strain_energy(u, v, w)
        }
    
    def extra_repr(self) -> str:
        return f"h={self.h}, E={self.E:.2e}, R={self.R}"


class BeamNeuron3D(nn.Module):
    """
    Kiriş Nöronu: Euler-Bernoulli ve Timoshenko.
    
    Euler-Bernoulli (ince kiriş):
    EI·d⁴w/dx⁴ = q(x)
    
    Timoshenko (kalın kiriş, kayma etkili):
    κGA·(dψ/dx - d²w/dx²) = 0
    EI·d²ψ/dx² + κGA·(dw/dx - ψ) = 0
    
    Args:
        length: Kiriş uzunluğu (m)
        resolution: Grid çözünürlüğü
        E: Young modülü (Pa)
        I: Atalet momenti (m⁴)
        A: Kesit alanı (m²)
        kappa: Kayma düzeltme faktörü (dikdörtgen için ~5/6, dairesel için ~0.9)
        nu: Poisson oranı (boyutsuz). `material` verilmediyse zorunludur.
        material: MaterialProperties — verilirse E ve nu otomatik alınır.
    """

    def __init__(self, length: float, resolution: int,
                 E: Optional[float] = None, I: float = 1.0, A: float = 1.0,
                 rho: float = 7850, kappa: float = 5/6,
                 nu: Optional[float] = None,
                 material: Optional[MaterialProperties] = None):
        super().__init__()

        # Malzemeyi çözümle: material > (E, nu) > hata
        if material is not None:
            self.E = material.E
            self.nu = material.nu
        else:
            if E is None:
                raise ValueError(
                    "BeamNeuron3D: ya `material` ya da `E` verilmeli."
                )
            self.E = E
            if nu is None:
                # Geri-uyumluluk: eski API nu vermiyordu. Çelik varsayımı,
                # ama uyarı bas (kayma rijitliği yanlış olabilir).
                warnings.warn(
                    "BeamNeuron3D: `nu` verilmedi, varsayılan 0.3 (çelik) "
                    "kullanılıyor. Kayma rijitliği κGA bu varsayıma bağlı; "
                    "hassas hesap için `nu` veya `material` parametresini geç.",
                    stacklevel=2,
                )
                nu = 0.3
            self.nu = nu

        self.L = length
        self.resolution = resolution
        self.I = I
        self.A = A
        self.rho = rho
        self.kappa = kappa  # shear correction factor

        # Rijitlikler
        self.EI = self.E * I                          # Eğilme rijitliği [N·m²]
        # Kayma modülü: G = E / [2(1+ν)]
        self.G = self.E / (2.0 * (1.0 + self.nu))
        # Timoshenko kayma rijitliği κGA — κ shear correction factor
        # Standart Timoshenko/MITC formülasyonu: κGA·γ
        self.shear_stiffness = self.kappa * self.G * A   # κGA [N]
        # Geri-uyumluluk için aynı değeri `kGA` ve eski isim `GA` ile de tut.
        # NOT: Eski `self.GA` aslında κ olmadan G·A idi; burada düzeltildi —
        # eski kodda `self.GA` kullananlar artık κGA değerini görür.
        self.kGA = self.shear_stiffness
        self.GA = self.shear_stiffness
        
        # 1D Grid
        x = torch.linspace(0, length, resolution)
        self.register_buffer('x', x)
        
        # Dalga sayıları (1D)
        k = fftfreq(resolution, d=length/resolution) * 2 * math.pi
        self.register_buffer('k', k)
        self.register_buffer('k2', k**2)
        self.register_buffer('k4', k**4)
        
        # Öğrenilebilir parametreler
        self.stiffness_modulator = nn.Parameter(torch.ones(1))
    
    def fft1(self, f: torch.Tensor) -> torch.Tensor:
        """1D FFT."""
        return torch.fft.fft(f)
    
    def ifft1(self, f_hat: torch.Tensor) -> torch.Tensor:
        """1D IFFT."""
        return torch.fft.ifft(f_hat)
    
    def derivative(self, f: torch.Tensor, order: int = 1) -> torch.Tensor:
        """Spektral türev."""
        f_hat = self.fft1(f)
        
        if order == 1:
            df = self.ifft1(1j * self.k * f_hat).real
        elif order == 2:
            df = self.ifft1(-self.k2 * f_hat).real
        elif order == 4:
            df = self.ifft1(self.k4 * f_hat).real
        else:
            df = self.ifft1((1j * self.k)**order * f_hat).real
        
        return df
    
    def euler_bernoulli_solve(self, q: torch.Tensor) -> torch.Tensor:
        """
        Euler-Bernoulli denklemi çözümü.
        
        EI·d⁴w/dx⁴ = q
        
        Simply supported BC için Navier serisi.
        """
        EI = self.EI * self.stiffness_modulator
        
        w = torch.zeros_like(self.x)
        
        # Fourier serisi çözümü
        for n in range(1, 20):
            # Yük katsayısı
            phi_n = torch.sin(n * math.pi * self.x / self.L)
            q_n = 2 / self.L * (q * phi_n).sum() * (self.L / self.resolution)
            
            # Deplasman katsayısı
            alpha_n = n * math.pi / self.L
            w_n = q_n / (EI * alpha_n**4)
            
            w = w + w_n * phi_n
        
        return w
    
    def natural_frequency(self, n: int = 1, theory: str = 'euler') -> float:
        """
        Doğal frekans (Hz).
        
        Euler-Bernoulli (simply supported):
        ωn = (nπ/L)² · √(EI/ρA)
        """
        EI = self.EI * self.stiffness_modulator.item()
        rhoA = self.rho * self.A
        
        if theory == 'euler':
            omega = (n * math.pi / self.L)**2 * math.sqrt(EI / rhoA)
        else:
            # Timoshenko düzeltmesi (yaklaşık)
            r = math.sqrt(self.I / self.A)  # Gyration yarıçapı
            s = 2 * r / self.L
            omega_euler = (n * math.pi / self.L)**2 * math.sqrt(EI / rhoA)
            omega = omega_euler / math.sqrt(1 + (n * math.pi * s)**2)
        
        return omega / (2 * math.pi)  # Hz
    
    def critical_buckling_load(self, n: int = 1, K: float = 1.0) -> float:
        """
        Kritik burkulma yükü (Euler kolon burkulması).

        P_cr = n²·π²·EI / (K·L)²

        Effective length factor (K):
            - Pinned-pinned (basit mesnetli)        : K = 1.0   (varsayılan)
            - Fixed-fixed   (iki ucu ankastre)      : K = 0.5
            - Fixed-free    (cantilever / konsol)   : K = 2.0
            - Pinned-fixed  (bir ucu basit, biri ankastre): K = 0.7

        Args:
            n: Burkulma modu (1 = ilk mod, kritik mod).
            K: Effective length factor. Sınır koşullarına bağlı (yukarı bkz).

        Returns:
            P_cr [N]: Kritik eksenel basma yükü.
        """
        if K <= 0:
            raise ValueError(f"K (effective length factor) > 0 olmalı, alındı: {K}")
        EI = self.EI * self.stiffness_modulator.item()
        L_eff = K * self.L
        P_cr = n**2 * math.pi**2 * EI / L_eff**2
        return P_cr
    
    def mode_shape(self, n: int = 1) -> torch.Tensor:
        """
        n. mod şekli (simply supported).
        
        φn(x) = sin(nπx/L)
        """
        return torch.sin(n * math.pi * self.x / self.L)
    
    def bending_moment(self, w: torch.Tensor) -> torch.Tensor:
        """
        Eğilme momenti.
        
        M = -EI·d²w/dx²
        """
        EI = self.EI * self.stiffness_modulator
        d2w = self.derivative(w, order=2)
        return -EI * d2w
    
    def shear_force(self, w: torch.Tensor) -> torch.Tensor:
        """
        Kesme kuvveti.
        
        V = -EI·d³w/dx³
        """
        EI = self.EI * self.stiffness_modulator
        d3w = self.derivative(w, order=3)
        return -EI * d3w
    
    def max_stress(self, w: torch.Tensor, c: float) -> float:
        """
        Maksimum eğilme gerilmesi.
        
        σ = Mc/I
        
        Args:
            c: Nötr eksene mesafe (m)
        """
        M = self.bending_moment(w)
        sigma = M * c / self.I
        return sigma.abs().max().item()
    
    def forward(self, q: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Kiriş analizi.
        """
        w = self.euler_bernoulli_solve(q)
        M = self.bending_moment(w)
        V = self.shear_force(w)
        
        return {
            'displacement': w,
            'moment': M,
            'shear': V,
            'max_displacement': w.abs().max().item()
        }
    
    def extra_repr(self) -> str:
        return f"L={self.L}, EI={self.EI:.2e}"


# =============================================================================
# EKSENEL ÇUBUK NÖRONU (1D Rod / Bar) — Faz 5b
# =============================================================================

class RodNeuron(nn.Module):
    r"""
    Eksenel Çubuk Nöronu (Axial Rod/Bar).

    Yöneten denklem (1D eksenel denge + bünye):

        d/dx [ E·A(x) · du/dx ] + f(x) = 0

    burada:
        u(x) : eksenel deplasman (m)
        E(x) : Young modülü (Pa)        — segment/konum bağımlı olabilir
        A(x) : kesit alanı (m²)         — değişken kesit (koni/taper) destekli
        f(x) : dağıtılmış eksenel yük (N/m, +x yönü pozitif) — öz-ağırlık dahil
        N(x) = E·A·du/dx : iç normal kuvvet (çekme +)

    Çözüm stratejisi — mühendis kafasıyla, iki şeffaf adım (FFT YOK):
        1) STATİK (denge):    dN/dx = -f  →  N(x), uç/ara yüklerden integralle.
        2) KİNEMATİK (bünye): ε = N/(E·A),  u(x) = ∫ ε dx   (ankastre uçtan).

    1D'de integraller doğrudan kuadratür (trapez) ile alınır; bu hem analitik
    çözümle bire bir, hem de değişken katsayılı E·A(x) için spektral FFT'nin
    aksine doğal olarak doğrudur.

    SPINE felsefesi: fizik nöronun *içine* gömülüdür. Tek öğrenilebilir
    parametre `stiffness_modulator` (EA çarpanı); deterministik modda 1.0,
    ters/veri-uydurma problemlerinde gradyanla ayarlanabilir.

    Koordinat sözleşmesi: x ∈ [0, L]. `fixed_end='left'` (varsayılan) → x=0
    ankastre (u=0), x=L serbest uç. Tüm tekil yükler ve dağıtılmış yük çekme
    (serbest uca doğru) yönünde pozitiftir.

    Args:
        length: Çubuk boyu L (m).
        resolution: Grid nokta sayısı (varsayılan 401 — tek sayı Simpson dostu).
        dtype: torch.float64 (varsayılan, deterministik referans doğruluğu için).
        device: Hesap cihazı (varsayılan CPU — 1D, küçük, MPS gerekmez).
    """

    def __init__(self, length: float, resolution: int = 401,
                 dtype: torch.dtype = torch.float64,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.L = float(length)
        self.resolution = int(resolution)
        self.dtype = dtype
        self._device = device if device is not None else torch.device("cpu")

        x = torch.linspace(0.0, self.L, self.resolution,
                           dtype=dtype, device=self._device)
        self.register_buffer("x", x)

        # Tek öğrenilebilir parametre — EA çarpanı (SPINE modulator paterni)
        self.stiffness_modulator = nn.Parameter(
            torch.ones(1, dtype=dtype, device=self._device)
        )

    # ------------------------------------------------------------------ utils
    def _resolve_field(self, spec: Union[float, int, Callable, torch.Tensor],
                       name: str) -> torch.Tensor:
        """float | callable(x) | tensor → x ile aynı boyutta tensör."""
        x = self.x
        if callable(spec):
            field = spec(x)
            if not isinstance(field, torch.Tensor):
                field = torch.as_tensor(field, dtype=self.dtype, device=self._device)
            field = field.to(dtype=self.dtype, device=self._device)
            if field.shape != x.shape:
                raise ValueError(
                    f"RodNeuron: `{name}` callable çıktısı {tuple(field.shape)}, "
                    f"beklenen {tuple(x.shape)}."
                )
            return field
        if isinstance(spec, torch.Tensor):
            if spec.shape != x.shape:
                raise ValueError(
                    f"RodNeuron: `{name}` tensörü {tuple(spec.shape)}, "
                    f"beklenen {tuple(x.shape)}."
                )
            return spec.to(dtype=self.dtype, device=self._device)
        return torch.full_like(x, float(spec))

    @staticmethod
    def _cumtrapz_from_left(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """u(x) = ∫₀ˣ y ds (trapez). Çıktı uzunluğu N, u[0]=0."""
        dx = x[1:] - x[:-1]
        seg = 0.5 * (y[1:] + y[:-1]) * dx
        out = torch.zeros_like(y)
        out[1:] = torch.cumsum(seg, dim=0)
        return out

    @staticmethod
    def _cumtrapz_from_right(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """I(x) = ∫ₓᴸ y ds (trapez). Çıktı uzunluğu N, I[-1]=0."""
        dx = x[1:] - x[:-1]
        seg = 0.5 * (y[1:] + y[:-1]) * dx
        out = torch.zeros_like(y)
        out[:-1] = torch.flip(torch.cumsum(torch.flip(seg, [0]), dim=0), [0])
        return out

    # ------------------------------------------------------------------ solve
    def solve(self,
              E: Union[float, Callable, torch.Tensor],
              A: Union[float, Callable, torch.Tensor],
              distributed_load: Union[float, Callable, torch.Tensor] = 0.0,
              point_loads: Optional[List[Tuple[float, float]]] = None,
              fixed_end: str = "left",
              rho: Optional[Union[float, Callable, torch.Tensor]] = None,
              g: float = 9.81,
              self_weight: bool = False,
              explain: bool = False,
              label: str = "") -> Dict[str, torch.Tensor]:
        r"""
        Eksenel çubuğu çöz.

        Args:
            E: Young modülü alanı — float | callable(x) | tensör[N].
            A: Kesit alanı alanı  — float | callable(x) | tensör[N].
            distributed_load: f(x) dağıtılmış eksenel yük [N/m], +x (serbest uca
                doğru) pozitif. Öz-ağırlık DIŞI yükler için.
            point_loads: [(x_i, P_i), ...] tekil eksenel yükler; P_i çekme
                (serbest uca doğru) pozitif. Uç yükü de buraya konur (x_i=L).
            fixed_end: 'left' (x=0 ankastre, varsayılan) veya 'right'.
            rho: yoğunluk (kg/m³) — `self_weight=True` ise öz-ağırlık için.
            g: yerçekimi ivmesi (m/s², varsayılan 9.81).
            self_weight: True ise f(x) += ρ·g·A(x) (serbest uca doğru, asılı
                çubuk varsayımı). Alternatif: distributed_load'a γ·A(x) elle ver.
            explain: True ise her çözüm adımını şeffaf biçimde yazdırır.
            label: explain çıktısında problemi etiketler.

        Returns:
            dict: x, N (iç kuvvet), sigma (gerilme), epsilon (şekil değiştirme),
                  u (deplasman alanı), tip_displacement (serbest uç deplasmanı),
                  EA (rijitlik alanı).
        """
        if fixed_end not in ("left", "right"):
            raise ValueError("fixed_end 'left' veya 'right' olmalı.")
        x = self.x
        mod = self.stiffness_modulator
        eps_pos = 1e-9 * self.L  # tekil yük konum toleransı

        E_field = self._resolve_field(E, "E")
        A_field = self._resolve_field(A, "A")
        f_field = self._resolve_field(distributed_load, "distributed_load")

        # Öz-ağırlık: ρ·g·A, serbest uca doğru (asılı çubuk)
        sw_field = None
        if self_weight:
            if rho is None:
                raise ValueError("self_weight=True ise `rho` verilmeli.")
            rho_field = self._resolve_field(rho, "rho")
            sw_field = rho_field * g * A_field  # [N/m], büyüklük
            # Serbest uca doğru işaret: fixed=left → +x; fixed=right → -x
            sw_signed = sw_field if fixed_end == "left" else -sw_field
            f_field = f_field + sw_signed

        EA = E_field * A_field * mod  # rijitlik alanı [N]

        # --- 1) STATİK: N(x), dN/dx = -f ---
        if fixed_end == "left":   # serbest uç sağda (x=L)
            N = self._cumtrapz_from_right(f_field, x)          # ∫ₓᴸ f ds
            if point_loads:
                for xi, Pi in point_loads:
                    N = N + float(Pi) * (x <= (float(xi) + eps_pos)).to(self.dtype)
        else:                     # serbest uç solda (x=0)
            N = -self._cumtrapz_from_left(f_field, x)          # -∫₀ˣ f ds
            if point_loads:
                for xi, Pi in point_loads:
                    N = N + float(Pi) * (x >= (float(xi) - eps_pos)).to(self.dtype)

        # --- 2) KİNEMATİK: ε = N/(EA), u = ∫ ε dx ---
        # Sivri uç (koni tepesi) A→0 iken N→0; oranın fiziksel limiti sonludur
        # ama 0/0 sayısal NaN verir. A=0 düğümlerinde σ=ε=0 (limit) alınır.
        A_safe = torch.where(A_field.abs() > 0, A_field, torch.ones_like(A_field))
        EA_safe = torch.where(EA.abs() > 0, EA, torch.ones_like(EA))
        sigma = torch.where(A_field.abs() > 0, N / A_safe, torch.zeros_like(N))
        epsilon = torch.where(EA.abs() > 0, N / EA_safe, torch.zeros_like(N))
        if fixed_end == "left":
            u = self._cumtrapz_from_left(epsilon, x)   # u(0)=0
            tip_index = -1
        else:
            u = -self._cumtrapz_from_right(epsilon, x)  # u(L)=0
            tip_index = 0

        tip = u[tip_index]

        if explain:
            self._explain(label, fixed_end, E_field, A_field, f_field,
                          point_loads, sw_field, N, sigma, epsilon, u, tip)

        return {
            "x": x,
            "N": N,
            "sigma": sigma,
            "epsilon": epsilon,
            "u": u,
            "tip_displacement": tip,
            "EA": EA,
        }

    def _explain(self, label, fixed_end, E_field, A_field, f_field,
                 point_loads, sw_field, N, sigma, epsilon, u, tip):
        """Çözümü adım adım, şeffaf biçimde yazdır (hocanın istediği mod)."""
        sep = "=" * 68
        title = f" RodNeuron — Adım Adım Eksenel Çözüm "
        if label:
            title += f"[{label}] "
        print(f"\n{sep}\n{title}\n{sep}")

        # 0) Kurulum
        free = "x=L (sağ)" if fixed_end == "left" else "x=0 (sol)"
        fix = "x=0 (sol)" if fixed_end == "left" else "x=L (sağ)"
        print(f"[0] KURULUM")
        print(f"    Boy L = {self.L:.4g} m, grid = {self.resolution} nokta")
        print(f"    Ankastre uç: {fix}   |   Serbest uç: {free}")
        E0, EL = E_field[0].item(), E_field[-1].item()
        A0, AL = A_field[0].item(), A_field[-1].item()
        if abs(E0 - EL) / max(abs(E0), 1e-30) < 1e-9:
            print(f"    E = {E0:.4g} Pa (sabit)")
        else:
            print(f"    E(x): {E0:.4g} → {EL:.4g} Pa (konum bağımlı)")
        if abs(A0 - AL) / max(abs(A0), 1e-30) < 1e-9:
            print(f"    A = {A0:.6g} m² (sabit kesit)")
        else:
            print(f"    A(x): {A0:.6g} → {AL:.6g} m² (değişken kesit)")
        if sw_field is not None:
            W = self._cumtrapz_from_left(sw_field, self.x)[-1].item()
            print(f"    Öz-ağırlık dahil — toplam ağırlık ≈ {W:.4g} N")
        if point_loads:
            pl = ", ".join(f"P({xi:.3g})={Pi:.4g} N" for xi, Pi in point_loads)
            print(f"    Tekil yükler: {pl}")

        # 1) Statik
        print(f"[1] STATİK — iç normal kuvvet N(x)   (dN/dx = -f)")
        print(f"    N(ankastre) = {N[0].item() if fixed_end=='left' else N[-1].item():.6g} N"
              f"   |   N(serbest) = {N[-1].item() if fixed_end=='left' else N[0].item():.6g} N")
        print(f"    N aralığı: [{N.min().item():.6g}, {N.max().item():.6g}] N")

        # 2) Bünye
        print(f"[2] BÜNYE — gerilme ve şekil değiştirme")
        print(f"    σ = N/A : max |σ| = {sigma.abs().max().item():.6g} Pa")
        print(f"    ε = σ/E = N/(EA) : max |ε| = {epsilon.abs().max().item():.6g}")

        # 3) Kinematik
        print(f"[3] KİNEMATİK — u(x) = ∫ ε dx   (ankastre uçtan)")
        print(f"    u(ankastre) = 0 (sınır koşulu)")
        print(f"    SONUÇ → serbest uç deplasmanı δ = {tip.item():.6g} m"
              f"  ({tip.item()*1e3:.6g} mm, {tip.item()*1e6:.6g} µm)")
        print(sep)

    def forward(self, E, A, **kwargs) -> Dict[str, torch.Tensor]:
        """nn.Module arayüzü — solve() sarmalayıcısı."""
        return self.solve(E, A, **kwargs)

    def extra_repr(self) -> str:
        return f"L={self.L}, N={self.resolution}, dtype={self.dtype}"


# =============================================================================
# ANA MODEL CONTAINER
# =============================================================================

class SPINE(nn.Module):
    """
    SPINE: Structural Physics-Inherent Neural Engine

    Tüm yapısal mekanik nöronlarını birleştiren ana container.

    Kullanım:
        material = MaterialProperties(E=200e9, nu=0.3, h=0.005)
        model = SPINE(resolution=64, Lx=1.0, Ly=0.5, material=material)

        # Kritik yük
        N_cr = model.get_critical_load(m=1, n=1)

        # Mod şekli
        w = model.get_mode_shape(m=1, n=1)

        # PDE rezidüeli
        residual = model.buckling_residual(w, N_cr)
    """

    def __init__(self, resolution: int, Lx: float, Ly: float,
                 material: MaterialProperties,
                 bc_type: BoundaryConditionType = BoundaryConditionType.SIMPLY_SUPPORTED):
        super().__init__()

        self.resolution = resolution
        self.Lx = Lx
        self.Ly = Ly
        self.material = material
        self.bc_type = bc_type

        # Paylaşılan spektral operatörler
        self.spectral_ops = SpectralOps2DStruct(resolution, Lx, Ly)

        # Nöronlar
        self.biharmonic = BiharmonicNeuron(resolution, Lx, Ly, self.spectral_ops)
        self.buckling = BucklingNeuron(resolution, Lx, Ly, material, self.spectral_ops)
        self.stress = StressNeuron(material)
        self.strain = StrainNeuron(resolution, Lx, Ly, self.spectral_ops)
        self.boundary = BoundaryNeuron(resolution, Lx, Ly, bc_type)

        # Sine basis (simply supported için)
        self.sine_basis = SineBasis(resolution, Lx, Ly)

    def get_critical_load(self, m: int = 1, n: int = 1) -> float:
        """Analitik kritik yük."""
        return self.buckling.analytical_critical_load(m, n)

    def get_mode_shape(self, m: int = 1, n: int = 1) -> torch.Tensor:
        """Burkulma mod şekli."""
        return self.buckling.mode_shape(m, n)

    def buckling_residual(self, w: torch.Tensor,
                          Nx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Burkulma denkleminin rezidüeli."""
        return self.buckling(w, Nx)

    def forward(self, w: torch.Tensor) -> PlateState:
        """
        Tam plaka durumunu hesapla.

        Args:
            w: Deplasman alanı [B, Ny, Nx]

        Returns:
            PlateState: Tüm durum değişkenleri
        """
        # Eğimler
        theta_x, theta_y = self.spectral_ops.gradient(w)

        # Eğrilikler (StrainNeuron konvansiyonu: κ_ij = -w_ij)
        kappa_xx, kappa_yy, kappa_xy = self.strain(w)

        # Momentler (standart Kirchhoff, ShellNeuron3D.bending_moments ile aynı form):
        # M_x = -D(w_xx + ν·w_yy) = D(κ_xx + ν·κ_yy)
        # M_xy = -D(1-ν)·w_xy = D(1-ν)·κ_xy  (κ_xy = -w_xy olduğundan çarpan 1-ν, 2'ye bölünmez)
        D = self.material.D
        nu = self.material.nu
        M_xx = D * (kappa_xx + nu * kappa_yy)
        M_yy = D * (kappa_yy + nu * kappa_xx)
        M_xy = D * (1 - nu) * kappa_xy

        return PlateState(
            w=w,
            theta_x=theta_x,
            theta_y=theta_y,
            M_xx=M_xx,
            M_yy=M_yy,
            M_xy=M_xy
        )

    def get_diagnostics(self, w: torch.Tensor) -> Dict[str, float]:
        """Diagnostic değerler."""
        state = self.forward(w)

        return {
            'max_displacement': state.max_displacement().item(),
            'strain_energy': state.strain_energy(self.material.D, self.material.nu).mean().item(),
            'bc_loss': self.boundary.bc_loss(w).item()
        }

    def extra_repr(self) -> str:
        return f"resolution={self.resolution}, Lx={self.Lx}, Ly={self.Ly}, D={self.material.D:.2e}"
    
    # =========================================================================
    # MALZEME KÜTÜPHANESİ ENTEGRASYONU
    # =========================================================================
    
    @classmethod
    def from_material(cls, material_name: str, h: float, Lx: float, Ly: float,
                      resolution: int = 64, 
                      bc_type: BoundaryConditionType = BoundaryConditionType.SIMPLY_SUPPORTED):
        """
        Malzeme kütüphanesinden SPINE oluştur.
        
        Kullanım:
            model = SPINE.from_material("steel_304", h=0.005, Lx=1.0, Ly=0.5)
            N_cr = model.get_critical_load()
        
        Args:
            material_name: Malzeme adı (örn: "steel_304", "carbon_epoxy_T300")
            h: Kalınlık (m)
            Lx, Ly: Plaka boyutları (m)
            resolution: Grid çözünürlüğü
            bc_type: Sınır koşulu tipi
        
        Returns:
            SPINE: Hazır model
        """
        if not MATERIALS_AVAILABLE:
            raise ImportError(
                "materials.py bulunamadı. Lütfen materials.py dosyasını "
                "spine.py ile aynı klasöre koyun veya MaterialProperties kullanın."
            )
        
        mat = MaterialLibrary.get(material_name)
        
        if isinstance(mat, IsotropicMaterial):
            # İzotropik malzeme
            material = MaterialProperties(E=mat.E, nu=mat.nu, h=h)
        elif isinstance(mat, OrthotropicMaterial):
            # Ortotropik malzeme - eşdeğer izotropik yaklaşım
            # D_eq ≈ (D11 * D22)^0.5
            E_eq = (mat.E1 * mat.E2) ** 0.5
            nu_eq = mat.nu12
            material = MaterialProperties(E=E_eq, nu=nu_eq, h=h)
            warnings.warn(
                f"Ortotropik malzeme '{material_name}' eşdeğer izotropik olarak kullanıldı. "
                f"Gerçek ortotropik analiz için SPINE.from_laminate() kullanın."
            )
        else:
            raise ValueError(f"Desteklenmeyen malzeme tipi: {type(mat)}")
        
        return cls(resolution=resolution, Lx=Lx, Ly=Ly, material=material, bc_type=bc_type)
    
    @classmethod
    def from_laminate(cls, laminate: 'CompositeLaminate', Lx: float, Ly: float,
                      resolution: int = 64,
                      bc_type: BoundaryConditionType = BoundaryConditionType.SIMPLY_SUPPORTED):
        """
        Kompozit laminattan SPINE oluştur.
        
        Kullanım:
            from materials import CompositeLaminate, Ply
            
            laminate = CompositeLaminate([
                Ply(0, 0.125e-3, "carbon_epoxy_T300"),
                Ply(45, 0.125e-3, "carbon_epoxy_T300"),
                Ply(-45, 0.125e-3, "carbon_epoxy_T300"),
                Ply(90, 0.125e-3, "carbon_epoxy_T300"),
            ])
            
            model = SPINE.from_laminate(laminate, Lx=1.0, Ly=0.5)
        
        Args:
            laminate: CompositeLaminate objesi
            Lx, Ly: Plaka boyutları (m)
            resolution: Grid çözünürlüğü
            bc_type: Sınır koşulu tipi
        
        Returns:
            SPINE: Hazır model (eşdeğer rijitlikle)
        """
        if not MATERIALS_AVAILABLE:
            raise ImportError("materials.py bulunamadı.")
        
        # Laminat ABD matrisleri
        A, B, D_matrix = laminate.get_ABD_matrices()
        
        # Eşdeğer izotropik rijitlik: D_eq = (D11 * D22)^0.5
        D_eq = (D_matrix[0, 0] * D_matrix[1, 1]) ** 0.5
        
        # D = Eh³/[12(1-ν²)] formülünden E ve ν tahmini
        h = laminate.total_thickness
        # Varsayılan ν = 0.3 kullan, E'yi D'den hesapla
        nu_eq = 0.3
        E_eq = D_eq * 12 * (1 - nu_eq**2) / h**3
        
        material = MaterialProperties(E=E_eq, nu=nu_eq, h=h)
        
        # Gerçek D matrisini sakla (ortotropik analiz için)
        model = cls(resolution=resolution, Lx=Lx, Ly=Ly, material=material, bc_type=bc_type)
        model._laminate = laminate
        model._D_matrix = D_matrix
        
        return model
    
    def get_critical_load_orthotropic(self, m: int = 1, n: int = 1) -> float:
        """
        Ortotropik/laminat için kritik burkulma yükü.
        
        CLT formülü kullanır (SPINE.from_laminate ile oluşturulmuşsa).
        """
        if hasattr(self, '_D_matrix'):
            D = self._D_matrix
            D11, D12, D22, D66 = D[0,0], D[0,1], D[1,1], D[2,2]
            
            a, b = self.Lx, self.Ly
            
            term1 = D11 * (m**4) / (a**4)
            term2 = 2 * (D12 + 2*D66) * (m**2 * n**2) / (a**2 * b**2)
            term3 = D22 * (n**4) / (b**4)
            
            N_cr = (math.pi**2) * (term1 + term2 + term3) / (m**2 / a**2)
            return N_cr
        else:
            return self.get_critical_load(m, n)
    
    @staticmethod
    def list_materials() -> List[str]:
        """Kullanılabilir malzemeleri listele."""
        if not MATERIALS_AVAILABLE:
            return ["materials.py bulunamadı"]
        return MaterialLibrary.list_all()
    
    @staticmethod
    def get_material_info(name: str) -> str:
        """Malzeme bilgisi al."""
        if not MATERIALS_AVAILABLE:
            return "materials.py bulunamadı"
        return MaterialLibrary.info(name)
    
    @staticmethod
    def search_materials(keyword: str) -> List[str]:
        """Malzeme ara."""
        if not MATERIALS_AVAILABLE:
            return []
        return MaterialLibrary.search(keyword)


# =============================================================================
# HIZLI ANALİZ FONKSİYONLARI
# =============================================================================

def quick_analysis(material_name: str, h: float, Lx: float, Ly: float,
                   analysis_type: str = "buckling") -> Dict:
    """
    Hızlı analiz fonksiyonu.
    
    Kullanım:
        result = quick_analysis("steel_304", h=0.005, Lx=1.0, Ly=0.5)
        print(f"N_cr = {result['N_cr']/1000:.2f} kN/m")
    
    Args:
        material_name: Malzeme adı
        h: Kalınlık (m)
        Lx, Ly: Boyutlar (m)
        analysis_type: "buckling" veya "static"
    
    Returns:
        Dict: Analiz sonuçları
    """
    model = SPINE.from_material(material_name, h, Lx, Ly, resolution=32)
    
    result = {
        'material': material_name,
        'h': h,
        'Lx': Lx,
        'Ly': Ly,
    }
    
    if analysis_type == "buckling":
        result['N_cr'] = model.get_critical_load(m=1, n=1)
        result['modes'] = model.buckling.find_critical_modes(max_m=3, max_n=3)[:5]
    
    return result




