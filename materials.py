"""
SPINE Malzeme Modülü

İzotropik, Ortotropik ve Kompozit malzeme desteği.
Classical Laminate Theory (CLT) implementasyonu.

Kullanım:
    from materials import MaterialLibrary, CompositeLaminate, Ply
    
    # Hazır malzeme kullan
    steel = MaterialLibrary.get("steel_304")
    carbon = MaterialLibrary.get("carbon_epoxy_T300")
    
    # Kompozit laminat tanımla
    laminate = CompositeLaminate([
        Ply(angle=0, material="carbon_epoxy_T300", thickness=0.125e-3),
        Ply(angle=45, material="carbon_epoxy_T300", thickness=0.125e-3),
        Ply(angle=-45, material="carbon_epoxy_T300", thickness=0.125e-3),
        Ply(angle=90, material="carbon_epoxy_T300", thickness=0.125e-3),
    ])
    
    # ABD matrisleri al
    A, B, D = laminate.get_ABD_matrices()

Yazar: SPINE Project
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# MALZEME TİPLERİ
# =============================================================================

class MaterialType(Enum):
    """Malzeme tipi enum"""
    ISOTROPIC = "isotropic"           # Çelik, alüminyum, vb.
    ORTHOTROPIC = "orthotropic"       # Tek yönlü fiber, ahşap
    ANISOTROPIC = "anisotropic"       # Genel anizotropik
    COMPOSITE = "composite"           # Laminat kompozit


# =============================================================================
# İZOTROPİK MALZEME
# =============================================================================

@dataclass
class IsotropicMaterial:
    """
    İzotropik malzeme - tüm yönlerde aynı özellikler.
    
    Attributes:
        name: Malzeme adı
        E: Young modülü (Pa)
        nu: Poisson oranı
        rho: Yoğunluk (kg/m³)
        sigma_yield: Akma gerilmesi (Pa), opsiyonel
        sigma_ultimate: Kopma gerilmesi (Pa), opsiyonel
        alpha: Termal genleşme katsayısı (1/°C), opsiyonel
        K_IC: Kırılma tokluğu (Pa√m), opsiyonel
    """
    name: str
    E: float                          # Young modülü (Pa)
    nu: float                         # Poisson oranı
    rho: float = 1000.0               # Yoğunluk (kg/m³)
    sigma_yield: Optional[float] = None
    sigma_ultimate: Optional[float] = None
    alpha: Optional[float] = None     # Termal genleşme
    K_IC: Optional[float] = None      # Kırılma tokluğu
    
    material_type: MaterialType = field(default=MaterialType.ISOTROPIC, init=False)
    
    @property
    def G(self) -> float:
        """Kayma modülü: G = E / [2(1+ν)]"""
        return self.E / (2 * (1 + self.nu))
    
    @property
    def K(self) -> float:
        """Bulk modülü: K = E / [3(1-2ν)]"""
        if self.nu >= 0.5:
            return float('inf')  # Incompressible
        return self.E / (3 * (1 - 2 * self.nu))
    
    @property
    def lambda_lame(self) -> float:
        """Lamé's first parameter: λ = Eν / [(1+ν)(1-2ν)]"""
        if self.nu >= 0.5:
            return float('inf')
        return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
    
    def plate_rigidity(self, h: float) -> float:
        """Plaka rijitliği: D = Eh³ / [12(1-ν²)]"""
        return self.E * h**3 / (12 * (1 - self.nu**2))
    
    def get_C_matrix(self) -> np.ndarray:
        """
        2D düzlem gerilme rijitlik matrisi [3x3].
        
        [σ_xx]   [C11 C12  0 ] [ε_xx]
        [σ_yy] = [C12 C22  0 ] [ε_yy]
        [τ_xy]   [ 0   0  C66] [γ_xy]
        """
        E, nu = self.E, self.nu
        factor = E / (1 - nu**2)
        
        C = np.array([
            [factor, factor * nu, 0],
            [factor * nu, factor, 0],
            [0, 0, E / (2 * (1 + nu))]
        ])
        return C
    
    def get_C_matrix_3D(self) -> np.ndarray:
        """
        3D rijitlik matrisi (Voigt notasyonu) [6x6].
        """
        E, nu = self.E, self.nu
        G = self.G
        
        factor = E / ((1 + nu) * (1 - 2*nu))
        
        C = np.zeros((6, 6))
        C[0, 0] = C[1, 1] = C[2, 2] = factor * (1 - nu)
        C[0, 1] = C[1, 0] = C[0, 2] = C[2, 0] = C[1, 2] = C[2, 1] = factor * nu
        C[3, 3] = C[4, 4] = C[5, 5] = G
        
        return C
    
    def to_tensor(self) -> torch.Tensor:
        """Malzeme parametrelerini tensor olarak döndür."""
        return torch.tensor([self.E, self.nu, self.rho], dtype=torch.float32)
    
    def __repr__(self):
        return f"IsotropicMaterial('{self.name}', E={self.E:.2e} Pa, ν={self.nu}, ρ={self.rho} kg/m³)"


# =============================================================================
# ORTOTROPİK MALZEME
# =============================================================================

@dataclass
class OrthotropicMaterial:
    """
    Ortotropik malzeme - 3 ana yönde farklı özellikler.
    
    Kullanım alanı: Tek yönlü fiber kompozitler, ahşap.
    
    Attributes:
        name: Malzeme adı
        E1: Fiber yönü (1) Young modülü (Pa)
        E2: Enine yön (2) Young modülü (Pa)
        G12: Düzlem içi kayma modülü (Pa)
        nu12: Büyük Poisson oranı (1 yönünde yükleme, 2 yönünde daralma)
        rho: Yoğunluk (kg/m³)
        
    Opsiyonel 3D:
        E3: Kalınlık yönü (3) Young modülü
        G13, G23: Düzlem dışı kayma modülleri
        nu13, nu23: Ek Poisson oranları
        
    Mukavemet:
        Xt, Xc: Fiber yönü çekme/basma dayanımı
        Yt, Yc: Enine yön çekme/basma dayanımı
        S12: Düzlem içi kayma dayanımı
    """
    name: str
    E1: float                         # Fiber yönü modülü (Pa)
    E2: float                         # Enine yön modülü (Pa)
    G12: float                        # Kayma modülü (Pa)
    nu12: float                       # Poisson oranı (major)
    rho: float = 1600.0               # Yoğunluk (kg/m³)
    
    # 3D özellikler (opsiyonel)
    E3: Optional[float] = None
    G13: Optional[float] = None
    G23: Optional[float] = None
    nu13: Optional[float] = None
    nu23: Optional[float] = None
    
    # Mukavemet değerleri (opsiyonel)
    Xt: Optional[float] = None        # Fiber çekme dayanımı
    Xc: Optional[float] = None        # Fiber basma dayanımı
    Yt: Optional[float] = None        # Enine çekme dayanımı
    Yc: Optional[float] = None        # Enine basma dayanımı
    S12: Optional[float] = None       # Kayma dayanımı
    
    # Termal
    alpha1: Optional[float] = None    # Fiber yönü termal genleşme
    alpha2: Optional[float] = None    # Enine yön termal genleşme
    
    material_type: MaterialType = field(default=MaterialType.ORTHOTROPIC, init=False)
    
    @property
    def nu21(self) -> float:
        """Minor Poisson oranı: ν21 = ν12 × E2/E1"""
        return self.nu12 * self.E2 / self.E1
    
    def get_Q_matrix(self) -> np.ndarray:
        """
        Düzlem gerilme rijitlik matrisi [Q] - malzeme koordinatlarında.
        
        [σ_1]   [Q11 Q12  0 ] [ε_1]
        [σ_2] = [Q12 Q22  0 ] [ε_2]
        [τ_12]  [ 0   0  Q66] [γ_12]
        """
        E1, E2, G12 = self.E1, self.E2, self.G12
        nu12, nu21 = self.nu12, self.nu21
        
        denom = 1 - nu12 * nu21
        
        Q = np.array([
            [E1 / denom,      nu12 * E2 / denom, 0],
            [nu12 * E2 / denom, E2 / denom,      0],
            [0,               0,                 G12]
        ])
        return Q
    
    def get_Q_bar_matrix(self, theta_deg: float) -> np.ndarray:
        """
        Dönüştürülmüş rijitlik matrisi [Q̄] - global koordinatlarda.
        
        Args:
            theta_deg: Fiber açısı (derece)
            
        Returns:
            Q_bar: Dönüştürülmüş [3x3] rijitlik matrisi
        """
        theta = math.radians(theta_deg)
        c = math.cos(theta)
        s = math.sin(theta)
        
        Q = self.get_Q_matrix()
        
        # Dönüşüm matrisi
        T = np.array([
            [c**2,    s**2,    2*c*s],
            [s**2,    c**2,   -2*c*s],
            [-c*s,    c*s,    c**2-s**2]
        ])
        
        T_inv = np.array([
            [c**2,    s**2,   -2*c*s],
            [s**2,    c**2,    2*c*s],
            [c*s,    -c*s,    c**2-s**2]
        ])
        
        # Reuter matrisi (mühendislik ↔ tensör strain dönüşümü)
        R = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 2]
        ])
        R_inv = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0.5]
        ])
        
        # Q̄ = T_inv × Q × R × T × R_inv
        Q_bar = T_inv @ Q @ R @ T @ R_inv
        
        return Q_bar
    
    def plate_rigidity_11(self, h: float) -> float:
        """Eğilme rijitliği D11 (fiber yönü)"""
        Q = self.get_Q_matrix()
        return Q[0, 0] * h**3 / 12
    
    def plate_rigidity_22(self, h: float) -> float:
        """Eğilme rijitliği D22 (enine yön)"""
        Q = self.get_Q_matrix()
        return Q[1, 1] * h**3 / 12
    
    def to_tensor(self) -> torch.Tensor:
        """Malzeme parametrelerini tensor olarak döndür."""
        return torch.tensor([self.E1, self.E2, self.G12, self.nu12, self.rho], 
                           dtype=torch.float32)
    
    def __repr__(self):
        return (f"OrthotropicMaterial('{self.name}', "
                f"E1={self.E1:.2e}, E2={self.E2:.2e}, "
                f"G12={self.G12:.2e}, ν12={self.nu12})")


# =============================================================================
# KOMPOZİT KATMAN (PLY)
# =============================================================================

@dataclass
class Ply:
    """
    Tek bir kompozit katman.
    
    Attributes:
        angle: Fiber açısı (derece)
        thickness: Katman kalınlığı (m)
        material: Malzeme adı veya OrthotropicMaterial objesi
    """
    angle: float                      # Fiber açısı (derece)
    thickness: float                  # Kalınlık (m)
    material: Union[str, OrthotropicMaterial]  # Malzeme
    
    def get_material(self) -> OrthotropicMaterial:
        """Malzeme objesini döndür."""
        if isinstance(self.material, OrthotropicMaterial):
            return self.material
        else:
            return MaterialLibrary.get(self.material)
    
    def __repr__(self):
        mat_name = self.material if isinstance(self.material, str) else self.material.name
        return f"Ply({self.angle}°, t={self.thickness*1000:.3f}mm, {mat_name})"


# =============================================================================
# KOMPOZİT LAMİNAT (CLT)
# =============================================================================

class CompositeLaminate:
    """
    Kompozit laminat - Classical Laminate Theory (CLT) implementasyonu.
    
    Kullanım:
        laminate = CompositeLaminate([
            Ply(angle=0, thickness=0.125e-3, material="carbon_epoxy_T300"),
            Ply(angle=45, thickness=0.125e-3, material="carbon_epoxy_T300"),
            Ply(angle=-45, thickness=0.125e-3, material="carbon_epoxy_T300"),
            Ply(angle=90, thickness=0.125e-3, material="carbon_epoxy_T300"),
        ])
        
        A, B, D = laminate.get_ABD_matrices()
        D_eq = laminate.equivalent_flexural_rigidity()
    """
    
    def __init__(self, plies: List[Ply]):
        """
        Args:
            plies: Ply listesi (alttan üste sıralı)
        """
        self.plies = plies
        self._compute_geometry()
        self._compute_ABD()
    
    def _compute_geometry(self):
        """Laminat geometrisini hesapla."""
        self.total_thickness = sum(ply.thickness for ply in self.plies)
        
        # Katman z koordinatları (ortadan ölçülü)
        self.z_coords = []
        z = -self.total_thickness / 2
        for ply in self.plies:
            z_bottom = z
            z_top = z + ply.thickness
            self.z_coords.append((z_bottom, z_top))
            z = z_top
    
    def _compute_ABD(self):
        """[A], [B], [D] matrislerini hesapla."""
        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        D = np.zeros((3, 3))
        
        for ply, (z_bot, z_top) in zip(self.plies, self.z_coords):
            material = ply.get_material()
            Q_bar = material.get_Q_bar_matrix(ply.angle)
            
            # A: membran rijitliği
            A += Q_bar * (z_top - z_bot)
            
            # B: coupling (eğilme-membran bağlaşımı)
            B += 0.5 * Q_bar * (z_top**2 - z_bot**2)
            
            # D: eğilme rijitliği
            D += (1/3) * Q_bar * (z_top**3 - z_bot**3)
        
        self.A = A
        self.B = B
        self.D = D
    
    def get_ABD_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """[A], [B], [D] matrislerini döndür."""
        return self.A, self.B, self.D
    
    def get_ABD_tensor(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ABD matrislerini PyTorch tensor olarak döndür."""
        return (
            torch.from_numpy(self.A).float(),
            torch.from_numpy(self.B).float(),
            torch.from_numpy(self.D).float()
        )
    
    def is_symmetric(self, tol: float = 1e-6) -> bool:
        """Laminat simetrik mi? (B ≈ 0)"""
        return np.allclose(self.B, 0, atol=tol * np.max(np.abs(self.A)))
    
    def is_balanced(self, tol: float = 1e-6) -> bool:
        """Laminat dengeli mi? (A16 ≈ A26 ≈ 0)"""
        scale = np.max(np.abs(self.A))
        return (abs(self.A[0, 2]) < tol * scale and 
                abs(self.A[1, 2]) < tol * scale)
    
    def equivalent_flexural_rigidity(self) -> float:
        """
        Eşdeğer izotropik eğilme rijitliği.
        
        D_eq = (D11 × D22)^0.5 için yaklaşık değer.
        Burkulma hesaplarında kullanılır.
        """
        return math.sqrt(self.D[0, 0] * self.D[1, 1])
    
    def equivalent_Ex(self) -> float:
        """Eşdeğer membran modülü (x yönü)."""
        h = self.total_thickness
        a = np.linalg.inv(self.A)
        return 1 / (h * a[0, 0])
    
    def equivalent_Ey(self) -> float:
        """Eşdeğer membran modülü (y yönü)."""
        h = self.total_thickness
        a = np.linalg.inv(self.A)
        return 1 / (h * a[1, 1])
    
    def buckling_load_uniaxial(self, Lx: float, Ly: float, m: int = 1, n: int = 1) -> float:
        """
        Tek eksenli basma altında burkulma yükü (basit mesnetli).
        
        N_cr = π² × [D11×m⁴/a⁴ + 2(D12+2D66)×m²n²/(a²b²) + D22×n⁴/b⁴] / m²
        
        Args:
            Lx: x yönü uzunluk (m)
            Ly: y yönü uzunluk (m)
            m, n: Yarım dalga sayıları
            
        Returns:
            N_cr: Kritik yük (N/m)
        """
        D11 = self.D[0, 0]
        D12 = self.D[0, 1]
        D22 = self.D[1, 1]
        D66 = self.D[2, 2]
        
        a, b = Lx, Ly
        
        term1 = D11 * (m**4) / (a**4)
        term2 = 2 * (D12 + 2*D66) * (m**2 * n**2) / (a**2 * b**2)
        term3 = D22 * (n**4) / (b**4)
        
        N_cr = (math.pi**2) * (term1 + term2 + term3) / (m**2 / a**2)
        
        return N_cr
    
    def find_critical_buckling(self, Lx: float, Ly: float, max_m: int = 5, max_n: int = 5) -> Dict:
        """
        Kritik burkulma modunu bul.
        
        Returns:
            Dict: {'N_cr': float, 'm': int, 'n': int}
        """
        min_N_cr = float('inf')
        best_mode = (1, 1)
        
        for m in range(1, max_m + 1):
            for n in range(1, max_n + 1):
                N_cr = self.buckling_load_uniaxial(Lx, Ly, m, n)
                if N_cr < min_N_cr:
                    min_N_cr = N_cr
                    best_mode = (m, n)
        
        return {
            'N_cr': min_N_cr,
            'm': best_mode[0],
            'n': best_mode[1]
        }
    
    def average_density(self) -> float:
        """Ortalama yoğunluk (kg/m³)."""
        total_mass_per_area = 0
        for ply in self.plies:
            material = ply.get_material()
            total_mass_per_area += material.rho * ply.thickness
        return total_mass_per_area / self.total_thickness
    
    def stacking_sequence(self) -> str:
        """Katman dizilimi string olarak."""
        angles = [f"{int(ply.angle)}" for ply in self.plies]
        return f"[{'/'.join(angles)}]"
    
    def __repr__(self):
        sym = "S" if self.is_symmetric() else ""
        bal = "B" if self.is_balanced() else ""
        flags = f" ({sym}{bal})" if sym or bal else ""
        return f"CompositeLaminate({self.stacking_sequence()}, h={self.total_thickness*1000:.2f}mm{flags})"


# =============================================================================
# MALZEME KÜTÜPHANESİ
# =============================================================================

class MaterialLibrary:
    """
    Hazır malzeme kütüphanesi.
    
    50+ mühendislik malzemesi içerir:
    - Metaller (çelik, alüminyum, titanyum, vb.)
    - Polimerler (ABS, PLA, PEEK, vb.)
    - Seramikler
    - Kompozitler (karbon fiber, cam elyaf, kevlar, vb.)
    - Doğal malzemeler (ahşap)
    - Özel malzemeler (grafen, shape memory alloys)
    """
    
    # =========================================================================
    # METALLER - İzotropik
    # =========================================================================
    
    _isotropic_metals = {
        # ÇELİKLER
        "steel_mild": IsotropicMaterial(
            "Yumuşak Çelik (Mild Steel)", 
            E=200e9, nu=0.29, rho=7850,
            sigma_yield=250e6, sigma_ultimate=400e6,
            alpha=12e-6, K_IC=50e6
        ),
        "steel_304": IsotropicMaterial(
            "Paslanmaz Çelik 304", 
            E=193e9, nu=0.29, rho=8000,
            sigma_yield=215e6, sigma_ultimate=505e6,
            alpha=17.2e-6, K_IC=100e6
        ),
        "steel_316": IsotropicMaterial(
            "Paslanmaz Çelik 316", 
            E=193e9, nu=0.27, rho=8000,
            sigma_yield=290e6, sigma_ultimate=580e6,
            alpha=16e-6, K_IC=100e6
        ),
        "steel_4340": IsotropicMaterial(
            "Alaşımlı Çelik 4340", 
            E=205e9, nu=0.29, rho=7850,
            sigma_yield=860e6, sigma_ultimate=1080e6,
            alpha=12.3e-6, K_IC=50e6
        ),
        "steel_maraging": IsotropicMaterial(
            "Maraging Çelik (18Ni-250)", 
            E=190e9, nu=0.30, rho=8000,
            sigma_yield=1700e6, sigma_ultimate=1760e6,
            alpha=10.1e-6
        ),
        
        # ALÜMİNYUM
        "aluminum_pure": IsotropicMaterial(
            "Saf Alüminyum", 
            E=70e9, nu=0.33, rho=2700,
            sigma_yield=35e6, sigma_ultimate=90e6,
            alpha=23.6e-6
        ),
        "aluminum_6061_T6": IsotropicMaterial(
            "Alüminyum 6061-T6", 
            E=68.9e9, nu=0.33, rho=2700,
            sigma_yield=276e6, sigma_ultimate=310e6,
            alpha=23.6e-6, K_IC=29e6
        ),
        "aluminum_7075_T6": IsotropicMaterial(
            "Alüminyum 7075-T6 (Havacılık)", 
            E=71.7e9, nu=0.33, rho=2810,
            sigma_yield=503e6, sigma_ultimate=572e6,
            alpha=23.4e-6, K_IC=24e6
        ),
        "aluminum_2024_T3": IsotropicMaterial(
            "Alüminyum 2024-T3 (Havacılık)", 
            E=73.1e9, nu=0.33, rho=2780,
            sigma_yield=345e6, sigma_ultimate=483e6,
            alpha=23.2e-6, K_IC=37e6
        ),
        
        # TİTANYUM
        "titanium_pure": IsotropicMaterial(
            "Saf Titanyum (Grade 2)", 
            E=103e9, nu=0.34, rho=4510,
            sigma_yield=275e6, sigma_ultimate=345e6,
            alpha=8.6e-6
        ),
        "titanium_Ti6Al4V": IsotropicMaterial(
            "Ti-6Al-4V (Havacılık)", 
            E=113.8e9, nu=0.342, rho=4430,
            sigma_yield=880e6, sigma_ultimate=950e6,
            alpha=8.6e-6, K_IC=75e6
        ),
        "titanium_Ti6Al4V_ELI": IsotropicMaterial(
            "Ti-6Al-4V ELI (Medikal)", 
            E=114e9, nu=0.34, rho=4430,
            sigma_yield=795e6, sigma_ultimate=860e6,
            alpha=8.6e-6
        ),
        
        # BAKIR VE ALAŞIMLARI
        "copper_pure": IsotropicMaterial(
            "Saf Bakır", 
            E=117e9, nu=0.34, rho=8960,
            sigma_yield=70e6, sigma_ultimate=220e6,
            alpha=17e-6
        ),
        "bronze_phosphor": IsotropicMaterial(
            "Fosfor Bronz", 
            E=110e9, nu=0.34, rho=8800,
            sigma_yield=380e6, sigma_ultimate=520e6,
            alpha=17.8e-6
        ),
        "brass_70_30": IsotropicMaterial(
            "Pirinç 70-30", 
            E=110e9, nu=0.35, rho=8530,
            sigma_yield=200e6, sigma_ultimate=525e6,
            alpha=20e-6
        ),
        
        # NİKEL ALAŞIMLARI
        "inconel_718": IsotropicMaterial(
            "Inconel 718 (Süperalaşım)", 
            E=200e9, nu=0.29, rho=8190,
            sigma_yield=1035e6, sigma_ultimate=1240e6,
            alpha=13e-6
        ),
        "hastelloy_X": IsotropicMaterial(
            "Hastelloy X (Yüksek Sıcaklık)", 
            E=205e9, nu=0.32, rho=8220,
            sigma_yield=355e6, sigma_ultimate=755e6,
            alpha=13.9e-6
        ),
        
        # MAGNEZYUM
        "magnesium_AZ31": IsotropicMaterial(
            "Magnezyum AZ31B", 
            E=45e9, nu=0.35, rho=1770,
            sigma_yield=200e6, sigma_ultimate=290e6,
            alpha=26e-6
        ),
        
        # TUNGSTEN VE DİĞER
        "tungsten": IsotropicMaterial(
            "Tungsten", 
            E=411e9, nu=0.28, rho=19300,
            sigma_yield=750e6, sigma_ultimate=980e6,
            alpha=4.5e-6
        ),
        "beryllium": IsotropicMaterial(
            "Berilyum (Uzay)", 
            E=287e9, nu=0.032, rho=1850,
            sigma_yield=240e6, sigma_ultimate=370e6,
            alpha=11.3e-6
        ),
    }
    
    # =========================================================================
    # POLİMERLER - İzotropik
    # =========================================================================
    
    _isotropic_polymers = {
        # 3D BASKI MALZEMELERİ
        "pla": IsotropicMaterial(
            "PLA (3D Baskı)", 
            E=3.5e9, nu=0.36, rho=1240,
            sigma_yield=60e6, sigma_ultimate=65e6
        ),
        "abs": IsotropicMaterial(
            "ABS (3D Baskı)", 
            E=2.3e9, nu=0.35, rho=1050,
            sigma_yield=40e6, sigma_ultimate=45e6
        ),
        "petg": IsotropicMaterial(
            "PETG (3D Baskı)", 
            E=2.1e9, nu=0.38, rho=1270,
            sigma_yield=50e6, sigma_ultimate=53e6
        ),
        "nylon_pa12": IsotropicMaterial(
            "Naylon PA12", 
            E=1.7e9, nu=0.40, rho=1010,
            sigma_yield=45e6, sigma_ultimate=50e6
        ),
        
        # MÜHENDİSLİK PLASTİKLERİ
        "peek": IsotropicMaterial(
            "PEEK (Yüksek Performans)", 
            E=3.6e9, nu=0.38, rho=1300,
            sigma_yield=100e6, sigma_ultimate=100e6,
            alpha=47e-6
        ),
        "ultem_pei": IsotropicMaterial(
            "ULTEM (PEI)", 
            E=3.3e9, nu=0.36, rho=1270,
            sigma_yield=105e6, sigma_ultimate=105e6
        ),
        "polycarbonate": IsotropicMaterial(
            "Polikarbonat", 
            E=2.4e9, nu=0.37, rho=1200,
            sigma_yield=62e6, sigma_ultimate=66e6,
            alpha=70e-6
        ),
        "acrylic_pmma": IsotropicMaterial(
            "Akrilik (PMMA)", 
            E=3.0e9, nu=0.35, rho=1180,
            sigma_yield=72e6, sigma_ultimate=72e6
        ),
        "delrin_pom": IsotropicMaterial(
            "Delrin (POM)", 
            E=3.1e9, nu=0.35, rho=1410,
            sigma_yield=65e6, sigma_ultimate=70e6
        ),
        "ptfe_teflon": IsotropicMaterial(
            "PTFE (Teflon)", 
            E=0.5e9, nu=0.46, rho=2200,
            sigma_yield=20e6, sigma_ultimate=25e6
        ),
        
        # ELASTOMERLER
        "silicone_rubber": IsotropicMaterial(
            "Silikon Kauçuk", 
            E=0.01e9, nu=0.48, rho=1100,
            sigma_ultimate=7e6
        ),
        "neoprene": IsotropicMaterial(
            "Neopren", 
            E=0.007e9, nu=0.49, rho=1230,
            sigma_ultimate=20e6
        ),
    }
    
    # =========================================================================
    # SERAMİKLER VE CAMLAR - İzotropik
    # =========================================================================
    
    _isotropic_ceramics = {
        "alumina_Al2O3": IsotropicMaterial(
            "Alümina (Al2O3)", 
            E=380e9, nu=0.22, rho=3960,
            sigma_ultimate=300e6, K_IC=4e6
        ),
        "silicon_carbide": IsotropicMaterial(
            "Silisyum Karbür (SiC)", 
            E=410e9, nu=0.14, rho=3100,
            sigma_ultimate=400e6, K_IC=3.5e6
        ),
        "zirconia_ZrO2": IsotropicMaterial(
            "Zirkonya (ZrO2)", 
            E=200e9, nu=0.31, rho=6000,
            sigma_ultimate=900e6, K_IC=10e6
        ),
        "borosilicate_glass": IsotropicMaterial(
            "Borosilikat Cam (Pyrex)", 
            E=64e9, nu=0.20, rho=2230,
            sigma_ultimate=70e6, K_IC=0.7e6,
            alpha=3.3e-6
        ),
        "soda_lime_glass": IsotropicMaterial(
            "Soda Kireç Camı", 
            E=70e9, nu=0.22, rho=2500,
            sigma_ultimate=40e6, K_IC=0.75e6,
            alpha=9e-6
        ),
        "fused_silica": IsotropicMaterial(
            "Kaynaşık Silika", 
            E=73e9, nu=0.17, rho=2200,
            sigma_ultimate=110e6,
            alpha=0.5e-6  # Çok düşük termal genleşme
        ),
    }
    
    # =========================================================================
    # ÖZEL VE İLERİ MALZEMELER - İzotropik
    # =========================================================================
    
    _isotropic_special = {
        "concrete": IsotropicMaterial(
            "Beton (C30)", 
            E=30e9, nu=0.20, rho=2400,
            sigma_ultimate=30e6  # Basma
        ),
        "granite": IsotropicMaterial(
            "Granit", 
            E=50e9, nu=0.25, rho=2700,
            sigma_ultimate=200e6  # Basma
        ),
        "cork": IsotropicMaterial(
            "Mantar", 
            E=0.02e9, nu=0.0, rho=120,  # Poisson ≈ 0!
            sigma_ultimate=1e6
        ),
        "foam_pu_rigid": IsotropicMaterial(
            "Sert PU Köpük", 
            E=0.025e9, nu=0.30, rho=50,
            sigma_ultimate=0.5e6
        ),
        "graphite": IsotropicMaterial(
            "Grafit", 
            E=11e9, nu=0.20, rho=1750
        ),
        "diamond": IsotropicMaterial(
            "Elmas", 
            E=1050e9, nu=0.10, rho=3520,
            alpha=1.0e-6
        ),
        
        # SHAPE MEMORY ALLOYS
        "nitinol_austenite": IsotropicMaterial(
            "Nitinol (Austenit fazı)", 
            E=83e9, nu=0.33, rho=6450,
            sigma_yield=560e6  # Dönüşüm gerilmesi
        ),
        "nitinol_martensite": IsotropicMaterial(
            "Nitinol (Martensit fazı)", 
            E=28e9, nu=0.33, rho=6450
        ),
    }
    
    # =========================================================================
    # KOMPOZİT MALZEMELER - Ortotropik
    # =========================================================================
    
    _orthotropic_composites = {
        # KARBON FİBER / EPOKSİ
        "carbon_epoxy_T300": OrthotropicMaterial(
            "Karbon/Epoksi T300 (Standart)", 
            E1=181e9, E2=10.3e9, G12=7.17e9, nu12=0.28, rho=1600,
            Xt=1500e6, Xc=1200e6, Yt=50e6, Yc=200e6, S12=70e6,
            alpha1=-0.02e-6, alpha2=22e-6
        ),
        "carbon_epoxy_T700": OrthotropicMaterial(
            "Karbon/Epoksi T700 (Yüksek Mukavemet)", 
            E1=230e9, E2=10e9, G12=8e9, nu12=0.25, rho=1550,
            Xt=2100e6, Xc=1400e6, Yt=55e6, Yc=220e6, S12=75e6,
            alpha1=-0.4e-6, alpha2=25e-6
        ),
        "carbon_epoxy_IM7": OrthotropicMaterial(
            "Karbon/Epoksi IM7 (Ara Modül)", 
            E1=165e9, E2=8.4e9, G12=5.6e9, nu12=0.34, rho=1580,
            Xt=2700e6, Xc=1600e6, Yt=60e6, Yc=250e6, S12=100e6
        ),
        "carbon_epoxy_M55J": OrthotropicMaterial(
            "Karbon/Epoksi M55J (Yüksek Modül)", 
            E1=338e9, E2=6.9e9, G12=5.2e9, nu12=0.30, rho=1600,
            Xt=1740e6, Xc=1000e6, Yt=40e6, Yc=180e6, S12=50e6,
            alpha1=-1.0e-6, alpha2=30e-6
        ),
        
        # CAM ELYAF / EPOKSİ
        "glass_epoxy_E": OrthotropicMaterial(
            "E-Cam/Epoksi", 
            E1=45e9, E2=12e9, G12=5.5e9, nu12=0.28, rho=2100,
            Xt=1020e6, Xc=620e6, Yt=40e6, Yc=140e6, S12=60e6,
            alpha1=6.5e-6, alpha2=22e-6
        ),
        "glass_epoxy_S2": OrthotropicMaterial(
            "S2-Cam/Epoksi (Yüksek Mukavemet)", 
            E1=52e9, E2=13e9, G12=6.5e9, nu12=0.26, rho=2000,
            Xt=1280e6, Xc=700e6, Yt=55e6, Yc=160e6, S12=70e6
        ),
        
        # KEVLAR / ARAMİD
        "kevlar_49_epoxy": OrthotropicMaterial(
            "Kevlar 49/Epoksi", 
            E1=80e9, E2=5.5e9, G12=2.2e9, nu12=0.34, rho=1380,
            Xt=1400e6, Xc=280e6, Yt=30e6, Yc=140e6, S12=50e6,
            alpha1=-4.0e-6, alpha2=60e-6
        ),
        "kevlar_29_epoxy": OrthotropicMaterial(
            "Kevlar 29/Epoksi (Balistik)", 
            E1=62e9, E2=5.0e9, G12=2.0e9, nu12=0.35, rho=1440,
            Xt=1200e6, Xc=250e6, Yt=28e6, Yc=130e6, S12=45e6
        ),
        
        # BORU / UNİ-TAPE
        "boron_epoxy": OrthotropicMaterial(
            "Bor/Epoksi (Havacılık)", 
            E1=210e9, E2=19e9, G12=5.6e9, nu12=0.21, rho=2000,
            Xt=1400e6, Xc=2300e6, Yt=65e6, Yc=280e6, S12=125e6
        ),
        
        # BAZALT FİBER
        "basalt_epoxy": OrthotropicMaterial(
            "Bazalt/Epoksi", 
            E1=50e9, E2=11e9, G12=5e9, nu12=0.30, rho=1900,
            Xt=950e6, Xc=550e6, Yt=35e6, Yc=120e6, S12=55e6
        ),
        
        # NATURAL FİBER
        "flax_epoxy": OrthotropicMaterial(
            "Keten/Epoksi (Doğal)", 
            E1=35e9, E2=5e9, G12=2.5e9, nu12=0.30, rho=1300,
            Xt=280e6, Xc=150e6, Yt=20e6, Yc=80e6, S12=30e6
        ),
        "hemp_epoxy": OrthotropicMaterial(
            "Kenevir/Epoksi (Doğal)", 
            E1=25e9, E2=4e9, G12=2e9, nu12=0.32, rho=1200,
            Xt=200e6, Xc=120e6, Yt=15e6, Yc=60e6, S12=25e6
        ),
        
        # TERMOPLASTİK KOMPOZİT
        "carbon_peek": OrthotropicMaterial(
            "Karbon/PEEK (Termoplastik)", 
            E1=145e9, E2=10e9, G12=6e9, nu12=0.30, rho=1600,
            Xt=2000e6, Xc=1000e6, Yt=80e6, Yc=200e6, S12=100e6
        ),
        "carbon_pps": OrthotropicMaterial(
            "Karbon/PPS", 
            E1=135e9, E2=9e9, G12=5.5e9, nu12=0.32, rho=1580
        ),
        
        # METAL MATRİS KOMPOZİT
        "SiC_aluminum": OrthotropicMaterial(
            "SiC/Alüminyum (MMC)", 
            E1=210e9, E2=150e9, G12=60e9, nu12=0.25, rho=2900,
            Xt=1500e6, Xc=1500e6, Yt=400e6, Yc=400e6
        ),
        
        # AHŞAP
        "wood_oak": OrthotropicMaterial(
            "Meşe Ahşap", 
            E1=12e9, E2=1e9, G12=0.7e9, nu12=0.37, rho=700,
            Xt=100e6, Xc=50e6, Yt=10e6, Yc=25e6, S12=12e6
        ),
        "wood_pine": OrthotropicMaterial(
            "Çam Ahşap", 
            E1=10e9, E2=0.5e9, G12=0.6e9, nu12=0.35, rho=500,
            Xt=80e6, Xc=40e6, Yt=5e6, Yc=20e6, S12=8e6
        ),
        "wood_balsa": OrthotropicMaterial(
            "Balsa (Ultra Hafif)", 
            E1=3.5e9, E2=0.15e9, G12=0.18e9, nu12=0.30, rho=160,
            Xt=20e6, Xc=12e6, Yt=2e6, Yc=6e6, S12=3e6
        ),
        "bamboo": OrthotropicMaterial(
            "Bambu", 
            E1=15e9, E2=1.5e9, G12=0.9e9, nu12=0.30, rho=700,
            Xt=150e6, Xc=60e6, Yt=15e6, Yc=30e6
        ),
    }
    
    # Tüm malzemeleri birleştir
    _all_materials = {
        **_isotropic_metals,
        **_isotropic_polymers,
        **_isotropic_ceramics,
        **_isotropic_special,
        **_orthotropic_composites
    }
    
    @classmethod
    def get(cls, name: str) -> Union[IsotropicMaterial, OrthotropicMaterial]:
        """Malzeme adına göre malzeme döndür."""
        if name not in cls._all_materials:
            raise ValueError(f"Malzeme bulunamadı: '{name}'. "
                           f"Kullanılabilir: {list(cls._all_materials.keys())}")
        return cls._all_materials[name]
    
    @classmethod
    def list_all(cls) -> List[str]:
        """Tüm malzeme adlarını listele."""
        return list(cls._all_materials.keys())
    
    @classmethod
    def list_isotropic(cls) -> List[str]:
        """İzotropik malzeme adlarını listele."""
        return [name for name, mat in cls._all_materials.items() 
                if mat.material_type == MaterialType.ISOTROPIC]
    
    @classmethod
    def list_orthotropic(cls) -> List[str]:
        """Ortotropik malzeme adlarını listele."""
        return [name for name, mat in cls._all_materials.items() 
                if mat.material_type == MaterialType.ORTHOTROPIC]
    
    @classmethod
    def list_metals(cls) -> List[str]:
        """Metal malzeme adlarını listele."""
        return list(cls._isotropic_metals.keys())
    
    @classmethod
    def list_composites(cls) -> List[str]:
        """Kompozit malzeme adlarını listele."""
        return list(cls._orthotropic_composites.keys())
    
    @classmethod
    def list_polymers(cls) -> List[str]:
        """Polimer malzeme adlarını listele."""
        return list(cls._isotropic_polymers.keys())
    
    @classmethod
    def search(cls, keyword: str) -> List[str]:
        """Anahtar kelimeye göre malzeme ara."""
        keyword = keyword.lower()
        return [name for name in cls._all_materials.keys() 
                if keyword in name.lower() or 
                keyword in cls._all_materials[name].name.lower()]
    
    @classmethod
    def count(cls) -> int:
        """Toplam malzeme sayısı."""
        return len(cls._all_materials)
    
    @classmethod
    def add_custom(cls, key: str, material: Union[IsotropicMaterial, OrthotropicMaterial]):
        """Özel malzeme ekle."""
        cls._all_materials[key] = material
    
    @classmethod
    def info(cls, name: str) -> str:
        """Malzeme bilgisi string olarak."""
        mat = cls.get(name)
        info = f"\n{'='*60}\n"
        info += f"Malzeme: {mat.name}\n"
        info += f"{'='*60}\n"
        
        if mat.material_type == MaterialType.ISOTROPIC:
            info += f"Tip: İzotropik\n"
            info += f"E  = {mat.E/1e9:.1f} GPa\n"
            info += f"ν  = {mat.nu:.3f}\n"
            info += f"ρ  = {mat.rho:.0f} kg/m³\n"
            info += f"G  = {mat.G/1e9:.1f} GPa\n"
            if mat.sigma_yield:
                info += f"σ_yield = {mat.sigma_yield/1e6:.0f} MPa\n"
            if mat.sigma_ultimate:
                info += f"σ_ult   = {mat.sigma_ultimate/1e6:.0f} MPa\n"
            if mat.alpha:
                info += f"α  = {mat.alpha*1e6:.1f} μm/m·°C\n"
            if mat.K_IC:
                info += f"K_IC = {mat.K_IC/1e6:.1f} MPa√m\n"
        else:
            info += f"Tip: Ortotropik\n"
            info += f"E1  = {mat.E1/1e9:.1f} GPa\n"
            info += f"E2  = {mat.E2/1e9:.1f} GPa\n"
            info += f"G12 = {mat.G12/1e9:.1f} GPa\n"
            info += f"ν12 = {mat.nu12:.3f}\n"
            info += f"ρ   = {mat.rho:.0f} kg/m³\n"
            if mat.Xt:
                info += f"\nMukavemet:\n"
                info += f"  Xt = {mat.Xt/1e6:.0f} MPa (Fiber çekme)\n"
                info += f"  Xc = {mat.Xc/1e6:.0f} MPa (Fiber basma)\n"
                info += f"  Yt = {mat.Yt/1e6:.0f} MPa (Enine çekme)\n"
                info += f"  Yc = {mat.Yc/1e6:.0f} MPa (Enine basma)\n"
                info += f"  S12 = {mat.S12/1e6:.0f} MPa (Kayma)\n"
        
        return info


# =============================================================================
# MALZEME NÖRONU (PyTorch)
# =============================================================================

class AdvancedMaterialNeuron(nn.Module):
    """
    Gelişmiş Malzeme Nöronu.
    
    Hem izotropik hem ortotropik malzemeleri destekler.
    CLT hesaplamaları için kompozit desteği.
    """
    
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
    
    def plate_rigidity_isotropic(self, E: torch.Tensor, nu: torch.Tensor, 
                                  h: torch.Tensor) -> torch.Tensor:
        """İzotropik plaka rijitliği: D = Eh³/[12(1-ν²)]"""
        return self.scale * E * h**3 / (12 * (1 - nu**2))
    
    def plate_rigidity_orthotropic(self, E1: torch.Tensor, E2: torch.Tensor,
                                    G12: torch.Tensor, nu12: torch.Tensor,
                                    h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Ortotropik plaka rijitlik matrisi [D].
        
        Returns:
            Dict: D11, D12, D22, D66
        """
        nu21 = nu12 * E2 / E1
        denom = 1 - nu12 * nu21
        
        Q11 = E1 / denom
        Q12 = nu12 * E2 / denom
        Q22 = E2 / denom
        Q66 = G12
        
        factor = h**3 / 12
        
        return {
            'D11': self.scale * Q11 * factor,
            'D12': self.scale * Q12 * factor,
            'D22': self.scale * Q22 * factor,
            'D66': self.scale * Q66 * factor
        }
    
    def forward(self, material: Union[IsotropicMaterial, OrthotropicMaterial],
                h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Malzemeden rijitlik hesapla.
        """
        if material.material_type == MaterialType.ISOTROPIC:
            E = torch.tensor(material.E, dtype=torch.float32)
            nu = torch.tensor(material.nu, dtype=torch.float32)
            D = self.plate_rigidity_isotropic(E, nu, h)
            return {'D': D, 'type': 'isotropic'}
        else:
            E1 = torch.tensor(material.E1, dtype=torch.float32)
            E2 = torch.tensor(material.E2, dtype=torch.float32)
            G12 = torch.tensor(material.G12, dtype=torch.float32)
            nu12 = torch.tensor(material.nu12, dtype=torch.float32)
            D_dict = self.plate_rigidity_orthotropic(E1, E2, G12, nu12, h)
            D_dict['type'] = 'orthotropic'
            return D_dict


# =============================================================================
# __all__
# =============================================================================

__all__ = [
    # Enums
    'MaterialType',
    
    # Malzeme sınıfları
    'IsotropicMaterial',
    'OrthotropicMaterial',
    'Ply',
    'CompositeLaminate',
    
    # Kütüphane
    'MaterialLibrary',
    
    # Nöron
    'AdvancedMaterialNeuron',
]


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SPINE Malzeme Modülü Test")
    print("=" * 60)
    
    # Malzeme sayısı
    print(f"\nToplam malzeme sayısı: {MaterialLibrary.count()}")
    print(f"  - İzotropik: {len(MaterialLibrary.list_isotropic())}")
    print(f"  - Ortotropik: {len(MaterialLibrary.list_orthotropic())}")
    
    # Örnek malzemeler
    print("\n" + "=" * 60)
    print("ÖRNEK MALZEMELER")
    print("=" * 60)
    
    # Çelik
    steel = MaterialLibrary.get("steel_304")
    print(f"\n1. {steel}")
    print(f"   G = {steel.G/1e9:.1f} GPa")
    print(f"   D (h=5mm) = {steel.plate_rigidity(0.005):.2f} N·m")
    
    # Karbon fiber
    carbon = MaterialLibrary.get("carbon_epoxy_T300")
    print(f"\n2. {carbon}")
    print(f"   E1/E2 = {carbon.E1/carbon.E2:.1f}")
    
    # Kompozit laminat
    print("\n" + "=" * 60)
    print("KOMPOZİT LAMİNAT TESTİ")
    print("=" * 60)
    
    laminate = CompositeLaminate([
        Ply(angle=0, thickness=0.125e-3, material="carbon_epoxy_T300"),
        Ply(angle=45, thickness=0.125e-3, material="carbon_epoxy_T300"),
        Ply(angle=-45, thickness=0.125e-3, material="carbon_epoxy_T300"),
        Ply(angle=90, thickness=0.125e-3, material="carbon_epoxy_T300"),
        Ply(angle=90, thickness=0.125e-3, material="carbon_epoxy_T300"),
        Ply(angle=-45, thickness=0.125e-3, material="carbon_epoxy_T300"),
        Ply(angle=45, thickness=0.125e-3, material="carbon_epoxy_T300"),
        Ply(angle=0, thickness=0.125e-3, material="carbon_epoxy_T300"),
    ])
    
    print(f"\nLaminat: {laminate}")
    print(f"Dizilim: {laminate.stacking_sequence()}")
    print(f"Kalınlık: {laminate.total_thickness*1000:.2f} mm")
    print(f"Simetrik: {laminate.is_symmetric()}")
    print(f"Dengeli: {laminate.is_balanced()}")
    
    A, B, D = laminate.get_ABD_matrices()
    print(f"\nD11 = {D[0,0]/1e3:.2f} kN·m")
    print(f"D22 = {D[1,1]/1e3:.2f} kN·m")
    print(f"D12 = {D[0,1]/1e3:.2f} kN·m")
    
    # Burkulma
    result = laminate.find_critical_buckling(0.5, 0.3)
    print(f"\nBurkulma (0.5m × 0.3m):")
    print(f"  N_cr = {result['N_cr']/1000:.2f} kN/m")
    print(f"  Mod: ({result['m']}, {result['n']})")
    
    print("\n" + "=" * 60)
    print("TEST TAMAMLANDI ✅")
    print("=" * 60)

