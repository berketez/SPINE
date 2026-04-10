"""
SPINE Deformation Module: Domain Deformation ile Karmaşık Geometri

Karmaşık geometriyi düz (regular) hesaplama alanına deforme eder,
SPINE'ın FFT-tabanlı spektral operatörlerini korur, sonra Jacobian
düzeltmesiyle fiziksel türevlere dönüştürür.

Pipeline:
    Fiziksel domain (irregular) → IPHI → Hesaplama domain (regular) → FFT → Jacobian → Fiziksel türevler

Geo-FNO (Li et al., 2022) ve GINO (Li et al., NeurIPS 2023) yaklaşımlarından
esinlenilmiştir.

Kullanım:
    from spine_deformation import DeformationNet, DeformedSpectralOps2D

    # Deformasyon ağı
    iphi = DeformationNet(width=32, code_dim=16)

    # Deformed spektral operatörler
    deformed_ops = DeformedSpectralOps2D(resolution=64, Lx=1.0, Ly=1.0, iphi=iphi)

    # Fiziksel gradient
    df_dx, df_dy = deformed_ops.gradient(f, x_phys, code)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Dict
from torch.fft import fft2, ifft2, fftfreq

# SPINE'dan import
from spine import (
    SpectralOps2DStruct, SpectralOps3D,
    safe_fft2, safe_ifft2, get_device, DEVICE
)


# =============================================================================
# DEFORMATION NETWORK (IPHI)
# =============================================================================

class DeformationNet(nn.Module):
    """
    Inverse Phi (IPHI): x_physical → x_computational

    Fiziksel domain'deki noktaları düz hesaplama domain'ine map'ler.
    SPINE'ın FFT operatörleri hesaplama domain'inde çalışır.

    Geo-FNO v2'den adapte:
    - Multiplicative residual: return x + x * xd
    - NeRF positional encoding
    - Geometry code conditioning

    Args:
        width: Gizli katman genişliği
        code_dim: Geometri code vektörü boyutu (0=kod yok)
        n_layers: FC katman sayısı
        use_positional_encoding: NeRF sin/cos encoding
    """

    def __init__(self, width: int = 32, code_dim: int = 0,
                 n_layers: int = 3, use_positional_encoding: bool = True):
        super().__init__()

        self.width = width
        self.code_dim = code_dim
        self.use_pe = use_positional_encoding

        # Feature engineering: [x, y, angle, radius] = 4 features
        input_dim = 4

        # Positional encoding
        n_freq = width // 4
        self.register_buffer(
            'B_freq',
            math.pi * torch.pow(2, torch.arange(0, n_freq, dtype=torch.float)).reshape(1, 1, 1, n_freq)
        )

        if use_positional_encoding:
            # input: fc0(4→w) + sin(B*x)(4*n_freq) + cos(B*x)(4*n_freq) = 3*width
            pe_dim = 3 * width
        else:
            pe_dim = width

        self.fc0 = nn.Linear(input_dim, width)

        # Code conditioning
        if code_dim > 0:
            self.fc_code = nn.Linear(code_dim, width)
            layer_input = pe_dim + width
        else:
            self.fc_no_code = nn.Linear(pe_dim, 4 * width)
            layer_input = 4 * width

        # Hidden layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Linear(layer_input, layer_input))

        # Output: 2D deformasyon
        self.fc_out = nn.Linear(layer_input, 2)

        # Center (referans noktası — genelde domain merkezi)
        self.register_buffer('center', torch.tensor([0.5, 0.5]).reshape(1, 1, 2))

        # Scale factor (deformasyonu sınırlamak için)
        self.max_deformation = nn.Parameter(torch.tensor(0.5))

    def set_center(self, cx: float, cy: float):
        """Domain merkezini ayarla."""
        self.center = torch.tensor([cx, cy], device=self.center.device).reshape(1, 1, 2)

    def forward(self, x: torch.Tensor, code: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Fiziksel koordinatları hesaplama koordinatlarına dönüştür.

        Args:
            x: [batch, N, 2] fiziksel koordinatlar
            code: [batch, code_dim] geometri kodu (opsiyonel)

        Returns:
            xi: [batch, N, 2] hesaplama koordinatları
        """
        # Feature engineering
        angle = torch.atan2(
            x[..., 1] - self.center[..., 1],
            x[..., 0] - self.center[..., 0]
        )
        radius = torch.norm(x - self.center, dim=-1, p=2)
        xd = torch.stack([x[..., 0], x[..., 1], angle, radius], dim=-1)

        # Positional encoding
        if self.use_pe:
            b, n, d = xd.shape[0], xd.shape[1], xd.shape[2]
            x_expanded = xd.view(b, n, d, 1)  # [b, n, 4, 1]
            x_sin = torch.sin(self.B_freq * x_expanded).view(b, n, -1)
            x_cos = torch.cos(self.B_freq * x_expanded).view(b, n, -1)
            xd_fc = self.fc0(xd)
            xd = torch.cat([xd_fc, x_sin, x_cos], dim=-1)
        else:
            xd = self.fc0(xd)

        # Code conditioning
        if self.code_dim > 0 and code is not None:
            cd = self.fc_code(code)
            cd = cd.unsqueeze(1).expand(-1, xd.shape[1], -1)
            xd = torch.cat([cd, xd], dim=-1)
        else:
            if hasattr(self, 'fc_no_code'):
                xd = self.fc_no_code(xd)

        # Hidden layers
        for layer in self.layers:
            xd = layer(xd)
            xd = F.gelu(xd)

        # Output deformasyon
        xd = self.fc_out(xd)

        # Deformasyonu sınırla (tanh ile)
        xd = torch.tanh(xd) * torch.sigmoid(self.max_deformation)

        # Multiplicative residual: xi = x * (1 + deformation)
        return x * (1.0 + xd)

    def jacobian(self, x: torch.Tensor, code: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Deformasyon Jacobian'ı hesapla: J_ij = ∂xi_i/∂x_j

        Args:
            x: [batch, N, 2]

        Returns:
            J: [batch, N, 2, 2] Jacobian matrisi
        """
        x_req = x.detach().requires_grad_(True)
        xi = self.forward(x_req, code)

        batch, N, _ = x.shape
        J = torch.zeros(batch, N, 2, 2, device=x.device)

        for i in range(2):
            grad_outputs = torch.zeros_like(xi)
            grad_outputs[..., i] = 1.0

            grads = torch.autograd.grad(
                xi, x_req, grad_outputs=grad_outputs,
                create_graph=True, retain_graph=True
            )[0]

            J[..., i, :] = grads

        return J

    def det_jacobian(self, x: torch.Tensor, code: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Jacobian determinantı (bijectivity kontrolü için).

        det(J) > 0 olmalı — aksi halde mesh kendi üzerine katlanır.
        """
        J = self.jacobian(x, code)
        return J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]

    def bijectivity_loss(self, x: torch.Tensor, code: Optional[torch.Tensor] = None,
                         epsilon: float = 0.01) -> torch.Tensor:
        """
        Bijectivity regularization loss.

        det(J) > epsilon olmasını zorlar.
        det(J) ≤ 0 → mesh çöker, NaN üretir.
        """
        det_J = self.det_jacobian(x, code)
        # ReLU penalty: det(J) < epsilon olursa cezalandır
        violation = F.relu(epsilon - det_J)
        return violation.mean()


class GeometryEncoder(nn.Module):
    """
    SDF/mesh'ten geometri code vektörü çıkarır.

    Basit MLP encoder — PointNet benzeri ama daha hafif.
    SDF değerlerini ve sınır bilgilerini encode eder.

    Args:
        sdf_resolution: SDF grid çözünürlüğü
        code_dim: Çıkış code boyutu
    """

    def __init__(self, sdf_resolution: int = 64, code_dim: int = 16):
        super().__init__()

        self.code_dim = code_dim

        # SDF → code: basit CNN
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),   # N→N/2
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # N/2→N/4
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # N/4→N/8
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),                     # → [B, 64, 1, 1]
            nn.Flatten(),
            nn.Linear(64, code_dim)
        )

    def forward(self, sdf: torch.Tensor) -> torch.Tensor:
        """
        SDF → geometry code.

        Args:
            sdf: [batch, N, N] veya [N, N]

        Returns:
            code: [batch, code_dim]
        """
        if sdf.dim() == 2:
            sdf = sdf.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
        elif sdf.dim() == 3:
            sdf = sdf.unsqueeze(1)  # [B, 1, N, N]

        return self.encoder(sdf)


# =============================================================================
# DEFORMED SPECTRAL OPERATORS
# =============================================================================

class DeformedSpectralOps2D(nn.Module):
    """
    Deformed domain için spektral türev operatörleri.

    İş akışı:
    1. Fiziksel koordinatları (x, y) hesaplama koordinatlarına (ξ, η) dönüştür
    2. FFT türevlerini hesaplama uzayında al (SpectralOps2DStruct)
    3. Jacobian ile fiziksel türevlere dönüştür

    Mevcut SpectralOps2DStruct'ı wrap eder — spine.py'ye dokunmaz.

    Args:
        resolution: Grid çözünürlüğü
        Lx, Ly: Hesaplama domain boyutları
        iphi: DeformationNet instance
    """

    def __init__(self, resolution: int, Lx: float = 1.0, Ly: float = 1.0,
                 iphi: Optional[DeformationNet] = None):
        super().__init__()

        self.resolution = resolution
        self.Lx = Lx
        self.Ly = Ly

        # Mevcut SPINE SpectralOps (DEĞİŞTİRİLMEZ)
        self.spectral_ops = SpectralOps2DStruct(resolution, Lx, Ly)

        # Deformasyon ağı
        self.iphi = iphi

        # Hesaplama domain'i grid'i: [0, Lx] × [0, Ly]
        x = torch.linspace(0, Lx, resolution)
        y = torch.linspace(0, Ly, resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        self.register_buffer('X_comp', X)
        self.register_buffer('Y_comp', Y)

        # Grid koordinatları [1, N*N, 2] formatında (IPHI için)
        grid_flat = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
        self.register_buffer('grid_coords', grid_flat.unsqueeze(0))

    def compute_jacobian_on_grid(self, x_phys: torch.Tensor,
                                  code: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Grid üzerinde Jacobian bileşenlerini hesapla ve cache'le.

        Args:
            x_phys: [1, N*N, 2] fiziksel grid koordinatları
            code: [1, code_dim] geometri kodu

        Returns:
            Dict: J_inv bileşenleri ve determinant
        """
        J = self.iphi.jacobian(x_phys, code)  # [1, N*N, 2, 2]

        N = self.resolution

        # Determinant
        det_J = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]  # [1, N*N]
        inv_det = 1.0 / (det_J + 1e-8)

        # Inverse Jacobian bileşenleri
        # J^{-1} = (1/det) * [[J_11, -J_01], [-J_10, J_00]]
        J_inv_00 = (J[..., 1, 1] * inv_det).reshape(N, N)   # ∂ξ/∂x
        J_inv_01 = (-J[..., 0, 1] * inv_det).reshape(N, N)  # ∂ξ/∂y
        J_inv_10 = (-J[..., 1, 0] * inv_det).reshape(N, N)  # ∂η/∂x
        J_inv_11 = (J[..., 0, 0] * inv_det).reshape(N, N)   # ∂η/∂y

        return {
            'J_inv_00': J_inv_00, 'J_inv_01': J_inv_01,
            'J_inv_10': J_inv_10, 'J_inv_11': J_inv_11,
            'det_J': det_J.reshape(N, N)
        }

    def gradient(self, f: torch.Tensor,
                 J_cache: Optional[Dict] = None,
                 x_phys: Optional[torch.Tensor] = None,
                 code: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fiziksel gradient: ∂f/∂x_phys, ∂f/∂y_phys

        Hesaplama uzayında FFT türevi alır, Jacobian ile düzeltir:
            ∂f/∂x = (∂ξ/∂x)(∂f/∂ξ) + (∂η/∂x)(∂f/∂η)
            ∂f/∂y = (∂ξ/∂y)(∂f/∂ξ) + (∂η/∂y)(∂f/∂η)

        Args:
            f: [N, N] veya [B, N, N] — hesaplama grid'inde alan değerleri
            J_cache: Önceden hesaplanmış Jacobian (opsiyonel, performans için)
            x_phys: [1, N*N, 2] fiziksel koordinatlar (J_cache yoksa)
            code: Geometri kodu (J_cache yoksa)
        """
        if self.iphi is None:
            # Deformasyon yok — standart türev
            return self.spectral_ops.gradient(f)

        # Hesaplama uzayında türevler (FFT ile — tam doğru)
        df_dxi, df_deta = self.spectral_ops.gradient(f)

        # Jacobian
        if J_cache is None:
            if x_phys is None:
                x_phys = self.grid_coords
            J_cache = self.compute_jacobian_on_grid(x_phys, code)

        # Fiziksel türevler (chain rule)
        df_dx = J_cache['J_inv_00'] * df_dxi + J_cache['J_inv_10'] * df_deta
        df_dy = J_cache['J_inv_01'] * df_dxi + J_cache['J_inv_11'] * df_deta

        return df_dx, df_dy

    def laplacian(self, f: torch.Tensor,
                  J_cache: Optional[Dict] = None,
                  x_phys: Optional[torch.Tensor] = None,
                  code: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Fiziksel Laplacian: ∇²f = ∂²f/∂x² + ∂²f/∂y²

        İki kez gradient alarak hesaplar (chain rule iki kez uygulanır).
        Bu, doğrudan 2. türev chain rule'undan daha stabil.
        """
        if self.iphi is None:
            return self.spectral_ops.laplacian(f)

        if J_cache is None:
            if x_phys is None:
                x_phys = self.grid_coords
            J_cache = self.compute_jacobian_on_grid(x_phys, code)

        # 1. türevler
        df_dx, df_dy = self.gradient(f, J_cache)

        # 2. türevler (gradient'in gradient'i)
        d2f_dx2, _ = self.gradient(df_dx, J_cache)
        _, d2f_dy2 = self.gradient(df_dy, J_cache)

        return d2f_dx2 + d2f_dy2

    def biharmonic(self, f: torch.Tensor,
                   J_cache: Optional[Dict] = None,
                   x_phys: Optional[torch.Tensor] = None,
                   code: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Fiziksel biharmonik: ∇⁴f = ∇²(∇²f)

        4. türev doğrudan mapping turevleriyle hesaplanamaz (çok kararsız).
        Bunun yerine ∇²(∇²f) decomposition kullanır.
        """
        if self.iphi is None:
            return self.spectral_ops.biharmonic(f)

        if J_cache is None:
            if x_phys is None:
                x_phys = self.grid_coords
            J_cache = self.compute_jacobian_on_grid(x_phys, code)

        # ∇⁴f = ∇²(∇²f)
        lap_f = self.laplacian(f, J_cache)
        return self.laplacian(lap_f, J_cache)

    def second_derivatives(self, f: torch.Tensor,
                           J_cache: Optional[Dict] = None,
                           x_phys: Optional[torch.Tensor] = None,
                           code: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fiziksel ikinci türevler: (∂²f/∂x², ∂²f/∂y², ∂²f/∂x∂y)
        """
        if self.iphi is None:
            return self.spectral_ops.second_derivatives(f)

        if J_cache is None:
            if x_phys is None:
                x_phys = self.grid_coords
            J_cache = self.compute_jacobian_on_grid(x_phys, code)

        df_dx, df_dy = self.gradient(f, J_cache)

        f_xx, f_xy_1 = self.gradient(df_dx, J_cache)
        f_xy_2, f_yy = self.gradient(df_dy, J_cache)

        # Simetri ortalaması
        f_xy = 0.5 * (f_xy_1 + f_xy_2)

        return f_xx, f_yy, f_xy


# =============================================================================
# INTERPOLATION LAYER (Mesh ↔ Grid)
# =============================================================================

class MeshToGridLayer(nn.Module):
    """
    Differentiable mesh → grid interpolasyon katmanı.

    grid_sample kullanır (autograd destekli, GPU uyumlu).

    Kullanım:
        layer = MeshToGridLayer(resolution=64, bounds=((0, 1), (0, 1)))
        grid_values = layer(node_values, node_coords)
    """

    def __init__(self, resolution: int, bounds: Tuple[Tuple[float, float], ...]):
        super().__init__()
        self.resolution = resolution
        self.bounds = bounds
        self.ndim = len(bounds)

    def forward(self, node_values: torch.Tensor, node_coords: torch.Tensor,
                grid_values_init: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Mesh node değerlerini regular grid'e scatter et.

        Bilinear splatting (scatter-based, differentiable).

        Args:
            node_values: [N_nodes] değerler
            node_coords: [N_nodes, 2] koordinatlar
            grid_values_init: [N, N] başlangıç grid (opsiyonel)
        """
        N = self.resolution
        device = node_values.device

        # Normalize → [0, N-1]
        grid_idx = torch.zeros_like(node_coords)
        for d in range(self.ndim):
            lo, hi = self.bounds[d]
            grid_idx[:, d] = (node_coords[:, d] - lo) / (hi - lo + 1e-10) * (N - 1)

        grid_idx = grid_idx.clamp(0, N - 2)
        floor_idx = grid_idx.long()
        frac = grid_idx - floor_idx.float()

        if self.ndim == 2:
            # Bilinear ağırlıklar
            w00 = (1 - frac[:, 0]) * (1 - frac[:, 1])
            w10 = frac[:, 0] * (1 - frac[:, 1])
            w01 = (1 - frac[:, 0]) * frac[:, 1]
            w11 = frac[:, 0] * frac[:, 1]

            # Flat indices
            idx00 = floor_idx[:, 0] * N + floor_idx[:, 1]
            idx10 = (floor_idx[:, 0] + 1) * N + floor_idx[:, 1]
            idx01 = floor_idx[:, 0] * N + (floor_idx[:, 1] + 1)
            idx11 = (floor_idx[:, 0] + 1) * N + (floor_idx[:, 1] + 1)

            grid_flat = torch.zeros(N * N, device=device)
            weight_sum = torch.zeros(N * N, device=device)

            for idx, w in [(idx00, w00), (idx10, w10), (idx01, w01), (idx11, w11)]:
                grid_flat.scatter_add_(0, idx, node_values * w)
                weight_sum.scatter_add_(0, idx, w)

            grid_flat = grid_flat / (weight_sum + 1e-10)
            return grid_flat.reshape(N, N)

        raise NotImplementedError("3D scatter henüz desteklenmiyor")


class GridToMeshLayer(nn.Module):
    """
    Differentiable grid → mesh interpolasyon katmanı.

    torch.grid_sample wrapper (autograd destekli).
    """

    def __init__(self, bounds: Tuple[Tuple[float, float], ...]):
        super().__init__()
        self.bounds = bounds
        self.ndim = len(bounds)

    def forward(self, grid_values: torch.Tensor, node_coords: torch.Tensor) -> torch.Tensor:
        """
        Grid değerlerini mesh node'larına interpolate et.

        Args:
            grid_values: [N, N] veya [B, N, N]
            node_coords: [N_nodes, 2]

        Returns:
            [N_nodes] interpolated values
        """
        if grid_values.dim() == 2:
            grid_input = grid_values.unsqueeze(0).unsqueeze(0)
        elif grid_values.dim() == 3:
            grid_input = grid_values.unsqueeze(1)
        else:
            grid_input = grid_values

        # Normalize [-1, 1]
        coords_norm = torch.zeros_like(node_coords[:, :self.ndim])
        for d in range(self.ndim):
            lo, hi = self.bounds[d]
            coords_norm[:, d] = 2.0 * (node_coords[:, d] - lo) / (hi - lo + 1e-10) - 1.0

        # grid_sample expects (y, x) order in 2D
        if self.ndim == 2:
            sample_grid = coords_norm[:, [1, 0]].unsqueeze(0).unsqueeze(0)
            result = F.grid_sample(
                grid_input, sample_grid,
                mode='bilinear', padding_mode='border', align_corners=True
            )
            return result.squeeze()

        raise NotImplementedError("3D grid_sample henüz desteklenmiyor")


# =============================================================================
# COMPLEX GEOMETRY SOLVER (Ana wrapper)
# =============================================================================

class ComplexGeometrySolver(nn.Module):
    """
    SPINE + Karmaşık Geometri wrapper.

    Pipeline:
    1. SDF → GeometryEncoder → code vektörü
    2. Fiziksel grid → IPHI → hesaplama grid
    3. Hesaplama grid'inde FFT türevleri
    4. Jacobian düzeltmesiyle fiziksel türevler
    5. Mevcut SPINE nöronları (BiharmonicNeuron, StressNeuron, vb.)

    Kullanım:
        from spine import MaterialProperties
        from spine_geometry import GeometryProcessor

        geo = GeometryProcessor()
        mesh = geo.create_plate_with_hole()
        grid = geo.compute_sdf_2d(mesh, resolution=64)

        solver = ComplexGeometrySolver(
            resolution=64, Lx=grid.Lx, Ly=grid.Ly,
            code_dim=16, deformation_width=32
        )

        # Eğitim
        loss = solver.physics_loss(w_field, grid.sdf)
    """

    def __init__(self, resolution: int, Lx: float = 1.0, Ly: float = 1.0,
                 code_dim: int = 16, deformation_width: int = 32,
                 use_deformation: bool = True):
        super().__init__()

        self.resolution = resolution
        self.Lx = Lx
        self.Ly = Ly
        self.use_deformation = use_deformation

        # Geometry encoder: SDF → code
        self.geometry_encoder = GeometryEncoder(
            sdf_resolution=resolution, code_dim=code_dim
        ) if code_dim > 0 else None

        # Deformation network
        if use_deformation:
            self.iphi = DeformationNet(
                width=deformation_width,
                code_dim=code_dim,
                n_layers=3,
                use_positional_encoding=True
            )
        else:
            self.iphi = None

        # Deformed spectral operators
        self.deformed_ops = DeformedSpectralOps2D(
            resolution=resolution, Lx=Lx, Ly=Ly, iphi=self.iphi
        )

        # Standart (undeformed) spectral operators da tut (karşılaştırma için)
        self.spectral_ops = SpectralOps2DStruct(resolution, Lx, Ly)

        # Fiziksel grid
        x = torch.linspace(0, Lx, resolution)
        y = torch.linspace(0, Ly, resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        grid_flat = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
        self.register_buffer('grid_coords', grid_flat.unsqueeze(0))  # [1, N*N, 2]

        # Interpolasyon katmanları
        self.mesh_to_grid = MeshToGridLayer(resolution, ((0, Lx), (0, Ly)))
        self.grid_to_mesh = GridToMeshLayer(((0, Lx), (0, Ly)))

    def encode_geometry(self, sdf: torch.Tensor) -> Optional[torch.Tensor]:
        """SDF → geometry code."""
        if self.geometry_encoder is not None:
            return self.geometry_encoder(sdf)
        return None

    def physical_gradient(self, f: torch.Tensor, sdf: Optional[torch.Tensor] = None,
                          code: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Karmaşık geometride fiziksel gradient."""
        if code is None and sdf is not None:
            code = self.encode_geometry(sdf)

        if self.use_deformation:
            J_cache = self.deformed_ops.compute_jacobian_on_grid(self.grid_coords, code)
            return self.deformed_ops.gradient(f, J_cache)
        else:
            return self.spectral_ops.gradient(f)

    def physical_laplacian(self, f: torch.Tensor, sdf: Optional[torch.Tensor] = None,
                           code: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Karmaşık geometride fiziksel Laplacian."""
        if code is None and sdf is not None:
            code = self.encode_geometry(sdf)

        if self.use_deformation:
            J_cache = self.deformed_ops.compute_jacobian_on_grid(self.grid_coords, code)
            return self.deformed_ops.laplacian(f, J_cache)
        else:
            return self.spectral_ops.laplacian(f)

    def physical_biharmonic(self, f: torch.Tensor, sdf: Optional[torch.Tensor] = None,
                            code: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Karmaşık geometride fiziksel biharmonik ∇⁴f."""
        if code is None and sdf is not None:
            code = self.encode_geometry(sdf)

        if self.use_deformation:
            J_cache = self.deformed_ops.compute_jacobian_on_grid(self.grid_coords, code)
            return self.deformed_ops.biharmonic(f, J_cache)
        else:
            return self.spectral_ops.biharmonic(f)

    def deformation_loss(self, sdf: Optional[torch.Tensor] = None,
                         code: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Deformation regularization loss.

        - Bijectivity: det(J) > 0
        - Smoothness: Jacobian değişimi küçük olmalı
        """
        if not self.use_deformation or self.iphi is None:
            return torch.tensor(0.0, device=self.grid_coords.device)

        if code is None and sdf is not None:
            code = self.encode_geometry(sdf)

        # Bijectivity loss
        loss_bij = self.iphi.bijectivity_loss(self.grid_coords, code)

        # Smoothness loss: Jacobian'ın spatial gradient'i küçük olsun
        det_J = self.iphi.det_jacobian(self.grid_coords, code)
        det_J_grid = det_J.reshape(self.resolution, self.resolution)
        det_J_dx = det_J_grid[1:, :] - det_J_grid[:-1, :]
        det_J_dy = det_J_grid[:, 1:] - det_J_grid[:, :-1]
        loss_smooth = (det_J_dx**2).mean() + (det_J_dy**2).mean()

        return loss_bij + 0.1 * loss_smooth

    def forward(self, f: torch.Tensor, sdf: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Tam fiziksel analiz.

        Args:
            f: [N, N] alan değeri (deplasman, sıcaklık, vb.)
            sdf: [N, N] signed distance function

        Returns:
            Dict: gradient, laplacian, biharmonic, deformation_loss
        """
        code = self.encode_geometry(sdf) if sdf is not None else None

        J_cache = None
        if self.use_deformation and self.iphi is not None:
            J_cache = self.deformed_ops.compute_jacobian_on_grid(self.grid_coords, code)

        df_dx, df_dy = self.deformed_ops.gradient(f, J_cache) if J_cache else self.spectral_ops.gradient(f)
        lap_f = self.deformed_ops.laplacian(f, J_cache) if J_cache else self.spectral_ops.laplacian(f)
        biharm_f = self.deformed_ops.biharmonic(f, J_cache) if J_cache else self.spectral_ops.biharmonic(f)

        result = {
            'df_dx': df_dx,
            'df_dy': df_dy,
            'laplacian': lap_f,
            'biharmonic': biharm_f,
        }

        if self.use_deformation:
            result['deformation_loss'] = self.deformation_loss(sdf, code)
            result['det_J'] = J_cache['det_J'] if J_cache else None

        return result

    def extra_repr(self) -> str:
        return (f"resolution={self.resolution}, Lx={self.Lx:.2f}, Ly={self.Ly:.2f}, "
                f"use_deformation={self.use_deformation}")
