"""
SPINE Geometry Module: Karmaşık Geometri Desteği

STEP/IGES/STL dosyalarını okur, mesh oluşturur, SDF hesaplar,
mesh↔grid interpolasyonu yapar.

Pipeline:
    STEP dosyası → gmsh mesh → SDF → regular grid → SPINE FFT

Bağımlılıklar:
    pip install gmsh meshio

Kullanım:
    from spine_geometry import GeometryProcessor, MeshData

    # STEP dosyasından yükle
    geo = GeometryProcessor()
    mesh = geo.load_step("part.step", mesh_size=0.01)

    # SDF hesapla
    sdf = geo.compute_sdf(mesh, grid_resolution=64)

    # NACA airfoil oluştur (parametrik)
    mesh = geo.create_naca_airfoil("2412", chord=0.1, resolution=64)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import warnings
import os
import tempfile
from typing import Optional, Tuple, Dict, List, Union
from dataclasses import dataclass, field

try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False
    warnings.warn("gmsh bulunamadı. pip install gmsh")

try:
    import meshio
    MESHIO_AVAILABLE = True
except ImportError:
    MESHIO_AVAILABLE = False
    warnings.warn("meshio bulunamadı. pip install meshio")

from scipy.spatial import KDTree, Delaunay
from scipy.interpolate import griddata
from matplotlib.path import Path


# =============================================================================
# VERİ YAPILARI
# =============================================================================

@dataclass
class MeshData:
    """Mesh verisi container."""
    points: np.ndarray            # [N_nodes, ndim] node koordinatları
    cells: Optional[np.ndarray]   # [N_elements, nodes_per_elem] connectivity
    cell_type: str = "triangle"   # triangle, tetra, quad, hex
    ndim: int = 2                 # 2D veya 3D

    # Boundary bilgileri
    boundary_nodes: Optional[Dict[str, np.ndarray]] = None  # {name: node_indices}
    boundary_types: Optional[Dict[str, str]] = None          # {name: bc_type}

    # Bounding box
    bbox_min: Optional[np.ndarray] = None
    bbox_max: Optional[np.ndarray] = None

    # Orijinal CAD bilgisi
    source_file: Optional[str] = None

    def __post_init__(self):
        if self.bbox_min is None:
            self.bbox_min = self.points.min(axis=0)
        if self.bbox_max is None:
            self.bbox_max = self.points.max(axis=0)
        self.ndim = self.points.shape[1]

    @property
    def n_nodes(self) -> int:
        return len(self.points)

    @property
    def n_elements(self) -> int:
        return len(self.cells) if self.cells is not None else 0

    @property
    def extent(self) -> np.ndarray:
        return self.bbox_max - self.bbox_min

    def boundary_polygon(self) -> np.ndarray:
        """2D mesh'in dış sınır poligonunu çıkar."""
        if self.ndim != 2 or self.cells is None:
            raise ValueError("boundary_polygon sadece 2D triangle mesh için")
        return _extract_boundary_polygon(self.points, self.cells)


@dataclass
class GridData:
    """Regular grid verisi."""
    sdf: torch.Tensor                 # [N, N] veya [N, N, N] signed distance
    mask: torch.Tensor                # [N, N] veya [N, N, N] domain mask (smooth)
    Lx: float                         # x domain uzunluğu
    Ly: float                         # y domain uzunluğu
    Lz: float = 0.0                   # z domain uzunluğu (3D)
    origin: Optional[np.ndarray] = None  # grid orijini
    resolution: int = 64
    ndim: int = 2

    # Boundary bilgileri (grid üzerinde)
    boundary_mask: Optional[torch.Tensor] = None      # sınıra yakın bölge
    normal_x: Optional[torch.Tensor] = None            # sınır normali x
    normal_y: Optional[torch.Tensor] = None            # sınır normali y


# =============================================================================
# ANA İŞLEMCİ
# =============================================================================

class GeometryProcessor:
    """
    CAD dosyası → SPINE-ready grid pipeline.

    Kullanım:
        geo = GeometryProcessor()
        mesh = geo.load_step("turbine_blade.step")
        grid = geo.mesh_to_grid(mesh, resolution=64)
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    # -----------------------------------------------------------------
    # STEP / IGES / STL Okuma
    # -----------------------------------------------------------------

    def load_step(self, filepath: str, mesh_size: float = 0.01,
                  dim: int = 2, element_order: int = 1) -> MeshData:
        """
        STEP dosyasını oku ve mesh oluştur.

        Args:
            filepath: .step/.stp/.iges/.igs dosya yolu
            mesh_size: Hedef eleman boyutu (m)
            dim: Mesh boyutu (2=yüzey, 3=hacim)
            element_order: Eleman derecesi (1=lineer, 2=kuadratik)

        Returns:
            MeshData: Mesh verisi
        """
        if not GMSH_AVAILABLE:
            raise ImportError("gmsh gerekli: pip install gmsh")

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0)

        try:
            # CAD dosyasını yükle
            gmsh.model.occ.importShapes(filepath)
            gmsh.model.occ.synchronize()

            # Mesh parametreleri
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.1)

            if element_order > 1:
                gmsh.option.setNumber("Mesh.ElementOrder", element_order)

            # Mesh oluştur
            gmsh.model.mesh.generate(dim)

            # Geçici dosyaya kaydet
            tmp_msh = tempfile.mktemp(suffix='.msh')
            gmsh.write(tmp_msh)

            # Boundary bilgilerini çıkar
            boundaries = self._extract_gmsh_boundaries()

        finally:
            gmsh.finalize()

        # meshio ile oku
        msh = meshio.read(tmp_msh)
        os.unlink(tmp_msh)

        # Cells ve points çıkar
        points = msh.points[:, :dim] if dim < 3 else msh.points
        cells, cell_type = self._extract_cells(msh, dim)

        return MeshData(
            points=points,
            cells=cells,
            cell_type=cell_type,
            ndim=dim,
            boundary_nodes=boundaries.get('nodes'),
            boundary_types=boundaries.get('types'),
            source_file=filepath
        )

    def load_stl(self, filepath: str) -> MeshData:
        """STL dosyasını oku (sadece yüzey mesh)."""
        if not MESHIO_AVAILABLE:
            raise ImportError("meshio gerekli: pip install meshio")

        msh = meshio.read(filepath)
        points = msh.points
        cells = None
        for cb in msh.cells:
            if cb.type == "triangle":
                cells = cb.data
                break

        return MeshData(
            points=points,
            cells=cells,
            cell_type="triangle",
            ndim=3,
            source_file=filepath
        )

    # -----------------------------------------------------------------
    # Parametrik Geometri Oluşturma
    # -----------------------------------------------------------------

    def create_naca_airfoil(self, naca: str = "2412", chord: float = 1.0,
                            n_points: int = 100, thickness_ratio: float = 1.0) -> MeshData:
        """
        NACA 4-digit airfoil profili oluştur.

        Args:
            naca: 4 haneli NACA kodu (ör: "2412")
            chord: Veter uzunluğu (m)
            n_points: Profil nokta sayısı
            thickness_ratio: Kalınlık çarpanı

        Returns:
            MeshData: Airfoil sınır noktaları
        """
        m = int(naca[0]) / 100.0      # max camber
        p = int(naca[1]) / 10.0       # max camber konumu
        t = int(naca[2:]) / 100.0     # kalınlık oranı
        t *= thickness_ratio

        # x koordinatları (cosine spacing — leading edge'de sık)
        beta = np.linspace(0, np.pi, n_points)
        x = 0.5 * (1 - np.cos(beta)) * chord

        # Kalınlık dağılımı
        yt = 5 * t * chord * (
            0.2969 * np.sqrt(x / chord)
            - 0.1260 * (x / chord)
            - 0.3516 * (x / chord)**2
            + 0.2843 * (x / chord)**3
            - 0.1015 * (x / chord)**4
        )

        # Camber çizgisi
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)

        if m > 0 and p > 0:
            front = x <= p * chord
            back = ~front

            yc[front] = (m / p**2) * (2*p*(x[front]/chord) - (x[front]/chord)**2) * chord
            yc[back] = (m / (1-p)**2) * ((1-2*p) + 2*p*(x[back]/chord) - (x[back]/chord)**2) * chord

            dyc_dx[front] = (2*m / p**2) * (p - x[front]/chord)
            dyc_dx[back] = (2*m / (1-p)**2) * (p - x[back]/chord)

        theta = np.arctan(dyc_dx)

        # Üst ve alt yüzey
        x_upper = x - yt * np.sin(theta)
        y_upper = yc + yt * np.cos(theta)
        x_lower = x + yt * np.sin(theta)
        y_lower = yc - yt * np.cos(theta)

        # Kapalı profil (trailing edge'den başla, üst, leading edge, alt, geri)
        x_profile = np.concatenate([x_upper, x_lower[::-1][1:]])
        y_profile = np.concatenate([y_upper, y_lower[::-1][1:]])

        points = np.column_stack([x_profile, y_profile])

        return MeshData(
            points=points,
            cells=None,
            cell_type="polygon",
            ndim=2,
            source_file=f"NACA_{naca}"
        )

    def create_plate_with_hole(self, Lx: float = 1.0, Ly: float = 1.0,
                               hole_center: Tuple[float, float] = (0.5, 0.5),
                               hole_radius: float = 0.15,
                               mesh_size: float = 0.02) -> MeshData:
        """
        Delikli plaka geometrisi oluştur (Kirsch problemi).

        Args:
            Lx, Ly: Plaka boyutları
            hole_center: Delik merkezi
            hole_radius: Delik yarıçapı
            mesh_size: Mesh boyutu
        """
        if not GMSH_AVAILABLE:
            raise ImportError("gmsh gerekli: pip install gmsh")

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0)

        try:
            # Dikdörtgen plaka
            rect = gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly)
            # Dairesel delik
            disk = gmsh.model.occ.addDisk(hole_center[0], hole_center[1], 0,
                                           hole_radius, hole_radius)
            # Boolean fark (plaka - delik)
            gmsh.model.occ.cut([(2, rect)], [(2, disk)])
            gmsh.model.occ.synchronize()

            # Physical groups (BC tanımı)
            # Sol kenar: fixed
            left_edges = []
            right_edges = []
            for dim, tag in gmsh.model.getEntities(1):
                com = gmsh.model.occ.getCenterOfMass(dim, tag)
                if abs(com[0]) < 1e-6:
                    left_edges.append(tag)
                elif abs(com[0] - Lx) < 1e-6:
                    right_edges.append(tag)

            if left_edges:
                gmsh.model.addPhysicalGroup(1, left_edges, name="fixed_support")
            if right_edges:
                gmsh.model.addPhysicalGroup(1, right_edges, name="load_surface")

            # Tüm yüzeyi physical group yap
            surfaces = [tag for dim, tag in gmsh.model.getEntities(2)]
            gmsh.model.addPhysicalGroup(2, surfaces, name="domain")

            # Mesh
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
            # Delik etrafında ince mesh
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.3)
            gmsh.model.mesh.generate(2)

            tmp_msh = tempfile.mktemp(suffix='.msh')
            gmsh.write(tmp_msh)
            boundaries = self._extract_gmsh_boundaries()
        finally:
            gmsh.finalize()

        msh = meshio.read(tmp_msh)
        os.unlink(tmp_msh)

        points = msh.points[:, :2]
        cells, cell_type = self._extract_cells(msh, 2)

        return MeshData(
            points=points,
            cells=cells,
            cell_type=cell_type,
            ndim=2,
            boundary_nodes=boundaries.get('nodes'),
            boundary_types=boundaries.get('types'),
            source_file=f"plate_with_hole_R{hole_radius}"
        )

    def create_l_bracket(self, arm1_length: float = 0.1, arm1_width: float = 0.05,
                         arm2_length: float = 0.1, arm2_width: float = 0.05,
                         fillet_radius: float = 0.005,
                         mesh_size: float = 0.005) -> MeshData:
        """L-bracket geometrisi oluştur."""
        if not GMSH_AVAILABLE:
            raise ImportError("gmsh gerekli")

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0)

        try:
            # L-şeklini iki dikdörtgen birleşimi olarak oluştur
            rect1 = gmsh.model.occ.addRectangle(0, 0, 0, arm1_length, arm1_width)
            rect2 = gmsh.model.occ.addRectangle(0, 0, 0, arm2_width, arm1_width + arm2_length)

            # Birleştir (fuse)
            gmsh.model.occ.fuse([(2, rect1)], [(2, rect2)])

            if fillet_radius > 0:
                gmsh.model.occ.synchronize()
                # İç köşeye fillet
                curves = gmsh.model.getEntities(1)
                for dim, tag in curves:
                    com = gmsh.model.occ.getCenterOfMass(dim, tag)
                    # İç köşeye yakın kenarları bul
                    if (abs(com[0] - arm2_width) < arm2_width * 0.6 and
                        abs(com[1] - arm1_width) < arm1_width * 0.6):
                        try:
                            gmsh.model.occ.fillet([(2, 1)], [tag], [fillet_radius])
                        except Exception:
                            pass

            gmsh.model.occ.synchronize()

            surfaces = [tag for dim, tag in gmsh.model.getEntities(2)]
            gmsh.model.addPhysicalGroup(2, surfaces, name="domain")

            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
            gmsh.model.mesh.generate(2)

            tmp_msh = tempfile.mktemp(suffix='.msh')
            gmsh.write(tmp_msh)
            boundaries = self._extract_gmsh_boundaries()
        finally:
            gmsh.finalize()

        msh = meshio.read(tmp_msh)
        os.unlink(tmp_msh)

        points = msh.points[:, :2]
        cells, cell_type = self._extract_cells(msh, 2)

        return MeshData(
            points=points, cells=cells, cell_type=cell_type, ndim=2,
            boundary_nodes=boundaries.get('nodes'),
            boundary_types=boundaries.get('types'),
            source_file="l_bracket"
        )

    # -----------------------------------------------------------------
    # SDF Hesaplama
    # -----------------------------------------------------------------

    def compute_sdf_2d(self, mesh: MeshData, grid_resolution: int,
                       padding: float = 0.1) -> GridData:
        """
        2D mesh'ten regular grid üzerinde SDF hesapla.

        Args:
            mesh: MeshData (2D)
            grid_resolution: Grid çözünürlüğü
            padding: Bounding box padding oranı

        Returns:
            GridData: SDF, mask, ve grid bilgileri
        """
        # Boundary polygon çıkar
        if mesh.cell_type == "polygon":
            boundary = mesh.points
        else:
            boundary = mesh.boundary_polygon()

        # Bounding box + padding
        bbox_min = boundary.min(axis=0)
        bbox_max = boundary.max(axis=0)
        extent = bbox_max - bbox_min
        origin = bbox_min - padding * extent
        Lx = extent[0] * (1 + 2 * padding)
        Ly = extent[1] * (1 + 2 * padding)

        N = grid_resolution

        # Vektörize SDF hesaplama
        x = np.linspace(origin[0], origin[0] + Lx, N)
        y = np.linspace(origin[1], origin[1] + Ly, N)
        X, Y = np.meshgrid(x, y, indexing='ij')
        points = np.column_stack([X.ravel(), Y.ravel()])

        # Her noktanın sınıra mesafesi
        n_verts = len(boundary)
        min_dist = np.full(len(points), np.inf)

        for i in range(n_verts):
            a = boundary[i]
            b = boundary[(i + 1) % n_verts]
            ab = b - a
            ap = points - a
            t = np.clip(np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12), 0, 1)
            proj = a + np.outer(t, ab)
            dist = np.linalg.norm(points - proj, axis=1)
            min_dist = np.minimum(min_dist, dist)

        # İç/dış tespiti (matplotlib Path — winding number tabanlı)
        path = Path(boundary)
        inside = path.contains_points(points)

        sdf_flat = np.where(inside, -min_dist, min_dist)
        sdf = torch.tensor(sdf_flat.reshape(N, N), dtype=torch.float32, device=self.device)

        # Smooth mask (differentiable)
        epsilon = 2.0 * max(Lx, Ly) / N
        mask = torch.sigmoid(-sdf / epsilon)

        # Sınır normalleri (SDF gradient'inden)
        # Basit finite difference
        dx = Lx / (N - 1)
        dy = Ly / (N - 1)
        sdf_dx = torch.zeros_like(sdf)
        sdf_dy = torch.zeros_like(sdf)
        sdf_dx[1:-1, :] = (sdf[2:, :] - sdf[:-2, :]) / (2 * dx)
        sdf_dy[:, 1:-1] = (sdf[:, 2:] - sdf[:, :-2]) / (2 * dy)
        grad_mag = torch.sqrt(sdf_dx**2 + sdf_dy**2 + 1e-10)

        # Boundary mask (SDF ≈ 0 bölgesi)
        boundary_width = 2 * max(dx, dy)
        boundary_mask = torch.exp(-sdf**2 / (2 * boundary_width**2))

        return GridData(
            sdf=sdf,
            mask=mask,
            Lx=Lx,
            Ly=Ly,
            origin=origin,
            resolution=N,
            ndim=2,
            boundary_mask=boundary_mask,
            normal_x=sdf_dx / grad_mag,
            normal_y=sdf_dy / grad_mag
        )

    def compute_sdf_3d(self, mesh: MeshData, grid_resolution: int,
                       padding: float = 0.1) -> GridData:
        """
        3D mesh'ten regular grid üzerinde SDF hesapla.

        trimesh kullanır (KD-tree + winding number).
        """
        try:
            import trimesh
        except ImportError:
            raise ImportError("3D SDF için trimesh gerekli: pip install trimesh")

        # Yüzey mesh'i çıkar
        if mesh.cell_type in ("tetra", "tetra10"):
            surface = _extract_surface_from_tets(mesh.points, mesh.cells)
            tri_mesh = trimesh.Trimesh(vertices=mesh.points, faces=surface)
        elif mesh.cell_type == "triangle":
            tri_mesh = trimesh.Trimesh(vertices=mesh.points, faces=mesh.cells)
        else:
            raise ValueError(f"Desteklenmeyen cell type: {mesh.cell_type}")

        # Bounding box
        bbox_min = tri_mesh.bounds[0]
        bbox_max = tri_mesh.bounds[1]
        extent = bbox_max - bbox_min
        origin = bbox_min - padding * extent
        Lx = extent[0] * (1 + 2 * padding)
        Ly = extent[1] * (1 + 2 * padding)
        Lz = extent[2] * (1 + 2 * padding)

        N = grid_resolution
        x = np.linspace(origin[0], origin[0] + Lx, N)
        y = np.linspace(origin[1], origin[1] + Ly, N)
        z = np.linspace(origin[2], origin[2] + Lz, N)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        query_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        # SDF (trimesh: negatif = içerde)
        sdf_values = tri_mesh.nearest.signed_distance(query_points)
        sdf = torch.tensor(sdf_values.reshape(N, N, N),
                          dtype=torch.float32, device=self.device)

        epsilon = 2.0 * max(Lx, Ly, Lz) / N
        mask = torch.sigmoid(-sdf / epsilon)

        return GridData(
            sdf=sdf, mask=mask,
            Lx=Lx, Ly=Ly, Lz=Lz,
            origin=origin, resolution=N, ndim=3
        )

    # -----------------------------------------------------------------
    # Mesh ↔ Grid İnterpolasyon
    # -----------------------------------------------------------------

    def mesh_to_grid(self, node_coords: np.ndarray, node_values: np.ndarray,
                     grid: GridData, method: str = 'linear') -> torch.Tensor:
        """
        Unstructured mesh değerlerini regular grid'e interpolate et.

        Args:
            node_coords: [N_nodes, 2/3] mesh koordinatları
            node_values: [N_nodes] veya [N_nodes, C] değerler
            grid: GridData (hedef grid bilgileri)
            method: 'linear' veya 'nearest'
        """
        N = grid.resolution

        if grid.ndim == 2:
            x = np.linspace(grid.origin[0], grid.origin[0] + grid.Lx, N)
            y = np.linspace(grid.origin[1], grid.origin[1] + grid.Ly, N)
            X, Y = np.meshgrid(x, y, indexing='ij')

            result = griddata(node_coords[:, :2], node_values, (X, Y),
                            method=method, fill_value=0.0)
        else:
            x = np.linspace(grid.origin[0], grid.origin[0] + grid.Lx, N)
            y = np.linspace(grid.origin[1], grid.origin[1] + grid.Ly, N)
            z = np.linspace(grid.origin[2], grid.origin[2] + grid.Lz, N)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

            result = griddata(node_coords, node_values, pts,
                            method=method, fill_value=0.0)
            shape = [N, N, N]
            if node_values.ndim > 1:
                shape.append(node_values.shape[1])
            result = result.reshape(shape)

        return torch.tensor(result, dtype=torch.float32, device=self.device)

    def grid_to_mesh(self, grid_values: torch.Tensor, node_coords: np.ndarray,
                     grid: GridData) -> torch.Tensor:
        """
        Regular grid değerlerini mesh node'larına interpolate et.

        torch.grid_sample kullanır (autograd uyumlu).
        """
        N = grid.resolution
        ndim = grid.ndim

        # Koordinatları [-1, 1] aralığına normalize et
        coords_norm = np.zeros_like(node_coords[:, :ndim], dtype=np.float32)

        bounds = [(grid.origin[0], grid.origin[0] + grid.Lx),
                  (grid.origin[1], grid.origin[1] + grid.Ly)]
        if ndim == 3:
            bounds.append((grid.origin[2], grid.origin[2] + grid.Lz))

        for d in range(ndim):
            lo, hi = bounds[d]
            coords_norm[:, d] = 2.0 * (node_coords[:, d] - lo) / (hi - lo + 1e-10) - 1.0

        if ndim == 2:
            grid_input = grid_values.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
            # grid_sample expects (y, x) order
            sample_coords = torch.tensor(
                coords_norm[:, ::-1].copy(), dtype=torch.float32, device=self.device
            ).unsqueeze(0).unsqueeze(0)  # [1, 1, N_nodes, 2]

            result = F.grid_sample(
                grid_input, sample_coords,
                mode='bilinear', padding_mode='border', align_corners=True
            )
            return result.squeeze()

        elif ndim == 3:
            grid_input = grid_values.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
            sample_coords = torch.tensor(
                coords_norm[:, ::-1].copy(), dtype=torch.float32, device=self.device
            ).reshape(1, 1, 1, -1, 3)

            result = F.grid_sample(
                grid_input, sample_coords,
                mode='bilinear', padding_mode='border', align_corners=True
            )
            return result.squeeze()

    # -----------------------------------------------------------------
    # Precomputed Interpolator (Eğitim loop'u için)
    # -----------------------------------------------------------------

    def create_interpolator(self, node_coords: np.ndarray, grid: GridData) -> 'MeshGridInterpolator':
        """
        Bir kez Delaunay hesapla, sonra hızlı interpolasyon yap.
        Training loop'unda her epoch'ta mesh_to_grid yerine bunu kullan.
        """
        return MeshGridInterpolator(node_coords, grid, device=self.device)

    # -----------------------------------------------------------------
    # Yardımcı Metodlar
    # -----------------------------------------------------------------

    def _extract_cells(self, msh: meshio.Mesh, dim: int) -> Tuple[np.ndarray, str]:
        """meshio mesh'inden hücreleri çıkar."""
        priority_2d = ["triangle", "triangle6", "quad", "quad8"]
        priority_3d = ["tetra", "tetra10", "hexahedron", "hexahedron20"]
        priority = priority_3d if dim == 3 else priority_2d

        for cell_type in priority:
            for cb in msh.cells:
                if cb.type == cell_type:
                    return cb.data, cell_type

        # Bulunamazsa ilk cell block'u al
        if msh.cells:
            cb = msh.cells[0]
            return cb.data, cb.type

        return np.array([]), "unknown"

    def _extract_gmsh_boundaries(self) -> Dict:
        """gmsh'ten boundary bilgilerini çıkar."""
        boundaries = {'nodes': {}, 'types': {}}

        try:
            groups = gmsh.model.getPhysicalGroups()
            for dim, tag in groups:
                if dim >= 2:
                    continue  # Sadece çizgi/nokta boundary'ler

                name = gmsh.model.getPhysicalName(dim, tag)
                if not name:
                    name = f"boundary_{tag}"

                entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
                all_nodes = set()

                for entity in entities:
                    node_tags, _, _ = gmsh.model.mesh.getNodes(dim, entity)
                    all_nodes.update(node_tags)

                if all_nodes:
                    # 0-indexed
                    boundaries['nodes'][name] = np.array(sorted(all_nodes), dtype=np.int64) - 1
                    boundaries['types'][name] = _infer_bc_type(name)
        except Exception:
            pass

        return boundaries


class MeshGridInterpolator:
    """
    Önbellek kullanan interpolator.

    Delaunay triangülasyonu bir kez hesaplanır, sonra her interpolasyon
    sadece barycentric ağırlıklarla matris çarpımı olur.

    Kullanım:
        interp = MeshGridInterpolator(node_coords, grid)
        grid_values = interp.forward(node_values)  # Her epoch'ta hızlı
    """

    def __init__(self, node_coords: np.ndarray, grid: GridData, device: str = "cpu"):
        self.device = device
        self.grid = grid
        N = grid.resolution
        ndim = grid.ndim

        # Grid noktaları
        if ndim == 2:
            x = np.linspace(grid.origin[0], grid.origin[0] + grid.Lx, N)
            y = np.linspace(grid.origin[1], grid.origin[1] + grid.Ly, N)
            X, Y = np.meshgrid(x, y, indexing='ij')
            self.grid_points = np.column_stack([X.ravel(), Y.ravel()])
            self.grid_shape = (N, N)
        else:
            x = np.linspace(grid.origin[0], grid.origin[0] + grid.Lx, N)
            y = np.linspace(grid.origin[1], grid.origin[1] + grid.Ly, N)
            z = np.linspace(grid.origin[2], grid.origin[2] + grid.Lz, N)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            self.grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
            self.grid_shape = (N, N, N)

        # Delaunay bir kez
        coords = node_coords[:, :ndim]
        self.tri = Delaunay(coords)

        # Her grid noktasının simplex'i
        self.simplex_indices = self.tri.find_simplex(self.grid_points)
        self.inside_mask = self.simplex_indices >= 0

        # Barycentric ağırlıklar (önceden hesapla)
        self._precompute_barycentric()

        # Dışarda kalan noktalar için nearest neighbor
        self.outside_mask = ~self.inside_mask
        if self.outside_mask.any():
            kdtree = KDTree(coords)
            _, self.nearest_indices = kdtree.query(self.grid_points[self.outside_mask])
        else:
            self.nearest_indices = np.array([], dtype=np.int64)

    def _precompute_barycentric(self):
        inside_points = self.grid_points[self.inside_mask]
        inside_simplices = self.simplex_indices[self.inside_mask]

        transforms = self.tri.transform[inside_simplices]
        delta = inside_points - transforms[:, -1, :]
        bary = np.einsum('ijk,ik->ij', transforms[:, :-1, :], delta)
        bary_last = 1.0 - bary.sum(axis=1, keepdims=True)

        self.bary_coords = np.hstack([bary, bary_last])
        self.vertex_indices = self.tri.simplices[inside_simplices]

    def forward(self, values: np.ndarray) -> torch.Tensor:
        """
        Hızlı interpolasyon (önceden hesaplanmış ağırlıklarla).

        Args:
            values: [N_nodes] veya [N_nodes, C]

        Returns:
            torch.Tensor: grid_shape şeklinde
        """
        n_grid = len(self.grid_points)

        if values.ndim == 1:
            result = np.zeros(n_grid)
            vertex_vals = values[self.vertex_indices]
            result[self.inside_mask] = (self.bary_coords * vertex_vals).sum(axis=1)
            if self.outside_mask.any():
                result[self.outside_mask] = values[self.nearest_indices]
            result = result.reshape(self.grid_shape)
        else:
            n_comp = values.shape[1]
            result = np.zeros((n_grid, n_comp))
            vertex_vals = values[self.vertex_indices]
            result[self.inside_mask] = np.einsum('ij,ijk->ik', self.bary_coords, vertex_vals)
            if self.outside_mask.any():
                result[self.outside_mask] = values[self.nearest_indices]
            result = result.reshape(self.grid_shape + (n_comp,))

        return torch.tensor(result, dtype=torch.float32, device=self.device)


# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def _extract_boundary_polygon(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Triangle mesh'in dış sınır poligonunu çıkar."""
    edges = {}
    for tri in cells:
        for i in range(3):
            edge = tuple(sorted([tri[i], tri[(i + 1) % 3]]))
            edges[edge] = edges.get(edge, 0) + 1

    # Sadece 1 üçgende geçen kenarlar = dış sınır
    boundary_edges = [e for e, count in edges.items() if count == 1]

    if not boundary_edges:
        return np.array([])

    # Kenarları zincirle
    chain = list(boundary_edges[0])
    used = {0}

    while len(used) < len(boundary_edges):
        last = chain[-1]
        found = False
        for i, edge in enumerate(boundary_edges):
            if i in used:
                continue
            if edge[0] == last:
                chain.append(edge[1])
                used.add(i)
                found = True
                break
            elif edge[1] == last:
                chain.append(edge[0])
                used.add(i)
                found = True
                break
        if not found:
            break

    return points[chain, :2]


def _extract_surface_from_tets(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Tet mesh'ten yüzey üçgenlerini çıkar."""
    face_count = {}
    for tet in tets:
        faces = [
            tuple(sorted([tet[0], tet[1], tet[2]])),
            tuple(sorted([tet[0], tet[1], tet[3]])),
            tuple(sorted([tet[0], tet[2], tet[3]])),
            tuple(sorted([tet[1], tet[2], tet[3]])),
        ]
        for f in faces:
            face_count[f] = face_count.get(f, 0) + 1

    surface = np.array([list(f) for f, count in face_count.items() if count == 1])
    return surface


def _infer_bc_type(name: str) -> str:
    """Physical group isminden BC tipini çıkar."""
    name_lower = name.lower()
    if any(kw in name_lower for kw in ['fixed', 'clamp', 'wall', 'dirichlet', 'support']):
        return 'dirichlet'
    elif any(kw in name_lower for kw in ['load', 'force', 'neumann', 'pressure', 'traction']):
        return 'neumann'
    elif any(kw in name_lower for kw in ['free', 'symmetry']):
        return 'free'
    return 'dirichlet'
