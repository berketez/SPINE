"""
SPINE Karmaşık Geometri Demo

Bu script, SPINE'ın karmaşık geometri desteğini gösterir:
1. Delikli plaka (Kirsch problemi) — analitik doğrulama
2. NACA airfoil profili — parametrik geometri
3. L-bracket — gerçekçi mühendislik parçası
4. STEP dosyası okuma (varsa)

Çalıştırma:
    python demo_complex_geometry.py

Gereksinimler:
    pip install gmsh meshio torch matplotlib
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import time

# SPINE modüllerini import et
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spine import (
    SpectralOps2DStruct, BiharmonicNeuron, StressNeuron,
    MaterialProperties, DEVICE, get_device
)
from spine_geometry import GeometryProcessor, MeshData, GridData
from spine_deformation import (
    DeformationNet, DeformedSpectralOps2D, GeometryEncoder,
    ComplexGeometrySolver
)


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# =============================================================================
# DEMO 1: Delikli Plaka (Kirsch Problemi)
# =============================================================================

def demo_plate_with_hole():
    separator("Demo 1: Delikli Plaka (Kirsch)")

    geo = GeometryProcessor(device="cpu")

    print("Geometri oluşturuluyor...")
    t0 = time.time()
    mesh = geo.create_plate_with_hole(
        Lx=1.0, Ly=1.0,
        hole_center=(0.5, 0.5),
        hole_radius=0.15,
        mesh_size=0.02
    )
    t1 = time.time()
    print(f"  Mesh: {mesh.n_nodes} node, {mesh.n_elements} element ({t1-t0:.2f}s)")

    # SDF hesapla
    print("SDF hesaplanıyor...")
    t0 = time.time()
    grid = geo.compute_sdf_2d(mesh, grid_resolution=64, padding=0.1)
    t1 = time.time()
    print(f"  SDF: {grid.resolution}x{grid.resolution} grid ({t1-t0:.2f}s)")
    print(f"  Domain: {grid.Lx:.3f} x {grid.Ly:.3f} m")
    print(f"  SDF range: [{grid.sdf.min():.4f}, {grid.sdf.max():.4f}]")
    print(f"  Mask sum: {grid.mask.sum():.0f} / {grid.resolution**2} (domain oranı: {grid.mask.sum()/grid.resolution**2:.1%})")

    # ComplexGeometrySolver ile test
    print("\nComplexGeometrySolver oluşturuluyor...")
    solver = ComplexGeometrySolver(
        resolution=64, Lx=grid.Lx, Ly=grid.Ly,
        code_dim=16, deformation_width=32,
        use_deformation=True
    )
    n_params = sum(p.numel() for p in solver.parameters())
    print(f"  Toplam parametre: {n_params:,}")

    # Test field: basit sinüzoidal
    x = torch.linspace(0, grid.Lx, 64)
    y = torch.linspace(0, grid.Ly, 64)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    f_test = torch.sin(2 * np.pi * X / grid.Lx) * torch.sin(2 * np.pi * Y / grid.Ly)

    # Forward pass (deformasyon olmadan)
    print("\nSpektral operatörler test ediliyor...")
    t0 = time.time()
    result = solver(f_test, grid.sdf)
    t1 = time.time()

    print(f"  Gradient max: |df/dx|={result['df_dx'].abs().max():.4f}, |df/dy|={result['df_dy'].abs().max():.4f}")
    print(f"  Laplacian max: {result['laplacian'].abs().max():.4f}")
    print(f"  Biharmonic max: {result['biharmonic'].abs().max():.4f}")
    if result.get('deformation_loss') is not None:
        print(f"  Deformation loss: {result['deformation_loss'].item():.6f}")
    print(f"  Forward pass süresi: {(t1-t0)*1000:.1f}ms")

    # Kirsch analitik çözüm karşılaştırması
    print("\n[Kirsch Analitik Referans]")
    sigma_inf = 100e6  # 100 MPa uzak alan gerilmesi
    R = 0.15           # Delik yarıçapı
    # Delik kenarında (r=R, θ=π/2): σ_θθ = 3·σ_inf
    sigma_max_kirsch = 3 * sigma_inf
    print(f"  Gerilme yoğunlaşma faktörü (SCF): 3.0")
    print(f"  σ_max = {sigma_max_kirsch/1e6:.0f} MPa (σ_∞ = {sigma_inf/1e6:.0f} MPa)")

    print("\n[OK] Delikli plaka demosu başarılı!")
    return grid, solver


# =============================================================================
# DEMO 2: NACA Airfoil Profili
# =============================================================================

def demo_naca_airfoil():
    separator("Demo 2: NACA Airfoil Profili")

    geo = GeometryProcessor(device="cpu")

    # Farklı NACA profilleri
    profiles = ["0012", "2412", "4415", "6412"]

    for naca in profiles:
        mesh = geo.create_naca_airfoil(naca=naca, chord=0.1, n_points=80)
        print(f"  NACA {naca}: {mesh.n_nodes} nokta, "
              f"bbox=[{mesh.bbox_min[0]:.4f}, {mesh.bbox_max[0]:.4f}] x "
              f"[{mesh.bbox_min[1]:.4f}, {mesh.bbox_max[1]:.4f}]")

    # Bir profil için SDF hesapla
    print("\nNACA 2412 SDF hesaplanıyor...")
    mesh = geo.create_naca_airfoil("2412", chord=0.1, n_points=100)
    grid = geo.compute_sdf_2d(mesh, grid_resolution=64, padding=0.3)
    print(f"  SDF grid: {grid.resolution}x{grid.resolution}")
    print(f"  Domain: {grid.Lx:.4f} x {grid.Ly:.4f} m")
    print(f"  İç alan oranı: {(grid.sdf <= 0).float().mean():.1%}")

    # Geometry encoder test
    print("\nGeometry encoder test...")
    encoder = GeometryEncoder(sdf_resolution=64, code_dim=16)
    code = encoder(grid.sdf)
    print(f"  Code shape: {code.shape}")
    print(f"  Code range: [{code.min():.4f}, {code.max():.4f}]")

    print("\n[OK] Airfoil demosu başarılı!")
    return grid


# =============================================================================
# DEMO 3: L-Bracket
# =============================================================================

def demo_l_bracket():
    separator("Demo 3: L-Bracket")

    geo = GeometryProcessor(device="cpu")

    print("L-bracket oluşturuluyor...")
    t0 = time.time()
    mesh = geo.create_l_bracket(
        arm1_length=0.1, arm1_width=0.05,
        arm2_length=0.1, arm2_width=0.05,
        mesh_size=0.003
    )
    t1 = time.time()
    print(f"  Mesh: {mesh.n_nodes} node, {mesh.n_elements} element ({t1-t0:.2f}s)")

    # SDF
    print("SDF hesaplanıyor...")
    grid = geo.compute_sdf_2d(mesh, grid_resolution=64, padding=0.1)
    print(f"  Domain: {grid.Lx:.4f} x {grid.Ly:.4f} m")
    print(f"  İç alan oranı: {(grid.sdf <= 0).float().mean():.1%}")

    print("\n[OK] L-bracket demosu başarılı!")
    return grid


# =============================================================================
# DEMO 4: Deformation Network Eğitimi
# =============================================================================

def demo_deformation_training():
    separator("Demo 4: Deformation Network Eğitimi")

    # Basit test: birim kare → deformed domain
    resolution = 32  # Küçük grid (hızlı test)

    print("DeformationNet oluşturuluyor...")
    iphi = DeformationNet(width=16, code_dim=0, n_layers=2, use_positional_encoding=False)
    n_params = sum(p.numel() for p in iphi.parameters())
    print(f"  Parametre: {n_params:,}")

    # Test noktaları
    x = torch.linspace(0.1, 0.9, resolution)
    y = torch.linspace(0.1, 0.9, resolution)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    x_phys = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1).unsqueeze(0)  # [1, N*N, 2]

    # Forward pass
    print("\nForward pass...")
    xi = iphi(x_phys)
    print(f"  Input range:  x=[{x_phys[...,0].min():.2f}, {x_phys[...,0].max():.2f}], "
          f"y=[{x_phys[...,1].min():.2f}, {x_phys[...,1].max():.2f}]")
    print(f"  Output range: ξ=[{xi[...,0].min():.2f}, {xi[...,0].max():.2f}], "
          f"η=[{xi[...,1].min():.2f}, {xi[...,1].max():.2f}]")

    # Jacobian
    print("\nJacobian hesaplanıyor...")
    det_J = iphi.det_jacobian(x_phys)
    print(f"  det(J) range: [{det_J.min():.4f}, {det_J.max():.4f}]")
    print(f"  det(J) > 0: {(det_J > 0).float().mean():.1%} (bijectivity)")

    # Bijectivity loss
    loss_bij = iphi.bijectivity_loss(x_phys)
    print(f"  Bijectivity loss: {loss_bij.item():.6f}")

    # Mini training loop (bijectivity + identity mapping yakınsama)
    print("\nMini eğitim (10 iterasyon)...")
    optimizer = torch.optim.Adam(iphi.parameters(), lr=1e-3)

    for i in range(10):
        optimizer.zero_grad()

        xi = iphi(x_phys)

        # Loss: identity mapping'e yakın kal + bijectivity
        loss_identity = F.mse_loss(xi, x_phys)
        loss_bij = iphi.bijectivity_loss(x_phys)
        loss = loss_identity + 10.0 * loss_bij

        loss.backward()
        optimizer.step()

        if i % 3 == 0 or i == 9:
            print(f"  iter {i:2d}: loss={loss.item():.6f} "
                  f"(identity={loss_identity.item():.6f}, bij={loss_bij.item():.6f})")

    print("\n[OK] Deformation training demosu başarılı!")


# =============================================================================
# DEMO 5: DeformedSpectralOps Doğrulama
# =============================================================================

def demo_deformed_spectral_ops():
    separator("Demo 5: DeformedSpectralOps Doğrulama")

    resolution = 32
    Lx, Ly = 1.0, 1.0

    # Standart (deformasyonsuz) operatörler
    std_ops = SpectralOps2DStruct(resolution, Lx, Ly)

    # Deformed operatörler (identity deformation = standart olmalı)
    deformed_ops = DeformedSpectralOps2D(resolution, Lx, Ly, iphi=None)

    # Test field: f(x,y) = sin(2πx)·cos(2πy)
    x = torch.linspace(0, Lx, resolution)
    y = torch.linspace(0, Ly, resolution)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    f = torch.sin(2 * np.pi * X / Lx) * torch.cos(2 * np.pi * Y / Ly)

    # Analitik türevler
    df_dx_exact = (2*np.pi/Lx) * torch.cos(2*np.pi*X/Lx) * torch.cos(2*np.pi*Y/Ly)
    df_dy_exact = -(2*np.pi/Ly) * torch.sin(2*np.pi*X/Lx) * torch.sin(2*np.pi*Y/Ly)
    lap_exact = -(2*np.pi/Lx)**2 * f - (2*np.pi/Ly)**2 * f

    # Standart SpectralOps
    df_dx_std, df_dy_std = std_ops.gradient(f)
    lap_std = std_ops.laplacian(f)

    # Deformed (identity = aynı sonuç olmalı)
    df_dx_def, df_dy_def = deformed_ops.gradient(f)
    lap_def = deformed_ops.laplacian(f)

    # Hata hesapla
    err_grad_std = ((df_dx_std - df_dx_exact)**2).mean().sqrt().item()
    err_grad_def = ((df_dx_def - df_dx_exact)**2).mean().sqrt().item()
    err_lap_std = ((lap_std - lap_exact)**2).mean().sqrt().item()
    err_lap_def = ((lap_def - lap_exact)**2).mean().sqrt().item()

    print("Gradient doğruluğu (RMSE):")
    print(f"  SpectralOps2DStruct:    {err_grad_std:.2e}")
    print(f"  DeformedSpectralOps2D:  {err_grad_def:.2e}")
    print(f"  Fark: {abs(err_grad_std - err_grad_def):.2e}")

    print("\nLaplacian doğruluğu (RMSE):")
    print(f"  SpectralOps2DStruct:    {err_lap_std:.2e}")
    print(f"  DeformedSpectralOps2D:  {err_lap_def:.2e}")
    print(f"  Fark: {abs(err_lap_std - err_lap_def):.2e}")

    # Biharmonic
    bih_std = std_ops.biharmonic(f)
    bih_def = deformed_ops.biharmonic(f)
    bih_exact = ((2*np.pi/Lx)**2 + (2*np.pi/Ly)**2)**2 * f

    err_bih_std = ((bih_std - bih_exact)**2).mean().sqrt().item()
    err_bih_def = ((bih_def - bih_exact)**2).mean().sqrt().item()

    print("\nBiharmonik doğruluğu (RMSE):")
    print(f"  SpectralOps2DStruct:    {err_bih_std:.2e}")
    print(f"  DeformedSpectralOps2D:  {err_bih_def:.2e}")

    print("\n[OK] Spectral ops doğrulama başarılı!")


# =============================================================================
# DEMO 6: STEP Dosyası (varsa)
# =============================================================================

def demo_step_file():
    separator("Demo 6: STEP Dosyası Okuma")

    # Demo STEP dosyaları ara
    step_files = []
    search_dirs = [
        "/Users/apple/Desktop/buckneuron",
        "/Users/apple/Desktop",
    ]

    for d in search_dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.lower().endswith(('.step', '.stp', '.iges', '.igs')):
                    step_files.append(os.path.join(d, f))

    if not step_files:
        print("STEP/IGES dosyası bulunamadı.")
        print("Test etmek için bir .step dosyasını buckneuron/ klasörüne koyun.")
        print("\nGmsh STEP okuma testi (boş geometri)...")

        # gmsh'in çalıştığını doğrula
        try:
            import gmsh
            gmsh.initialize()
            gmsh.option.setNumber("General.Verbosity", 0)
            print(f"  gmsh version: {gmsh.GMSH_API_VERSION}")
            gmsh.finalize()
            print("  [OK] gmsh başarıyla yüklendi!")
        except Exception as e:
            print(f"  [HATA] gmsh: {e}")

        return

    print(f"Bulunan dosyalar: {len(step_files)}")
    for f in step_files[:3]:
        print(f"  {f}")

    # İlk dosyayı oku
    geo = GeometryProcessor(device="cpu")
    filepath = step_files[0]
    print(f"\n'{os.path.basename(filepath)}' okunuyor...")

    try:
        mesh = geo.load_step(filepath, mesh_size=0.01, dim=2)
        print(f"  Mesh: {mesh.n_nodes} node, {mesh.n_elements} element")
        print(f"  Bbox: {mesh.bbox_min} → {mesh.bbox_max}")

        grid = geo.compute_sdf_2d(mesh, grid_resolution=64)
        print(f"  SDF: {grid.resolution}x{grid.resolution}")
        print(f"  İç alan oranı: {(grid.sdf <= 0).float().mean():.1%}")
        print("  [OK] STEP dosyası başarıyla okundu!")
    except Exception as e:
        print(f"  [HATA] {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("  SPINE Karmaşık Geometri Demo")
    print("  Structural Physics-Inherent Neural Engine")
    print("="*60)
    print(f"\nDevice: {DEVICE}")
    print(f"PyTorch: {torch.__version__}")

    try:
        import gmsh
        print(f"gmsh: {gmsh.GMSH_API_VERSION}")
    except ImportError:
        print("gmsh: NOT INSTALLED")
        print("  pip install gmsh")
        return

    try:
        import meshio
        print(f"meshio: {meshio.__version__}")
    except ImportError:
        print("meshio: NOT INSTALLED")

    # Demo'ları çalıştır
    try:
        demo_plate_with_hole()
    except Exception as e:
        print(f"[HATA] Demo 1: {e}")

    try:
        demo_naca_airfoil()
    except Exception as e:
        print(f"[HATA] Demo 2: {e}")

    try:
        demo_l_bracket()
    except Exception as e:
        print(f"[HATA] Demo 3: {e}")

    try:
        demo_deformation_training()
    except Exception as e:
        print(f"[HATA] Demo 4: {e}")

    try:
        demo_deformed_spectral_ops()
    except Exception as e:
        print(f"[HATA] Demo 5: {e}")

    try:
        demo_step_file()
    except Exception as e:
        print(f"[HATA] Demo 6: {e}")

    separator("SONUÇ")
    print("Tüm demolar tamamlandı!")
    print("\nYeni dosyalar:")
    print("  spine_geometry.py     — STEP okuma, mesh, SDF pipeline")
    print("  spine_deformation.py  — Deformation network, DeformedSpectralOps")
    print("  demo_complex_geometry.py — Bu demo script")
    print("\nKullanım:")
    print("  from spine_geometry import GeometryProcessor")
    print("  from spine_deformation import ComplexGeometrySolver")


if __name__ == "__main__":
    main()
