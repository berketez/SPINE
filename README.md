<p align="center">
  <h1 align="center">SPINE</h1>
  <p align="center"><strong>Structural Physics-Inherent Neural Engine</strong></p>
  <p align="center">
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" alt="Python"></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  </p>
</p>

---

A Fourier-based neural operator framework for structural mechanics that embeds physics directly into neuron architectures rather than loss functions. SPINE replaces PINN-style autograd chains with single-step spectral operations, achieving exact derivative computation for equations up to 4th order (biharmonic).

## Why SPINE?

Traditional PINNs compute 4th-order derivatives through chained autograd -- each differentiation compounds numerical error. SPINE uses spectral methods in Fourier space where the biharmonic operator reduces to a single multiplication by k^4:

```
PINN:  w -> dw/dx -> d2w/dx2 -> d3w/dx3 -> d4w/dx4   (4 autograd passes, error accumulates)
SPINE: w -> FFT(w) * k^4 -> IFFT                       (1 spectral step, machine-precision)
```

## Features

| Category | Components | Description |
|----------|-----------|-------------|
| **2D Core** | BiharmonicNeuron, BucklingNeuron, StressNeuron, StrainNeuron, BoundaryNeuron | Plate bending, critical loads, stress-strain tensors, boundary conditions |
| **2D Advanced** | StaticNeuron, ModalNeuron, CrackNeuron, ThermalNeuron, FatigueNeuron, DynamicNeuron, NonlinearNeuron | Static equilibrium, modal analysis, fracture (K_I/K_II), thermal coupling, fatigue life, transient dynamics, geometric nonlinearity |
| **3D** | SolidNeuron3D, ShellNeuron3D, BeamNeuron3D | Full 3D elasticity, shell elements, Timoshenko beams |
| **Materials** | MaterialLibrary (50+ materials) | Metals, polymers, ceramics, composites with full CLT support |
| **Geometry** | GeometryProcessor, DeformationNet, ComplexGeometrySolver | STEP/STL import, SDF computation, domain deformation for irregular shapes |
| **Benchmarks** | 3D L-Bracket, FEM vs SPINE velocity | Validated against finite element solutions |

## Architecture

```
                        +--------------------------------------------------+
                        |                     SPINE                        |
                        |   Structural Physics-Inherent Neural Engine      |
                        +--------------------------------------------------+
                                            |
                    +-----------------------+-----------------------+
                    |                       |                       |
            +-------v--------+     +--------v-------+     +--------v-------+
            | SpectralOps2D  |     | SpectralOps3D  |     |  SineBasis     |
            | FFT-based      |     | 3D wavenumbers |     |  Non-periodic  |
            | derivatives    |     |                |     |  BC support    |
            +---+---+---+----+     +--------+-------+     +--------+-------+
                |   |   |                   |                      |
         +------+   |   +------+            |                      |
         |          |          |            |                      |
   +-----v---+ +---v----+ +---v-----+  +---v----------+  +--------v--------+
   |Biharmonic| |Buckling| | Stress  |  | Solid3D      |  |  BoundaryNeuron |
   |  nabla^4 | | N_cr   | | sigma=  |  | Shell3D      |  |  Simply-Supp.   |
   |          | | modes  | | C:eps   |  | Beam3D       |  |  Clamped/Free   |
   +----------+ +--------+ +--------+  +--------------+  +-----------------+
         |          |          |                |
         +----------+----------+----------------+
                    |
            +-------v--------+
            | MaterialLibrary|
            | 50+ materials  |
            | CLT composites |
            +----------------+
                    |
            +-------v--------+          +-------------------+
            |  SPINE (main)  |<-------->| DeformationNet    |
            |  Container     |          | Geo-FNO style     |
            |  Full analysis |          | Complex geometry   |
            +----------------+          +-------------------+
```

## Quick Start

### Installation

```bash
git clone https://github.com/berketez/SPINE.git
cd SPINE
pip install torch numpy scipy matplotlib
# Optional: complex geometry support
pip install gmsh meshio
```

### Basic Usage

```python
from spine import SPINE, MaterialProperties, BoundaryConditionType

# Define material and geometry
material = MaterialProperties(E=200e9, nu=0.3, h=0.005)  # Steel plate, 5mm thick
model = SPINE(resolution=64, Lx=1.0, Ly=0.5, material=material)

# Critical buckling load (analytical, instant)
N_cr = model.get_critical_load(m=1, n=1)
print(f"Critical load: {N_cr/1000:.2f} kN/m")

# Mode shape
w = model.get_mode_shape(m=1, n=1)

# Full plate state (displacement, rotations, moments)
state = model.forward(w)
print(f"Max displacement: {state.max_displacement():.6e} m")
```

### Using the Material Library

```python
from spine import SPINE

# Build from 50+ built-in materials
model = SPINE.from_material("aluminum_7075_T6", h=0.003, Lx=0.8, Ly=0.4)
N_cr = model.get_critical_load()

# Search materials
results = SPINE.search_materials("carbon")
print(results)  # ['carbon_epoxy_T300', 'carbon_epoxy_IM7', ...]
```

### Composite Laminate Analysis (CLT)

```python
from materials import CompositeLaminate, Ply
from spine import SPINE

laminate = CompositeLaminate([
    Ply(angle=0,   thickness=0.125e-3, material="carbon_epoxy_T300"),
    Ply(angle=45,  thickness=0.125e-3, material="carbon_epoxy_T300"),
    Ply(angle=-45, thickness=0.125e-3, material="carbon_epoxy_T300"),
    Ply(angle=90,  thickness=0.125e-3, material="carbon_epoxy_T300"),
])

A, B, D = laminate.get_ABD_matrices()
model = SPINE.from_laminate(laminate, Lx=1.0, Ly=0.5)
N_cr = model.get_critical_load_orthotropic()
```

### Complex Geometry (Domain Deformation)

```python
from spine_geometry import GeometryProcessor
from spine_deformation import ComplexGeometrySolver

geo = GeometryProcessor()
mesh = geo.create_plate_with_hole(Lx=1.0, Ly=1.0, hole_radius=0.15)
sdf = geo.compute_sdf_2d(mesh, resolution=64)

solver = ComplexGeometrySolver(resolution=64, code_dim=16)
result = solver(displacement_field, sdf.sdf)
# result: gradient, laplacian, biharmonic on irregular domain
```

## What It Solves

SPINE targets structural mechanics problems governed by:

- **Plate bending** (Kirchhoff-Love): nabla^4 w = q/D
- **Buckling**: D nabla^4 w + N_x d2w/dx2 = 0
- **Stress-strain**: sigma = C : epsilon (isotropic and orthotropic)
- **Modal analysis**: (K - omega^2 M) phi = 0
- **Fracture mechanics**: Stress intensity factors K_I, K_II near crack tips
- **Thermal-structural coupling**: Thermal expansion and thermal buckling
- **3D elasticity**: Full Cauchy stress tensor, shell and beam elements
- **Composite laminates**: Classical Lamination Theory (CLT) with ABD matrices

### Boundary Conditions

| Type | Constraints | Method |
|------|------------|--------|
| Simply Supported | w = 0, M = 0 at edges | Sine basis (natural satisfaction) |
| Clamped | w = 0, dw/dn = 0 at edges | Penalty + spectral correction |
| Free | M = 0, V = 0 at edges | Spectral operators |

## Project Structure

```
SPINE/
├── spine.py                  # Core engine: 16 physics neurons, spectral operators, SPINE container
├── spine_deformation.py      # Domain deformation for complex geometries (Geo-FNO style)
├── spine_geometry.py         # STEP/STL import, meshing, SDF computation
├── materials.py              # 50+ materials, CLT composites, ABD matrices
├── benchmark3D.py            # 3D L-Bracket: FEM vs SPINE comparison
├── benchmarkVelocity.py      # Speed benchmark: SPINE vs FEM timing
├── demo_complex_geometry.py  # Complex geometry demos (plate with hole, airfoil, L-bracket)
└── tests/                    # Test suite
    ├── test_buckling.py      # Buckling neuron tests
    ├── test_faz3.py          # Phase 3 neuron tests (modal, crack, thermal)
    ├── test_faz4_3d.py       # Phase 4 3D neuron tests
    ├── test_materials.py     # Material library tests
    └── train_*.py            # Training scripts for individual neurons
```

## Key Physics

### Spectral Derivative Computation

All derivatives are computed in Fourier space. For a function f(x, y):

```
f_hat = FFT2(f)

df/dx   =  IFFT2(i * kx * f_hat)           # 1st derivative
d2f/dx2 =  IFFT2(-kx^2 * f_hat)            # 2nd derivative
nabla^4 f = IFFT2((kx^2 + ky^2)^2 * f_hat) # Biharmonic (single step)
```

This gives machine-precision derivatives regardless of order, unlike finite differences or autograd chains.

### Classical Lamination Theory

For composite laminates, SPINE implements full CLT:

```
[N]   [A  B] [eps0]
[ ] = [    ] [    ]
[M]   [B  D] [kap ]
```

Where A (extensional), B (coupling), and D (bending) stiffness matrices are assembled from ply-level transformed stiffness matrices Q-bar.

## Benchmarks

**3D L-Bracket** (`benchmark3D.py`): Tetrahedral FEM solution compared against SPINE neural operator on a standard L-bracket with hole geometry.

**Speed Comparison** (`benchmarkVelocity.py`): SPINE inference vs FEM assembly+solve timing across resolutions.

## Hardware Support

| Device | Status | Notes |
|--------|--------|-------|
| CUDA GPU | Supported | cuDNN benchmark enabled, TF32 allowed |
| Apple Silicon (MPS) | Supported | Automatic FFT fallback to CPU when needed |
| CPU | Supported | Default fallback |

## Author

**Berke Tezgöçen**

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
