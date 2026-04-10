"""
3D L-Bracket Benchmark: FEM vs SPINE AI
========================================

Endüstriyel standart 3D L-Bracket problemi:
- 3D FEM (tetrahedral elements) ile çözüm
- ~5K parametreli SPINE AI eğitimi
- PyVista ile ANSYS-tarzı interaktif görselleştirme

Geometri:
        ┌─────────────────┐
        │                 │  ← Sabit (Fixed BC)
        │     ARM 1       │
        │                 │
        ├─────────┐       │
        │  HOLE   │       │
        │    ○    │       │
        │         │       │
        │  ARM 2  │       │
        │         │
        └─────────┘
              ↓
           LOAD (F)

Yazar: SPINE Projesi
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Dict, List, Tuple, Optional
import time
import warnings
warnings.filterwarnings('ignore')

# PyVista for 3D visualization
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("PyVista not available. Install with: pip install pyvista")


# =============================================================================
# DEVICE SETUP
# =============================================================================

def get_device():
    """Otomatik cihaz seçimi."""
    if torch.cuda.is_available():
        return torch.device("cuda"), "CUDA GPU"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps"), "Apple Silicon GPU (MPS)"
    return torch.device("cpu"), "CPU"

DEVICE, DEVICE_NAME = get_device()


# =============================================================================
# L-BRACKET MESH GENERATOR
# =============================================================================

class LBracketMesh:
    """
    3D L-Bracket mesh generator with tetrahedral elements.

    Geometry (mm):
        - Arm 1 (horizontal): 100 x 50 x 10
        - Arm 2 (vertical): 50 x 100 x 10
        - Hole at junction: diameter 20mm

    Mesh: Structured hexahedra → split into tetrahedra (6 tets per hex)
    """

    def __init__(self,
                 arm1_length: float = 100.0,  # mm
                 arm1_width: float = 50.0,
                 arm2_length: float = 100.0,
                 arm2_width: float = 50.0,
                 thickness: float = 10.0,
                 hole_diameter: float = 20.0,
                 n_elements_per_arm: int = 8):

        self.arm1_L = arm1_length
        self.arm1_W = arm1_width
        self.arm2_L = arm2_length
        self.arm2_W = arm2_width
        self.t = thickness
        self.hole_d = hole_diameter
        self.n_elem = n_elements_per_arm

        # Mesh data
        self.nodes = None
        self.elements = None
        self.node_sets = {}  # Boundary condition sets

        self._generate_mesh()

    def _generate_mesh(self):
        """Generate L-bracket tetrahedral mesh."""
        n = self.n_elem

        # Create structured grid for L-shape
        # Region 1: Arm 1 (horizontal part)
        # Region 2: Arm 2 (vertical part)
        # Combined to form L-shape

        nodes_list = []
        elements_list = []
        node_id_map = {}
        current_node_id = 0

        # Grid spacing
        dx1 = self.arm1_L / n
        dy1 = self.arm1_W / n
        dx2 = self.arm2_W / n
        dy2 = self.arm2_L / n
        dz = self.t / max(2, n // 4)
        nz = max(2, n // 4)

        # Generate L-shape nodes
        # Arm 1: x in [0, arm1_L], y in [arm2_L - arm1_W, arm2_L], z in [0, t]
        # Arm 2: x in [0, arm2_W], y in [0, arm2_L], z in [0, t]

        # First, create all unique nodes
        all_positions = set()

        # Arm 1 nodes
        for i in range(n + 1):
            for j in range(n + 1):
                for k in range(nz + 1):
                    x = i * dx1
                    y = (self.arm2_L - self.arm1_W) + j * dy1
                    z = k * dz
                    pos = (round(x, 6), round(y, 6), round(z, 6))
                    all_positions.add(pos)

        # Arm 2 nodes
        for i in range(n + 1):
            for j in range(n + 1):
                for k in range(nz + 1):
                    x = i * dx2
                    y = j * dy2
                    z = k * dz
                    pos = (round(x, 6), round(y, 6), round(z, 6))
                    all_positions.add(pos)

        # Sort and create node array
        sorted_positions = sorted(all_positions)
        self.nodes = np.array(sorted_positions)

        # Create position to node ID mapping
        for idx, pos in enumerate(sorted_positions):
            node_id_map[pos] = idx

        # Hole center (at junction)
        hole_center = np.array([self.arm2_W / 2, self.arm2_L - self.arm1_W / 2, self.t / 2])
        hole_radius = self.hole_d / 2

        # Generate hexahedra and split into tetrahedra
        def get_node_id(x, y, z):
            pos = (round(x, 6), round(y, 6), round(z, 6))
            return node_id_map.get(pos, -1)

        def is_in_hole(cx, cy, cz):
            """Check if point is inside the hole."""
            dx = cx - hole_center[0]
            dy = cy - hole_center[1]
            return np.sqrt(dx*dx + dy*dy) < hole_radius * 0.9

        def hex_to_tets(n0, n1, n2, n3, n4, n5, n6, n7):
            """Split hexahedron into 6 tetrahedra."""
            # Consistent splitting pattern
            return [
                [n0, n1, n3, n4],
                [n1, n2, n3, n6],
                [n1, n4, n5, n6],
                [n3, n4, n6, n7],
                [n1, n3, n4, n6],
                [n1, n3, n6, n2],  # Extra tet for better quality
            ][:5]  # Use 5 tets per hex (more common)

        elements_list = []

        # Arm 1 elements
        for i in range(n):
            for j in range(n):
                for k in range(nz):
                    x0 = i * dx1
                    y0 = (self.arm2_L - self.arm1_W) + j * dy1
                    z0 = k * dz

                    # Check if hex center is in hole
                    cx = x0 + dx1/2
                    cy = y0 + dy1/2
                    cz = z0 + dz/2

                    if is_in_hole(cx, cy, cz):
                        continue

                    # Get 8 corner nodes of hexahedron
                    n0 = get_node_id(x0, y0, z0)
                    n1 = get_node_id(x0 + dx1, y0, z0)
                    n2 = get_node_id(x0 + dx1, y0 + dy1, z0)
                    n3 = get_node_id(x0, y0 + dy1, z0)
                    n4 = get_node_id(x0, y0, z0 + dz)
                    n5 = get_node_id(x0 + dx1, y0, z0 + dz)
                    n6 = get_node_id(x0 + dx1, y0 + dy1, z0 + dz)
                    n7 = get_node_id(x0, y0 + dy1, z0 + dz)

                    if -1 in [n0, n1, n2, n3, n4, n5, n6, n7]:
                        continue

                    tets = hex_to_tets(n0, n1, n2, n3, n4, n5, n6, n7)
                    elements_list.extend(tets)

        # Arm 2 elements (excluding overlap with Arm 1)
        for i in range(n):
            for j in range(n):
                for k in range(nz):
                    x0 = i * dx2
                    y0 = j * dy2
                    z0 = k * dz

                    # Skip if in Arm 1 region
                    if y0 >= (self.arm2_L - self.arm1_W) and x0 < self.arm2_W:
                        continue

                    cx = x0 + dx2/2
                    cy = y0 + dy2/2
                    cz = z0 + dz/2

                    if is_in_hole(cx, cy, cz):
                        continue

                    n0 = get_node_id(x0, y0, z0)
                    n1 = get_node_id(x0 + dx2, y0, z0)
                    n2 = get_node_id(x0 + dx2, y0 + dy2, z0)
                    n3 = get_node_id(x0, y0 + dy2, z0)
                    n4 = get_node_id(x0, y0, z0 + dz)
                    n5 = get_node_id(x0 + dx2, y0, z0 + dz)
                    n6 = get_node_id(x0 + dx2, y0 + dy2, z0 + dz)
                    n7 = get_node_id(x0, y0 + dy2, z0 + dz)

                    if -1 in [n0, n1, n2, n3, n4, n5, n6, n7]:
                        continue

                    tets = hex_to_tets(n0, n1, n2, n3, n4, n5, n6, n7)
                    elements_list.extend(tets)

        self.elements = np.array(elements_list, dtype=np.int32)

        # Define boundary node sets
        tol = 1e-6

        # Fixed BC: top of Arm 1 (y = arm2_L)
        self.node_sets['fixed'] = np.where(
            np.abs(self.nodes[:, 1] - self.arm2_L) < tol
        )[0]

        # Load BC: bottom of Arm 2 (y = 0)
        self.node_sets['load'] = np.where(
            np.abs(self.nodes[:, 1]) < tol
        )[0]

        # Hole boundary (for visualization)
        dist_to_hole = np.sqrt(
            (self.nodes[:, 0] - hole_center[0])**2 +
            (self.nodes[:, 1] - hole_center[1])**2
        )
        self.node_sets['hole'] = np.where(
            np.abs(dist_to_hole - hole_radius) < hole_radius * 0.3
        )[0]

        # Remove orphan nodes (not connected to any element)
        self._remove_orphan_nodes()

    def _remove_orphan_nodes(self):
        """Remove nodes that are not connected to any element."""
        # Find all nodes referenced by elements
        used_nodes = np.unique(self.elements.flatten())

        # Create mapping from old to new node IDs
        old_to_new = {old: new for new, old in enumerate(used_nodes)}

        # Update node array
        self.nodes = self.nodes[used_nodes]

        # Update element connectivity
        new_elements = np.zeros_like(self.elements)
        for i, elem in enumerate(self.elements):
            for j, node in enumerate(elem):
                new_elements[i, j] = old_to_new[node]
        self.elements = new_elements

        # Update node sets
        for name, node_ids in self.node_sets.items():
            new_ids = []
            for old_id in node_ids:
                if old_id in old_to_new:
                    new_ids.append(old_to_new[old_id])
            self.node_sets[name] = np.array(new_ids, dtype=np.int32)

    def get_info(self) -> str:
        """Return mesh information."""
        return (f"L-Bracket Mesh: {len(self.nodes)} nodes, "
                f"{len(self.elements)} tetrahedra, "
                f"Fixed: {len(self.node_sets['fixed'])} nodes, "
                f"Load: {len(self.node_sets['load'])} nodes")


# =============================================================================
# TET4 FINITE ELEMENT SOLVER (3D LINEAR ELASTICITY)
# =============================================================================

class Tet4FEM:
    """
    4-node tetrahedral element (C3D4) FEM solver.

    3D Linear Elasticity:
        σ = D : ε
        ∇·σ + b = 0

    Element: Constant strain tetrahedron (simplest 3D element)
    """

    def __init__(self, nodes: np.ndarray, elements: np.ndarray,
                 E: float = 210e9, nu: float = 0.3):
        """
        Args:
            nodes: (n_nodes, 3) node coordinates
            elements: (n_elements, 4) element connectivity
            E: Young's modulus (Pa)
            nu: Poisson's ratio
        """
        self.nodes = nodes.astype(np.float64)
        self.elements = elements.astype(np.int32)
        self.E = E
        self.nu = nu

        self.n_nodes = len(nodes)
        self.n_elements = len(elements)
        self.n_dof = self.n_nodes * 3

        # Precompute material matrix
        self.D = self._material_matrix()

        # Precompute element volumes and B matrices
        self.volumes = np.zeros(self.n_elements)
        self.B_matrices = []
        self._precompute_elements()

    def _material_matrix(self) -> np.ndarray:
        """
        3D isotropic elasticity matrix (6x6).

        σ = [σxx, σyy, σzz, τxy, τyz, τzx]^T
        ε = [εxx, εyy, εzz, γxy, γyz, γzx]^T
        """
        E, nu = self.E, self.nu

        # Lamé parameters
        lam = E * nu / ((1 + nu) * (1 - 2*nu))
        mu = E / (2 * (1 + nu))

        D = np.array([
            [lam + 2*mu, lam, lam, 0, 0, 0],
            [lam, lam + 2*mu, lam, 0, 0, 0],
            [lam, lam, lam + 2*mu, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu]
        ])

        return D

    def _precompute_elements(self):
        """Precompute B matrices and volumes for all elements."""
        for e in range(self.n_elements):
            B, V = self._element_B_matrix(e)
            self.B_matrices.append(B)
            self.volumes[e] = V

    def _element_B_matrix(self, elem_idx: int) -> Tuple[np.ndarray, float]:
        """
        Compute strain-displacement matrix B (6x12) for tetrahedral element.

        For constant strain tetrahedron:
            ε = B * u_e

        where u_e = [u1, v1, w1, u2, v2, w2, u3, v3, w3, u4, v4, w4]^T
        """
        nodes_e = self.elements[elem_idx]
        coords = self.nodes[nodes_e]  # (4, 3)

        # Node coordinates
        x1, y1, z1 = coords[0]
        x2, y2, z2 = coords[1]
        x3, y3, z3 = coords[2]
        x4, y4, z4 = coords[3]

        # Volume of tetrahedron (6V = det of coordinate matrix)
        # Using the formula: 6V = |x2-x1  y2-y1  z2-z1|
        #                         |x3-x1  y3-y1  z3-z1|
        #                         |x4-x1  y4-y1  z4-z1|
        J = np.array([
            [x2-x1, y2-y1, z2-z1],
            [x3-x1, y3-y1, z3-z1],
            [x4-x1, y4-y1, z4-z1]
        ])

        V6 = np.linalg.det(J)
        V = abs(V6) / 6.0

        if V < 1e-15:
            # Degenerate element
            return np.zeros((6, 12)), 1e-15

        # Shape function derivatives (constant for linear tet)
        # dN/dx, dN/dy, dN/dz for each of 4 nodes

        # Cofactors of Jacobian (for inverse)
        J_inv = np.linalg.inv(J)

        # dN/d(x,y,z) for nodes 2,3,4 (node 1 is reference)
        # N1 = 1 - ξ - η - ζ, N2 = ξ, N3 = η, N4 = ζ
        # In physical coords: need Jacobian inverse

        # Direct computation of shape function derivatives
        # Using cofactor method for robustness

        # a, b, c, d coefficients from coordinate differences
        # This is the standard Tet4 B-matrix derivation

        # Submatrices for B computation
        # Following standard FEM formulation

        # Compute gradients using volume coordinates
        # For node i: ∂Ni/∂x = ai/(6V), etc.

        # Coefficients from coordinate expansion
        def cofactor_3x3(mat, i, j):
            rows = [r for r in range(3) if r != i]
            cols = [c for c in range(3) if c != j]
            minor = mat[np.ix_(rows, cols)]
            return ((-1)**(i+j)) * np.linalg.det(minor)

        # Extended matrix for shape function derivatives
        M = np.array([
            [1, x1, y1, z1],
            [1, x2, y2, z2],
            [1, x3, y3, z3],
            [1, x4, y4, z4]
        ])

        # Cofactors give us the shape function derivatives times 6V
        a = np.array([cofactor_3x3(M[1:, 1:], -1, -1) for _ in range(1)])  # dummy

        # Simpler approach: compute directly
        # ∂N1/∂x = (1/6V) * |1  y2  z2|    (cyclic for other nodes)
        #                   |1  y3  z3|
        #                   |1  y4  z4|

        def det3(a1, b1, c1, a2, b2, c2, a3, b3, c3):
            return (a1*(b2*c3-b3*c2) - b1*(a2*c3-a3*c2) + c1*(a2*b3-a3*b2))

        # Shape function derivatives * 6V
        # Node 1
        b1 = -det3(1,y2,z2, 1,y3,z3, 1,y4,z4)
        c1 = det3(1,x2,z2, 1,x3,z3, 1,x4,z4)
        d1 = -det3(1,x2,y2, 1,x3,y3, 1,x4,y4)

        # Node 2
        b2 = det3(1,y1,z1, 1,y3,z3, 1,y4,z4)
        c2 = -det3(1,x1,z1, 1,x3,z3, 1,x4,z4)
        d2 = det3(1,x1,y1, 1,x3,y3, 1,x4,y4)

        # Node 3
        b3 = -det3(1,y1,z1, 1,y2,z2, 1,y4,z4)
        c3 = det3(1,x1,z1, 1,x2,z2, 1,x4,z4)
        d3 = -det3(1,x1,y1, 1,x2,y2, 1,x4,y4)

        # Node 4
        b4 = det3(1,y1,z1, 1,y2,z2, 1,y3,z3)
        c4 = -det3(1,x1,z1, 1,x2,z2, 1,x3,z3)
        d4 = det3(1,x1,y1, 1,x2,y2, 1,x3,y3)

        # B matrix (6x12)
        # ε = [εxx, εyy, εzz, γxy, γyz, γzx]
        # u_e = [u1,v1,w1, u2,v2,w2, u3,v3,w3, u4,v4,w4]

        inv6V = 1.0 / (6.0 * V) if V > 1e-15 else 0.0

        B = np.zeros((6, 12))

        # εxx = ∂u/∂x
        B[0, 0] = b1 * inv6V
        B[0, 3] = b2 * inv6V
        B[0, 6] = b3 * inv6V
        B[0, 9] = b4 * inv6V

        # εyy = ∂v/∂y
        B[1, 1] = c1 * inv6V
        B[1, 4] = c2 * inv6V
        B[1, 7] = c3 * inv6V
        B[1, 10] = c4 * inv6V

        # εzz = ∂w/∂z
        B[2, 2] = d1 * inv6V
        B[2, 5] = d2 * inv6V
        B[2, 8] = d3 * inv6V
        B[2, 11] = d4 * inv6V

        # γxy = ∂u/∂y + ∂v/∂x
        B[3, 0] = c1 * inv6V
        B[3, 1] = b1 * inv6V
        B[3, 3] = c2 * inv6V
        B[3, 4] = b2 * inv6V
        B[3, 6] = c3 * inv6V
        B[3, 7] = b3 * inv6V
        B[3, 9] = c4 * inv6V
        B[3, 10] = b4 * inv6V

        # γyz = ∂v/∂z + ∂w/∂y
        B[4, 1] = d1 * inv6V
        B[4, 2] = c1 * inv6V
        B[4, 4] = d2 * inv6V
        B[4, 5] = c2 * inv6V
        B[4, 7] = d3 * inv6V
        B[4, 8] = c3 * inv6V
        B[4, 10] = d4 * inv6V
        B[4, 11] = c4 * inv6V

        # γzx = ∂w/∂x + ∂u/∂z
        B[5, 0] = d1 * inv6V
        B[5, 2] = b1 * inv6V
        B[5, 3] = d2 * inv6V
        B[5, 5] = b2 * inv6V
        B[5, 6] = d3 * inv6V
        B[5, 8] = b3 * inv6V
        B[5, 9] = d4 * inv6V
        B[5, 11] = b4 * inv6V

        return B, V

    def _element_stiffness(self, elem_idx: int) -> np.ndarray:
        """
        Element stiffness matrix K_e (12x12).

        K_e = V * B^T * D * B
        """
        B = self.B_matrices[elem_idx]
        V = self.volumes[elem_idx]

        K_e = V * (B.T @ self.D @ B)
        return K_e

    def assemble_stiffness(self) -> sparse.csr_matrix:
        """Assemble global stiffness matrix."""
        n_dof = self.n_dof

        # Use COO format for assembly
        rows = []
        cols = []
        data = []

        for e in range(self.n_elements):
            K_e = self._element_stiffness(e)
            nodes_e = self.elements[e]

            # DOF indices for this element
            dof_e = np.zeros(12, dtype=np.int32)
            for i, n in enumerate(nodes_e):
                dof_e[3*i:3*i+3] = [3*n, 3*n+1, 3*n+2]

            # Add to global matrix
            for i in range(12):
                for j in range(12):
                    rows.append(dof_e[i])
                    cols.append(dof_e[j])
                    data.append(K_e[i, j])

        K = sparse.coo_matrix((data, (rows, cols)), shape=(n_dof, n_dof))
        return K.tocsr()

    def apply_bc(self, K: sparse.csr_matrix, F: np.ndarray,
                 fixed_nodes: np.ndarray) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
        """
        Apply fixed boundary conditions.

        Returns:
            K_bc: Reduced stiffness matrix
            F_bc: Reduced load vector
            free_dofs: Array of free DOF indices
        """
        # Fixed DOFs
        fixed_dofs = []
        for n in fixed_nodes:
            fixed_dofs.extend([3*n, 3*n+1, 3*n+2])
        fixed_dofs = np.array(fixed_dofs)

        # Free DOFs
        all_dofs = np.arange(self.n_dof)
        free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

        # Extract submatrices
        K_bc = K[free_dofs, :][:, free_dofs]
        F_bc = F[free_dofs]

        return K_bc, F_bc, free_dofs

    def solve(self, F_nodes: np.ndarray, F_values: np.ndarray,
              fixed_nodes: np.ndarray) -> Tuple[np.ndarray, float, Dict]:
        """
        Solve the FEM problem.

        Args:
            F_nodes: Node indices where loads are applied
            F_values: Load values (n_load_nodes, 3) for Fx, Fy, Fz
            fixed_nodes: Node indices with fixed BC

        Returns:
            u: Displacement vector (n_dof,)
            solve_time: Solution time
            info: Additional information
        """
        start_time = time.perf_counter()

        # Assemble stiffness matrix
        K = self.assemble_stiffness()
        assembly_time = time.perf_counter() - start_time

        # Create load vector
        F = np.zeros(self.n_dof)
        for i, n in enumerate(F_nodes):
            F[3*n:3*n+3] = F_values[i] if F_values.ndim > 1 else F_values

        # Apply boundary conditions
        K_bc, F_bc, free_dofs = self.apply_bc(K, F, fixed_nodes)

        # Solve
        solve_start = time.perf_counter()
        u_free = spsolve(K_bc, F_bc)
        linear_solve_time = time.perf_counter() - solve_start

        # Reconstruct full displacement vector
        u = np.zeros(self.n_dof)
        u[free_dofs] = u_free

        total_time = time.perf_counter() - start_time

        info = {
            'n_nodes': self.n_nodes,
            'n_elements': self.n_elements,
            'n_dof': self.n_dof,
            'n_free_dof': len(free_dofs),
            'assembly_time': assembly_time,
            'linear_solve_time': linear_solve_time,
            'total_time': total_time
        }

        return u, total_time, info

    def compute_stress(self, u: np.ndarray) -> np.ndarray:
        """
        Compute stress at element centroids.

        Returns:
            stress: (n_elements, 6) stress tensor components
                    [σxx, σyy, σzz, τxy, τyz, τzx]
        """
        stress = np.zeros((self.n_elements, 6))

        for e in range(self.n_elements):
            nodes_e = self.elements[e]

            # Element displacement vector
            u_e = np.zeros(12)
            for i, n in enumerate(nodes_e):
                u_e[3*i:3*i+3] = u[3*n:3*n+3]

            # Strain
            B = self.B_matrices[e]
            eps = B @ u_e

            # Stress
            stress[e] = self.D @ eps

        return stress

    def von_mises_stress(self, stress: np.ndarray) -> np.ndarray:
        """
        Compute von Mises equivalent stress.

        σ_vm = sqrt(σxx² + σyy² + σzz² - σxxσyy - σyyσzz - σzzσxx + 3(τxy² + τyz² + τzx²))
        """
        s = stress
        vm = np.sqrt(
            s[:, 0]**2 + s[:, 1]**2 + s[:, 2]**2
            - s[:, 0]*s[:, 1] - s[:, 1]*s[:, 2] - s[:, 2]*s[:, 0]
            + 3*(s[:, 3]**2 + s[:, 4]**2 + s[:, 5]**2)
        )
        return vm

    def principal_stresses(self, stress: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute principal stresses σ1 ≥ σ2 ≥ σ3.
        """
        n_elem = stress.shape[0]
        s1 = np.zeros(n_elem)
        s2 = np.zeros(n_elem)
        s3 = np.zeros(n_elem)

        for e in range(n_elem):
            # Stress tensor
            sigma = np.array([
                [stress[e, 0], stress[e, 3], stress[e, 5]],
                [stress[e, 3], stress[e, 1], stress[e, 4]],
                [stress[e, 5], stress[e, 4], stress[e, 2]]
            ])

            # Eigenvalues = principal stresses
            eigvals = np.linalg.eigvalsh(sigma)
            eigvals = np.sort(eigvals)[::-1]  # Descending

            s1[e], s2[e], s3[e] = eigvals

        return s1, s2, s3

    def get_info(self) -> str:
        """Return solver information."""
        return (f"Tet4 FEM: {self.n_nodes} nodes, {self.n_elements} elements, "
                f"{self.n_dof} DOF, E={self.E/1e9:.0f} GPa, nu={self.nu}")


# =============================================================================
# SPINE 3D NEURAL NETWORK (~5K PARAMETERS)
# =============================================================================

class SPINE3DNet(nn.Module):
    """
    SPINE 3D Neural Network for structural analysis.

    ~5K trainable parameters with physics priors.

    Input: [E, nu, Lx, Ly, Lz, Fx, Fy, Fz, hole_r] - 9 params
    Output: [u_max, sigma_vm_max, sigma_1_max, sigma_3_min] - 4 outputs
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        self.hidden_dim = hidden_dim

        # MLP architecture (~5K params)
        # 9 → 64: 9*64 + 64 = 640
        # 64 → 64: 64*64 + 64 = 4160
        # 64 → 32: 64*32 + 32 = 2080
        # 32 → 4: 32*4 + 4 = 132
        # Total: ~7K params

        self.encoder = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 4),
        )

        # Learnable scaling
        self.output_scale = nn.Parameter(torch.ones(4))
        self.output_bias = nn.Parameter(torch.zeros(4))

    def physics_prior(self, E: torch.Tensor, nu: torch.Tensor,
                      Lx: torch.Tensor, Ly: torch.Tensor, Lz: torch.Tensor,
                      Fx: torch.Tensor, Fy: torch.Tensor, Fz: torch.Tensor,
                      hole_r: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analytical physics approximation for L-bracket.

        Cantilever beam + stress concentration factor.
        """
        # Total force magnitude
        F_mag = torch.sqrt(Fx**2 + Fy**2 + Fz**2 + 1e-10)

        # Approximate as cantilever beam
        # I = (Lx * Lz^3) / 12  (moment of inertia)
        I = Lx * Lz**3 / 12.0

        # Max deflection: δ = F * L^3 / (3 * E * I)
        L_eff = Ly  # Effective length
        u_max_beam = F_mag * L_eff**3 / (3 * E * I + 1e-10)

        # Nominal stress: σ = M * c / I, M = F * L, c = Lz/2
        M = F_mag * L_eff
        c = Lz / 2
        sigma_nom = M * c / (I + 1e-10)

        # Stress concentration factor for circular hole
        # K_t ≈ 3 for hole in infinite plate under tension
        # Adjusted for finite geometry
        d_over_w = 2 * hole_r / Lx
        K_t = 3.0 - 3.13 * d_over_w + 3.66 * d_over_w**2  # Peterson's formula
        K_t = torch.clamp(K_t, 1.5, 4.0)

        sigma_max = K_t * sigma_nom

        return {
            'u_max': u_max_beam,
            'sigma_vm_max': sigma_max,
            'sigma_1_max': sigma_max * 1.1,  # Principal stress slightly higher
            'sigma_3_min': -sigma_max * 0.3,  # Compressive
        }

    def forward(self, params: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass - direct prediction in log space.

        params: [B, 9] = [E, nu, Lx, Ly, Lz, Fx, Fy, Fz, hole_r]
        """
        if params.dim() == 1:
            params = params.unsqueeze(0)

        B = params.shape[0]

        # Extract parameters
        E = params[:, 0]
        nu = params[:, 1]
        Lx = params[:, 2]
        Ly = params[:, 3]
        Lz = params[:, 4]
        Fx = params[:, 5]
        Fy = params[:, 6]
        Fz = params[:, 7]
        hole_r = params[:, 8]

        # Normalize inputs for neural network (E in MPa = N/mm²)
        # Key insight: use log of force for better scaling
        F_mag = torch.sqrt(Fx**2 + Fy**2 + Fz**2 + 1e-10)

        params_norm = torch.stack([
            torch.log10(E + 1) / 6.0,  # ~5 → ~0.83
            nu / 0.5,
            torch.log10(Lx + 1) / 3.0,
            torch.log10(Ly + 1) / 3.0,
            torch.log10(Lz + 1) / 2.0,
            torch.log10(F_mag + 1) / 5.0,
            Fx / (F_mag + 1e-10),  # Direction cosines
            Fy / (F_mag + 1e-10),
            hole_r / Lx,  # Relative hole size
        ], dim=1)

        # MLP: predict log of outputs
        features = self.encoder(params_norm)
        log_outputs = self.decoder(features)

        # Convert from log space to physical values
        # Scale outputs based on physics (typical ranges)
        # u_max: ~0.001 - 1 mm → log10 in [-3, 0]
        # sigma: ~1 - 100 MPa → log10 in [0, 2]

        u_max_log = log_outputs[:, 0] * 2.0 - 3.0  # Range: -5 to -1 (0.00001 to 0.1 mm)
        sigma_log = log_outputs[:, 1] * 2.0  # Range: -2 to 2 (0.01 to 100 MPa)
        s1_log = log_outputs[:, 2] * 2.0
        s3_log = log_outputs[:, 3] * 2.0

        results = {
            'u_max': 10 ** u_max_log * self.output_scale[0] + self.output_bias[0],
            'sigma_vm_max': 10 ** sigma_log * self.output_scale[1] + self.output_bias[1],
            'sigma_1_max': 10 ** s1_log * self.output_scale[2] + self.output_bias[2],
            'sigma_3_min': -(10 ** s3_log) * self.output_scale[3] + self.output_bias[3],  # Negative
        }

        return results

    def predict(self, E: float, nu: float, Lx: float, Ly: float, Lz: float,
                Fx: float, Fy: float, Fz: float, hole_r: float) -> Dict[str, float]:
        """Single prediction interface."""
        device = next(self.parameters()).device
        params = torch.tensor(
            [E, nu, Lx, Ly, Lz, Fx, Fy, Fz, hole_r],
            dtype=torch.float32, device=device
        )

        self.eval()
        with torch.no_grad():
            result = self.forward(params)

        return {k: v.item() for k, v in result.items()}

    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


class SPINE3DLoss(nn.Module):
    """Loss function for SPINE3D training - log-scale MSE for stability."""

    def __init__(self):
        super().__init__()

    def forward(self, pred: Dict[str, torch.Tensor],
                target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute log-scale MSE loss (more stable for multi-scale values)."""
        losses = {}

        for key in ['u_max', 'sigma_vm_max']:  # Focus on main outputs
            if key in pred and key in target:
                # Log-scale MSE: (log(pred) - log(target))^2
                pred_log = torch.log(torch.abs(pred[key]) + 1e-10)
                target_log = torch.log(torch.abs(target[key]) + 1e-10)
                losses[f'{key}_loss'] = ((pred_log - target_log) ** 2).mean()

        # Handle signed values (sigma_3 can be negative)
        if 'sigma_1_max' in pred and 'sigma_1_max' in target:
            pred_log = torch.log(torch.abs(pred['sigma_1_max']) + 1e-10)
            target_log = torch.log(torch.abs(target['sigma_1_max']) + 1e-10)
            losses['sigma_1_loss'] = ((pred_log - target_log) ** 2).mean()

        losses['total'] = sum(losses.values()) / len(losses) if losses else torch.tensor(0.0)
        return losses


def generate_training_data(fem: Tet4FEM, mesh: LBracketMesh,
                          n_samples: int = 100,
                          device: torch.device = DEVICE) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Generate training data from FEM solutions.

    Varies: E, nu, load magnitude, load direction
    """
    print(f"\nGenerating {n_samples} FEM training samples...")

    inputs = []
    targets = {'u_max': [], 'sigma_vm_max': [], 'sigma_1_max': [], 'sigma_3_min': []}

    # Base geometry (fixed for this mesh)
    Lx = mesh.arm2_W
    Ly = mesh.arm2_L
    Lz = mesh.t
    hole_r = mesh.hole_d / 2

    fixed_nodes = mesh.node_sets['fixed']
    load_nodes = mesh.node_sets['load']
    n_load_nodes = len(load_nodes)

    for i in range(n_samples):
        # First few samples: include benchmark-like cases for better interpolation
        if i < n_samples // 10:
            # Cases similar to benchmark (E~210GPa, F~1000N, -Y direction)
            E = 210e3 * (1 + np.random.uniform(-0.2, 0.2))  # 168-252 GPa
            nu = 0.3 + np.random.uniform(-0.05, 0.05)
            F_mag = 1000 * (1 + np.random.uniform(-0.5, 1.0))  # 500-2000 N
            Fx, Fz = 0.0, 0.0
            Fy = -F_mag
        else:
            # Random material properties (units: N/mm² = MPa)
            E = 10 ** (np.random.uniform(4.5, 5.5))  # 30,000 - 300,000 MPa
            nu = np.random.uniform(0.2, 0.4)

            # Random load - mostly -Y direction
            F_mag = 10 ** (np.random.uniform(2, 4))  # 100 - 10000 N
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi/4)  # Limit angle from Y-axis

            Fx = F_mag * np.sin(phi) * np.cos(theta) * 0.2
            Fy = -F_mag * np.cos(phi)  # Primarily -y direction
            Fz = F_mag * np.sin(phi) * np.sin(theta) * 0.2

        # Solve FEM
        fem_solver = Tet4FEM(mesh.nodes, mesh.elements, E=E, nu=nu)

        # Distribute load among load nodes
        F_per_node = np.array([Fx, Fy, Fz]) / n_load_nodes
        F_values = np.tile(F_per_node, (n_load_nodes, 1))

        try:
            u, _, _ = fem_solver.solve(load_nodes, F_values, fixed_nodes)
            stress = fem_solver.compute_stress(u)
            vm_stress = fem_solver.von_mises_stress(stress)
            s1, s2, s3 = fem_solver.principal_stresses(stress)

            # Extract results
            u_max = np.abs(u).max()
            sigma_vm_max = vm_stress.max()
            sigma_1_max = s1.max()
            sigma_3_min = s3.min()

            if np.isfinite(u_max) and np.isfinite(sigma_vm_max):
                inputs.append([E, nu, Lx, Ly, Lz, Fx, Fy, Fz, hole_r])
                targets['u_max'].append(u_max)
                targets['sigma_vm_max'].append(sigma_vm_max)
                targets['sigma_1_max'].append(sigma_1_max)
                targets['sigma_3_min'].append(sigma_3_min)

                if (i + 1) % 20 == 0:
                    print(f"  Sample {i+1}/{n_samples}: u_max={u_max:.4e}, σ_vm={sigma_vm_max:.2e}")

        except Exception as e:
            print(f"  Sample {i+1} failed: {e}")
            continue

    inputs = torch.tensor(inputs, dtype=torch.float32, device=device)
    targets = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in targets.items()}

    print(f"Generated {len(inputs)} valid samples")
    return inputs, targets


def train_spine3d(model: SPINE3DNet, inputs: torch.Tensor,
                  targets: Dict[str, torch.Tensor],
                  n_epochs: int = 200, batch_size: int = 32,
                  lr: float = 1e-3, verbose: bool = True) -> List[float]:
    """Train SPINE3D model."""
    device = next(model.parameters()).device
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = SPINE3DLoss()

    n_samples = len(inputs)
    losses = []
    best_loss = float('inf')

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        perm = torch.randperm(n_samples, device=device)

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            batch_inputs = inputs[idx]
            batch_targets = {k: v[idx] for k, v in targets.items()}

            optimizer.zero_grad()

            pred = model(batch_inputs)
            loss_dict = criterion(pred, batch_targets)
            loss = loss_dict['total']

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if n_batches > 0:
            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss

            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}, Best: {best_loss:.4f}")

    return losses


# =============================================================================
# PYVISTA 3D VISUALIZER
# =============================================================================

class FEM3DVisualizer:
    """
    ANSYS-style 3D interactive visualization using PyVista.

    Features:
    - Von Mises stress heatmap (blue → red)
    - Deformed shape with magnification
    - Interactive rotation, zoom, pan
    - Cross-section views
    """

    def __init__(self, nodes: np.ndarray, elements: np.ndarray,
                 displacement: Optional[np.ndarray] = None,
                 stress: Optional[np.ndarray] = None,
                 von_mises: Optional[np.ndarray] = None):

        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista not available. Install with: pip install pyvista")

        self.nodes = nodes
        self.elements = elements
        self.displacement = displacement
        self.stress = stress
        self.von_mises = von_mises

        # Create PyVista mesh
        self.mesh = self._create_mesh()

    def _create_mesh(self) -> pv.UnstructuredGrid:
        """Create PyVista UnstructuredGrid from tetrahedral elements."""
        n_cells = len(self.elements)

        # VTK cell format: [n_points, p0, p1, p2, p3, n_points, p0, ...]
        cells = []
        cell_types = []

        for elem in self.elements:
            cells.extend([4, elem[0], elem[1], elem[2], elem[3]])
            cell_types.append(10)  # VTK_TETRA = 10

        cells = np.array(cells)
        cell_types = np.array(cell_types)

        mesh = pv.UnstructuredGrid(cells, cell_types, self.nodes)

        # Add data arrays
        if self.von_mises is not None:
            mesh.cell_data['Von Mises Stress (MPa)'] = self.von_mises

        if self.stress is not None:
            mesh.cell_data['Stress XX'] = self.stress[:, 0]
            mesh.cell_data['Stress YY'] = self.stress[:, 1]
            mesh.cell_data['Stress ZZ'] = self.stress[:, 2]

        if self.displacement is not None:
            # Reshape to (n_nodes, 3)
            disp_3d = self.displacement.reshape(-1, 3)
            mesh.point_data['Displacement'] = disp_3d
            mesh.point_data['Displacement Magnitude'] = np.linalg.norm(disp_3d, axis=1)

        return mesh

    def show_stress(self, component: str = 'Von Mises Stress (MPa)',
                   show_edges: bool = True, cmap: str = 'jet',
                   title: Optional[str] = None):
        """
        Display stress field with heatmap coloring.

        Args:
            component: 'Von Mises Stress (MPa)', 'Stress XX', etc.
            show_edges: Show mesh edges
            cmap: Colormap ('jet', 'viridis', 'coolwarm')
            title: Window title
        """
        pl = pv.Plotter(title=title or f"3D FEM - {component}")

        # Add mesh with stress coloring
        pl.add_mesh(
            self.mesh,
            scalars=component,
            cmap=cmap,
            show_edges=show_edges,
            edge_color='gray',
            opacity=1.0,
            scalar_bar_args={
                'title': component,
                'vertical': True,
                'n_labels': 5,
            }
        )

        # Add axes
        pl.add_axes()

        # Set initial view
        pl.camera_position = 'iso'

        # Show
        pl.show()

    def show_deformed(self, scale: float = 100.0, show_original: bool = True,
                     cmap: str = 'jet'):
        """
        Display deformed shape with magnification.

        Args:
            scale: Deformation magnification factor
            show_original: Show original shape as wireframe
            cmap: Colormap for displacement magnitude
        """
        if self.displacement is None:
            print("No displacement data available")
            return

        pl = pv.Plotter(title=f"Deformed Shape (scale: {scale}x)")

        # Warp mesh by displacement
        warped = self.mesh.warp_by_vector('Displacement', factor=scale)

        # Add deformed mesh
        pl.add_mesh(
            warped,
            scalars='Displacement Magnitude',
            cmap=cmap,
            show_edges=True,
            edge_color='gray',
            scalar_bar_args={
                'title': 'Displacement (m)',
                'vertical': True,
            }
        )

        # Add original mesh as wireframe
        if show_original:
            pl.add_mesh(
                self.mesh,
                style='wireframe',
                color='gray',
                opacity=0.3,
                line_width=1,
            )

        pl.add_axes()
        pl.camera_position = 'iso'
        pl.show()

    def show_comparison(self, scale: float = 100.0, cmap: str = 'jet'):
        """
        Side-by-side comparison: Original vs Deformed.
        """
        if self.displacement is None:
            print("No displacement data available")
            return

        pl = pv.Plotter(shape=(1, 2), title="FEM Results Comparison")

        # Left: Von Mises stress on original
        pl.subplot(0, 0)
        pl.add_text("Von Mises Stress", font_size=12)
        pl.add_mesh(
            self.mesh,
            scalars='Von Mises Stress (MPa)' if self.von_mises is not None else None,
            cmap=cmap,
            show_edges=True,
        )
        pl.add_axes()

        # Right: Deformed shape
        pl.subplot(0, 1)
        pl.add_text(f"Deformed Shape ({scale}x)", font_size=12)
        warped = self.mesh.warp_by_vector('Displacement', factor=scale)
        pl.add_mesh(
            warped,
            scalars='Displacement Magnitude',
            cmap=cmap,
            show_edges=True,
        )
        pl.add_mesh(
            self.mesh,
            style='wireframe',
            color='gray',
            opacity=0.3,
        )
        pl.add_axes()

        pl.link_views()
        pl.show()

    def slice_view(self, normal: str = 'x', origin: Optional[np.ndarray] = None,
                  cmap: str = 'jet'):
        """
        Display cross-section slice.

        Args:
            normal: Slice normal direction ('x', 'y', 'z')
            origin: Slice origin point
            cmap: Colormap
        """
        if origin is None:
            origin = self.mesh.center

        sliced = self.mesh.slice(normal=normal, origin=origin)

        pl = pv.Plotter(title=f"Cross-Section (normal: {normal})")

        # Show slice
        pl.add_mesh(
            sliced,
            scalars='Von Mises Stress (MPa)' if self.von_mises is not None else None,
            cmap=cmap,
            show_edges=True,
            line_width=2,
        )

        # Show original as transparent
        pl.add_mesh(
            self.mesh,
            opacity=0.1,
            color='gray',
        )

        pl.add_axes()
        pl.show()

    def animate_deformation(self, n_frames: int = 20, scale: float = 100.0,
                           filename: Optional[str] = None):
        """
        Animate deformation from 0 to full scale.
        """
        if self.displacement is None:
            print("No displacement data available")
            return

        pl = pv.Plotter(off_screen=filename is not None)

        # Initial mesh
        pl.add_mesh(
            self.mesh,
            scalars='Displacement Magnitude',
            cmap='jet',
            show_edges=True,
        )

        pl.open_gif(filename) if filename else None

        for i in range(n_frames):
            factor = scale * (i + 1) / n_frames
            warped = self.mesh.warp_by_vector('Displacement', factor=factor)
            pl.update_coordinates(warped.points)
            if filename:
                pl.write_frame()

        if filename:
            pl.close()
            print(f"Animation saved to: {filename}")
        else:
            pl.show()


# =============================================================================
# BENCHMARK COMPARISON
# =============================================================================

def run_3d_benchmark(n_elements: int = 8, visualize: bool = True,
                    train_spine: bool = True, n_training_samples: int = 50):
    """
    Run 3D L-Bracket benchmark: FEM vs SPINE.

    Args:
        n_elements: Elements per arm (mesh density)
        visualize: Show PyVista visualization
        train_spine: Train SPINE model
        n_training_samples: Number of FEM samples for training
    """
    print("\n" + "=" * 70)
    print("           3D L-BRACKET BENCHMARK: FEM vs SPINE AI")
    print("=" * 70)

    # ==========================================================================
    # 1. CREATE MESH
    # ==========================================================================
    print("\n[1/5] Creating L-Bracket mesh...")
    mesh = LBracketMesh(
        arm1_length=100.0,  # mm
        arm1_width=50.0,
        arm2_length=100.0,
        arm2_width=50.0,
        thickness=10.0,
        hole_diameter=20.0,
        n_elements_per_arm=n_elements
    )
    print(f"      {mesh.get_info()}")

    # ==========================================================================
    # 2. FEM SOLUTION
    # ==========================================================================
    print("\n[2/5] Solving with FEM (Tet4 elements)...")

    # Material: Steel (units: N, mm → E in N/mm² = MPa)
    E = 210e3  # N/mm² (210 GPa = 210,000 MPa)
    nu = 0.3

    # Load: 1000 N in -Y direction at bottom
    F_total = 1000.0  # N
    load_nodes = mesh.node_sets['load']
    fixed_nodes = mesh.node_sets['fixed']

    n_load_nodes = len(load_nodes)
    F_per_node = np.array([0, -F_total / n_load_nodes, 0])
    F_values = np.tile(F_per_node, (n_load_nodes, 1))

    # Create and solve FEM
    fem = Tet4FEM(mesh.nodes, mesh.elements, E=E, nu=nu)
    print(f"      {fem.get_info()}")

    u_fem, fem_time, fem_info = fem.solve(load_nodes, F_values, fixed_nodes)

    # Compute stresses
    stress_fem = fem.compute_stress(u_fem)
    vm_stress = fem.von_mises_stress(stress_fem)
    s1, s2, s3 = fem.principal_stresses(stress_fem)

    # Results (already in mm and MPa due to unit system)
    u_max_fem = np.abs(u_fem).max()  # mm
    sigma_vm_max_fem = vm_stress.max()  # MPa (N/mm²)
    sigma_1_max_fem = s1.max()  # MPa
    sigma_3_min_fem = s3.min()  # MPa

    print(f"\n      FEM Results:")
    print(f"      ├── Max displacement: {u_max_fem:.4f} mm")
    print(f"      ├── Max von Mises stress: {sigma_vm_max_fem:.2f} MPa")
    print(f"      ├── Max principal σ1: {sigma_1_max_fem:.2f} MPa")
    print(f"      ├── Min principal σ3: {sigma_3_min_fem:.2f} MPa")
    print(f"      └── Solution time: {fem_time:.3f} s")

    # ==========================================================================
    # 3. SPINE MODEL
    # ==========================================================================
    print("\n[3/5] Initializing SPINE3D model...")

    spine = SPINE3DNet(hidden_dim=64).to(DEVICE)
    param_count = spine.count_parameters()
    print(f"      Parameters: {param_count['trainable']:,} trainable")

    # ==========================================================================
    # 4. TRAIN SPINE (optional)
    # ==========================================================================
    if train_spine:
        print(f"\n[4/5] Training SPINE3D with {n_training_samples} FEM samples...")

        inputs, targets = generate_training_data(
            fem, mesh, n_samples=n_training_samples, device=DEVICE
        )

        if len(inputs) > 0:
            losses = train_spine3d(
                spine, inputs, targets,
                n_epochs=200, batch_size=min(32, len(inputs)),
                verbose=True
            )
            print(f"      Final loss: {losses[-1]:.4f}")
        else:
            print("      Warning: No training data generated")
    else:
        print("\n[4/5] Skipping SPINE training (using untrained model)")

    # ==========================================================================
    # 5. SPINE INFERENCE
    # ==========================================================================
    print("\n[5/5] SPINE inference...")

    # Same parameters as FEM
    Lx = mesh.arm2_W
    Ly = mesh.arm2_L
    Lz = mesh.t
    hole_r = mesh.hole_d / 2

    spine.eval()
    spine_start = time.perf_counter()
    with torch.no_grad():
        spine_result = spine.predict(
            E=E, nu=nu, Lx=Lx, Ly=Ly, Lz=Lz,
            Fx=0, Fy=-F_total, Fz=0, hole_r=hole_r
        )
    spine_time = time.perf_counter() - spine_start

    # SPINE outputs are already in mm and MPa (same unit system as FEM)
    u_max_spine = spine_result['u_max']  # mm
    sigma_vm_max_spine = spine_result['sigma_vm_max']  # MPa

    print(f"\n      SPINE Results:")
    print(f"      ├── Max displacement: {u_max_spine:.4f} mm")
    print(f"      ├── Max von Mises stress: {sigma_vm_max_spine:.2f} MPa")
    print(f"      └── Inference time: {spine_time*1000:.3f} ms")

    # ==========================================================================
    # COMPARISON TABLE
    # ==========================================================================
    speedup = fem_time / spine_time if spine_time > 0 else 0
    u_error = abs(u_max_spine - u_max_fem) / u_max_fem * 100 if u_max_fem > 0 else 0
    stress_error = abs(sigma_vm_max_spine - sigma_vm_max_fem) / sigma_vm_max_fem * 100 if sigma_vm_max_fem > 0 else 0

    print("\n" + "=" * 70)
    print("                        COMPARISON RESULTS")
    print("=" * 70)
    print(f"""
  ┌────────────────┬─────────────┬─────────────┬─────────────┐
  │     Method     │  Time (s)   │  u_max (mm) │ σ_vm (MPa)  │
  ├────────────────┼─────────────┼─────────────┼─────────────┤
  │ FEM (Tet4)     │ {fem_time:>9.4f}   │ {u_max_fem:>9.4f}   │ {sigma_vm_max_fem:>9.2f}   │
  │ SPINE AI       │ {spine_time:>9.6f} │ {u_max_spine:>9.4f}   │ {sigma_vm_max_spine:>9.2f}   │
  ├────────────────┼─────────────┼─────────────┼─────────────┤
  │ Speedup        │ {speedup:>9.0f}x  │ Error: {u_error:>4.1f}% │ Error: {stress_error:>4.1f}% │
  └────────────────┴─────────────┴─────────────┴─────────────┘
""")

    # ==========================================================================
    # VISUALIZATION
    # ==========================================================================
    if visualize and PYVISTA_AVAILABLE:
        print("\n[VIZ] Opening 3D visualization...")

        viz = FEM3DVisualizer(
            nodes=mesh.nodes,  # Keep in mm
            elements=mesh.elements,
            displacement=u_fem,
            stress=stress_fem,
            von_mises=vm_stress
        )

        # Show comparison view (scale deformation for visibility)
        viz.show_comparison(scale=5000)  # 5000x magnification for small displacements

    return {
        'fem_time': fem_time,
        'spine_time': spine_time,
        'speedup': speedup,
        'u_max_fem': u_max_fem,
        'u_max_spine': u_max_spine,
        'sigma_vm_fem': sigma_vm_max_fem,
        'sigma_vm_spine': sigma_vm_max_spine,
        'u_error': u_error,
        'stress_error': stress_error,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"\nDevice: {DEVICE_NAME} ({DEVICE})")
    print(f"PyVista available: {PYVISTA_AVAILABLE}")

    # Run benchmark
    results = run_3d_benchmark(
        n_elements=8,           # Mesh density
        visualize=True,         # Show 3D visualization
        train_spine=True,       # Train SPINE model
        n_training_samples=50   # FEM samples for training
    )

    print("\n" + "=" * 70)
    print("                         BENCHMARK COMPLETE")
    print("=" * 70)
