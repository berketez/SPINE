"""
RodNeuron (eksenel çubuk) regresyon + doğrulama testleri.

Hocanın 3 Hibbeler eksenel problemi + manufactured solution + enerji dengesi
+ sınır koşulu + grid yakınsaması.

Çalıştırma:
    python3 tests/test_rod_axial.py
"""

import math
import os
import sys

import torch

# Repo kökünü dinamik bul (tests/ dizininin bir üstü) — hardcoded yol yok.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spine import RodNeuron

# --- Ortak parametreler (Berke onayı: çelik γ=77 kN/m³, E=200 GPa) ---
G = 9.81
GAMMA_STEEL = 77000.0            # N/m³
RHO_STEEL = GAMMA_STEEL / G      # kg/m³
E_STEEL = 200e9                  # Pa
E_ALU = 70e9                     # Pa
D_ROD = 0.012                    # m (∅12 mm)
A_ROD = math.pi / 4 * D_ROD**2   # ≈ 1.13097e-4 m²


def rel_err(a, b):
    return abs(a - b) / abs(b)


# ---------------------------------------------------------------------------
# Problem 1 — Asılı koni, öz-ağırlık (Hibbeler 4-9): δ = γL²/(6E)
# ---------------------------------------------------------------------------

def test_problem1_cone_self_weight():
    L, r0 = 4.0, 0.2
    rod = RodNeuron(length=L, resolution=2001)
    A_cone = lambda x: math.pi * r0**2 * (1.0 - x / L) ** 2
    res = rod.solve(E=E_STEEL, A=A_cone, self_weight=True, rho=RHO_STEEL, g=G,
                    fixed_end="left")
    delta = res["tip_displacement"].item()
    delta_ana = GAMMA_STEEL * L**2 / (6 * E_STEEL)
    e = rel_err(delta, delta_ana)
    assert e < 1e-3, f"Koni δ hatası %{e*100:.3f} (>0.1%)"

    # r0'dan bağımsızlık (analitik özellik): r0 iki katına çıksa δ değişmemeli
    rod2 = RodNeuron(length=L, resolution=2001)
    A_cone2 = lambda x: math.pi * (2 * r0) ** 2 * (1.0 - x / L) ** 2
    d2 = rod2.solve(E=E_STEEL, A=A_cone2, self_weight=True, rho=RHO_STEEL, g=G,
                    fixed_end="left")["tip_displacement"].item()
    assert rel_err(d2, delta) < 1e-6, "Koni δ, r0'dan bağımsız olmalı"
    print(f"[P1] Koni δ={delta*1e6:.4f} µm (analitik {delta_ana*1e6:.4f}), "
          f"hata %{e*100:.4f}, r0-bağımsızlık OK")


# ---------------------------------------------------------------------------
# Problem 2 — Çelik(3m)+Alüminyum(2m) çubuk (Hibbeler 4-5)
# ---------------------------------------------------------------------------

def test_problem2_steel_aluminum():
    L = 5.0
    rod = RodNeuron(length=L, resolution=4001)
    E_field = lambda x: torch.where(x < 3.0, torch.full_like(x, E_STEEL),
                                    torch.full_like(x, E_ALU))
    res = rod.solve(E=E_field, A=A_ROD, point_loads=[(5.0, 18e3), (3.0, -6e3)],
                    fixed_end="left")
    x, u = res["x"], res["u"]
    iB = int(torch.argmin((x - 3.0).abs()))
    dB, dA = u[iB].item(), res["tip_displacement"].item()
    dB_ana = 12e3 * 3.0 / (A_ROD * E_STEEL)
    dA_ana = dB_ana + 18e3 * 2.0 / (A_ROD * E_ALU)
    eB, eA = rel_err(dB, dB_ana), rel_err(dA, dA_ana)
    assert eB < 1e-3, f"δ_B hatası %{eB*100:.3f}"
    assert eA < 1e-3, f"δ_A hatası %{eA*100:.3f}"
    # Kitap cevabı (yuvarlanmış): 1.59 mm / 6.14 mm
    assert abs(dB * 1e3 - 1.59) < 0.02
    assert abs(dA * 1e3 - 6.14) < 0.02
    print(f"[P2] δ_B={dB*1e3:.4f} mm (%{eB*100:.4f}), "
          f"δ_A={dA*1e3:.4f} mm (%{eA*100:.4f}) — kitap 1.59/6.14 OK")


# ---------------------------------------------------------------------------
# Problem 3 — P + öz-ağırlık (Hibbeler 4-13): δ = PL/(AE) + γL²/(2E)
# ---------------------------------------------------------------------------

def test_problem3_P_plus_self_weight():
    L, P = 4.0, 18e3
    rod = RodNeuron(length=L, resolution=2001)
    res = rod.solve(E=E_STEEL, A=A_ROD, self_weight=True, rho=RHO_STEEL, g=G,
                    point_loads=[(L, P)], fixed_end="left")
    delta = res["tip_displacement"].item()
    delta_ana = P * L / (A_ROD * E_STEEL) + GAMMA_STEEL * L**2 / (2 * E_STEEL)
    e = rel_err(delta, delta_ana)
    assert e < 1e-3, f"δ hatası %{e*100:.3f}"
    # Süperpozisyon: P katkısı + öz-ağırlık katkısı ayrı ayrı doğru
    d_P = P * L / (A_ROD * E_STEEL)
    d_sw = GAMMA_STEEL * L**2 / (2 * E_STEEL)
    print(f"[P3] δ={delta*1e3:.4f} mm = PL/AE({d_P*1e3:.4f}) + γL²/2E({d_sw*1e6:.3f}µm), "
          f"hata %{e*100:.4f}")


# ---------------------------------------------------------------------------
# Manufactured solution — u(x)=sin(πx/2L), f geri hesaplanır
# ---------------------------------------------------------------------------

def test_manufactured_solution():
    L, EA = 2.0, 1.0e6
    rod = RodNeuron(length=L, resolution=4001)
    k = math.pi / (2 * L)             # u(0)=0, u'(L)=0 → serbest uç N(L)=0
    A_m = 1.0
    E_m = EA / A_m
    f_man = lambda x: EA * k**2 * torch.sin(k * x)   # f = -d/dx(EA u')
    res = rod.solve(E=E_m, A=A_m, distributed_load=f_man, fixed_end="left")
    u_num, xm = res["u"], res["x"]
    u_exact = torch.sin(k * xm)
    l2 = (torch.linalg.norm(u_num - u_exact) /
          torch.linalg.norm(u_exact)).item()
    assert l2 < 1e-6, f"Manufactured L2 hatası {l2:.2e}"
    print(f"[MMS] u=sin(πx/2L) L2 rel hata = {l2:.3e} — OK")


# ---------------------------------------------------------------------------
# Sınır koşulu — ankastre uçta u=0 makine hassasiyeti
# ---------------------------------------------------------------------------

def test_boundary_condition():
    rod = RodNeuron(length=3.0, resolution=1001)
    res = rod.solve(E=E_STEEL, A=A_ROD, point_loads=[(3.0, 10e3)],
                    fixed_end="left")
    assert abs(res["u"][0].item()) < 1e-15, "u(0)=0 sağlanmıyor"
    # Serbest uçta N = uygulanan uç yükü
    assert rel_err(res["N"][-1].item(), 10e3) < 1e-9, "N(L)=P sağlanmıyor"
    print(f"[BC] u(0)={res['u'][0].item():.2e}, N(L)={res['N'][-1].item():.1f} N — OK")


# ---------------------------------------------------------------------------
# Clapeyron enerji dengesi — W_dış = 2·U_iç (lineer elastik)
# ---------------------------------------------------------------------------

def test_clapeyron_energy_balance():
    L = 5.0
    rod = RodNeuron(length=L, resolution=4001)
    E_field = lambda x: torch.where(x < 3.0, torch.full_like(x, E_STEEL),
                                    torch.full_like(x, E_ALU))
    pl = [(5.0, 18e3), (3.0, -6e3)]
    res = rod.solve(E=E_field, A=A_ROD, point_loads=pl, fixed_end="left")
    x, u, N, EA = res["x"], res["u"], res["N"], res["EA"]

    # Dış iş (tam, faktörsüz): Σ P_i·u_i
    W_ext = 0.0
    for xi, Pi in pl:
        i = int(torch.argmin((x - xi).abs()))
        W_ext += Pi * u[i].item()
    # 2·U_iç = ∫ N²/(EA) dx
    integrand = N**2 / EA
    two_U = torch.trapezoid(integrand, x).item()
    e = rel_err(W_ext, two_U)
    assert e < 1e-3, f"Clapeyron dengesi tutmuyor: W_ext={W_ext:.4f}, 2U={two_U:.4f}, hata %{e*100:.3f}"
    print(f"[Enerji] W_dış={W_ext:.4f} J, 2·U_iç={two_U:.4f} J, fark %{e*100:.4f} — OK")


# ---------------------------------------------------------------------------
# Grid yakınsaması — trapez O(dx²), hata düzgün düşer
# ---------------------------------------------------------------------------

def test_grid_convergence():
    L, r0 = 4.0, 0.2
    A_cone = lambda x: math.pi * r0**2 * (1.0 - x / L) ** 2
    delta_ana = GAMMA_STEEL * L**2 / (6 * E_STEEL)
    errs = []
    for res_n in (101, 201, 401, 801):
        rod = RodNeuron(length=L, resolution=res_n)
        d = rod.solve(E=E_STEEL, A=A_cone, self_weight=True, rho=RHO_STEEL,
                      g=G, fixed_end="left")["tip_displacement"].item()
        errs.append(rel_err(d, delta_ana))
    # Hata monoton azalmalı
    for i in range(1, len(errs)):
        assert errs[i] <= errs[i-1] * 1.05, f"Yakınsama bozuk: {errs}"
    print(f"[Yakınsama] hata 101→801: "
          f"{', '.join(f'{e:.2e}' for e in errs)} — OK")


# ---------------------------------------------------------------------------
# Aynalı sınır — fixed_end='right' sabit çubukta aynı büyüklüğü vermeli
# ---------------------------------------------------------------------------

def test_fixed_end_right_mirror():
    L, P = 2.0, 12e3
    # fixed=left: P uçta (x=L)
    rl = RodNeuron(length=L, resolution=1001)
    dl = rl.solve(E=E_STEEL, A=A_ROD, point_loads=[(L, P)],
                  fixed_end="left")["tip_displacement"].item()
    # fixed=right: serbest uç solda (x=0), P orada (çekme +)
    rr = RodNeuron(length=L, resolution=1001)
    dr = rr.solve(E=E_STEEL, A=A_ROD, point_loads=[(0.0, P)],
                  fixed_end="right")["tip_displacement"].item()
    pl_ae = P * L / (A_ROD * E_STEEL)
    assert rel_err(dl, pl_ae) < 1e-6
    assert rel_err(abs(dr), pl_ae) < 1e-6
    print(f"[Ayna] fixed=left δ={dl*1e3:.4f} mm, fixed=right |δ|={abs(dr)*1e3:.4f} mm, "
          f"PL/AE={pl_ae*1e3:.4f} — OK")


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_all():
    tests = [
        test_problem1_cone_self_weight,
        test_problem2_steel_aluminum,
        test_problem3_P_plus_self_weight,
        test_manufactured_solution,
        test_boundary_condition,
        test_clapeyron_energy_balance,
        test_grid_convergence,
        test_fixed_end_right_mirror,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {t.__name__}: {exc}")
    print()
    if failed:
        print(f"=== {failed}/{len(tests)} test KALDI ===")
        sys.exit(1)
    print(f"=== TÜM {len(tests)} TEST GEÇTİ ===")


if __name__ == "__main__":
    run_all()
