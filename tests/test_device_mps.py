"""
Device tutarlılığı regresyon testleri (commit 8ff029f'in koruması).

Bug geçmişi: BucklingNeuron.mode_shape grid'i global DEVICE'a (Apple
Silicon'da MPS) üretiyordu; model buffer'ları CPU'da kaldığından README
quick-start MPS'li her Mac'te çöküyordu. Düzeltme: grid, modülün kendi
parametresinin (load_scale) device'ını izler.

Bu testler CPU'da her makinede koşar; MPS eşitlik testi yalnız Apple
Silicon'da aktiftir.

Çalıştırma:
    PYTHONPATH=. python3 tests/test_device_mps.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from spine import SPINE, MaterialProperties

MAT = MaterialProperties(E=200e9, nu=0.3, h=0.005)


def _new_model():
    return SPINE(resolution=64, Lx=1.0, Ly=0.5, material=MAT)


def test_default_model_stays_on_cpu():
    """Taşınmamış model CPU'da kalmalı: mode_shape çıktısı da CPU olmalı.

    Regresyon: mode_shape tekrar global DEVICE'a dönerse bu test MPS'li
    makinede kırılır (w mps:0 olur) — 8ff029f öncesi bug tam buydu.
    """
    model = _new_model()
    w = model.get_mode_shape(m=1, n=1)
    assert w.device.type == "cpu", f"varsayılan model CPU olmalı, w {w.device}'ta"

    # forward uçtan uca çalışmalı (eski bug burada RuntimeError veriyordu)
    state = model.forward(w)
    assert torch.isfinite(state.M_xx).all()
    print(f"[Device] varsayılan akış CPU'da uçtan uca — OK")


def test_mps_transfer_and_equivalence():
    """.to('mps') sonrası mode_shape MPS'te üretilmeli ve sonuçlar CPU ile
    eşleşmeli. MPS yoksa sessizce atlanır."""
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        print("[Device] MPS yok — eşitlik testi atlandı")
        return

    model_cpu = _new_model()
    w_cpu = model_cpu.get_mode_shape(m=1, n=1)
    state_cpu = model_cpu.forward(w_cpu)

    model_mps = _new_model().to("mps")
    w_mps = model_mps.get_mode_shape(m=1, n=1)
    assert w_mps.device.type == "mps", f"mode_shape modülü izlemeli, w {w_mps.device}'ta"
    state_mps = model_mps.forward(w_mps)

    d_cpu = state_cpu.max_displacement().item()
    d_mps = state_mps.max_displacement().item()
    assert abs(d_cpu - d_mps) / abs(d_cpu) < 1e-4, f"CPU={d_cpu} vs MPS={d_mps}"

    m_cpu = state_cpu.M_xx.abs().max().item()
    m_mps = state_mps.M_xx.abs().max().cpu().item()
    assert abs(m_cpu - m_mps) / m_cpu < 1e-3, f"M_xx CPU={m_cpu} vs MPS={m_mps}"
    print(f"[Device] MPS eşitliği: w_max fark {abs(d_cpu-d_mps):.1e} — OK")


if __name__ == "__main__":
    test_default_model_stays_on_cpu()
    test_mps_transfer_and_equivalence()
    print("\n=== TÜM 2 TEST GEÇTİ ===")
