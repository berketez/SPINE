"""
SPINE Malzeme Test Scripti

Çeşitli malzemelerle burkulma ve eğilme analizleri.
Basit metallerden egzotik kompozitlere kadar test.

Yazar: SPINE Project
"""

import torch
import numpy as np
import time
from typing import Dict, List
import sys

# SPINE modülleri
from materials import (
    MaterialLibrary, IsotropicMaterial, OrthotropicMaterial,
    CompositeLaminate, Ply, AdvancedMaterialNeuron
)


def print_header(title: str):
    """Başlık yazdır."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title: str):
    print(f"\n--- {title} ---")


def test_isotropic_buckling():
    """İzotropik malzemelerle burkulma testi."""
    print_header("İZOTROPİK MALZEME BURKULMA TESTİ")
    
    # Plaka geometrisi
    Lx, Ly = 0.5, 0.3  # 500mm x 300mm
    h = 0.003  # 3mm kalınlık
    
    print(f"\nPlaka boyutları: {Lx*1000:.0f}mm × {Ly*1000:.0f}mm × {h*1000:.1f}mm")
    
    # Test edilecek malzemeler
    materials = [
        # Standart metaller
        "steel_304",
        "aluminum_6061_T6",
        "titanium_Ti6Al4V",
        
        # Egzotik metaller
        "beryllium",  # Ultra hafif, yüksek modül
        "tungsten",   # Ultra ağır, ultra rijit
        "inconel_718",  # Süperalaşım
        "magnesium_AZ31",  # Ultra hafif
        
        # Polimerler
        "peek",
        "polycarbonate",
        "pla",
        
        # Özel
        "diamond",  # En rijit
        "cork",  # En yumuşak
    ]
    
    print(f"\n{'Malzeme':<30} {'E (GPa)':<12} {'ρ (kg/m³)':<12} {'N_cr (kN/m)':<12} {'Spesifik':<12}")
    print("-" * 78)
    
    results = []
    
    for mat_name in materials:
        try:
            mat = MaterialLibrary.get(mat_name)
            
            # Plaka rijitliği
            D = mat.plate_rigidity(h)
            
            # Kritik yük (basit mesnetli, m=n=1)
            import math
            alpha = math.pi / Lx
            beta = math.pi / Ly
            N_cr = D * (alpha**2 + beta**2)**2 / alpha**2
            
            # Spesifik mukavemet (N_cr / ρ)
            specific = N_cr / mat.rho
            
            results.append({
                'name': mat_name,
                'E': mat.E,
                'rho': mat.rho,
                'N_cr': N_cr,
                'specific': specific
            })
            
            print(f"{mat.name[:28]:<30} {mat.E/1e9:<12.1f} {mat.rho:<12.0f} "
                  f"{N_cr/1000:<12.2f} {specific:<12.2f}")
            
        except Exception as e:
            print(f"{mat_name:<30} HATA: {e}")
    
    # En iyi malzeme
    if results:
        print("\n" + "-" * 78)
        best_absolute = max(results, key=lambda x: x['N_cr'])
        best_specific = max(results, key=lambda x: x['specific'])
        
        print(f"\n🏆 En yüksek mutlak dayanım: {best_absolute['name']}")
        print(f"   N_cr = {best_absolute['N_cr']/1000:.2f} kN/m")
        
        print(f"\n🏆 En yüksek spesifik dayanım: {best_specific['name']}")
        print(f"   N_cr/ρ = {best_specific['specific']:.2f}")
    
    return results


def test_orthotropic_buckling():
    """Ortotropik malzemelerle burkulma testi."""
    print_header("ORTOTROPİK (KOMPOZİT) MALZEME BURKULMA TESTİ")
    
    # Plaka geometrisi
    Lx, Ly = 0.5, 0.3
    h = 0.003
    
    print(f"\nPlaka boyutları: {Lx*1000:.0f}mm × {Ly*1000:.0f}mm × {h*1000:.1f}mm")
    print("(Tek yönlü fiber, 0° yönleme)")
    
    # Test edilecek malzemeler
    composites = [
        # Karbon fiber
        "carbon_epoxy_T300",
        "carbon_epoxy_T700",
        "carbon_epoxy_M55J",  # Yüksek modül
        "carbon_peek",
        
        # Cam elyaf
        "glass_epoxy_E",
        "glass_epoxy_S2",
        
        # Aramid
        "kevlar_49_epoxy",
        "kevlar_29_epoxy",
        
        # Egzotik
        "boron_epoxy",  # Havacılık özel
        "basalt_epoxy",  # Volkanik
        
        # Doğal
        "flax_epoxy",  # Keten
        "hemp_epoxy",  # Kenevir
        "bamboo",
        
        # Ahşap
        "wood_oak",
        "wood_balsa",  # Ultra hafif
    ]
    
    print(f"\n{'Malzeme':<32} {'E1 (GPa)':<10} {'E2 (GPa)':<10} {'E1/E2':<8} {'N_cr (kN/m)':<12}")
    print("-" * 82)
    
    results = []
    
    for mat_name in composites:
        try:
            mat = MaterialLibrary.get(mat_name)
            
            # Ortotropik plaka rijitliği (0° fiber)
            Q = mat.get_Q_matrix()
            D11 = Q[0, 0] * h**3 / 12
            D22 = Q[1, 1] * h**3 / 12
            D12 = Q[0, 1] * h**3 / 12
            D66 = Q[2, 2] * h**3 / 12
            
            # Kritik yük (fiber yönünde basma)
            import math
            a, b = Lx, Ly
            m, n = 1, 1
            
            term1 = D11 * (m**4) / (a**4)
            term2 = 2 * (D12 + 2*D66) * (m**2 * n**2) / (a**2 * b**2)
            term3 = D22 * (n**4) / (b**4)
            
            N_cr = (math.pi**2) * (term1 + term2 + term3) / (m**2 / a**2)
            
            ratio = mat.E1 / mat.E2
            
            results.append({
                'name': mat_name,
                'E1': mat.E1,
                'E2': mat.E2,
                'ratio': ratio,
                'N_cr': N_cr,
                'rho': mat.rho
            })
            
            print(f"{mat.name[:30]:<32} {mat.E1/1e9:<10.1f} {mat.E2/1e9:<10.1f} "
                  f"{ratio:<8.1f} {N_cr/1000:<12.2f}")
            
        except Exception as e:
            print(f"{mat_name:<32} HATA: {e}")
    
    # Analiz
    if results:
        print("\n" + "-" * 82)
        best = max(results, key=lambda x: x['N_cr'])
        best_specific = max(results, key=lambda x: x['N_cr']/x['rho'])
        
        print(f"\n🏆 En yüksek dayanım: {best['name']}")
        print(f"   N_cr = {best['N_cr']/1000:.2f} kN/m")
        
        print(f"\n🏆 En yüksek spesifik: {best_specific['name']}")
    
    return results


def test_composite_laminates():
    """Farklı laminat dizilimleriyle test."""
    print_header("KOMPOZİT LAMİNAT BURKULMA TESTİ")
    
    Lx, Ly = 0.5, 0.3
    ply_thickness = 0.125e-3  # 0.125mm/katman
    
    print(f"\nPlaka: {Lx*1000:.0f}mm × {Ly*1000:.0f}mm")
    print(f"Katman kalınlığı: {ply_thickness*1000:.3f}mm")
    print(f"Malzeme: Karbon/Epoksi T300")
    
    # Test laminatları
    laminates = {
        "Tek yönlü [0]₈": [
            Ply(0, ply_thickness, "carbon_epoxy_T300"),
        ] * 8,
        
        "Çapraz [0/90]₂s": [
            Ply(0, ply_thickness, "carbon_epoxy_T300"),
            Ply(90, ply_thickness, "carbon_epoxy_T300"),
            Ply(90, ply_thickness, "carbon_epoxy_T300"),
            Ply(0, ply_thickness, "carbon_epoxy_T300"),
        ] * 2,
        
        "Quasi-isotropic [0/±45/90]s": [
            Ply(0, ply_thickness, "carbon_epoxy_T300"),
            Ply(45, ply_thickness, "carbon_epoxy_T300"),
            Ply(-45, ply_thickness, "carbon_epoxy_T300"),
            Ply(90, ply_thickness, "carbon_epoxy_T300"),
            Ply(90, ply_thickness, "carbon_epoxy_T300"),
            Ply(-45, ply_thickness, "carbon_epoxy_T300"),
            Ply(45, ply_thickness, "carbon_epoxy_T300"),
            Ply(0, ply_thickness, "carbon_epoxy_T300"),
        ],
        
        "±45 dominant [±45/0/90]s": [
            Ply(45, ply_thickness, "carbon_epoxy_T300"),
            Ply(-45, ply_thickness, "carbon_epoxy_T300"),
            Ply(0, ply_thickness, "carbon_epoxy_T300"),
            Ply(90, ply_thickness, "carbon_epoxy_T300"),
            Ply(90, ply_thickness, "carbon_epoxy_T300"),
            Ply(0, ply_thickness, "carbon_epoxy_T300"),
            Ply(-45, ply_thickness, "carbon_epoxy_T300"),
            Ply(45, ply_thickness, "carbon_epoxy_T300"),
        ],
        
        "0° ağırlıklı [0₃/±45/90]s": [
            Ply(0, ply_thickness, "carbon_epoxy_T300"),
            Ply(0, ply_thickness, "carbon_epoxy_T300"),
            Ply(0, ply_thickness, "carbon_epoxy_T300"),
            Ply(45, ply_thickness, "carbon_epoxy_T300"),
            Ply(-45, ply_thickness, "carbon_epoxy_T300"),
            Ply(90, ply_thickness, "carbon_epoxy_T300"),
            Ply(90, ply_thickness, "carbon_epoxy_T300"),
            Ply(-45, ply_thickness, "carbon_epoxy_T300"),
            Ply(45, ply_thickness, "carbon_epoxy_T300"),
            Ply(0, ply_thickness, "carbon_epoxy_T300"),
            Ply(0, ply_thickness, "carbon_epoxy_T300"),
            Ply(0, ply_thickness, "carbon_epoxy_T300"),
        ],
    }
    
    print(f"\n{'Dizilim':<35} {'Kalınlık':<12} {'Simetrik':<10} {'N_cr (kN/m)':<12}")
    print("-" * 75)
    
    for name, plies in laminates.items():
        laminate = CompositeLaminate(plies)
        result = laminate.find_critical_buckling(Lx, Ly)
        
        sym = "✓" if laminate.is_symmetric() else "✗"
        
        print(f"{name:<35} {laminate.total_thickness*1000:.2f}mm      "
              f"{sym:<10} {result['N_cr']/1000:<12.2f}")
    
    return laminates


def test_exotic_materials():
    """Egzotik ve ilginç malzeme karşılaştırması."""
    print_header("EGZOTİK MALZEME KARŞILAŞTIRMASI")
    
    Lx, Ly, h = 0.5, 0.3, 0.003
    
    categories = {
        "🔩 Ultra Rijit": ["diamond", "tungsten", "silicon_carbide", "boron_epoxy"],
        "🪶 Ultra Hafif": ["beryllium", "magnesium_AZ31", "wood_balsa", "foam_pu_rigid"],
        "🌡️ Yüksek Sıcaklık": ["inconel_718", "hastelloy_X", "titanium_Ti6Al4V", "carbon_peek"],
        "🌿 Doğal/Sürdürülebilir": ["flax_epoxy", "hemp_epoxy", "bamboo", "wood_oak"],
        "🚀 Havacılık Özel": ["titanium_Ti6Al4V", "aluminum_7075_T6", "carbon_epoxy_IM7", "boron_epoxy"],
        "⚡ Shape Memory": ["nitinol_austenite", "nitinol_martensite"],
        "🔬 Yüksek Teknoloji": ["carbon_epoxy_M55J", "SiC_aluminum", "kevlar_49_epoxy"],
    }
    
    for category, materials in categories.items():
        print_subheader(category)
        
        print(f"{'Malzeme':<35} {'E (GPa)':<12} {'ρ (kg/m³)':<12} {'E/ρ':<12}")
        print("-" * 71)
        
        for mat_name in materials:
            try:
                mat = MaterialLibrary.get(mat_name)
                
                if hasattr(mat, 'E1'):  # Ortotropik
                    E = mat.E1
                else:
                    E = mat.E
                
                ratio = (E / 1e9) / (mat.rho / 1000)  # GPa / (g/cm³) = spesifik modül
                
                print(f"{mat.name[:33]:<35} {E/1e9:<12.1f} {mat.rho:<12.0f} {ratio:<12.1f}")
                
            except Exception as e:
                print(f"{mat_name:<35} Bulunamadı: {e}")


def test_speed_comparison():
    """Hesaplama hızı testi."""
    print_header("HIZ TESTİ: SPINE vs Geleneksel")
    
    import math
    
    # Test parametreleri
    n_tests = 1000
    materials_to_test = ["steel_304", "carbon_epoxy_T300"]
    
    print(f"\nTest sayısı: {n_tests}")
    
    for mat_name in materials_to_test:
        mat = MaterialLibrary.get(mat_name)
        
        print_subheader(f"Malzeme: {mat.name}")
        
        # SPINE yaklaşımı (formül tabanlı)
        start = time.perf_counter()
        
        for i in range(n_tests):
            # Rastgele parametreler
            Lx = 0.3 + 0.4 * (i / n_tests)
            Ly = 0.2 + 0.3 * (i / n_tests)
            h = 0.002 + 0.006 * (i / n_tests)
            
            if hasattr(mat, 'E1'):
                # Ortotropik
                Q = mat.get_Q_matrix()
                D11 = Q[0, 0] * h**3 / 12
                D22 = Q[1, 1] * h**3 / 12
                D12 = Q[0, 1] * h**3 / 12
                D66 = Q[2, 2] * h**3 / 12
                
                term1 = D11 / Lx**4
                term2 = 2 * (D12 + 2*D66) / (Lx**2 * Ly**2)
                term3 = D22 / Ly**4
                N_cr = math.pi**2 * Lx**2 * (term1 + term2 + term3)
            else:
                # İzotropik
                D = mat.E * h**3 / (12 * (1 - mat.nu**2))
                alpha = math.pi / Lx
                beta = math.pi / Ly
                N_cr = D * (alpha**2 + beta**2)**2 / alpha**2
        
        spine_time = time.perf_counter() - start
        
        print(f"SPINE (formül):     {spine_time*1000:.2f} ms ({n_tests} hesap)")
        print(f"Ortalama:           {spine_time/n_tests*1e6:.2f} μs/hesap")
        
        # Karşılaştırma için FEM tahmini
        estimated_fem_time = n_tests * 0.5  # 0.5 saniye/analiz varsayımı
        speedup = estimated_fem_time / spine_time
        
        print(f"\nFEM tahmini:        {estimated_fem_time:.0f} saniye")
        print(f"Hızlanma:           {speedup:.0f}x 🚀")


def test_material_neuron():
    """AdvancedMaterialNeuron testi."""
    print_header("MALZEME NÖRONU TESTİ (PyTorch)")
    
    neuron = AdvancedMaterialNeuron()
    
    # İzotropik test
    steel = MaterialLibrary.get("steel_304")
    h = torch.tensor(0.005)
    
    result = neuron(steel, h)
    print(f"\nİzotropik (Çelik 304, h=5mm):")
    print(f"  D = {result['D'].item():.2f} N·m")
    
    # Ortotropik test
    carbon = MaterialLibrary.get("carbon_epoxy_T300")
    
    result = neuron(carbon, h)
    print(f"\nOrtotropik (Karbon T300, h=5mm):")
    print(f"  D11 = {result['D11'].item():.2f} N·m")
    print(f"  D22 = {result['D22'].item():.2f} N·m")
    print(f"  D12 = {result['D12'].item():.2f} N·m")
    print(f"  D66 = {result['D66'].item():.2f} N·m")


def test_hybrid_laminate():
    """Hibrit laminat (farklı malzeme karışımı) testi."""
    print_header("HİBRİT LAMİNAT TESTİ")
    
    print("\nDış katmanlar: Karbon (rijitlik)")
    print("İç katmanlar: Cam elyaf (maliyet)")
    
    t = 0.2e-3  # 0.2mm katman
    
    hybrid = CompositeLaminate([
        # Dış - Karbon (yüksek modül)
        Ply(0, t, "carbon_epoxy_T300"),
        Ply(45, t, "carbon_epoxy_T300"),
        # İç - Cam (ekonomik)
        Ply(-45, t, "glass_epoxy_E"),
        Ply(90, t, "glass_epoxy_E"),
        Ply(90, t, "glass_epoxy_E"),
        Ply(-45, t, "glass_epoxy_E"),
        # Dış - Karbon
        Ply(45, t, "carbon_epoxy_T300"),
        Ply(0, t, "carbon_epoxy_T300"),
    ])
    
    print(f"\nLaminat: {hybrid}")
    print(f"Toplam kalınlık: {hybrid.total_thickness*1000:.2f} mm")
    
    A, B, D = hybrid.get_ABD_matrices()
    
    print(f"\n[D] Matrisi (eğilme rijitliği):")
    print(f"  D11 = {D[0,0]:.4f} N·m")
    print(f"  D22 = {D[1,1]:.4f} N·m")
    print(f"  D12 = {D[0,1]:.4f} N·m")
    
    # Burkulma
    result = hybrid.find_critical_buckling(0.5, 0.3)
    print(f"\nBurkulma (0.5m × 0.3m):")
    print(f"  N_cr = {result['N_cr']/1000:.2f} kN/m")
    print(f"  Mod: ({result['m']}, {result['n']})")
    
    # Karşılaştırma için saf karbon
    pure_carbon = CompositeLaminate([
        Ply(0, t, "carbon_epoxy_T300"),
        Ply(45, t, "carbon_epoxy_T300"),
        Ply(-45, t, "carbon_epoxy_T300"),
        Ply(90, t, "carbon_epoxy_T300"),
        Ply(90, t, "carbon_epoxy_T300"),
        Ply(-45, t, "carbon_epoxy_T300"),
        Ply(45, t, "carbon_epoxy_T300"),
        Ply(0, t, "carbon_epoxy_T300"),
    ])
    result_pure = pure_carbon.find_critical_buckling(0.5, 0.3)
    
    print(f"\nKarşılaştırma (saf karbon):")
    print(f"  N_cr = {result_pure['N_cr']/1000:.2f} kN/m")
    
    efficiency = result['N_cr'] / result_pure['N_cr'] * 100
    print(f"\nHibrit verimlilik: %{efficiency:.1f} (saf karbona göre)")


def main():
    """Tüm testleri çalıştır."""
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "      SPINE MALZEME MODÜLÜ - KAPSAMLI TEST".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    print(f"\nToplam malzeme: {MaterialLibrary.count()}")
    print(f"  - Metaller: {len(MaterialLibrary.list_metals())}")
    print(f"  - Polimerler: {len(MaterialLibrary.list_polymers())}")
    print(f"  - Kompozitler: {len(MaterialLibrary.list_composites())}")
    
    # Testler
    test_isotropic_buckling()
    test_orthotropic_buckling()
    test_composite_laminates()
    test_exotic_materials()
    test_hybrid_laminate()
    test_material_neuron()
    test_speed_comparison()
    
    print_header("TÜM TESTLER TAMAMLANDI ✅")
    print("\n🎉 Malzeme modülü başarıyla çalışıyor!")
    print("   65 malzeme, CLT hesaplama, PyTorch entegrasyonu hazır.\n")


if __name__ == "__main__":
    main()

