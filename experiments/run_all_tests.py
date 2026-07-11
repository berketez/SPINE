"""
TUM EGITIM TESTLERINI CALISTIR
Tek Noron Testleri (7 adet):
1. StaticNeuron
2. ModalNeuron
3. CrackNeuron
4. ThermalNeuron
5. SolidNeuron3D
6. ShellNeuron3D
7. BeamNeuron3D

Entegrasyon Testleri (6 adet):
8. Termal + Statik
9. Statik + Catlak
10. Modal + Catlak
11. Termal + Burkulma
12. Shell + Beam
13. Tam Entegrasyon
"""

import os
import subprocess
import sys
import time

TESTS = [
    # Tek Noron Testleri
    ("1. StaticNeuron", "train_static.py"),
    ("2. ModalNeuron", "train_modal.py"),
    ("3. CrackNeuron", "train_crack.py"),
    ("4. ThermalNeuron", "train_thermal.py"),
    ("5. SolidNeuron3D", "train_solid3d.py"),
    ("6. ShellNeuron3D", "train_shell3d.py"),
    ("7. BeamNeuron3D", "train_beam3d.py"),
    # Entegrasyon Testleri
    ("8. Termal + Statik", "train_thermal_static.py"),
    ("9. Statik + Catlak", "train_static_crack.py"),
    ("10. Modal + Catlak", "train_modal_crack.py"),
    ("11. Termal + Burkulma", "train_thermal_buckling.py"),
    ("12. Shell + Beam", "train_shell_beam.py"),
    ("13. Tam Entegrasyon", "train_full_integration.py"),
]

def run_test(name, script):
    print("\n" + "=" * 70)
    print(f"🚀 {name} BAŞLIYOR")
    print("=" * 70)

    start_time = time.time()

    # Betikler bu dosyayla aynı dizinde — cwd'den bağımsız çalışsın
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result = subprocess.run(
        [sys.executable, os.path.join(script_dir, script)],
        capture_output=False,
        cwd=script_dir
    )

    elapsed = time.time() - start_time
    print(f"\n⏱️  Süre: {elapsed:.1f} saniye")

    if result.returncode != 0:
        print(f"❌ {name} BAŞARISIZ!")
        return False
    else:
        print(f"✅ {name} TAMAMLANDI!")
        return True


def main():
    print("=" * 70)
    print("SPINE EĞİTİM TESTLERİ")
    print("7 Test - Static, Modal, Thermal, Crack, 3D, Entegrasyon")
    print("=" * 70)

    total_start = time.time()
    results = []

    for name, script in TESTS:
        success = run_test(name, script)
        results.append((name, success))

    total_time = time.time() - total_start

    # Özet
    print("\n" + "=" * 70)
    print("ÖZET")
    print("=" * 70)

    passed = sum(1 for _, s in results if s)
    failed = len(results) - passed

    for name, success in results:
        status = "✅ GEÇTI" if success else "❌ BAŞARISIZ"
        print(f"  {name}: {status}")

    print("-" * 70)
    print(f"Toplam: {passed}/{len(results)} test geçti")
    print(f"Toplam süre: {total_time/60:.1f} dakika")
    print("=" * 70)


if __name__ == "__main__":
    main()
