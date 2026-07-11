"""
YENI EGITIM TESTLERINI CALISTIR
Tek Noron (3 adet):
1. ThermalNeuron
2. ShellNeuron3D
3. BeamNeuron3D

Entegrasyon (3 adet):
4. Modal + Catlak
5. Termal + Burkulma
6. Shell + Beam
"""

import subprocess
import sys
import time

TESTS = [
    ("1. ThermalNeuron", "train_thermal.py"),
    ("2. ShellNeuron3D", "train_shell3d.py"),
    ("3. BeamNeuron3D", "train_beam3d.py"),
    ("4. Modal + Catlak", "train_modal_crack.py"),
    ("5. Termal + Burkulma", "train_thermal_buckling.py"),
    ("6. Shell + Beam", "train_shell_beam.py"),
]

def run_test(name, script):
    print("\n" + "=" * 70)
    print(f"{name} BASLIYOR")
    print("=" * 70)

    start_time = time.time()

    result = subprocess.run(
        [sys.executable, script],
        capture_output=False
    )

    elapsed = time.time() - start_time
    print(f"\nSure: {elapsed:.1f} saniye")

    if result.returncode != 0:
        print(f"{name} BASARISIZ!")
        return False
    else:
        print(f"{name} TAMAMLANDI!")
        return True


def main():
    print("=" * 70)
    print("SPINE YENI EGITIM TESTLERI")
    print("6 Test - Thermal, Shell3D, Beam3D, Entegrasyonlar")
    print("=" * 70)

    total_start = time.time()
    results = []

    for name, script in TESTS:
        success = run_test(name, script)
        results.append((name, success))

    total_time = time.time() - total_start

    # Ozet
    print("\n" + "=" * 70)
    print("OZET")
    print("=" * 70)

    passed = sum(1 for _, s in results if s)
    failed = len(results) - passed

    for name, success in results:
        status = "GECTI" if success else "BASARISIZ"
        print(f"  {name}: {status}")

    print("-" * 70)
    print(f"Toplam: {passed}/{len(results)} test gecti")
    print(f"Toplam sure: {total_time/60:.1f} dakika")
    print("=" * 70)


if __name__ == "__main__":
    main()
