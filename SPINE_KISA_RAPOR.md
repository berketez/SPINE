# SPINE - Kisa Rapor (2-3 sayfa)

SPINE (Structural Physics-Inherent Neural Engine), yapisal analizde fizik
kisitlarini dogrudan noronun icine gomerek hizli ve yorumlanabilir bir on-analiz
katmani olusturmayi hedefler. Bu rapor; problemin tanimi, SPINE yaklasimi,
mimari, mevcut kapsam, dogrulama durumu ve kisa bir kullanim ornegini sunar.

---

## 1. GIRIS & PROBLEM

### 1.1 Problem
- ANSYS/ABAQUS gibi klasik FEA araclari lisans maliyeti ve hesaplama suresi
  nedeniyle tasarim iterasyonlarini yavaslatir.
- PINN yaklasimlari 4. derece turevlerde (or. plaka burkulmasi) autograd zinciri
  nedeniyle hata birikimi ve kararsizlik yasar.

### 1.2 SPINE Cozumu
- Fizik, loss fonksiyonuna degil noronun yapisina gomuludur (hard-constraint).
- Spektral operatorler ile nabla, nabla^2, nabla^4 turevleri tek adimda ve yuksek
  dogrulukla hesaplanir.

### 1.3 Hedef Karsilastirma (kapsam ici hedefler)
```
+-------------------+------------+-------------------+
|                   |    PINN    |   SPINE (Hedef)   |
+-------------------+------------+-------------------+
| Dogruluk          | %5-10 hata | Hedef <2%         |
+-------------------+------------+-------------------+
| Hiz               | Yavas      | Hedef 10-100x     |
+-------------------+------------+-------------------+
| Yorumlanabilirlik | Kara kutu  | Her noron = fizik |
+-------------------+------------+-------------------+
```
Not: Dogruluk ve hiz hedefleri, tanimli domain (geometri, sinir kosulu, malzeme
araligi) icinde gecerlidir.

---

## 2. MIMARI & NORONLAR

### 2.1 Sistem Ozeti (cekirdek moduller)
- Toplam: ~4500 satir, 31 sinif
- `spine.py`: 2868 satir, 22 sinif
- `materials.py`: 1205 satir, 7 sinif
- `fem_solver.py`: 427 satir, 2 sinif
- PyTorch tabanli; CUDA/MPS/CPU destegi

### 2.2 Mevcut Noronlar ve Bilesenler
```
+----------+------------------------------------------------------------+
| Kategori |                          Noronlar                          |
+----------+------------------------------------------------------------+
| 2D Temel | Biharmonic, Buckling, Stress, Strain, Boundary             |
+----------+------------------------------------------------------------+
| 2D Ileri | Static, Modal, Crack, Thermal, Fatigue, Dynamic, Nonlinear |
+----------+------------------------------------------------------------+
| 3D       | SpectralOps3D, Solid3D, Shell3D, Beam3D                    |
+----------+------------------------------------------------------------+
| Malzeme  | AdvancedMaterialNeuron, 50+ hazir malzeme                  |
+----------+------------------------------------------------------------+
```

### 2.3 Planlanan (FAZ 5)
```
+-----------------+---------------------------------------------------+
|    Kategori     |                       Hedef                       |
+-----------------+---------------------------------------------------+
| Urunlesme       | REST API, Web UI, CLI araci, Pip paketi           |
+-----------------+---------------------------------------------------+
| Eksik Analizler | 3D Crack, 3D Thermal, Dinamik-Nonlineer birlesimi |
+-----------------+---------------------------------------------------+
| Dogrulama       | ANSYS benchmark testleri                          |
+-----------------+---------------------------------------------------+
```

---

## 3. DOGRULAMA & GELECEK

### 3.1 Dogrulama Protokolu
```
+--------------------+--------+------------------------------------------+
|       Asama        | Durum  |                Aciklama                  |
+--------------------+--------+------------------------------------------+
| Analitik formuller | DONE   | Kirchhoff-Love cozumleri ile kiyas        |
+--------------------+--------+------------------------------------------+
| fem_solver.py      | DONE   | Rayleigh-Ritz referans cozumler           |
+--------------------+--------+------------------------------------------+
| ANSYS kiyasi       | PLAN   | Endustriyel dogrulama                     |
+--------------------+--------+------------------------------------------+
```
Ilk sonuclar analitik cozumlerle uyumlu; hedef hata <%2 (kapsam ici).

### 3.2 Tamamlanan Fazlar
- FAZ 1-4 DONE (Temel noronlar, 2D ileri analizler, 3D destek)

### 3.3 Kullanim Senaryosu (Hedef)
SPINE, ANSYS/ABAQUS oncesi hizli on-analiz ve tasarim iterasyonu icin
kullanilir. Nihai onay her zaman klasik FEA ile yapilir.

### 3.4 Basit Kullanim Ornegi
```python
from spine import SPINE

model = SPINE.from_material("steel_304", h=0.005, Lx=1.0, Ly=0.5)
N_cr = model.get_critical_load(m=1, n=1)
print(f"Kritik yuk: {N_cr:.2f} N/m")
```

---

### Kisa Sonuc
SPINE, fizik-gomulu noronlar ile yapisal analizde hizli ve yorumlanabilir bir
on-analiz katmani sunmayi hedefler. Dogrulama, analitik ve Rayleigh-Ritz
tabanli referanslarla baslamis durumda; endustriyel benchmark adimi (ANSYS)
ile kapsamin guvenli sekilde genisletilmesi planlanmaktadir.
