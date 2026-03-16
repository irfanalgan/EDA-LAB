# EDA Laboratuvarı

Kredi riski ve ikili sınıflandırma problemleri için geliştirilmiş **yerel, interaktif EDA ve model geliştirme aracı**. Dash + Plotly tabanlı, tamamen Python ile çalışır. Hiçbir veri dışarı çıkmaz.

---

## Özellikler

| Sekme | İçerik |
|-------|--------|
| **Önizleme** | Ham veri tablosu, uzman değişken eleme |
| **Profiling** | Kolon bazında eksik %, kardinalite, tip analizi |
| **Target & IV** | Bad rate dağılımı, IV hesabı, zaman serisi |
| **Outlier Analizi** | IQR / Z-score aykırı değer tespiti, müşteri bazında outlier sayısı |
| **Değişken Analizi** | WoE/PSI/bivariate deep dive |
| **İstatistiksel Testler** | Korelasyon, Chi-Square, ANOVA, KS, VIF |
| **Değişken Özeti** | IV · Eksik% · PSI · Korelasyon · VIF tek tabloda |
| **Playground** | Grafik oluşturucu + hızlı model (LR / LightGBM / XGBoost / RF) + SHAP |

### Veri Kaynakları
- **MS SQL Server** — Windows Authentication, `config.toml` üzerinden bağlantı
- **CSV** — Sürükle-bırak yükleme, ayırıcı seçimi (`,` `;` `\t` `|`)

### Outlier Analizi
- **IQR** — 1.5× (normal sınır) veya 3.0× (aşırı aykırı) çarpanı
- **Z-Score** — ±2σ / ±2.5σ / ±3σ eşiği
- Değişken bazında özet tablo: N Outlier, % Outlier, alt/üst sınır, min/max
- **Müşteri bazında detay** — seçilen kimlik kolonu üzerinden her müşterinin kaç farklı değişkende aykırı olduğu

### İstatistiksel Testler
- **Korelasyon** — Pearson r matrisi, çift scatter, VIF
- **Chi-Square** — Cramér's V, contingency heatmap
- **ANOVA** — F-testi, grup istatistikleri, box plot
- **KS** — Ampirik CDF karşılaştırması, Good vs Bad
- **VIF Kum Havuzu** — IV ≥ 0.10 filtreli veya tüm numerik değişkenler

### Hızlı Model
Logistic Regression · LightGBM · XGBoost · Random Forest — train/test ayrımı (rastgele veya tarihe göre), AUC · Gini · KS · F1 · Precision · Recall, ROC eğrisi, Confusion Matrix, katsayı/importance tablosu.

- **Eşik optimizasyonu** — Sabit 0.50 / F1 Maks. / KS Noktası / Özel
- **SHAP Beeswarm** — Tree modeller için tüm test verisi üzerinde shap.summary_plot ile özellik katkı grafiği

---

## Mimari

```
SQL / CSV → df_original (server-side cache, UUID key)
                │
                ├─ apply_segment_filter() → df_active
                │
                ├─ modules/profiling.py
                ├─ modules/target_analysis.py
                ├─ modules/deep_dive.py      (WoE, PSI, IV)
                ├─ modules/correlation.py    (r matrisi, VIF)
                └─ modules/screening.py      (eksik/sabit eleme)
```

**Temel kural:** `df_original` hiçbir zaman değiştirilmez. Tüm analizler `df_active.copy()` üzerinde çalışır.

---

## Kurulum

```bash
pip install -r requirements.txt
python app.py
# http://localhost:8050 otomatik açılır
```

### Gereksinimler

```
dash>=2.14
dash-bootstrap-components>=1.5
pandas>=2.0
numpy
plotly>=5.18
scipy>=1.11
scikit-learn>=1.3
lightgbm>=4.0
xgboost>=2.0
pyodbc
ydata-profiling
shap
matplotlib
```

### Kurumsal Ortamda Pip Proxy

`pip_prefix.txt` dosyasına kurumun gerektirdiği pip ön komutunu yapıştır:

```
pip install --index-url https://... --trusted-host ...
```

Uygulama başlarken `setup_deps.py` eksik paketleri bu prefix ile otomatik yükler.

### SQL Server Bağlantısı

`config.toml` dosyasını düzenle:

```toml
[database]
driver   = "ODBC Driver 17 for SQL Server"
server   = "SUNUCU_ADI"
database = "VERITABANI_ADI"
```

Windows Authentication kullanılır, kullanıcı adı/şifre gerekmez.

---

## Klasör Yapısı

```
EDA-LAB/
├── app.py                  # Ana uygulama (~5000 satır)
├── setup_deps.py           # Otomatik bağımlılık yükleyici
├── pip_prefix.txt          # Kurumsal pip prefix (opsiyonel)
├── config.toml             # DB bağlantı ayarları
├── assets/
│   └── custom.css          # Dark tema stilleri
├── data/
│   └── loader.py           # SQL bağlantısı
├── modules/
│   ├── profiling.py        # Veri profiling
│   ├── target_analysis.py  # Target & IV hesaplamaları
│   ├── deep_dive.py        # WoE, PSI, bivariate
│   ├── correlation.py      # Korelasyon matrisi, VIF
│   └── screening.py        # Kolon kalite filtresi
└── utils/
    └── helpers.py          # Segment filtresi vb.
```

---

## Notlar

- 5M+ satır için optimize edilmiştir (server-side cache, aggregate-first istatistik)
- Segment filtresi tüm sekmelere anlık yansır; `df_original`'e dokunmaz
- WoE encode başarısız olan değişkenler ham değerle modele girer, sessizce atlanmaz — bildirim gösterilir
- SHAP hesabı tüm test seti üzerinde çalışır; büyük veri setlerinde birkaç saniye sürebilir

---

## Değişiklik Geçmişi

### v1.1
- **Outlier Analizi sekmesi** eklendi (IQR / Z-score, müşteri bazında detay tablosu)
- **SHAP Beeswarm** — shap.summary_plot ile tam test verisi üzerinde görselleştirme
- **Eşik optimizasyonu** — F1 Maks. / KS Noktası / Özel eşik seçeneği
- **CSV yükleme** — sidebar'a SQL yanına eklendi
- **LightGBM · XGBoost · Random Forest** Hızlı Model'e eklendi
- **Playground değişken seçici** — arama filtresi + dikey liste (100+ değişken için ölçeklenebilir)
- **Tab açıklama kartları** — her sekmenin üstünde amaç/yorum bilgisi
- Dark tema metin renkleri güçlendirildi

### v1.0
- İlk sürüm — Önizleme, Profiling, Target & IV, Değişken Analizi, İstatistiksel Testler, Değişken Özeti, Playground
