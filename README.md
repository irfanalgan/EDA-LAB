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
- **Korelasyon** — Pearson r matrisi (|r| ≥ 0.60 çiftler), scatter, VIF
- **Chi-Square** — Cramér's V, contingency heatmap
- **ANOVA** — F-testi, grup istatistikleri, box plot
- **KS** — Ampirik CDF karşılaştırması, Good vs Bad
- **VIF Kum Havuzu** — IV ≥ 0.10 filtreli veya tüm numerik değişkenler

### Hızlı Model
Logistic Regression · LightGBM · XGBoost · Random Forest — train/test ayrımı (rastgele veya tarihe göre), AUC · Gini · KS · F1 · Precision · Recall, ROC eğrisi, Confusion Matrix, katsayı/importance tablosu.

- **Eşik optimizasyonu** — Sabit 0.50 / F1 Maks. / KS Noktası / Özel
- **SHAP Beeswarm** — Tree modeller için tüm test verisi üzerinde shap.summary_plot ile özellik katkı grafiği

### Sistem Hazırlama (Precompute)
Yapılandırma onaylandıktan sonra bir popup açılır ve IV Ranking, Profiling, Korelasyon gibi ağır hesaplamalar arka planda adım adım yapılır. Her adımın durumu ve süresi görüntülenir. Tamamlandıktan sonra sekmeler açılışta bekleme olmadan çalışır.

---

## Performans — Benchmark Testi

**Test ortamı:** Windows 11 Pro · Python 3.13.3 · 16 çekirdek · 31.2 GB RAM
**Test verisi:** 5,745,504 satır × 37 kolon (4 Parquet klasörü + 1 CSV, merge edilmiş)
**Kütüphaneler:** Pandas 2.3.3 · scikit-learn 1.8.0 · LightGBM 4.6.0 · XGBoost 3.1.2 · SHAP 0.50.0

### Veri Yükleme

| İşlem | Süre | RAM Artışı |
|-------|------|-----------|
| Parquet + CSV okuma + 4 tablo merge | **4.4s** | +2,319 MB |

5.7M satırlık veri 4.4 saniyede belleğe alınıyor.

### Modül Bazında Performans

| Test | Süre | Durum |
|------|------|-------|
| Profiling (37 kolon) | **11.2s** | ✓ |
| Target İstatistikleri | **0.04s** | ✓ |
| IV Ranking — tüm numerik kolonlar | **282s** | ⚠ cache ile ilk seferlik |
| Ön Eleme (Screening) | **3.7s** | ✓ |
| Korelasyon Matrisi (30 kolon) | **7.3s** | ✓ |
| VIF (30 kolon) | **0.7s** | ✓ |
| WOE Detail (1 kolon) | **14.5s** | ✓ |
| PSI (1 kolon) | **4.1s** | ✓ |
| Variable Stats (1 kolon) | **0.6s** | ✓ |
| Target Over Time | **0.3s** | ✓ |
| High Corr Pairs (threshold=0.7) | **7.3s** | ✓ |
| Chi-Square testi | **13.2s** | ✓ |
| ANOVA testi (200k/grup örnekleme) | **0.1s** | ✓ |
| KS testi (tam veri) | **0.05s** | ✓ |
| Outlier IQR 1.5 (20 kolon) | **1.9s** | ✓ |
| Outlier IQR 3.0 (20 kolon) | **1.9s** | ✓ |
| Outlier Z-Score 3.0 (20 kolon) | **1.9s** | ✓ |
| Müşteri Outlier Tablosu (IQR 1.5) | **3.3s** | ✓ |

### Segment Filtresi Etkisi

| Veri Boyutu | Profiling | IV Ranking | Outlier IQR |
|-------------|-----------|-----------|-------------|
| %100 — 5,745,504 satır | 11.5s | 291s | 2.0s |
| %50 — 2,872,752 satır | 5.3s | 139s | 1.0s |
| %10 — 574,550 satır | 0.9s | **12s** | 0.2s |

Segment filtresi IV Ranking'i **23×** hızlandırıyor. Küçük segment ile çalışmak önerilir.

### Model Eğitimi

| Model | Fit Süresi | Notlar |
|-------|-----------|--------|
| Logistic Regression | **20.6s** | 4M satır, SAGA solver |
| LightGBM | **3.7s** | 100 estimator |
| XGBoost | **3.6s** | 100 estimator |
| Random Forest | **47.0s** | 100 estimator |

### SHAP (1,723,652 satır test seti)

| Model | SHAP Süresi | RAM Artışı |
|-------|------------|-----------|
| LightGBM | **59.4s** | +408 MB |
| XGBoost | **3.9s** | +414 MB |
| Random Forest | — | Atlandı (saat mertebesi) |

### Temel Bulgular

**Darboğaz:** IV Ranking, tüm veri üzerinde ~282 saniye sürmektedir. Bu değer segment filtresiyle dramatik biçimde düşmektedir (%10 segmentte 12s). Uygulama bu hesabı `key + segment` kombinasyonu başına bir kez yapıp cache'ler; aynı sekmede ikinci açılış anında gelir. Yapılandırma onayı sırasında açılan precompute popup'ı bu süreyi ilk kullanımda şeffaf hale getirir.

**Hızlı olanlar:** Veri yükleme (4.4s), Outlier (2s), KS testi (0.05s), VIF (0.7s), LightGBM fit (3.7s) ve XGBoost SHAP (3.9s) beklenen sürelerin çok altındadır.

**SHAP notu:** Random Forest SHAP, 1.7M satır ve 100 ağaç kombinasyonunda saat mertebesinde sürmektedir. Büyük veri setlerinde RF seçilmesi durumunda örnekleme önerilir.

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

**Temel kural:** `df_original` hiçbir zaman değiştirilmez. Tüm analizler `df_active` üzerinde çalışır.

### Precompute — Background Thread

Yapılandırma onaylanınca IV Ranking, Profiling ve Korelasyon hesaplamaları ayrı bir `threading.Thread` içinde çalışır. Dash ana thread'i bloklanmaz; diğer sekmeler hesaplama süresince kullanılabilir. İlerleme 300ms aralıklarla `dcc.Interval` üzerinden UI'ya yansır.

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
├── app.py                  # Giriş noktası (23 satır)
├── app_instance.py         # Dash app tanımı — tek yer
├── server_state.py         # Paylaşılan state (_SERVER_STORE, _PRECOMPUTE_PROGRESS)
├── benchmark.py            # Performans test scripti
├── setup_deps.py           # Otomatik bağımlılık yükleyici
├── pip_prefix.txt          # Kurumsal pip prefix (opsiyonel)
├── config.toml             # DB bağlantı ayarları
├── assets/
│   └── custom.css          # Dark tema stilleri
├── data/
│   └── loader.py           # SQL bağlantısı
├── layout/
│   └── __init__.py         # build_layout() — sidebar, tab yapıları
├── callbacks/
│   ├── __init__.py         # Tüm modülleri import eder (kayıt tetikler)
│   ├── data_loading.py     # CSV/SQL yükleme, config, segment
│   ├── precompute.py       # Background thread + progress modal
│   ├── preview.py          # Metrikler + veri önizleme + ön eleme raporu
│   ├── profiling.py        # Profiling sekmesi
│   ├── target_iv.py        # Target & IV sekmesi
│   ├── outlier.py          # Outlier Analizi sekmesi
│   ├── deep_dive.py        # Değişken Analizi (WoE, PSI, bivariate)
│   ├── correlation.py      # Korelasyon sekmesi
│   ├── stat_tests.py       # Chi-Square, ANOVA, KS, VIF
│   ├── var_summary.py      # Değişken Özeti
│   └── playground.py       # Grafik + Hızlı Model + SHAP
├── modules/
│   ├── profiling.py        # Veri profiling
│   ├── target_analysis.py  # Target & IV hesaplamaları
│   ├── deep_dive.py        # WoE, PSI, bivariate
│   ├── correlation.py      # Korelasyon matrisi, VIF
│   └── screening.py        # Kolon kalite filtresi
└── utils/
    ├── helpers.py          # Segment filtresi
    └── chart_helpers.py    # Grafik teması, DataTable stili, tab_info, r badge
```

---

## Notlar

- 5M+ satır için optimize edilmiştir (server-side cache, aggregate-first istatistik)
- Segment filtresi tüm sekmelere anlık yansır; `df_original`'e dokunmaz
- WoE encode başarısız olan değişkenler ham değerle modele girer, sessizce atlanmaz — bildirim gösterilir
- SHAP hesabı tüm test seti üzerinde çalışır; Random Forest için büyük veri setlerinde örnekleme önerilir
- PSI bin sınırları WOE analizi ile birebir aynıdır; tutarsız karşılaştırma olmaz

---

## Değişiklik Geçmişi

### v1.5
- **Loading Slideshow** — 8 eğitim slaytı (EDA Lab nedir, Veri Yükleme, Önizleme, Target & IV, Deep Dive, İstatistiksel Testler, Değişken Özeti, Playground); otomatik 8-saniye ilerleme; tıklanabilir navigasyon noktaları (●○○○); geçen süre sayacı
- **CDN Reachability Check** — Google Fonts ve Bootstrap Icons iş ağında bloke olunca düşüş kalmıyor; conditional loading ile graceful fallback
- **Playground WoE + LR** — Logistic Regression (statsmodels) için StandardScaler otomatik atlanıyor, sm.Logit doğrudan WoE değerlerine uygulanıyor
- **Train/Test Split** — Otomatik stratify (target ≤10 unique değerse)
- **Date Sorting** — Veri yüklendikten hemen sonra date column'a göre sıralanıyor (IV tutarlılığı)
- **Playground Test Oranı** — UI'dan kaldırıldı, config değeri doğrudan kullanılıyor
- **İstatistiksel Test Açıklamaları** — Tüm 5 test (Korelasyon, Chi², ANOVA, KS, VIF) yeniden yazıldı; kullanıcı-dostu "Çıktıyı nasıl okurum" bölümleri
- **CSS v2.1** — Slideshow animasyonları, responsive tasarım, progress bar, Dash 4.0 dropdown refactor

### v1.4
- **Target Tipi Tespiti** — `confirm_config` anında target kolonu otomatik olarak `binary | continuous | multiclass | categorical` olarak sınıflandırılır; tüm sekmeler bu bilgiyi kullanır
- **Target & IV sekmesi** — binary dışı targetlar için IV/WoE yerine dağılım grafiği, temel istatistikler ve zaman serisi gösterilir; IV'nin binary-only olduğuna dair açıklama notu eklendi
- **Değişken Özeti** — binary olmayan targetlarda IV yerine **Mutual Information** (sklearn) kullanılır; `MI` sütunu ve eşik notları eklendi
- **KS Testi** — binary: Good vs Bad; multiclass: dominant sınıf vs diğerleri; continuous: medyan split (Düşük / Yüksek)
- **Chi-Square** — continuous target seçildiğinde otomatik `qcut(5)` ile 5 gruba bölünür
- **ANOVA → Kruskal-Wallis** — multiclass ve 3+ sınıf varsa parametrik olmayan Kruskal-Wallis testi otomatik devreye girer
- **Playground Regresyon** — continuous target için Ridge, LightGBM Regressor, XGBoost Regressor, Random Forest Regressor eklendi; metrikler R² · RMSE · MAE; Actual vs Predicted scatter + artık dağılımı grafiği
- **Preview target kartı** — binary: bad rate; continuous: ortalama + min/max/std tooltip; multiclass: sınıf sayısı; categorical: unique değer sayısı
- **Genel numeric yükleme** — CSV/SQL yüklemede tüm object kolonlar %95 eşikle otomatik numerik dönüşüme tabi tutulur

### v1.3
- **Modüler mimari** — 5232 satırlık monolitik `app.py` 17 dosyaya bölündü; her sekme bağımsız bir `callbacks/` modülü
- **Background thread precompute** — IV Ranking ve ağır hesaplamalar ayrı thread'de çalışır, Dash ana thread'i bloklanmaz; diğer sekmeler hesaplama sırasında kullanılabilir
- **Otomatik precompute başlatma** — yapılandırma onaylanınca hesaplama kendiliğinden başlar, "Başlat" butonu kaldırıldı
- **Precompute buton kaydı düzeltildi** — butonlar baştan layout'a hidden olarak eklendi; Dash frontend ID'leri sayfa yüklenirken tanır

### v1.2
- **Precompute Popup** — yapılandırma onaylanınca IV Ranking, Profiling, Korelasyon adım adım hesaplanır; sekmelerde cold-start bekleme ortadan kalkar
- **PSI bin düzeltmesi** — PSI artık WOE ile aynı bin sınırlarını kullanır; bin sayısı ve aralıklar birebir eşleşir, x ekseni numerik sırayla görünür
- **Korelasyon çiftleri** — |r| < 0.60 çiftler tabloda gösterilmez, gürültü azaltıldı
- **Benchmark testi** — 5.7M satır veri seti üzerinde tüm modüller ölçüldü, sonuçlar README'ye eklendi
- `benchmark.py` scripti eklendi

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
