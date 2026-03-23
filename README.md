# EDA Laboratuvarı

Kredi riski ve ikili sınıflandırma problemleri için geliştirilmiş **yerel, interaktif EDA, model geliştirme ve model izleme aracı**. Dash + Plotly tabanlı, tamamen Python ile çalışır. Hiçbir veri dışarı çıkmaz.

---

## Özellikler

### Geliştirme (EDA + Model)

| Sekme | İçerik |
|-------|--------|
| **Önizleme** | Ham veri tablosu, uzman değişken eleme |
| **Describe** | Kolon bazında eksik %, kardinalite, tip analizi |
| **Target & IV** | Bad rate dağılımı, IV hesabı, zaman serisi |
| **Outlier Analizi** | IQR / Z-score aykırı değer tespiti, müşteri bazında outlier sayısı |
| **Değişken Analizi** | WoE/PSI/bivariate deep dive |
| **İstatistiksel Testler** | Korelasyon, Chi-Square, ANOVA, KS, VIF |
| **Değişken Özeti** | IV · Eksik% · PSI · Korelasyon · VIF tek tabloda |
| **Playground** | Grafik oluşturucu + hızlı model (LR / LightGBM / XGBoost / RF) + SHAP |
| **Sonuç** | Detaylı model raporu — metrikler, ROC, confusion matrix, SHAP, model özeti (accordion) |

### İzleme (Model Monitoring)

| Alt Tab | İçerik |
|---------|--------|
| **PSI** | Değişken bazlı PSI + Rating PSI; dönemsel trend grafiği (Rating PSI + her değişken ayrı çizgi) |
| **Gini/KS** | KS, Gini (Accuracy Ratio); dönemsel trend |
| **Temerrüt Oranı** | Dönem bazlı default oranı; trend çizgisi |
| **HHI** | Rating konsantrasyon indeksi; trend çizgisi + eşik çizgileri (0.06 / 0.10) |
| **Backtesting** | Binomial test — rating bazlı güven aralıkları, conservatism, monotonicity |
| **Göç Matrisi** | Referans vs izleme rating geçiş heatmap'i (ID eşleştirmeli) |

Her tab **Trend** (dönem dropdown + trend grafiği) ve **Kümülatif** (tüm dönemlerin birleşik sonucu) alt sekmelerinden oluşur.

### Veri Kaynakları
- **MS SQL Server** — Windows Authentication, `config.toml` üzerinden bağlantı
- **CSV** — Sürükle-bırak yükleme, ayırıcı seçimi (`,` `;` `\t` `|`)

### İzleme Sistemi — Nasıl Çalışır?

Canlıya alınmış modellerin performansını sürekli takip eder. Kullanıcı **Referans** (geliştirme sample'ı) ve **İzleme** (canlı tablo) verilerini yükler, yapılandırma onaylar, sistem tüm metrikleri dönemsel özetlere dönüştürür. **Ham veri saklanmaz** — profilde yalnızca özetler tutulur (~KB boyutunda).

**Yapılandırma alanları:** Target kolonu, Tarih kolonu, PD kolonu, Model değişkenleri, ID kolonu (opsiyonel — göç matrisi için), Olgunlaşma süresi (ay), Dönem frekansı (Aylık / Çeyreklik), WoE aktif/pasif + OPT pickle

**3 senaryo:**

1. **İlk hesaplama** — Referans + İzleme yüklenir → WoE dönüşümü (aktifse) → Referans özeti çıkarılır → İzleme verisi dönemlere ayrılır → Her dönem için özet hesaplanır → Ham veri atılır
2. **Profil yükle — yeni veri yok** — Özetler cache'den yüklenir, tab'lar anında dolar
3. **Profil yükle — yeni dönem verisi var** — SQL'den sadece yeni satırlar çekilir → yeni dönemlerin özetleri hesaplanır → mevcut listeye eklenir → eski sonuçlar korunur

**Kolon şeması koruması:** Profil kaydedilirken SQL tablosunun kolon listesi saklanır. Yeni veri çekildiğinde kolon yapısı değişmişse hesaplama durdurulur, kullanıcıya yeni profil açması veya tabloyu eski yapısına döndürmesi önerilir.

**Olgunlaşma filtresi:** PSI ve HHI gibi anlık metrikler tüm dönemleri kullanır. Gini, KS, Bad Rate, Backtesting gibi sonuç bağımlı metrikler yalnızca olgun dönemleri (default sonucu gözlemlenmiş) kullanır.

**WoE etkisi:** WoE yalnızca Değişken PSI'ı etkiler (bin edge'leri optb.splits'ten gelir). Diğer tüm metrikler (KS, Gini, Bad Rate, HHI, Backtesting, Göç Matrisi) PD/Rating kolonundan hesaplanır — WoE'den bağımsızdır.

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
- **Model Kaydet / Yükle** — Modeller profil içine kaydedilir, parametreleriyle birlikte geri yüklenir

### Sonuç Sekmesi
Playground'dan bağımsız, detaylı model raporu:
- **dbc.Accordion** ile kategorize edilmiş bölümler (Metrikler, Model Özeti, ROC, Confusion Matrix, SHAP)
- **Confusion Matrix** — Train / Test / OOT yan yana heatmap
- **Model Özeti** — LR: `model.summary()` monospace metin; Tree: feature importance tablosu
- **Ham / WoE alt sekmeleri** — Her iki model sonucu ayrı ayrı incelenebilir
- **Profil klasörüne export** — Model Pickle, OPT Pickle ve Excel doğrudan profil klasörüne kaydedilir (browser download yerine)
- **SQL Push** — Model sonuçlarını SQL Server'a yazma

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

### Geliştirme tarafı

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

### İzleme tarafı

```
Referans (SQL/CSV) ──┐
                     ├─► compute.py (background thread)
İzleme  (SQL/CSV) ──┘         │
                               ├─ ref_summary        (tek dict, ~10-30 KB)
                               ├─ period_summaries[]  (dönem başı ~10-50 KB)
                               └─ ref_df.pkl          (göç matrisi için ham ref)
                                       │
                   ┌───────────────────┼───────────────────┐
                   │                   │                   │
             _MON_STORE          profile.py           tabs/*.py
          (server-side cache)  (kaydet/yükle/sil)   (6 metrik tabı)
```

**Dönemsel özet yapısı:** Her dönem için `rating_counts[25]`, `rating_defaults[25]`, `var_psi{değişken→bin counts}` ve opsiyonel `migration_matrix[25×25]` saklanır. Tüm metrikler bu özetlerden hesaplanır — ham veriye gerek yoktur.

**Kümülatif hesaplama:** Dönemsel özetlerin count'ları element bazlı toplanır. Yeni bir hesaplama gerektirmez.

**Artımlı güncelleme:** Profil yüklendiğinde SQL'den `WHERE tarih > last_period` ile sadece yeni satırlar çekilir, yeni dönemlerin özetleri hesaplanıp mevcut listeye eklenir.

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
├── server_state.py         # Paylaşılan state (_SERVER_STORE, _PRECOMPUTE_PROGRESS, _MON_STORE)
├── benchmark.py            # Performans test scripti
├── setup_deps.py           # Otomatik bağımlılık yükleyici
├── pip_prefix.txt          # Kurumsal pip prefix (opsiyonel)
├── config.toml             # DB bağlantı ayarları
├── assets/
│   └── custom.css          # Dark tema stilleri
├── data/
│   └── loader.py           # SQL bağlantısı
├── layout/
│   ├── __init__.py         # build_layout() — sidebar, tab yapıları, store'lar
│   └── izleme.py           # İzleme sekmesi layout'u (config, tab'lar, progress modal)
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
│   ├── playground.py       # Grafik + Hızlı Model + SHAP
│   ├── results.py          # Sonuç sekmesi — detaylı model raporu
│   ├── profile.py          # Profil kaydet/yükle/sil + model kaydet/yükle
│   └── izleme/             # ── İzleme modülü ──
│       ├── __init__.py     # Alt modül import'ları
│       ├── compute.py      # Hesaplama motoru (ref/dönem özeti, kümülatif, background thread)
│       ├── data.py         # Veri yükleme, config, format modal, yapılandırma onayı
│       ├── nav.py          # İzleme içi navigasyon
│       ├── profile.py      # İzleme profil kaydet/yükle/sil + artımlı güncelleme
│       └── tabs/           # ── Metrik tab callback'leri ──
│           ├── psi.py      # Değişken PSI + Rating PSI (trend + kümülatif)
│           ├── disc.py     # Gini/KS — Accuracy Ratio (trend + kümülatif)
│           ├── badrate.py  # Temerrüt Oranı (trend + kümülatif)
│           ├── hhi.py      # HHI konsantrasyon (trend + kümülatif)
│           ├── backtest.py # Binomial test (trend + kümülatif)
│           └── migration.py# Göç matrisi heatmap (trend + kümülatif)
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

### v2.0 — İzleme (Model Monitoring)
- **İzleme Sekmesi** — Canlıya alınmış modellerin dönemsel performans takibi; Referans + İzleme veri yükleme, tek tıkla yapılandırma
- **6 Metrik Tabı** — PSI (Değişken + Rating), Gini/KS, Temerrüt Oranı, HHI, Backtesting (Binomial Test), Göç Matrisi
- **Trend + Kümülatif** — Her tab'da dönem seçici dropdown + trend çizgi grafiği + tüm dönemlerin birleşik kümülatif sonucu
- **Özet Tabanlı Mimari** — Ham veri saklanmaz; dönemsel özetler (~KB boyutunda) profilde tutulur, tüm metrikler özetlerden hesaplanır
- **Artımlı Güncelleme** — Yeni veri geldiğinde sadece yeni dönemlerin özetleri hesaplanır, eski sonuçlar korunur
- **Kolon Şeması Koruması** — SQL tablosunda kolon değişikliği algılanır, uyumsuzlukta hesaplama durdurulur
- **Olgunlaşma Filtresi** — Sonuç bağımlı metrikler (KS, Gini, Temerrüt Oranı, Backtesting) yalnızca olgun dönemleri kullanır
- **WoE Entegrasyonu** — OPT pickle yüklenerek değişken PSI'da WoE bin'leri kullanılabilir; diğer metrikler WoE'den bağımsız
- **İzleme Profilleri** — Ayrı profil sistemi (kaydet/yükle/sil); `ref_summary.pkl` + `period_summaries.pkl` + `ref_df.pkl` + `opt.pkl`
- **Background Compute** — Hesaplama ayrı thread'de çalışır, iptal edilebilir; ilerleme progress bar ile gösterilir
- **PSI Trend Grafiği** — Rating PSI (kalın kırmızı) + her değişkenin PSI'ı (ayrı renkli ince çizgiler) tek grafikte
- **Format Modal** — CSV ayırıcı seçimi ve önizleme (izleme tarafı için)

### v1.9
- **Modelleme Filtreleri** — HHI accordion, profil yükleme düzeltmeleri

### v1.8
- **Thread Cancellation** — Precompute iptal desteği, ham değerler precompute'a eklendi, mutlak korelasyon

### v1.7
- **Sonuç Sekmesi** — Playground'dan ayrı, detaylı model raporu: Metrikler, Model Özeti, ROC Eğrisi, Confusion Matrix (Train/Test/OOT yan yana), SHAP Beeswarm; tümü accordion ile aç/kapat
- **Playground Sadeleştirme** — Playground artık sadece Gini kartları + özet katsayı tablosu gösterir; detaylı grafikler Sonuç sekmesine taşındı
- **Ham / WoE Alt Sekmeleri** — Sonuç sekmesinde her iki model sonucu ayrı tab'larda incelenebilir
- **Profil Klasörüne Export** — Model Pickle, OPT Pickle ve Excel artık browser download yerine `profiles/<profil_adı>/` klasörüne kaydedilir
- **Tab-Aware Pickle** — Ham sekmede raw model, WoE sekmede WoE model pickle'ı oluşturulur; OPT pickle sadece WoE sekmesinde aktif
- **Model Kaydet / Yükle / Sil** — Modeller profil meta.json'a parametreleriyle birlikte kaydedilir, dropdown ile geri yüklenebilir, silinebilir
- **SQL Push** — Model sonuçlarını doğrudan SQL Server tablosuna yazma
- **State Reset** — Veri yüklendiğinde veya yapılandırma değiştiğinde eski model sonuçları, profil bağlantısı ve kayıtlı model paneli otomatik temizlenir
- **OPT Pickle Bugfix** — Eski 2-tuple WoE cache formatında `_opt_dict` boş geliyordu; `_build_woe_dataset` ile yeniden hesaplanarak düzeltildi
- **SmLogitWrapper** — statsmodels Logit sonucu modül düzeyinde sarmalanarak pickle uyumlu hale getirildi
- **Describe Sekmesi** — Profiling sekmesi "Describe" olarak yeniden adlandırıldı
- **Sticky Navbar** — Sekme navigasyonu sabitlendi, scroll ile kaybolmaz

### v1.5
- **Loading Slideshow** — 8 eğitim slaytı (EDA Lab nedir, Veri Yükleme, Önizleme, Target & IV, Deep Dive, İstatistiksel Testler, Değişken Özeti, Playground); otomatik 8-saniye ilerleme; tıklanabilir navigasyon noktaları (●○○○); geçen süre sayacı
- **CDN Reachability Check** — Google Fonts ve Bootstrap Icons iş ağında bloke olunca düşüş kalmıyor; conditional loading ile graceful fallback
- **Playground WoE + LR** — Logistic Regression (statsmodels) için StandardScaler otomatik atlanıyor, sm.Logit doğrudan WoE değerlerine uygulanıyor
- **Train/Test Split** — Otomatik stratify (target ≤10 unique değerse)
- **Date Sorting** — Veri yüklendikten hemen sonra date column'a göre sıralanıyor (IV tutarlılığı)
- **Playground Test Oranı** — UI'dan kaldırıldı, config değeri doğrudan kullanılıyor
- **İstatistiksel Test Açıklamaları** — Tüm 5 test (Korelasyon, Chi², ANOVA, KS, VIF) yeniden yazıldı; kullanıcı-dostu "Çıktıyı nasıl okurum" bölümleri
- **CSS v2.1** — Slideshow animasyonları, responsive tasarım, progress bar, Dash 4.0 dropdown refactor

### v1.6
- **Binary-Only Odak** — continuous, multiclass ve categorical target desteği kaldırıldı; tüm sekmeler yalnızca binary (0/1) target için çalışır
- **Kod Temizliği** — `detect_target_type()` fonksiyonu, Mutual Information hesabı, Ridge/Regressor modelleri, Pearson/Spearman korelasyon, Kruskal-Wallis testi, regression metrikleri (R²/RMSE/MAE), multiclass metrikleri (Accuracy/F1 Macro) silindi
- **.bak dosyaları silindi** — `playground.py.bak` ve `layout/__init__.py.bak` temizlendi

### v1.4
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
