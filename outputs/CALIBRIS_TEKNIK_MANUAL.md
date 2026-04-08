# Calibris — Teknik Bakim Manueli

> Bu dokuman, sistemde bir sorun oldugunda veya degisiklik yapilmasi gerektiginde
> hizlica mudahale edebilmeniz icin hazirlanmistir.

---

## 1. DOSYA YAPISI — GENEL BAKIS

```
Preprocessing/
├── app.py                  ← Uygulama baslatici (port 8060)
├── app_instance.py         ← Dash app singleton
├── server_state.py         ← Sunucu tarafi cache (_SERVER_STORE)
├── config.toml             ← SQL Server varsayilan ayarlar
├── setup_deps.py           ← Bagimliliklari kontrol/yukle
│
├── layout/
│   └── __init__.py         ← Tum UI yapisi (sekmeler, sidebar, modallar)
│
├── callbacks/              ← Tum kullanici etkilesim mantigi
│   ├── __init__.py         ← Callback import kaydi
│   ├── data_loading.py     ← Veri yukleme (SQL + CSV)
│   ├── precompute.py       ← Arka plan hesaplama orkestratoru
│   ├── preview.py          ← Onizleme + screening + uzman elemesi
│   ├── profiling.py        ← Describe (kolon istatistikleri)
│   ├── target_iv.py        ← Target & IV sekmesi
│   ├── deep_dive.py        ← Degisken Analizi (WoE tablo + grafik)
│   ├── correlation.py      ← Korelasyon + isi haritasi
│   ├── stat_tests.py       ← Chi², ANOVA, KS, VIF
│   ├── var_summary.py      ← Degisken Ozeti tablosu
│   ├── outlier.py          ← Outlier tespiti
│   ├── playground.py       ← Model kurma (LR, LightGBM, XGBoost)
│   ├── results.py          ← Model sonuclari + export
│   ├── profile.py          ← Profil kaydet/yukle
│   └── help_overlay.py     ← Yardim modali
│
├── modules/                ← Saf hesaplama fonksiyonlari (Dash bagimsiiz)
│   ├── target_analysis.py  ← Target istatistikleri
│   ├── screening.py        ← Degisken kalite filtresi
│   ├── profiling.py        ← Kolon bazli istatistikler
│   ├── correlation.py      ← Korelasyon matrisi + VIF
│   └── deep_dive.py        ← WoE/IV hesaplama + format + monotonluk
│
├── utils/                  ← Yardimci fonksiyonlar
│   ├── helpers.py          ← Segment filtre, train/test split, numerik coerce
│   ├── chart_helpers.py    ← Grafik tema, PSI hesaplama (TEK KAYNAK)
│   └── anomaly_hints.py    ← Uyari kartlari (dusuk IV, yuksek PSI, vs.)
│
└── data/
    └── loader.py           ← SQL Server baglanti + veri cekme
```

---

## 2. VERI AKISI — BASTAN SONA

```
KULLANICI                    SISTEM                         CACHE
─────────                    ──────                         ─────
Veri yukle          →  data_loading.py
                           ↓
                       _SERVER_STORE[uuid]              ← Ham veri (ASLA degismez)
                       _SERVER_STORE[uuid_quality]      ← Tip donusum bilgisi
                           ↓
Yapilandir          →  precompute.py (arka plan thread)
                           ↓
                       Adim 1: Screening               → uuid_screen
                       Adim 2: WoE / IV                → uuid_ds_*_woe_tables
                                                         uuid_ds_*_bins
                                                         uuid_ds_*_optb
                                                         uuid_ds_*_train_woe
                                                         uuid_ds_*_test_woe
                                                         uuid_ds_*_oot_woe
                                                         uuid_iv_*
                       Adim 3: Korelasyon              → uuid_corr_*
                       Adim 4: Degisken Ozeti          → uuid_varsummary_*
                           ↓
Sekme sec           →  Ilgili callback cache'den okur (hesaplama YAPMAZ)
```

**Kritik Prensip:** Hicbir callback kendi basina WoE/IV hesaplamaz. Hepsi `precompute.py`'nin yazdigi cache'i okur.

---

## 3. CACHE ANAHTAR YAPISI

Tum anahtarlar `_SERVER_STORE` sozlugunde tutulur (`server_state.py`).

**Prefix formati:**
- `key` = UUID (veri yukleme sirasinda olusur)
- `_pfx` = `{key}_ds_{seg_col}_{seg_val}`

| Anahtar | Icerik | Yazan | Okuyan |
|---------|--------|-------|--------|
| `{key}` | Ham DataFrame (orijinal) | data_loading | Hepsi |
| `{key}_quality` | Tip donusum raporu | data_loading | preview |
| `{key}_screen` | (gecen_kolonlar, eleme_raporu) | precompute | preview, deep_dive |
| `{_pfx}_woe_tables` | `{col: {train_table, test_table, oot_table, iv_train, iv_test, iv_oot, monoton_test, monoton_oot}}` | precompute | deep_dive, var_summary, playground |
| `{_pfx}_bins` | `{col: [edge1, edge2, ...]}` | precompute | deep_dive, var_summary |
| `{_pfx}_optb` | `{col: OptimalBinning nesnesi}` | precompute | var_summary |
| `{_pfx}_train_woe` | WoE donusumlu train DataFrame | precompute | korelasyon, playground |
| `{_pfx}_test_woe` | WoE donusumlu test DataFrame | precompute | playground |
| `{_pfx}_oot_woe` | WoE donusumlu OOT DataFrame | precompute | playground |
| `{_pfx}_train` | Raw train DataFrame | precompute | deep_dive, stat_tests |
| `{_pfx}_test` | Raw test DataFrame | precompute | deep_dive |
| `{_pfx}_oot` | Raw OOT DataFrame | precompute | deep_dive |
| `{key}_iv_{seg}_{val}` | IV siralama tablosu (DataFrame) | precompute | target_iv, var_summary, preview, playground |
| `{key}_corr_{seg}_{val}` | Korelasyon matrisi | precompute | correlation |
| `{key}_varsummary_{seg}_{val}` | Degisken ozeti tablosu | precompute | var_summary |
| `{key}_profile_{seg}_{val}` | Profiling sonucu | precompute | profiling |

---

## 4. WoE / IV HESAPLAMA DETAYI

### 4.1 Nerede hesaplaniyor?
**`callbacks/precompute.py`** — Adim 2 (satir ~165-295)

### 4.2 Hesaplama akisi

```python
for col in passed_cols:
    # 1. OptimalBinning fit
    optb = OptimalBinning(
        name=col, monotonic_trend="auto_asc_desc",
        max_n_bins=_mb, dtype="numerical", solver="cp",
        special_codes={'special_1': 9999999999, 'special_2': 8888888888},
    )
    optb.fit(df_train[col].values, df_train[target].values)

    # 2. Train tablosu
    raw_bt = optb.binning_table.build(show_digits=8)
    train_bt = format_binning_table(raw_bt)        # → UI formati

    # 3. IV: raw tablodan oku
    iv_train = round(float(raw_bt.loc["Totals", "IV"]), 4)

    # 4. WoE transform (3 ayri DataFrame — ayni optb nesnesi ile)
    train_woe_df[col] = optb.transform(df_train[col].values, metric="woe",
                         metric_missing="empirical", metric_special="empirical")
    test_woe_df[col]  = optb.transform(df_test[col].values, metric="woe",
                         metric_missing="empirical", metric_special="empirical")
    oot_woe_df[col]   = optb.transform(df_oot[col].values, metric="woe",
                         metric_missing="empirical", metric_special="empirical")

    # 5. Test/OOT gosterim tablolari (train WoE degerleri ile)
    test_bt = build_period_table(df_test, col, target, edges, train_bt)
    oot_bt  = build_period_table(df_oot,  col, target, edges, train_bt)
```

### 4.3 Format fonksiyonlari (`modules/deep_dive.py`)

| Fonksiyon | Ne yapar | Satir |
|-----------|----------|-------|
| `format_binning_table(bt)` | OptBinning ciktisini UI formatina cevirir | ~234 |
| `build_period_table(df, col, target, edges, train_bt)` | Test/OOT tablosu (train WoE ile) | ~299 |
| `_iv_label(iv)` | IV guc etiketi (Cok Zayif → Supheli) | ~225 |
| `_check_monotonicity(bt)` | Bad Rate yonu kontrol | ~410 |

### 4.4 Etiket donusumleri

| OptBinning | UI Formati |
|------------|-----------|
| `Totals` | `TOPLAM` |
| `Missing` | `Eksik` |
| `special_1` | `Special (9999999999)` |
| `special_2` | `Special (8888888888)` |
| `Count` | `Toplam` |
| `Count (%)` | `Toplam (%)` |
| `Event rate` | `Bad Rate %` (x100) |

### 4.5 Satir sirasi
```
bin araliklari → Eksik → Special(lar) → TOPLAM
```

### 4.6 Yuvarlama kurallari
- **IV, Toplam (%), Bad Rate %**: 4 ondalik
- **WoE**: 8 ondalik
- **Toplam, Non-event, Event**: Tam sayi (float olarak tutulur)
- **TOPLAM satirinda WoE**: `np.nan` (bos)

### 4.7 Special deger yonetimi
```python
SPECIAL_CODES = {'special_1': 9999999999, 'special_2': 8888888888}
```
- Degiskende special deger >= %2 ise → `max_n_bins=2` (config ile degistirilebilir)
- Diger degiskenlerde → `max_n_bins=4` (veya config degeri)
- `build_period_table` sadece train tablosunda bulunan special/eksik satirlari ekler

---

## 5. PSI HESAPLAMA

### 5.1 Tek kaynak: `utils/chart_helpers.py` → `calc_psi()`

**Iki modu var:**

| Mod | `discrete` | Kullanim | Binning yontemi |
|-----|-----------|----------|-----------------|
| WoE sekmesi | `True` | Bin bazli dagilim karsilastirmasi | WoE bin'leri |
| Ham sekme | `False` | Numerik dagilim karsilastirmasi | `np.percentile` (quantile) |

### 5.2 PSI esikleri
```
< 0.10  → Stabil
0.10-0.25 → Hafif Kayma
> 0.25  → Kritik Kayma
```

### 5.3 Deep dive'da PSI nereden okunur?
- **WoE sekmesi:** `woe_tables[col]` icindeki `train_table` ve `oot_table` satirlarindan
- **Ham sekme:** `calc_psi(df_train[col], df_oot[col], discrete=False)`

---

## 6. VIF HESAPLAMA

### 6.1 Nerede: `modules/correlation.py` → `compute_vif()`
### 6.2 Yontem: `statsmodels.variance_inflation_factor` + `add_constant`
### 6.3 Full sample kullanir (sampling yok)
### 6.4 Esikler: `<5 Normal` | `5-10 Orta` | `>10 Yuksek`

---

## 7. TRAIN / TEST / OOT SPLIT

### 7.1 Nerede: `utils/helpers.py` → `get_splits(df, config)`

### 7.2 Mantik:
```
OOT tarihi varsa:
  - OOT oncesi veri → train_test havuzu
  - OOT sonrasi veri → df_oot
  - has_test_split=True ise → train_test havuzundan %test_size ayrilir
  - has_test_split=False ise → df_test = None

OOT tarihi yoksa:
  - has_test_split=True ise → rastgele %test_size ayrilir
  - has_test_split=False ise → tum veri train, df_test = None, df_oot = None
```

---

## 8. PRECOMPUTE ADIMLARI

`callbacks/precompute.py` → `_run_precompute()` fonksiyonu (arka plan thread)

| Adim | Islem | Sure (tipik) | Cache anahtari |
|------|-------|-------------|----------------|
| 1 | Screening + Profiling | 1-3 sn | `{key}_screen`, `{key}_profile_*` |
| 2 | WoE / IV (tum degiskenler) | 10-60 sn | `{_pfx}_woe_tables`, `{_pfx}_bins`, `{_pfx}_optb`, `{_pfx}_*_woe`, `{key}_iv_*` |
| 3 | Korelasyon matrisi | 1-5 sn | `{key}_corr_*` |
| 4 | Degisken ozeti | 2-10 sn | `{key}_varsummary_*` |

### Ilerleme takibi:
```python
_PRECOMPUTE_PROGRESS[prog_key] = {
    "step": 2,           # mevcut adim
    "durations": {...},   # her adimin suresi
    "done": False         # tamamlandi mi
}
```
- `interval-precompute` (1000ms) ile UI yoklar
- Tamamlaninca `interval-precompute.disabled = True` → callback'ler tetiklenir

---

## 9. CALLBACK TETIKLEME ZAMANLARI

Bazi callback'ler `store-config` degistiginde hemen tetiklenir ama o anda precompute bitmemis olabilir. Bu yuzden su callback'lere `Input("interval-precompute", "disabled")` eklendi:

| Callback | Dosya | Neden |
|----------|-------|-------|
| `update_target_iv` | target_iv.py | IV tablosu precompute'dan geliyor |
| `update_var_summary` | var_summary.py | Degisken ozeti precompute'dan geliyor |
| `render_pg_var_summary_preview` | playground.py | IV onizleme precompute'dan geliyor |

**Deep dive etkilenmez** — kullanici degisken secene kadar precompute bitmis olur.

---

## 10. MODEL PLAYGROUND

### 10.1 Nerede: `callbacks/playground.py`

### 10.2 Model tipleri:
| Tip | Kutuphane | Ozel ayar |
|-----|-----------|-----------|
| Logistic Regression | `statsmodels.Logit` | `method="bfgs"`, WoE'de scaling yok |
| LightGBM | `lightgbm.LGBMClassifier` | 200 tree, lr=0.05 |
| XGBoost | `xgboost.XGBClassifier` | 200 tree, otomatik class weight |

### 10.3 Iki varyant otomatik:
1. **Ham Veri Modeli:** Orijinal degerler + StandardScaler (LR icin)
2. **WoE Modeli:** WoE donusumlu degerler (scaling yok)

### 10.4 Null stratejileri:
- LR: Median / Ortalama / Mod / Sifir / Reddet
- Tree: Koru (native) / Median / Ortalama / Mod / Sifir / Reddet

### 10.5 Esik yontemleri:
- Sabit (0.50)
- F1 Maksimizasyon
- KS Noktasi (max TPR-FPR)
- Ozel esik

### 10.6 Metrikler:
- AUC-ROC, Gini, KS, F1, Precision, Recall
- Confusion matrix, olasilik dagilimi
- LR: katsayilar + p-value + AIC/BIC
- Tree: feature importance + SHAP beeswarm

---

## 11. SORUN GIDERME REHBERI

### 11.1 "IV hesaplanamadi" veya IV tablosu bos
**Neden:** Precompute bitmeden callback tetiklendi.
**Kontrol:** `interval-precompute.disabled` Input'u var mi? (target_iv, var_summary, playground)
**Cozum:** Callback'e `Input("interval-precompute", "disabled")` ekle.

### 11.2 Test/OOT tablolari gorunmuyor
**Olasi nedenler:**
1. Config'de `has_test_split=True` veya `oot_date` yok → `get_splits()` None doner
2. `build_period_table` satir uyusmazligi → console'da `WARNING: build_period_table satir uyusmazligi` ara
3. `df_test` / `df_oot` bos

**Kontrol:** Precompute loglarinda `WoE/IV — df_test: ... df_oot: ...` satirina bak.

### 11.3 IV degeri yanlis (eskisinden farkli)
**Olasi neden:** Yuvarlama.
**Kontrol:** `format_binning_table` ve `build_period_table`'daki `.round(4)` degerleri.
**IV kaynagi:** `raw_bt.loc["Totals", "IV"]` (OptBinning'in kendi hesabi, yuvarlanmamis).
Sonra `round(..., 4)` uygulanir.

### 11.4 VIF uyusmuyor
**Kontrol:** `modules/correlation.py` → `compute_vif()` artik `statsmodels.variance_inflation_factor` kullaniyor, sampling yok (full data).
**Fark sebebi:** VIF degisken setine baglidir — farkli sayida degiskenle hesaplama farkli sonuc verir.

### 11.5 WoE degerleri cok uzun (15+ basamak)
**Kontrol:** `format_binning_table` ve `build_period_table`'da WoE `.round(8)` olmali.

### 11.6 PSI grafiklerinde tum veri tek bin'de
**Neden:** Eski `np.linspace` binning, carpik veride ilk bin'e yigilir.
**Cozum (mevcut):** `chart_helpers.py`'de `np.percentile` (quantile-based) kullaniyor.
**Sadece ham sekme icin gecerli** — WoE sekmesi binning table'dan hesaplar.

### 11.7 Uygulama baslamiyor / import hatasi
**Kontrol:** `setup_deps.py` tum bagimliliklari kontrol eder.
**Manuel:** `pip install dash dash-bootstrap-components pandas numpy plotly scipy scikit-learn statsmodels lightgbm xgboost shap optbinning pyodbc`

### 11.8 Special degerler tabloda gorunmuyor
**Kontrol:** `SPECIAL_CODES` dict'i `modules/deep_dive.py` basinda tanimli.
Degiskende special deger yoksa OptBinning satir olusturmaz → tabloda da gorunmez (normal).

### 11.9 Precompute takiliyor / bitmiyor
**Kontrol:** Console loglarinda `WoE basarisiz: {col} — {hata}` ara.
**Olasi neden:** Bir degiskende OptBinning exception firlatir (sabit kolon, tip uyusmazligi).
`failed_cols` listesine eklenir, diger degiskenler devam eder.

### 11.10 Callback "duplicate output" hatasi
**Neden:** Ayni Output iki farkli callback'te kullaniliyor.
**Cozum:** Ikinciye `allow_duplicate=True` ekle.

---

## 12. DEGISIKLIK YAPMADAN ONCE KONTROL LISTESI

1. **Cache anahtari degistiriyorsan** → o anahtari okuyan TUM callback'leri bul:
   ```
   grep -r "anahtar_adi" callbacks/ modules/ utils/
   ```

2. **Yeni kolon ekliyorsan tabloya** → `format_binning_table` ve `build_period_table`'daki kolon listelerini guncelle

3. **Yuvarlama degistiriyorsan** → su dosyalarda tutarli ol:
   - `modules/deep_dive.py` (format_binning_table + build_period_table)
   - `callbacks/precompute.py` (iv_train, iv_test, iv_oot, iv_df)

4. **Yeni sekme/callback ekliyorsan** → `callbacks/__init__.py`'ye import ekle

5. **OptBinning parametreleri degistiriyorsan** → sadece `precompute.py`'deki loop'u degistir (tek yer)

---

## 13. ONEMLI SABITLER

| Sabit | Deger | Dosya | Aciklama |
|-------|-------|-------|----------|
| `SPECIAL_CODES` | `{'special_1': 9999999999, 'special_2': 8888888888}` | modules/deep_dive.py | Ozel deger kodlari |
| `_SPECIAL_RATIO_THRESHOLD` | `0.02` (%2) | modules/deep_dive.py | Degiskeni "special" yapan esik |
| `max_n_bins` (special) | `2` | precompute.py | Special kolon icin max bin |
| `max_n_bins` (diger) | `4` (config'den) | precompute.py | Normal kolon icin max bin |
| `monotonic_trend` | `"auto_asc_desc"` | precompute.py | OptBinning monotonluk |
| `solver` | `"cp"` | precompute.py | OptBinning optimizasyon |
| Port | `8060` | app.py | Web sunucu portu |
| PSI stabil esik | `0.10` | chart_helpers.py | PSI < 0.10 = Stabil |
| PSI kayma esik | `0.25` | chart_helpers.py | PSI > 0.25 = Kritik |
| VIF normal esik | `5.0` | correlation.py | VIF < 5 = Normal |
| VIF yuksek esik | `10.0` | correlation.py | VIF > 10 = Yuksek |
| Eksik eleme esik | `0.60` (%60) | screening.py | Screening varsayilan |
| IV cok zayif | `< 0.02` | deep_dive.py | _iv_label() |
| IV zayif | `0.02 – 0.10` | deep_dive.py | _iv_label() |
| IV orta | `0.10 – 0.30` | deep_dive.py | _iv_label() |
| IV guclu | `0.30 – 0.50` | deep_dive.py | _iv_label() |
| IV supheli | `> 0.50` | deep_dive.py | _iv_label() |

---

## 14. HARICI KUTUPHANELER

| Kutuphane | Kullanim yeri | Ne icin |
|-----------|--------------|---------|
| `dash` | Tum proje | Web framework |
| `dash-bootstrap-components` | Layout + callbacks | UI bilesenleri |
| `pandas` | Tum proje | Veri manipulasyonu |
| `numpy` | Tum proje | Numerik hesaplama |
| `plotly` | Callbacks | Interaktif grafikler |
| `scipy` | stat_tests | Istatistiksel testler |
| `scikit-learn` | playground, deep_dive | ML algoritmalari + metrikler |
| `statsmodels` | playground, correlation | Logit model, VIF |
| `optbinning` | precompute, deep_dive | Optimal binning (WoE/IV) |
| `lightgbm` | playground | Gradient boosting |
| `xgboost` | playground | Gradient boosting |
| `shap` | playground, results | Model aciklanabilirligi |
| `pyodbc` | data/loader | SQL Server baglanti |

---

## 15. SIKCA YAPILAN ISLEMLER

### Yeni special deger eklemek:
1. `modules/deep_dive.py` → `SPECIAL_CODES` dict'ine ekle
2. Baska bir sey degistirmeye gerek yok — `format_binning_table` ve `build_period_table` otomatik uyum saglar

### IV esiklerini degistirmek:
1. `modules/deep_dive.py` → `_iv_label()` fonksiyonundaki if'leri degistir
2. `callbacks/target_iv.py` → vline esiklerini guncelle (satir ~192)
3. `callbacks/var_summary.py` → oneri kurallarindaki IV esiklerini guncelle

### Max bin sayisini degistirmek:
1. `callbacks/precompute.py` → `_max_bins = int(config.get("max_bins", 4))` satirini degistir
2. Veya config'e `max_bins` parametresi ekle

### Yeni model tipi eklemek:
1. `callbacks/playground.py` → `_MODEL_PARAMS` dict'ine ekle
2. `_fit_and_render()` fonksiyonuna elif blogu ekle
3. Model tipi dropdown'ina option ekle (satir ~448)
