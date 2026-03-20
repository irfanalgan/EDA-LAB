# "Yapılandır" Butonu — Hesaplama Akışı

> Bu belge, kullanıcı "Yapılandır" butonuna bastıktan sonra arka planda çalışan
> `_run_precompute_background()` fonksiyonundaki TÜM hesaplama adımlarını,
> sırasıyla, kullandığı fonksiyonları, formülleri ve cache key'lerini belgelemektedir.

---

## Giriş: Callback Tetiklenmesi

| | |
|---|---|
| **Dosya** | `callbacks/precompute.py:277` |
| **Callback** | `confirm_config()` — `btn-confirm` butonuna tıklanınca tetiklenir |
| **Girdi** | target_col, date_col, oot_date, segment_col, segment_val, train_test_split, test_size, key |
| **Yaptığı** | Config'i `store-config`'e yazar, progress modal'ı açar, arka plan thread'i başlatır |
| **Thread** | `_run_precompute_background(prog_key, key, target, date_col, seg_col, seg_val, config)` |

**Ön Hazırlık** (satır 114-127):
```
df_orig  = _SERVER_STORE[key]           ← SQL'den çekilen ham veri
df_active = apply_segment_filter(df_orig, seg_col, seg_val)   ← segment filtresi
_pfx     = f"{key}_ds_{seg_col}_{seg_val}"                    ← cache prefix
```

**Skip kontrolü** (satır 130): `{_pfx}_train` zaten varsa hiçbir şey yapmadan döner.

---

## ADIM 0: ÖN ELEME (Screening)

| | |
|---|---|
| **Dosya** | `modules/screening.py:5` |
| **Fonksiyon** | `screen_columns(df_active, target, date_col, seg_col, missing_threshold=0.80)` |
| **Girdi** | `df_active` (segment filtrelenmiş tüm veri) |
| **Cache Key** | `{key}_screen` → `(passed_list, screen_report_df)` |

### Mantık
Her kolon için (target, date, segment hariç):
1. `missing_pct = kolon.isna().sum() / len(df)`
2. `nunique = kolon.nunique(dropna=True)`

### Eleme Kuralları
| Kural | Koşul | Sonuç |
|---|---|---|
| Yüksek Eksik | `missing_pct > 0.80` | ELEME |
| Sabit Değişken | `nunique <= 1` | ELEME |
| Geçti | Diğer | `passed` listesine eklenir |

---

## ADIM 1: PROFİL ANALİZİ

| | |
|---|---|
| **Dosya** | `modules/profiling.py:5` |
| **Fonksiyon** | `compute_profile(df_active)` |
| **Girdi** | `df_active` |
| **Cache Key** | `{key}_profile_{seg_col}_{seg_val}` → DataFrame |

### Hesaplanan Metrikler (her kolon için)
| Metrik | Formül |
|---|---|
| Eksik Sayı | `s.isna().sum()` |
| Eksik % | `eksik / n * 100` |
| Tekil Değer | `s.nunique(dropna=True)` |
| En Sık Değer | `s.mode(dropna=True).iloc[0]` |
| En Sık % | `(s == mode_val).sum() / n * 100` |
| Ortalama | `non_null.mean()` (sadece numerik) |
| Std | `non_null.std()` (sadece numerik) |
| Min / Max | `non_null.min()` / `non_null.max()` |
| Percentiller | P1, P5, P10, P25, P50(Medyan), P75, P90, P95, P99 |

---

## ADIM 2: SPLIT + WoE + IV + WoE Tabloları

Bu en büyük adım. Alt bölümlere ayrılmıştır.

---

### ADIM 2a: Veriyi Böl (Train / Test / OOT)

| | |
|---|---|
| **Dosya** | `utils/helpers.py:153` |
| **Fonksiyon** | `get_splits(df_active, config)` |
| **Girdi** | `df_active`, config dict |
| **Cache Key'ler** | `{_pfx}_train`, `{_pfx}_test`, `{_pfx}_oot` |

### Mantık

```
EĞER date_col varsa:
    df = df.sort_values(date_col)          ← tarih sıralı, INDEX KORUNUR (reset_index YOK)

EĞER oot_date VE date_col varsa:
    dates = pd.to_datetime(df[date_col])
    df_pool = df[dates < oot_date]         ← OOT öncesi
    df_oot  = df[dates >= oot_date]        ← OOT dönemi

    EĞER has_test_split VE len(df_pool) >= 20:
        df_train, df_test = sklearn.train_test_split(
            df_pool, test_size=test_size%, random_state=42,
            stratify=target  EĞER target.nunique() <= 10)
    DEĞİLSE:
        df_train = df_pool, df_test = None
DEĞİLSE:
    df_oot = None
    EĞER has_test_split VE len(df) >= 20:
        df_train, df_test = sklearn.train_test_split(...)
    DEĞİLSE:
        df_train = df, df_test = None
```

**ÖNEMLİ:** `reset_index()` YAPILMAZ — orijinal index korunur. Bu sayede `df_active[idx]` ile `df_train[idx]` aynı satırı gösterir.

---

### ADIM 2b: Değişken Listesi Belirleme

| | |
|---|---|
| **Dosya** | `callbacks/precompute.py:162-168` |

```
EĞER screening geçtiyse:
    var_list = screen_data[0]  (target hariç)
DEĞİLSE:
    var_list = df_train.columns  (target, date, segment hariç)
```

---

### ADIM 2c: WoE Encoding + IV Hesaplama

| | |
|---|---|
| **Dosya** | `utils/chart_helpers.py:79` |
| **Fonksiyon** | `build_woe_datasets(df_train, df_test, df_oot, target, var_list)` |
| **Girdi** | 3 split DataFrame + target + değişken listesi |

#### Her değişken için tek döngü (satır 116-175):

**1. OptimalBinning FIT (sadece train üzerinde):**
```python
optb = OptimalBinning(
    name=col,
    monotonic_trend="auto_asc_desc",
    max_n_bins=4,
    solver="cp",
    dtype="numerical" veya "categorical",
    special_codes=[9999999999, 8888888888]    # sadece numerik
)
optb.fit(X_train, y_train)     # y_train: target, int, NaN temizlenmiş
```

**2. IV değeri (train'den):**
```python
bt = optb.binning_table.build(show_digits=8)
iv = float(bt.loc["Totals", "IV"])
```
OptBinning kütüphanesi kendi IV formülünü kullanır:
```
IV_i = (dist_bad_i - dist_good_i) × ln(dist_bad_i / dist_good_i)

dist_bad_i  = bad_sayısı_bin_i / toplam_bad
dist_good_i = good_sayısı_bin_i / toplam_good

IV_total = Σ IV_i  (tüm bin'ler)
```

**3. Bin Edges (sadece numerik):**
```python
splits = list(optb.splits)           # [3.5, 7.2, 12.0]  gibi
edges  = [-inf] + splits + [+inf]    # [-inf, 3.5, 7.2, 12.0, +inf]
```

**4. WoE Transform (her split için):**
```python
# TRAIN:
woe_train = optb.transform(df_train[col].values,
                            metric="woe",
                            metric_missing="empirical",
                            metric_special="empirical")

# TEST (varsa):
woe_test = optb.transform(df_test[col].values, ...)

# OOT (varsa):
woe_oot = optb.transform(df_oot[col].values, ...)
```
WoE formülü (optbinning dahili):
```
WoE_i = ln(dist_bad_i / dist_good_i)
```
`metric_missing="empirical"` → eksik değerler, kendi bin'lerindeki gerçek oranla WoE alır.
`metric_special="empirical"` → özel değerler de aynı şekilde.

**5. Eksik % hesabı:**
```python
EĞER df_test varsa:
    _eksik_df = pd.concat([df_train, df_test])
DEĞİLSE:
    _eksik_df = df_train

eksik_pct = round(_eksik_df[col].isna().mean() * 100, 2)
```
**NOT:** Eksik % hesabında OOT dahil DEĞİL, sadece train + test.

**6. Başarısız değişkenler:**
Exception fırlarsa → `failed` listesine eklenir, IV=0.0 atanır.

#### Çıktı Cache Key'leri:

| Cache Key | Değer |
|---|---|
| `{_pfx}_train_woe` | DataFrame (n_train × n_vars) — her hücre WoE değeri |
| `{_pfx}_test_woe` | DataFrame (n_test × n_vars) veya None |
| `{_pfx}_oot_woe` | DataFrame (n_oot × n_vars) veya None |
| `{key}_iv_{seg_col}_{seg_val}` | DataFrame: [Değişken, IV, Eksik %] — IV'ye göre azalan sıralı |
| `{_pfx}_optb` | dict: {col: OptimalBinning nesnesi} |
| `{_pfx}_bins` | dict: {col: [-inf, split1, ..., +inf]} |
| `{_pfx}_iv_tables` | dict: {col: optbinning binning_table DataFrame} |
| `{_pfx}_failed` | list: encode edilemeyen kolon adları |

---

### ADIM 2d: WoE Dağılım Tabloları + Monotonluk

| | |
|---|---|
| **Dosya** | `callbacks/precompute.py:183-245` |
| **Kullanılan** | `modules/deep_dive.py` → `get_woe_detail()`, `_build_binning_table_from_edges()` |
| **Cache Key** | `{_pfx}_woe_tables` → dict |

Her değişken için:

#### TRAIN tablosu:
```python
bt_train, iv_train, _, _ = get_woe_detail(
    df_train, var, target,
    fitted_optb=_optb,
    use_edges=False            # ← train: optb.binning_table.build() kullanır
)
```
- `use_edges=False` → `optb.binning_table.build(show_digits=8)` çağrılır
- Optbinning'in kendi tablosu kullanılır (Bin, Count, Event, Non-event, Event rate, WoE, IV)
- Çıktı formatı: [Bin, Toplam, Bad, Good, Bad Rate %, WOE, IV Katkı] + TOPLAM satırı

#### TEST tablosu (df_test varsa ve len > 0):
```python
bt_test, iv_test, _, _ = get_woe_detail(
    df_test, var, target,
    fitted_optb=_optb,
    use_edges=True             # ← test: train'in bin sınırlarıyla kendi dağılımını hesaplar
)
```
- `use_edges=True` → `_build_binning_table_from_edges(optb, X, y, col, is_numeric)` çağrılır

#### OOT tablosu (df_oot varsa ve len > 0):
```python
bt_oot, iv_oot, _, _ = get_woe_detail(
    df_oot, var, target,
    fitted_optb=_optb,
    use_edges=True             # ← oot: train'in bin sınırlarıyla kendi dağılımını hesaplar
)
```

#### `_build_binning_table_from_edges()` detayı (deep_dive.py:31-161):

Bu fonksiyon test/oot verisi için train'in bin sınırlarını kullanarak tablo oluşturur.

```
Girdiler:
  - optb: train'de fit edilmiş OptimalBinning nesnesi
  - X: test/oot'un kolon değerleri (array)
  - y: test/oot'un target değerleri (array, int)

Adımlar:
  1. Train'in binning tablosundan her bin'in WoE değerini al
  2. Train'in split noktalarından edges oluştur: [-inf, split1, ..., +inf]
  3. Veriyi 3 gruba ayır: special (9999999999, 8888888888), missing (NaN), normal
  4. Normal veriyi bin'lere ata: np.digitize(x_normal, edges[1:-1], right=True)

Her bin i için:
  - n_i    = bin'deki gözlem sayısı
  - bad_i  = bin'deki bad (target=1) sayısı
  - good_i = n_i - bad_i
  - bad_rate = bad_i / n_i

  - WOE = train'den SABIT (train_woe_list[i])      ← DEĞİŞMEZ

  - IV hesabı = KENDİ dağılımından:
    dist_bad_i  = max(bad_i / total_bad, 1e-9)
    dist_good_i = max(good_i / total_good, 1e-9)
    iv_part_i   = (dist_bad_i - dist_good_i) × ln(dist_bad_i / dist_good_i)

  - IV her zaman ≥ 0 (matematiksel olarak garanti:
    f(x) = (x - y) × ln(x/y) her zaman ≥ 0 çünkü x > y ise ln > 0, x < y ise ikisi de < 0)

Special ve Missing için aynı mantık uygulanır.

TOPLAM satırı: iv_total = Σ iv_part_i (tüm bin'ler + special + missing)
```

#### Monotonluk Kontrolü:

```python
def _mono_check(bt):
    m = bt[~bt["Bin"].isin(["TOPLAM", "Eksik", "Special"])]    # sadece normal bin'ler
    nums = [float(w) for w in m["WOE"].dropna() if isinstance(w, (int, float))]

    if len(nums) < 2: return "–"

    diffs = [nums[i+1] - nums[i] for i in range(len(nums)-1)]

    if all(d >= 0 for d in diffs): return "Artan ↑"       # WoE monoton artan
    if all(d <= 0 for d in diffs): return "Azalan ↓"      # WoE monoton azalan
    return "Monoton Değil ✗"
```
**NOT (train):** Train monotonluğu train WoE değerlerine bakılır.
**NOT (test/oot):** Test/OOT monotonluğu da tablodaki WOE sütununa bakılır — ancak WOE sütunu **train'den sabit** olduğu için, test ve oot monotonluğu her zaman train ile AYNI çıkmalı (çünkü WOE değerleri aynı). **BU DOĞRU MU?**

#### woe_tables dict yapısı:

```python
woe_tables[var] = {
    "train_table": bt_train.to_dict("records"),   # [{Bin, Toplam, Bad, Good, Bad Rate %, WOE, IV Katkı}, ...]
    "iv_train":    round(iv_train, 4),
    "monoton":     "Artan ↑" / "Azalan ↓" / "Monoton Değil ✗" / "–",

    # Eğer test varsa:
    "test_table":  bt_test.to_dict("records"),
    "iv_test":     round(iv_test, 4),
    "monoton_test": "..." ,

    # Eğer oot varsa:
    "oot_table":   bt_oot.to_dict("records"),
    "iv_oot":      round(iv_oot, 4),
    "monoton_oot": "...",
}
```

---

## ADIM 3: KORELASYON MATRİSİ

| | |
|---|---|
| **Dosya** | `callbacks/precompute.py:252-262` |
| **Fonksiyon** | `modules/correlation.py:25` → `compute_correlation_matrix()` |
| **Girdi** | `train_woe` DataFrame, ilk 30 kolon |
| **Cache Key** | `{key}_corr_{seg_col}_{seg_val}_precomp` → `(corr_df, num_cols_corr)` |

### Formül
```
Pearson korelasyon katsayısı (pandas .corr(method="pearson")):

r_ij = Σ((x_i - μ_xi) × (x_j - μ_xj)) / √(Σ(x_i - μ_xi)² × Σ(x_j - μ_xj)²)

NaN değerler pairwise olarak hariç tutulur.
```

### Not
- Sadece İLK 30 kolon alınır: `num_cols_corr = list(train_woe.columns)[:30]`
- Korelasyon, TRAIN WoE değerleri üzerinden hesaplanır

---

## ADIM 4: DEĞİŞKEN ÖZETİ TABLOSU

| | |
|---|---|
| **Dosya** | `callbacks/var_summary.py:15` |
| **Fonksiyon** | `compute_var_summary_table(config, key, seg_col, seg_val)` |
| **Cache Key** | `{key}_varsummary_{seg_col}_{seg_val}` ve `{key}_summary_{seg_col}_{seg_val}` |

Bu fonksiyon önceki adımlarda cache'e yazılan verileri okuyup tek bir özet tablo oluşturur.

### 4a: Temel Tablo
```python
iv_df = _SERVER_STORE[f"{key}_iv_{seg_col}_{seg_val}"]     # Adım 2c'den
summary = iv_df[["Değişken", "IV", "Eksik %"]].copy()
```

### 4b: PSI Hesabı

| | |
|---|---|
| **Dosya** | `utils/chart_helpers.py:197` |
| **Fonksiyon** | `calc_psi(base, comp, n_bins=10)` |
| **Girdi** | `base` = train WoE değerleri (array), `comp` = OOT WoE değerleri (array) |

**Her değişken için:**
```python
tr_vals  = train_woe[var].values     # train'deki WoE encode edilmiş değerler
oot_vals = oot_woe[var].values       # OOT'deki WoE encode edilmiş değerler
psi_val  = calc_psi(tr_vals, oot_vals)
```

**PSI Formülü:**
```
mn, mx = base.min(), base.max()
bins = np.linspace(mn, mx, 11)       # 10 eşit aralıklı bin
bins[0] = -inf, bins[-1] = +inf

b_pct = np.histogram(base, bins)[0] / len(base)     # train dağılımı (%)
c_pct = np.histogram(comp, bins)[0] / len(comp)     # OOT dağılımı (%)

eps = 1e-4
b_pct = max(b_pct, eps)   # log(0) koruması
c_pct = max(c_pct, eps)

PSI = Σ (c_pct_i - b_pct_i) × ln(c_pct_i / b_pct_i)
```

**NOT:** PSI bin'leri, Train WoE değerlerinin min-max aralığına göre linspace ile oluşturulur.
Bu, Adım 2d'deki IV tablosundaki bin'lerden FARKLI bir binning yöntemidir.
IV → OptimalBinning bin sınırları, PSI → WoE değerlerinin 10 eşit aralığı.

**PSI Yorumları:**
| PSI | Yorum |
|---|---|
| < 0.10 | Stabil |
| 0.10 - 0.25 | Hafif Kayma |
| ≥ 0.25 | Kritik Kayma |

### 4c: Monotonluk (Cache'ten)

```python
woe_tables = _SERVER_STORE[f"{_pfx}_woe_tables"]    # Adım 2d'den

# Her değişken için:
monoton_test = woe_tables[var]["monoton_test"]       # "Artan ↑", "Azalan ↓", "Monoton Değil ✗", "–"
monoton_oot  = woe_tables[var]["monoton_oot"]

# Tabloda gösterim:
"Artan ↑" veya "Azalan ↓"  → ✅
"Monoton Değil ✗"           → ❌
Boş veya "–"                → —
```
**Bu adımda YENİDEN hesaplama YAPILMAZ**, cache'ten okunur.

### 4d: Korelasyon (Yeniden Hesaplanır)

| | |
|---|---|
| **Dosya** | `callbacks/var_summary.py:78-94` |
| **Girdi** | `train_woe` DataFrame — TÜM değişkenler (Adım 3'teki gibi 30 limit YOK) |

```python
num_cols = [v for v in var_list if v in train_woe.columns]
corr_df = compute_correlation_matrix(train_woe[num_cols], num_cols)
high_pairs = find_high_corr_pairs(corr_df, threshold=0.0)    # TÜM çiftler (eşik=0)
```

**Her değişken için:** O değişkenin DİĞER TÜM değişkenlerle olan korelasyonlarından en büyük |r| değeri:
```python
corr_map[var] = max(|r_ij|)   tüm j ≠ var için
```

**NOT:** Bu korelasyon Adım 3'te hesaplanandan FARKLI olabilir çünkü:
- Adım 3: ilk 30 kolon
- Adım 4d: TÜM değişkenler

**NOT-2:** Bu hesaplama her çağrıda yeniden yapılır, cache'lenmez (kalan değişken setine bağlı).

### 4e: VIF Hesabı

| | |
|---|---|
| **Dosya** | `modules/correlation.py:56` |
| **Fonksiyon** | `compute_vif(train_woe[num_cols_vif], num_cols_vif, sample_size=50000)` |
| **Girdi** | `train_woe` — TÜM değişkenler |

```python
num_cols_vif = [v for v in var_list if v in train_woe.columns]
```

**Downsampling:** `len(veri) > 50_000` ise rastgele 50,000 satır (random_state=42).
**NaN:** Tüm NaN satırları düşürülür (`dropna()`).

**VIF Formülü:**
```
Her kolon i için:
    y = X[:, i]                          # hedef kolon
    X_rest = X (i hariç tüm kolonlar)   # diğer kolonlar
    A = [1, X_rest]                      # sabit terim + diğer kolonlar

    beta = np.linalg.lstsq(A, y)        # OLS regresyon
    y_hat = A @ beta

    SS_res = Σ(y - y_hat)²
    SS_tot = Σ(y - mean(y))²

    R² = 1 - (SS_res / SS_tot)          # max(0, ...)  negatif koruması var

    VIF = 1 / (1 - R²)                  # R² ≈ 1 ise VIF = 999
```

**VIF Yorumları:**
| VIF | Yorum |
|---|---|
| < 5 | ✓ Normal |
| 5 - 10 | ⚠ Orta |
| ≥ 10 | ✗ Yüksek |

**NOT:** VIF her çağrıda yeniden hesaplanır, cache'lenmez (kalan değişken setine bağlı).

### 4f: Öneri Mantığı

| | |
|---|---|
| **Dosya** | `callbacks/var_summary.py:112-150` |

**❌ Çıkar** — aşağıdakilerden BİRİ bile varsa:
| Koşul | Eşik |
|---|---|
| IV çok düşük | IV < 0.02 |
| Çok fazla eksik | Eksik % > 80 |
| Kritik dağılım kayması | PSI > 0.25 |

**⚠️ İncele** — "Çıkar" değilse ve aşağıdakilerden BİRİ varsa:
| Koşul | Eşik |
|---|---|
| IV düşük | IV < 0.10 |
| Yüksek eksik | Eksik % > 20 |
| Hafif dağılım kayması | PSI > 0.10 |
| Yüksek korelasyon | \|Korelasyon\| ≥ 0.75 |
| Yüksek çoklu doğrusallık | VIF > 5 |

**✅ Tut** — yukarıdakilerin hiçbiri yoksa.

**Sıralama:** Tut → İncele → Çıkar, ardından IV azalan.

### 4g: Son Tablo Kolonları

```
["Değişken", "Öneri", "Sebep", "IV",
 "Test Monoton", "OOT Monoton",
 "Korr Değeri", "PSI Değeri", "PSI Durumu",
 "Train VIF", "Eksik %"]
```

---

## TÜM CACHE KEY'LERİ ÖZETİ

| # | Cache Key | Adım | Tip | Açıklama |
|---|---|---|---|---|
| 1 | `{key}_screen` | 0 | (list, df) | Geçen kolonlar + eleme raporu |
| 2 | `{key}_profile_{seg}_{val}` | 1 | DataFrame | Profil istatistikleri |
| 3 | `{_pfx}_train` | 2a | DataFrame | Train ham veri |
| 4 | `{_pfx}_test` | 2a | DataFrame/None | Test ham veri |
| 5 | `{_pfx}_oot` | 2a | DataFrame/None | OOT ham veri |
| 6 | `{_pfx}_train_woe` | 2c | DataFrame | Train WoE encode (n_train × n_vars) |
| 7 | `{_pfx}_test_woe` | 2c | DataFrame/None | Test WoE encode |
| 8 | `{_pfx}_oot_woe` | 2c | DataFrame/None | OOT WoE encode |
| 9 | `{key}_iv_{seg}_{val}` | 2c | DataFrame | [Değişken, IV, Eksik %] sıralı |
| 10 | `{_pfx}_optb` | 2c | dict | {col: OptimalBinning nesnesi} |
| 11 | `{_pfx}_bins` | 2c | dict | {col: bin_edges array} |
| 12 | `{_pfx}_iv_tables` | 2c | dict | {col: optbinning binning_table} |
| 13 | `{_pfx}_failed` | 2c | list | Başarısız kolonlar |
| 14 | `{_pfx}_woe_tables` | 2d | dict | {col: {train_table, iv_train, monoton, ...}} |
| 15 | `{key}_corr_{seg}_{val}_precomp` | 3 | (df, list) | (Korelasyon matrisi, kolon listesi) |
| 16 | `{key}_varsummary_{seg}_{val}` | 4 | DataFrame | Değişken özeti tablosu |
| 17 | `{key}_summary_{seg}_{val}` | 4 | DataFrame | Değişken özeti (aynı, kopya) |

**Prefix:** `_pfx = f"{key}_ds_{seg_col}_{seg_val}"`

---

## SORU İŞARETLERİ / KONTROL EDİLECEKLER

1. **Monotonluk (test/oot):** `_build_binning_table_from_edges` fonksiyonunda WOE sütunu train'den SABİT alınıyor. Bu durumda test/oot monotonluğu her zaman train ile aynı çıkmalı. Farklı çıkması mantıken mümkün değil gibi görünüyor — AMA bin'e düşen gözlem sayısı 0 ise o bin'in WoE'su tabloda olmuyor olabilir, bu durumda farklılık ortaya çıkabilir. **Bu kasıtlı mı?**

2. **Eksik %:** Train + Test üzerinden hesaplanıyor, OOT dahil DEĞİL. Bu kasıtlı bir karar mı?

3. **Korelasyon farkı:** Adım 3'te ilk 30 kolon ile korelasyon matrisi hesaplanıp cache'e yazılıyor. Adım 4d'de ise TÜM değişkenler ile tekrar hesaplanıyor. Cache'teki kullanılmıyor. Korelasyon matrisi görselleştirmesinde (heatmap) Adım 3'teki mi yoksa Adım 4d'deki mi kullanılıyor?

4. **PSI vs IV binning farkı:**
   - PSI: WoE değerlerinin 10 eşit aralığına göre (linspace)
   - IV tablosu: OptimalBinning'in belirlediği optimal bin sınırlarına göre
   - Bu iki farklı binning doğru mu? (Genelde PSI bu şekilde WoE üzerinden hesaplanır, ama bazı yaklaşımlarda aynı bin'ler kullanılır.)
