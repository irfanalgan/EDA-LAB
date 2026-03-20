# Ham / WoE Yapı Düzenlemesi — Veri Kaynakları Notu

Bu dosya, her adımda hangi veri kaynağının kullanıldığını belgeler.

## 6 DataFrame
- `df_train`, `df_test`, `df_oot` → Ham (raw) veriler
- `df_train_woe`, `df_test_woe`, `df_oot_woe` → WoE dönüştürülmüş veriler
- `df_active` → Segment filtreli tüm veri (train+test+oot veya seçilen segment)

---

## 1. Precompute (callbacks/precompute.py)

| Hesaplama | Önceki Kaynak | Yeni Kaynak | Değişiklik |
|---|---|---|---|
| Önizleme | df_active (tümü) | df_active (tümü) | Değişiklik yok |
| Describe | df_active (tümü) | raw train+test (df_train + df_test) | **DEĞİŞTİ** |
| Outlier | df_active (tümü) | raw train+test (df_train + df_test) | **DEĞİŞTİ** |
| IV | raw train (df_train) | raw train (df_train) | Değişiklik yok |

## 2. Değişken Analizi Detay (callbacks/deep_dive.py)

| Hesaplama | Tab | Kaynak |
|---|---|---|
| Dağılım Analizi | Her iki tab | Aynı (mevcut kaynak) |
| Good vs Bad | Her iki tab | Aynı (mevcut kaynak) |
| WoE & Bad Rate | Sadece WoE tab | WoE binlerinden |
| PSI | WoE tab | TEK KAYNAK: calc_psi(train_woe vs oot_woe, discrete=True) |
| PSI | Ham tab | TEK KAYNAK: calc_psi(train vs oot, n_bins=10, discrete=False) |

## 3. İstatistiksel Testler (callbacks/stat_tests.py, callbacks/correlation.py)

| Test | Ham Tab | WoE Tab |
|---|---|---|
| Korelasyon | raw train+test | woe train+test |
| Chi-Square | raw train+test | woe train+test |
| ANOVA | raw train+test | woe train+test |
| KS | raw train+test | woe train+test |
| VIF Sandbox | raw train+test | woe train+test |

## 4. Değişken Özeti (callbacks/var_summary.py)

| Kolon | Ham Tab | WoE Tab |
|---|---|---|
| IV | Cache'den (train raw = train woe IV) | Cache'den |
| Eksik % | Ham veri | Ham veri |
| PSI (10 parça) | Cache'den (ham deep dive) | — |
| PSI (WoE bin) | — | Cache'den (woe deep dive) |
| Korelasyon | Ham train+test | WoE train+test |
| VIF | Ham train+test | WoE train+test |
| Test Monoton | — | Cache'den |
| OOT Monoton | — | Cache'den |

## 5. Modelleme (callbacks/playground.py)

| İşlem | Kaynak | Not |
|---|---|---|
| Grafik Oluşturucu | Ham değişkenler | Ayrım yok |
| Hızlı Model - Ham | raw train/test/oot | Null stratejisi uygulanır |
| Hızlı Model - WoE | woe train/test/oot | Null stratejisi ETKİLEMEZ |
| Katsayı tablosu | Model fit sonucu | Limit kaldırıldı (tüm değişkenler) |

## 6. Sonuç Sekmesi (callbacks/results.py)

| İçerik | Kaynak | Not |
|---|---|---|
| Metrikler, CM, ROC, SHAP, Rating | Model fit | Her tab kendi modelinden |
| Katsayı/Ağırlık | Model fit | Her tab kendi modelinden |
| PSI | Ham tab: raw train vs OOT (10 parça) / WoE tab: cache psi_map | Tab'a göre ayrı |
| WoE Dağılım | Cache'den | Sadece WoE tab |
| Describe | Cache'den (precompute profile) | Her iki tab aynı |
| VIF | Ham tab: raw model vars / WoE tab: woe model vars | Final değişkenlerle yeniden hesaplanır |
| Korelasyon | Ham tab: raw train / WoE tab: woe train | Final değişkenlerle yeniden hesaplanır |
