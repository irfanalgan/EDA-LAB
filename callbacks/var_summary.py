import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

from app_instance import app
from server_state import _SERVER_STORE, get_df as _get_df
from utils.chart_helpers import calc_psi, psi_label
from modules.correlation import compute_correlation_matrix, find_high_corr_pairs


# ── Hesaplama fonksiyonu (precompute + callback ortak) ────────────────────────
def compute_var_summary_table(config, key, seg_col, seg_val):
    """
    Merkezi cache'teki 6 DataFrame'den değişken özeti tablosu üretir.
    Precompute thread'inden veya callback'ten çağrılabilir.
    Returns: summary DataFrame
    """
    target   = config["target_col"]
    _pfx     = f"{key}_ds_{seg_col}_{seg_val}"

    # ── Cache'ten oku ─────────────────────────────────────────────────────────
    iv_df       = _SERVER_STORE.get(f"{key}_iv_{seg_col}_{seg_val}")
    train_woe   = _SERVER_STORE.get(f"{_pfx}_train_woe")
    oot_woe     = _SERVER_STORE.get(f"{_pfx}_oot_woe")

    if iv_df is None or iv_df.empty:
        return pd.DataFrame()

    summary = iv_df[["Değişken", "IV", "Eksik %"]].copy()
    var_list = summary["Değişken"].tolist()

    # ── Bin sayısı — bins_dict veya optb_dict'ten ────────────────────────────
    bins_dict = _SERVER_STORE.get(f"{_pfx}_bins", {})
    optb_dict = _SERVER_STORE.get(f"{_pfx}_optb", {})

    def _get_n_bins(var):
        if var in bins_dict:
            return len(bins_dict[var]) - 1  # edges - 1 = bin sayısı
        _ob = optb_dict.get(var)
        if _ob is not None:
            try:
                _bt = _ob.binning_table.build()
                _exclude = {"Totals", "Missing"}
                _data = _bt[~_bt["Bin"].isin(_exclude)]
                return len(_data)
            except Exception:
                pass
        return "—"
    summary["Bin"] = summary["Değişken"].map(_get_n_bins)

    # ── PSI — Train WoE vs OOT WoE ──────────────────────────────────────────
    psi_map = {}
    psi_label_map = {}
    if train_woe is not None and oot_woe is not None and not oot_woe.empty:
        for var in var_list:
            if var not in train_woe.columns:
                continue
            if var not in oot_woe.columns:
                continue
            try:
                tr_vals = train_woe[var].values
                oot_vals = oot_woe[var].values
                psi_val = round(calc_psi(tr_vals, oot_vals), 4)
                psi_map[var] = psi_val
                psi_label_map[var] = psi_label(psi_val)
            except Exception:
                pass

    # PSI map'i cache'e yaz (sonuç kısmı buradan okuyacak)
    if psi_map:
        _SERVER_STORE[f"{_pfx}_psi_map"] = psi_map

    summary["PSI Değeri"] = summary["Değişken"].map(
        lambda v: psi_map.get(v, "—")
    )
    summary["PSI Durumu"] = summary["Değişken"].map(
        lambda v: psi_label_map.get(v, "—")
    )

    # ── Monotonluk (Test / OOT) — precompute cache'ten oku ────────────────────
    woe_tables = _SERVER_STORE.get(f"{_pfx}_woe_tables", {})

    def _mono_symbol(raw):
        """Precompute'taki metin → emoji dönüşümü."""
        if not raw or raw in ("–", "—", "—"):
            return "—"
        if "Artan" in raw or "Azalan" in raw:
            return "✅"
        return "❌"

    summary["Test Monoton"] = summary["Değişken"].map(
        lambda v: _mono_symbol(woe_tables.get(v, {}).get("monoton_test", ""))
    )
    summary["OOT Monoton"] = summary["Değişken"].map(
        lambda v: _mono_symbol(woe_tables.get(v, {}).get("monoton_oot", ""))
    )

    # ── Korelasyon — Değişken vs Target (Train WoE) ─────────────────────────
    corr_map = {}
    df_train = _SERVER_STORE.get(f"{_pfx}_train")
    if train_woe is not None and not train_woe.empty and df_train is not None:
        try:
            y = df_train[target]
            for v in var_list:
                if v in train_woe.columns:
                    r = train_woe[v].corr(y)
                    if pd.notna(r):
                        corr_map[v] = round(abs(r), 4)
        except Exception:
            pass
    summary["Korr (Target)"] = summary["Değişken"].map(lambda v: corr_map.get(v, "—"))

    # ── Öneri mantığı ────────────────────────────────────────────────────────
    def _recommend_with_reason(row):
        iv_val    = row["IV"]
        eksik_val = row["Eksik %"]
        psi_val   = row["PSI Değeri"] if isinstance(row["PSI Değeri"], (int, float)) else None
        corr_val  = row["Korr (Target)"] if isinstance(row["Korr (Target)"], (int, float)) else None

        cik_reasons = []
        if iv_val < 0.02:
            cik_reasons.append(f"IV={iv_val:.4f}<0.02")
        if eksik_val > 60.0:
            cik_reasons.append(f"Eksik={eksik_val:.1f}%>60%")
        if psi_val is not None and psi_val > 0.25:
            cik_reasons.append(f"PSI={psi_val:.4f}>0.25")

        if cik_reasons:
            return "3 ❌ Çıkar", "; ".join(cik_reasons)

        inc_reasons = []
        if iv_val < 0.10:
            inc_reasons.append(f"IV={iv_val:.4f}<0.10")
        if eksik_val > 20.0:
            inc_reasons.append(f"Eksik={eksik_val:.1f}%>20%")
        if psi_val is not None and psi_val > 0.10:
            inc_reasons.append(f"PSI={psi_val:.4f}>0.10")
        if corr_val is not None and abs(corr_val) >= 0.80:
            inc_reasons.append(f"|Korr|={abs(corr_val):.4f}≥0.80")

        if inc_reasons:
            return "2 ⚠️ İncele", "; ".join(inc_reasons)

        return "1 ✅ Tut", "—"

    summary[["Öneri", "Sebep"]] = summary.apply(
        lambda r: pd.Series(_recommend_with_reason(r)), axis=1
    )

    # Sıralama: Tut > İncele > Çıkar
    _oneri_order = {"1 ✅ Tut": 0, "2 ⚠️ İncele": 1, "3 ❌ Çıkar": 2}
    summary["_sort"] = summary["Öneri"].map(_oneri_order).fillna(3)
    summary = summary.sort_values(["_sort", "IV"], ascending=[True, False]).drop(columns="_sort")
    summary = summary.reset_index(drop=True)

    col_order = ["Değişken", "Öneri", "Sebep", "IV", "Bin",
                 "Test Monoton", "OOT Monoton",
                 "Korr (Target)", "PSI Değeri", "PSI Durumu",
                 "Eksik %"]
    summary = summary[[c for c in col_order if c in summary.columns]]

    # Cache'e yaz
    _SERVER_STORE[f"{key}_varsummary_{seg_col}_{seg_val}"] = summary.copy()

    return summary


def compute_var_summary_raw(config, key, seg_col, seg_val):
    """Ham (raw) train+test üzerinden değişken özeti tablosu üretir.
    Kolonlar: IV, Eksik %, PSI (10 parça), Korelasyon"""
    target = config["target_col"]
    _pfx = f"{key}_ds_{seg_col}_{seg_val}"

    iv_df = _SERVER_STORE.get(f"{key}_iv_{seg_col}_{seg_val}")
    df_train = _SERVER_STORE.get(f"{_pfx}_train")
    df_test = _SERVER_STORE.get(f"{_pfx}_test")
    df_oot = _SERVER_STORE.get(f"{_pfx}_oot")

    if iv_df is None or iv_df.empty:
        return pd.DataFrame()

    summary = iv_df[["Değişken", "IV", "Eksik %"]].copy()
    var_list = summary["Değişken"].tolist()

    # Raw train+test
    if df_train is not None:
        df_raw = pd.concat([df_train, df_test], ignore_index=True) if df_test is not None else df_train.copy()
    else:
        return pd.DataFrame()

    # ── PSI — Raw 10 parça (pd.qcut) — Train vs OOT ─────────────────────
    psi_map = {}
    psi_label_map = {}
    if df_train is not None and df_oot is not None and not df_oot.empty:
        for var in var_list:
            if var not in df_train.columns or var not in df_oot.columns:
                continue
            try:
                tr_vals = df_train[var].dropna()
                oot_vals = df_oot[var].dropna()
                if len(tr_vals) < 10 or len(oot_vals) < 10:
                    continue
                psi_val = round(calc_psi(tr_vals.values, oot_vals.values, n_bins=10, discrete=False), 4)
                psi_map[var] = psi_val
                psi_label_map[var] = psi_label(psi_val)
            except Exception:
                pass

    # Raw PSI map'i cache'e yaz (deep dive, sonuç kısmı buradan okuyacak)
    if psi_map:
        _SERVER_STORE[f"{_pfx}_raw_psi_map"] = psi_map

    summary["PSI Değeri"] = summary["Değişken"].map(lambda v: psi_map.get(v, "—"))
    summary["PSI Durumu"] = summary["Değişken"].map(lambda v: psi_label_map.get(v, "—"))

    # ── Korelasyon — Değişken vs Target (Raw train) ─────────────────────────
    corr_map = {}
    try:
        y = df_train[target]
        for v in var_list:
            if v in df_train.columns and pd.api.types.is_numeric_dtype(df_train[v]):
                r = df_train[v].corr(y)
                if pd.notna(r):
                    corr_map[v] = round(abs(r), 4)
    except Exception:
        pass
    summary["Korr (Target)"] = summary["Değişken"].map(lambda v: corr_map.get(v, "—"))

    # ── Öneri mantığı (aynı) ──────────────────────────────────────────────
    def _recommend_with_reason(row):
        iv_val = row["IV"]
        eksik_val = row["Eksik %"]
        psi_val = row["PSI Değeri"] if isinstance(row["PSI Değeri"], (int, float)) else None
        corr_val = row["Korr (Target)"] if isinstance(row["Korr (Target)"], (int, float)) else None
        cik_reasons = []
        if iv_val < 0.02:
            cik_reasons.append(f"IV={iv_val:.4f}<0.02")
        if eksik_val > 60.0:
            cik_reasons.append(f"Eksik={eksik_val:.1f}%>60%")
        if psi_val is not None and psi_val > 0.25:
            cik_reasons.append(f"PSI={psi_val:.4f}>0.25")
        if cik_reasons:
            return "3 ❌ Çıkar", "; ".join(cik_reasons)
        inc_reasons = []
        if iv_val < 0.10:
            inc_reasons.append(f"IV={iv_val:.4f}<0.10")
        if eksik_val > 20.0:
            inc_reasons.append(f"Eksik={eksik_val:.1f}%>20%")
        if psi_val is not None and psi_val > 0.10:
            inc_reasons.append(f"PSI={psi_val:.4f}>0.10")
        if corr_val is not None and abs(corr_val) >= 0.80:
            inc_reasons.append(f"|Korr|={abs(corr_val):.4f}>=0.80")
        if inc_reasons:
            return "2 ⚠️ İncele", "; ".join(inc_reasons)
        return "1 ✅ Tut", "—"

    summary[["Öneri", "Sebep"]] = summary.apply(
        lambda r: pd.Series(_recommend_with_reason(r)), axis=1
    )
    _oneri_order = {"1 ✅ Tut": 0, "2 ⚠️ İncele": 1, "3 ❌ Çıkar": 2}
    summary["_sort"] = summary["Öneri"].map(_oneri_order).fillna(3)
    summary = summary.sort_values(["_sort", "IV"], ascending=[True, False]).drop(columns="_sort")
    summary = summary.reset_index(drop=True)

    # Ham tab: Monoton kolonları yok
    col_order = ["Değişken", "Öneri", "Sebep", "IV",
                 "Korr (Target)", "PSI Değeri", "PSI Durumu",
                 "Eksik %"]
    summary = summary[[c for c in col_order if c in summary.columns]]

    _SERVER_STORE[f"{key}_varsummary_raw_{seg_col}_{seg_val}"] = summary.copy()
    return summary


# ── Filtre yardımcıları ────────────────────────────────────────────────────────

def _apply_numeric_filter(df, col, op, val):
    """Sayısal filtre uygula. '—' gibi string değerler (NaN) filtreden muaf tutulur."""
    if col not in df.columns:
        return df
    try:
        val = float(val)
    except (TypeError, ValueError):
        return df
    nums = pd.to_numeric(df[col], errors="coerce")
    is_na = nums.isna()
    if op == "ge":
        mask = (nums >= val) | is_na
    elif op == "gt":
        mask = (nums > val) | is_na
    elif op == "le":
        mask = (nums <= val) | is_na
    elif op == "lt":
        mask = (nums < val) | is_na
    else:
        return df
    return df[mask]


def _greedy_corr_eliminate(summary, corr_matrix, threshold):
    """Değişkenler arası korelasyon greedy eleme. IV yüksek olan korunur."""
    if corr_matrix is None or corr_matrix.empty:
        return set()
    var_list = summary["Değişken"].tolist()
    iv_map = dict(zip(summary["Değişken"], summary["IV"]))
    # IV'ye göre büyükten küçüğe sırala
    sorted_vars = sorted(var_list, key=lambda v: iv_map.get(v, 0), reverse=True)
    kept = set()
    eliminated = set()
    for var in sorted_vars:
        if var in eliminated:
            continue
        kept.add(var)
        if var not in corr_matrix.index:
            continue
        for other in sorted_vars:
            if other in kept or other in eliminated or other not in corr_matrix.columns:
                continue
            try:
                c = abs(corr_matrix.loc[var, other])
                if c >= threshold:
                    eliminated.add(other)
            except (KeyError, TypeError):
                pass
    return eliminated


def _compute_filtered_set(summary, filters, corr_matrix, use_woe):
    """Filtre kriterlerine göre geçen değişken setini döndürür."""
    disp = summary.copy()
    disp = _apply_numeric_filter(disp, "IV", filters["iv_op"], filters["iv_val"])
    disp = _apply_numeric_filter(disp, "Korr (Target)", filters["corr_target_op"], filters["corr_target_val"])
    disp = _apply_numeric_filter(disp, "PSI Değeri", filters["psi_op"], filters["psi_val"])
    disp = _apply_numeric_filter(disp, "Eksik %", filters["missing_op"], filters["missing_val"])

    if use_woe:
        if filters.get("test_mono") and filters["test_mono"] != "Hepsi" and "Test Monoton" in disp.columns:
            disp = disp[disp["Test Monoton"] == filters["test_mono"]]
        if filters.get("oot_mono") and filters["oot_mono"] != "Hepsi" and "OOT Monoton" in disp.columns:
            disp = disp[disp["OOT Monoton"] == filters["oot_mono"]]

    passed = set(disp["Değişken"].tolist())

    # Greedy korelasyon eleme
    try:
        corr_var_val = float(filters.get("corr_var_val"))
    except (TypeError, ValueError):
        corr_var_val = None
    if corr_var_val is not None and corr_matrix is not None:
        elim = _greedy_corr_eliminate(
            summary[summary["Değişken"].isin(passed)],
            corr_matrix, corr_var_val)
        passed -= elim

    return passed


# ── Render yardımcısı ─────────────────────────────────────────────────────────
def _render_var_summary(summary, use_woe, selected_vars=None):
    """Cache'den veya taze hesaplamadan gelen summary DataFrame'ini HTML'e çevirir."""
    style_conditions = [
        {"if": {"filter_query": '{Öneri} = "1 ✅ Tut"',    "column_id": "Öneri"}, "color": "#10b981", "fontWeight": "700"},
        {"if": {"filter_query": '{Öneri} = "2 ⚠️ İncele"', "column_id": "Öneri"}, "color": "#f59e0b", "fontWeight": "600"},
        {"if": {"filter_query": '{Öneri} = "3 ❌ Çıkar"',  "column_id": "Öneri"}, "color": "#ef4444", "fontWeight": "700"},
        {"if": {"filter_query": '{Öneri} = "2 ⚠️ İncele"', "column_id": "Sebep"}, "color": "#f59e0b"},
        {"if": {"filter_query": '{Öneri} = "3 ❌ Çıkar"',  "column_id": "Sebep"}, "color": "#ef4444"},
        {"if": {"filter_query": '{PSI Durumu} = "Kritik Kayma"',  "column_id": "PSI Durumu"}, "color": "#ef4444"},
        {"if": {"filter_query": '{PSI Durumu} = "Hafif Kayma"',   "column_id": "PSI Durumu"}, "color": "#f59e0b"},
        {"if": {"filter_query": '{PSI Durumu} = "Stabil"',        "column_id": "PSI Durumu"}, "color": "#10b981"},
        {"if": {"filter_query": '{Test Monoton} = "✅"', "column_id": "Test Monoton"}, "color": "#10b981"},
        {"if": {"filter_query": '{Test Monoton} = "❌"', "column_id": "Test Monoton"}, "color": "#ef4444"},
        {"if": {"filter_query": '{OOT Monoton} = "✅"',  "column_id": "OOT Monoton"},  "color": "#10b981"},
        {"if": {"filter_query": '{OOT Monoton} = "❌"',  "column_id": "OOT Monoton"},  "color": "#ef4444"},
        {"if": {"filter_query": '{Korr (Target)} >= 0.80', "column_id": "Korr (Target)"}, "color": "#f59e0b", "fontWeight": "600"},
        {"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"},
    ]

    n_cik = (summary["Öneri"] == "3 ❌ Çıkar").sum()
    n_inc = (summary["Öneri"] == "2 ⚠️ İncele").sum()
    n_tut = (summary["Öneri"] == "1 ✅ Tut").sum()

    tsv = summary.to_csv(sep="\t", index=False)

    woe_note = html.Div(
        "★ Korr (Target) — Train üzerinden  ·  Monotonluk — Train bin sınırlarıyla",
        style={"color": "#a78bfa", "fontSize": "0.75rem", "marginBottom": "0.75rem"},
    ) if use_woe else html.Div()

    # Satır verisine id ekle + seçili satırları hesapla
    all_vars = summary["Değişken"].tolist()
    if selected_vars is None:
        selected_vars = set(all_vars)
    else:
        selected_vars = set(selected_vars)

    # Tikli değişkenler üstte sıralama
    summary = summary.copy()
    summary["_sel"] = summary["Değişken"].apply(lambda v: 0 if v in selected_vars else 1)
    summary = summary.sort_values(["_sel", "IV"], ascending=[True, False]).reset_index(drop=True)
    summary = summary.drop(columns="_sel")

    records = summary.to_dict("records")
    for r in records:
        r["id"] = r["Değişken"]

    selected_rows = [i for i, r in enumerate(records) if r["Değişken"] in selected_vars]

    return html.Div([
        woe_note,
        dbc.Row([
            dbc.Col(html.Div([
                html.Div(str(n_tut), className="metric-value", style={"color": "#10b981", "fontSize": "1.4rem"}),
                html.Div("Tut", className="metric-label"),
            ], className="metric-card", id="card-tut"), width=2),
            dbc.Col(html.Div([
                html.Div(str(n_inc), className="metric-value", style={"color": "#f59e0b", "fontSize": "1.4rem"}),
                html.Div("İncele", className="metric-label"),
            ], className="metric-card", id="card-incele"), width=2),
            dbc.Col(html.Div([
                html.Div(str(n_cik), className="metric-value", style={"color": "#ef4444", "fontSize": "1.4rem"}),
                html.Div("Çıkar", className="metric-label"),
            ], className="metric-card", id="card-cikar"), width=2),
            dbc.Col(html.Div([
                html.Div(str(len(summary)), className="metric-value", style={"fontSize": "1.4rem"}),
                html.Div("Toplam Değişken", className="metric-label"),
            ], className="metric-card"), width=3),
        ], className="mb-4"),
        dbc.Tooltip(
            "Tüm kriterler tatmin edici:\nIV ≥ 0.10, Eksik ≤ 20%, PSI ≤ 0.10, |Korr| < 0.80",
            target="card-tut", placement="top",
            style={"whiteSpace": "pre-line", "fontSize": "0.78rem"},
        ),
        dbc.Tooltip(
            "En az bir zayıf sinyal var:\nIV 0.02–0.10 · Eksik 20–60%\nPSI 0.10–0.25 · |Korr| ≥ 0.80",
            target="card-incele", placement="top",
            style={"whiteSpace": "pre-line", "fontSize": "0.78rem"},
        ),
        dbc.Tooltip(
            "Kritik sorun tespit edildi:\nIV < 0.02 · Eksik > 60% · PSI > 0.25",
            target="card-cikar", placement="top",
            style={"whiteSpace": "pre-line", "fontSize": "0.78rem"},
        ),
        html.Div([
            dcc.Clipboard(target_id="var-summary-tsv", title="Kopyala",
                          style={"cursor": "pointer", "fontSize": "0.72rem",
                                 "color": "#a8b2c2", "padding": "0.2rem 0.55rem",
                                 "border": "1px solid #2d3a4f", "borderRadius": "4px",
                                 "backgroundColor": "#1a2035", "float": "right",
                                 "marginBottom": "0.4rem", "letterSpacing": "0.04em"}),
            html.Pre(tsv, id="var-summary-tsv", style={"display": "none"}),
        ], style={"overflow": "hidden"}),
        dash_table.DataTable(
            id="var-summary-table",
            data=records,
            columns=[{"name": c, "id": c} for c in summary.columns],
            row_selectable="multi",
            selected_rows=selected_rows,
            sort_action="native",
            filter_action="native",
            page_size=25,
            tooltip_header={
                "Değişken":    {"value": "Değişkenin adı", "type": "markdown"},
                "Öneri":       {"value": "IV, Eksik%, PSI ve Korelasyona göre otomatik öneri:\n"
                                         "- **✅ Tut** — Tüm kriterler tatmin edici\n"
                                         "- **⚠️ İncele** — En az bir zayıf sinyal var\n"
                                         "- **❌ Çıkar** — Kritik sorun tespit edildi", "type": "markdown"},
                "Sebep":       {"value": "Önerinin gerekçesi — hangi kural(lar) tetiklendi", "type": "markdown"},
                "IV":          {"value": "**Information Value** — binary target ile doğrusal olmayan ilişki gücü\n\n"
                                         "| Aralık | Güç |\n|---|---|\n"
                                         "| < 0.02 | Çok Zayıf |\n"
                                         "| 0.02–0.10 | Zayıf |\n"
                                         "| 0.10–0.30 | Orta |\n"
                                         "| 0.30–0.50 | Güçlü |\n"
                                         "| > 0.50 | Şüpheli (overfit riski) |", "type": "markdown"},
                "Test Monoton": {"value": "**Test** verisinde WoE bin sıralamasının monoton olup olmadığı "
                                          "(Train'in bin sınırlarıyla).\n\n"
                                          "- ✅ Monoton\n- ❌ Monoton değil\n- — Test split yok veya hesaplanamadı", "type": "markdown"},
                "OOT Monoton":  {"value": "**OOT** verisinde WoE bin sıralamasının monoton olup olmadığı "
                                          "(Train'in bin sınırlarıyla).\n\n"
                                          "- ✅ Monoton\n- ❌ Monoton değil\n- — OOT split yok veya hesaplanamadı", "type": "markdown"},
                "Korr (Target)":  {"value": "Değişkenin **target** ile Pearson korelasyonu "
                                          "(Train verisi üzerinden).\n\n"
                                          "- Yüksek |r| → güçlü doğrusal ilişki", "type": "markdown"},
                "PSI Değeri":   {"value": "**Population Stability Index** — veri dağılımının zaman içinde kayması\n\n"
                                          "| PSI | Durum |\n|---|---|\n"
                                          "| < 0.10 | Stabil |\n"
                                          "| 0.10–0.25 | Hafif Kayma |\n"
                                          "| > 0.25 | Kritik Kayma |", "type": "markdown"},
                "PSI Durumu":   {"value": "PSI değerine göre stabilite etiketi", "type": "markdown"},
                "Eksik %":     {"value": "Değişkendeki boş (null/NaN) değerlerin yüzdesi\n\n"
                                         "- **> 60%** → Çıkar\n- **20–60%** → İncele", "type": "markdown"},
                "Bin":         {"value": "OptBinning'in oluşturduğu **WoE bin sayısı**\n\n"
                                         "Aşağıdaki panelden değişken bazında değiştirilebilir", "type": "markdown"},
            },
            tooltip_delay=0,
            tooltip_duration=None,
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#161d2e", "color": "#a8b2c2",
                          "fontWeight": "600", "fontSize": "0.72rem",
                          "border": "1px solid #2d3a4f", "textTransform": "uppercase",
                          "textDecoration": "underline dotted", "cursor": "help"},
            style_cell={"backgroundColor": "#111827", "color": "#d1d5db",
                        "fontSize": "0.82rem", "border": "1px solid #1f2a3c",
                        "padding": "6px 10px", "textAlign": "left",
                        "whiteSpace": "normal", "maxWidth": "220px"},
            style_filter={"backgroundColor": "#0e1117", "color": "#c8cdd8",
                          "border": "1px solid #2d3a4f"},
            css=[{"selector": ".dash-filter input",
                  "rule": "color: #c8cdd8 !important; background-color: #0e1117 !important; "
                          "border: 1px solid #2d3a4f !important; border-radius: 4px; "
                          "padding: 2px 6px; font-size: 0.75rem;"},
                 {"selector": ".dash-filter input::placeholder",
                  "rule": "color: #4a5568 !important;"}],
            style_data_conditional=style_conditions,
        ),
    ])


# ── Callback: Değişken Özeti ───────────────────────────────────────────────────
@app.callback(
    Output("div-var-summary", "children"),
    Input("store-config", "data"),
    Input("btn-var-summary", "n_clicks"),
    Input("store-expert-exclude", "data"),
    Input("main-tabs", "active_tab"),
    Input("varsummary-data-tab", "active_tab"),
    Input("interval-precompute", "disabled"),
    Input("store-active-vars", "data"),
    State("store-key", "data"),
    State("chk-varsummary-woe", "value"),
    prevent_initial_call=True,
)
def update_var_summary(config, n_clicks, expert_excluded, active_tab, vs_tab,
                       _precompute_done, active_vars, key, woe_toggle):
    if active_tab != "tab-var-summary":
        return dash.no_update
    if not key or not config or not config.get("target_col"):
        return html.Div()

    seg_col      = config.get("segment_col")
    seg_val      = config.get("segment_val")
    use_woe      = (vs_tab == "vs-tab-woe")
    excluded_set = set(expert_excluded or [])

    # Precompute tamamlanmamışsa bekle — bağımsız hesaplama YAPMA
    _pfx = f"{key}_ds_{seg_col}_{seg_val}"
    if f"{_pfx}_train" not in _SERVER_STORE:
        return html.Div("Hesaplama devam ediyor, lütfen bekleyin...",
                         className="alert-info-custom")

    if use_woe:
        cache_key = f"{key}_varsummary_{seg_col}_{seg_val}"
        cached = _SERVER_STORE.get(cache_key)
        if cached is not None and "Bin" in cached.columns:
            summary = cached.copy()
        else:
            summary = compute_var_summary_table(config, key, seg_col, seg_val)
    else:
        cache_key_raw = f"{key}_varsummary_raw_{seg_col}_{seg_val}"
        cached_raw = _SERVER_STORE.get(cache_key_raw)
        if cached_raw is not None:
            summary = cached_raw.copy()
        else:
            summary = compute_var_summary_raw(config, key, seg_col, seg_val)

    if summary.empty:
        return html.Div("Özet tablosu oluşturulamadı.", className="alert-info-custom")

    screen_result = _SERVER_STORE.get(f"{key}_screen")
    if screen_result:
        passed_set = set(screen_result[0])
        summary = summary[summary["Değişken"].isin(passed_set)].reset_index(drop=True)

    if excluded_set:
        summary = summary[~summary["Değişken"].isin(excluded_set)].reset_index(drop=True)

    return _render_var_summary(summary, use_woe, selected_vars=active_vars)


# ── Yardımcı: özet + filtreleme bağlamını hazırla ────────────────────────────
def _get_filter_context(key, config, expert_excluded, vs_tab):
    """Filtre callback'leri için ortak veri hazırlığı."""
    seg_col = config.get("segment_col")
    seg_val = config.get("segment_val")
    use_woe = (vs_tab == "vs-tab-woe")

    if use_woe:
        summary = _SERVER_STORE.get(f"{key}_varsummary_{seg_col}_{seg_val}")
    else:
        summary = _SERVER_STORE.get(f"{key}_varsummary_raw_{seg_col}_{seg_val}")

    if summary is None or summary.empty:
        return None, None, set(), 0

    screen_result = _SERVER_STORE.get(f"{key}_screen")
    if screen_result:
        summary = summary[summary["Değişken"].isin(set(screen_result[0]))]
    excluded_set = set(expert_excluded or [])
    if excluded_set:
        summary = summary[~summary["Değişken"].isin(excluded_set)]
    summary = summary.reset_index(drop=True)
    all_vars = set(summary["Değişken"].tolist())

    # Tab'a göre doğru korelasyon matrisini kullan
    if use_woe:
        corr_matrix = _SERVER_STORE.get(f"{key}_corr_{seg_col}_{seg_val}")
    else:
        corr_matrix = _SERVER_STORE.get(f"{key}_raw_corr_{seg_col}_{seg_val}")

    return summary, corr_matrix, all_vars, len(all_vars)


# ── Callback: Filtre ↔ Checkbox senkronizasyonu (tek callback) ───────────────
@app.callback(
    Output("store-active-vars", "data", allow_duplicate=True),
    Output("store-vs-overrides", "data", allow_duplicate=True),
    Output("vs-active-count", "children"),
    Output("store-active-snapshot", "data"),
    # Filtre input'ları
    Input("vs-filter-iv-op", "value"),
    Input("vs-filter-iv-val", "value"),
    Input("vs-filter-corr-target-op", "value"),
    Input("vs-filter-corr-target-val", "value"),
    Input("vs-filter-corr-var-op", "value"),
    Input("vs-filter-corr-var-val", "value"),
    Input("vs-filter-psi-op", "value"),
    Input("vs-filter-psi-val", "value"),
    Input("vs-filter-missing-op", "value"),
    Input("vs-filter-missing-val", "value"),
    Input("vs-filter-test-mono", "value"),
    Input("vs-filter-oot-mono", "value"),
    # Kullanıcı checkbox tıklaması
    Input("var-summary-table", "selected_row_ids"),
    # States
    State("store-active-snapshot", "data"),
    State("store-vs-overrides", "data"),
    State("store-key", "data"),
    State("store-config", "data"),
    State("store-expert-exclude", "data"),
    State("varsummary-data-tab", "active_tab"),
    prevent_initial_call=True,
)
def sync_var_selection(iv_op, iv_val, corr_t_op, corr_t_val,
                       corr_v_op, corr_v_val, psi_op, psi_val,
                       miss_op, miss_val, test_mono, oot_mono,
                       selected_row_ids,
                       snapshot, overrides, key, config,
                       expert_excluded, vs_tab):
    _no = dash.no_update
    triggered = dash.ctx.triggered_id

    # ── Tablo tetiklemesi → gerçek kullanıcı tıklaması mı kontrol et ────────
    if triggered == "var-summary-table":
        # Tablo DOM'da yok veya yeniden oluşturuluyor
        if selected_row_ids is None:
            return _no, _no, _no, _no
        # Programatik güncelleme (update_var_summary tabloyu yeniden çizdi)
        if snapshot is not None and sorted(selected_row_ids) == sorted(snapshot):
            return _no, _no, _no, _no

    if not key or not config or not config.get("target_col"):
        return _no, _no, "", _no

    summary, corr_matrix, all_vars, total_count = _get_filter_context(
        key, config, expert_excluded, vs_tab)
    if summary is None:
        return _no, _no, "", _no

    overrides = overrides or {"included": [], "excluded": []}
    filters = {
        "iv_op": iv_op, "iv_val": iv_val,
        "corr_target_op": corr_t_op, "corr_target_val": corr_t_val,
        "corr_var_val": corr_v_val,
        "psi_op": psi_op, "psi_val": psi_val,
        "missing_op": miss_op, "missing_val": miss_val,
        "test_mono": test_mono, "oot_mono": oot_mono,
    }
    use_woe = (vs_tab == "vs-tab-woe")

    if triggered == "var-summary-table":
        # ── Kullanıcı checkbox tıkladı ───────────────────────────────────────
        user_selected = set(selected_row_ids)
        filter_passed = _compute_filtered_set(summary, filters, corr_matrix, use_woe)

        new_overrides = {
            "included": list(user_selected - filter_passed),
            "excluded": list(filter_passed - user_selected),
        }
        active = sorted(user_selected & all_vars)
        count_txt = f"{len(active)} / {total_count} değişken seçili"
        return active, new_overrides, count_txt, active  # snapshot = active
    else:
        # ── Filtre değişti ───────────────────────────────────────────────────
        filter_passed = _compute_filtered_set(summary, filters, corr_matrix, use_woe)
        manual_included = set(overrides.get("included", []))
        manual_excluded = set(overrides.get("excluded", []))

        active_set = (filter_passed | manual_included) - manual_excluded
        active_set &= all_vars
        active = sorted(active_set)
        count_txt = f"{len(active)} / {total_count} değişken seçili"
        return active, _no, count_txt, active  # snapshot = active


# ── Callback: Temizle butonu ─────────────────────────────────────────────────
@app.callback(
    Output("vs-filter-iv-op", "value"),
    Output("vs-filter-iv-val", "value"),
    Output("vs-filter-corr-target-op", "value"),
    Output("vs-filter-corr-target-val", "value"),
    Output("vs-filter-corr-var-op", "value"),
    Output("vs-filter-corr-var-val", "value"),
    Output("vs-filter-psi-op", "value"),
    Output("vs-filter-psi-val", "value"),
    Output("vs-filter-missing-op", "value"),
    Output("vs-filter-missing-val", "value"),
    Output("vs-filter-test-mono", "value"),
    Output("vs-filter-oot-mono", "value"),
    Output("store-vs-overrides", "data"),
    Input("btn-vs-filter-reset", "n_clicks"),
    prevent_initial_call=True,
)
def reset_vs_filters(_n):
    # None değerler → filtre uygulanmaz → tüm değişkenler geçer → hepsi tikli
    return ("ge", None, "lt", None, "lt", None, "lt", None,
            "le", None, "Hepsi", "Hepsi", {"included": [], "excluded": []})


# ── Callback: store-active-vars başlatma (veri yüklendiğinde) ────────────────
@app.callback(
    Output("store-active-vars", "data"),
    Input("store-config", "data"),
    Input("store-expert-exclude", "data"),
    Input("interval-precompute", "disabled"),
    State("store-key", "data"),
    prevent_initial_call=True,
)
def init_active_vars(config, expert_excluded, _precompute_done, key):
    if not key or not config or not config.get("target_col"):
        return dash.no_update
    excluded = set(expert_excluded or [])
    screen_result = _SERVER_STORE.get(f"{key}_screen")
    if screen_result:
        base = [c for c in screen_result[0]
                if c != config["target_col"] and c not in excluded]
    else:
        df = _get_df(key)
        if df is None:
            return dash.no_update
        cfg = {c for c in [config.get("target_col"), config.get("date_col"),
                            config.get("segment_col")] if c}
        base = [c for c in df.columns if c not in cfg and c not in excluded]
    return base
