import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

from app_instance import app
from server_state import _SERVER_STORE, get_df as _get_df
from utils.chart_helpers import calc_psi, psi_label
from modules.correlation import compute_correlation_matrix, find_high_corr_pairs, compute_vif


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
        if not raw or raw == "–":
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

    # ── VIF — modelleme sonrası hesaplanır, burada boş bırakılır ────────────
    summary["Train VIF"] = "—"

    # ── Öneri mantığı ────────────────────────────────────────────────────────
    def _recommend_with_reason(row):
        iv_val    = row["IV"]
        eksik_val = row["Eksik %"]
        psi_val   = row["PSI Değeri"] if isinstance(row["PSI Değeri"], (int, float)) else None
        corr_val  = row["Korr (Target)"] if isinstance(row["Korr (Target)"], (int, float)) else None
        vif_val   = row["Train VIF"] if isinstance(row["Train VIF"], (int, float)) else None

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
        if vif_val is not None and vif_val > 5.0:
            inc_reasons.append(f"VIF={vif_val:.1f}>5")

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
                 "Train VIF", "Eksik %"]
    summary = summary[[c for c in col_order if c in summary.columns]]

    # Cache'e yaz
    _SERVER_STORE[f"{key}_varsummary_{seg_col}_{seg_val}"] = summary.copy()

    return summary


def compute_var_summary_raw(config, key, seg_col, seg_val):
    """Ham (raw) train+test üzerinden değişken özeti tablosu üretir.
    Kolonlar: IV, Eksik %, PSI (10 parça), Korelasyon, VIF"""
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

    # ── VIF — modelleme sonrası hesaplanır, burada boş bırakılır ────────────
    summary["Train VIF"] = "—"

    # ── Öneri mantığı (aynı) ──────────────────────────────────────────────
    def _recommend_with_reason(row):
        iv_val = row["IV"]
        eksik_val = row["Eksik %"]
        psi_val = row["PSI Değeri"] if isinstance(row["PSI Değeri"], (int, float)) else None
        corr_val = row["Korr (Target)"] if isinstance(row["Korr (Target)"], (int, float)) else None
        vif_val = row["Train VIF"] if isinstance(row["Train VIF"], (int, float)) else None
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
        if vif_val is not None and vif_val > 5.0:
            inc_reasons.append(f"VIF={vif_val:.1f}>5")
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
                 "Train VIF", "Eksik %"]
    summary = summary[[c for c in col_order if c in summary.columns]]

    _SERVER_STORE[f"{key}_varsummary_raw_{seg_col}_{seg_val}"] = summary.copy()
    return summary


# ── Render yardımcısı ─────────────────────────────────────────────────────────
def _render_var_summary(summary, use_woe):
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
        {"if": {"filter_query": '{Train VIF} >= 10',     "column_id": "Train VIF"},   "color": "#ef4444", "fontWeight": "600"},
        {"if": {"filter_query": '{Train VIF} >= 5 && {Train VIF} < 10', "column_id": "Train VIF"}, "color": "#f59e0b"},
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
            "Tüm kriterler tatmin edici:\nIV ≥ 0.10, Eksik ≤ 20%, PSI ≤ 0.10, |Korr| < 0.80, VIF ≤ 5",
            target="card-tut", placement="top",
            style={"whiteSpace": "pre-line", "fontSize": "0.78rem"},
        ),
        dbc.Tooltip(
            "En az bir zayıf sinyal var:\nIV 0.02–0.10 · Eksik 20–60%\nPSI 0.10–0.25 · |Korr| ≥ 0.80 · VIF > 5",
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
            data=summary.to_dict("records"),
            columns=[{"name": c, "id": c} for c in summary.columns],
            sort_action="native",
            filter_action="native",
            page_size=25,
            tooltip_header={
                "Değişken":    {"value": "Değişkenin adı", "type": "markdown"},
                "Öneri":       {"value": "IV, Eksik%, PSI, Korelasyon ve VIF'e göre otomatik öneri:\n"
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
                "Train VIF":   {"value": "**Variance Inflation Factor** — çoklu doğrusal bağlantı ölçüsü "
                                         "(Train WoE üzerinden)\n\n"
                                         "- **< 5** — Normal\n- **5–10** — Dikkat\n- **> 10** — Kritik", "type": "markdown"},
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
    State("store-key", "data"),
    State("chk-varsummary-woe", "value"),
    prevent_initial_call=True,
)
def update_var_summary(config, n_clicks, expert_excluded, active_tab, vs_tab, _precompute_done, key, woe_toggle):
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
        # WoE tab: mevcut hesaplama (cache'den veya taze)
        cache_key = f"{key}_varsummary_{seg_col}_{seg_val}"
        cached = _SERVER_STORE.get(cache_key)
        if cached is not None and "Bin" in cached.columns:
            summary = cached.copy()
        else:
            summary = compute_var_summary_table(config, key, seg_col, seg_val)
    else:
        # Ham tab: raw train+test üzerinden
        cache_key_raw = f"{key}_varsummary_raw_{seg_col}_{seg_val}"
        cached_raw = _SERVER_STORE.get(cache_key_raw)
        if cached_raw is not None:
            summary = cached_raw.copy()
        else:
            summary = compute_var_summary_raw(config, key, seg_col, seg_val)

    if summary.empty:
        return html.Div("Özet tablosu oluşturulamadı.", className="alert-info-custom")

    # Ön eleme (screen) ile filtrele — elenen değişkenleri çıkar
    screen_result = _SERVER_STORE.get(f"{key}_screen")
    if screen_result:
        passed_set = set(screen_result[0])
        summary = summary[summary["Değişken"].isin(passed_set)].reset_index(drop=True)

    if excluded_set:
        summary = summary[~summary["Değişken"].isin(excluded_set)].reset_index(drop=True)

    return _render_var_summary(summary, use_woe)
