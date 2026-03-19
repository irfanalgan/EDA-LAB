import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

from app_instance import app
from server_state import _SERVER_STORE, get_df as _get_df
from utils.helpers import apply_segment_filter, get_splits
from utils.chart_helpers import build_woe_datasets, calc_psi, psi_label
from modules.deep_dive import get_woe_detail
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
    optb_dict   = _SERVER_STORE.get(f"{_pfx}_optb", {})
    df_test     = _SERVER_STORE.get(f"{_pfx}_test")
    df_oot      = _SERVER_STORE.get(f"{_pfx}_oot")

    if iv_df is None or iv_df.empty:
        return pd.DataFrame()

    summary = iv_df[["Değişken", "IV", "Eksik %"]].copy()
    var_list = summary["Değişken"].tolist()

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

    summary["PSI Değeri"] = summary["Değişken"].map(
        lambda v: psi_map.get(v, "—")
    )
    summary["PSI Durumu"] = summary["Değişken"].map(
        lambda v: psi_label_map.get(v, "—")
    )

    # ── Monotonluk (Test / OOT) — train'in fitted optb'si ile ────────────────
    def _check_monotonic(df_split, var, target_col):
        if df_split is None or len(df_split) < 50:
            return None
        try:
            _fitted = optb_dict.get(var)
            woe_df, iv_val, _, _ = get_woe_detail(df_split, var, target_col,
                                                    fitted_optb=_fitted)
            if woe_df.empty or iv_val == 0:
                return None
            main = woe_df[~woe_df["Bin"].isin(["TOPLAM", "Eksik", "Special"])]
            woes = main["WOE"].dropna().tolist()
            if len(woes) < 2:
                return None
            woes_num = [float(w) for w in woes if w != ""]
            if len(woes_num) < 2:
                return None
            is_asc  = all(woes_num[i] <= woes_num[i+1] for i in range(len(woes_num)-1))
            is_desc = all(woes_num[i] >= woes_num[i+1] for i in range(len(woes_num)-1))
            return "✅" if (is_asc or is_desc) else "❌"
        except Exception:
            return None

    mono_test = {}
    mono_oot  = {}
    for var in var_list:
        mono_test[var] = _check_monotonic(df_test, var, target) or "—"
        mono_oot[var]  = _check_monotonic(df_oot,  var, target) or "—"

    summary["Test Monoton"] = summary["Değişken"].map(lambda v: mono_test.get(v, "—"))
    summary["OOT Monoton"]  = summary["Değişken"].map(lambda v: mono_oot.get(v, "—"))

    # ── Korelasyon — Train WoE, tüm değişkenler ─────────────────────────────
    corr_map = {}
    if train_woe is not None and not train_woe.empty:
        try:
            num_cols = [v for v in var_list if v in train_woe.columns]
            if len(num_cols) >= 2:
                corr_df = compute_correlation_matrix(train_woe[num_cols], num_cols)
                high_pairs = find_high_corr_pairs(corr_df, threshold=0.0)
                for _, row in high_pairs.iterrows():
                    v1, v2 = row["Değişken 1"], row["Değişken 2"]
                    r = abs(row.get("Korelasyon", row.get(high_pairs.columns[2], 0)))
                    for v in (v1, v2):
                        if v not in corr_map or r > corr_map[v]:
                            corr_map[v] = round(r, 2)
        except Exception:
            pass
    summary["Korr Değeri"] = summary["Değişken"].map(lambda v: corr_map.get(v, "—"))

    # ── VIF — Train WoE, tüm değişkenler ─────────────────────────────────────
    vif_map = {}
    if train_woe is not None and not train_woe.empty:
        try:
            num_cols_vif = [v for v in var_list if v in train_woe.columns]
            if len(num_cols_vif) >= 2:
                vif_res = compute_vif(train_woe[num_cols_vif], num_cols_vif)
                if not vif_res.empty and "Değişken" in vif_res.columns:
                    for _, row in vif_res.iterrows():
                        vif_map[row["Değişken"]] = row["VIF"]
        except Exception:
            pass
    summary["Train VIF"] = summary["Değişken"].map(
        lambda v: round(vif_map[v], 1) if v in vif_map else "—"
    )

    # ── Öneri mantığı ────────────────────────────────────────────────────────
    def _recommend_with_reason(row):
        iv_val    = row["IV"]
        eksik_val = row["Eksik %"]
        psi_val   = row["PSI Değeri"] if isinstance(row["PSI Değeri"], (int, float)) else None
        corr_val  = row["Korr Değeri"] if isinstance(row["Korr Değeri"], (int, float)) else None
        vif_val   = row["Train VIF"] if isinstance(row["Train VIF"], (int, float)) else None

        cik_reasons = []
        if iv_val < 0.02:
            cik_reasons.append(f"IV={iv_val:.4f}<0.02")
        if eksik_val > 80.0:
            cik_reasons.append(f"Eksik={eksik_val:.1f}%>80%")
        if psi_val is not None and psi_val > 0.25:
            cik_reasons.append(f"PSI={psi_val:.4f}>0.25")

        if cik_reasons:
            return "❌ Çıkar", "; ".join(cik_reasons)

        inc_reasons = []
        if iv_val < 0.10:
            inc_reasons.append(f"IV={iv_val:.4f}<0.10")
        if eksik_val > 20.0:
            inc_reasons.append(f"Eksik={eksik_val:.1f}%>20%")
        if psi_val is not None and psi_val > 0.10:
            inc_reasons.append(f"PSI={psi_val:.4f}>0.10")
        if corr_val is not None and corr_val >= 0.75:
            inc_reasons.append(f"Korr={corr_val:.2f}≥0.75")
        if vif_val is not None and vif_val > 5.0:
            inc_reasons.append(f"VIF={vif_val:.1f}>5")

        if inc_reasons:
            return "⚠️ İncele", "; ".join(inc_reasons)

        return "✅ Tut", "—"

    summary[["Öneri", "Sebep"]] = summary.apply(
        lambda r: pd.Series(_recommend_with_reason(r)), axis=1
    )

    # Sıralama: Tut > İncele > Çıkar
    _oneri_order = {"✅ Tut": 0, "⚠️ İncele": 1, "❌ Çıkar": 2}
    summary["_sort"] = summary["Öneri"].map(_oneri_order).fillna(3)
    summary = summary.sort_values(["_sort", "IV"], ascending=[True, False]).drop(columns="_sort")
    summary = summary.reset_index(drop=True)

    col_order = ["Değişken", "Öneri", "Sebep", "IV",
                 "Test Monoton", "OOT Monoton",
                 "Korr Değeri", "PSI Değeri", "PSI Durumu",
                 "Train VIF", "Eksik %"]
    summary = summary[[c for c in col_order if c in summary.columns]]

    # Cache'e yaz
    _SERVER_STORE[f"{key}_varsummary_{seg_col}_{seg_val}"] = summary.copy()
    _SERVER_STORE[f"{key}_summary_{seg_col}_{seg_val}"] = summary.copy()

    return summary


# ── Render yardımcısı ─────────────────────────────────────────────────────────
def _render_var_summary(summary, use_woe):
    """Cache'den veya taze hesaplamadan gelen summary DataFrame'ini HTML'e çevirir."""
    style_conditions = [
        {"if": {"filter_query": '{Öneri} = "✅ Tut"',    "column_id": "Öneri"}, "color": "#10b981", "fontWeight": "700"},
        {"if": {"filter_query": '{Öneri} = "⚠️ İncele"', "column_id": "Öneri"}, "color": "#f59e0b", "fontWeight": "600"},
        {"if": {"filter_query": '{Öneri} = "❌ Çıkar"',  "column_id": "Öneri"}, "color": "#ef4444", "fontWeight": "700"},
        {"if": {"filter_query": '{Öneri} = "⚠️ İncele"', "column_id": "Sebep"}, "color": "#f59e0b"},
        {"if": {"filter_query": '{Öneri} = "❌ Çıkar"',  "column_id": "Sebep"}, "color": "#ef4444"},
        {"if": {"filter_query": '{PSI Durumu} = "Kritik Kayma"',  "column_id": "PSI Durumu"}, "color": "#ef4444"},
        {"if": {"filter_query": '{PSI Durumu} = "Hafif Kayma"',   "column_id": "PSI Durumu"}, "color": "#f59e0b"},
        {"if": {"filter_query": '{PSI Durumu} = "Stabil"',        "column_id": "PSI Durumu"}, "color": "#10b981"},
        {"if": {"filter_query": '{Test Monoton} = "✅"', "column_id": "Test Monoton"}, "color": "#10b981"},
        {"if": {"filter_query": '{Test Monoton} = "❌"', "column_id": "Test Monoton"}, "color": "#ef4444"},
        {"if": {"filter_query": '{OOT Monoton} = "✅"',  "column_id": "OOT Monoton"},  "color": "#10b981"},
        {"if": {"filter_query": '{OOT Monoton} = "❌"',  "column_id": "OOT Monoton"},  "color": "#ef4444"},
        {"if": {"filter_query": '{Korr Değeri} >= 0.75', "column_id": "Korr Değeri"}, "color": "#f59e0b", "fontWeight": "600"},
        {"if": {"filter_query": '{Train VIF} >= 10',     "column_id": "Train VIF"},   "color": "#ef4444", "fontWeight": "600"},
        {"if": {"filter_query": '{Train VIF} >= 5 && {Train VIF} < 10', "column_id": "Train VIF"}, "color": "#f59e0b"},
        {"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"},
    ]

    n_cik = (summary["Öneri"] == "❌ Çıkar").sum()
    n_inc = (summary["Öneri"] == "⚠️ İncele").sum()
    n_tut = (summary["Öneri"] == "✅ Tut").sum()

    tsv = summary.to_csv(sep="\t", index=False)

    woe_note = html.Div(
        "★ Korelasyon · VIF — Train WoE üzerinden  ·  Monotonluk — Train bin sınırlarıyla",
        style={"color": "#a78bfa", "fontSize": "0.75rem", "marginBottom": "0.75rem"},
    ) if use_woe else html.Div()

    return html.Div([
        woe_note,
        dbc.Row([
            dbc.Col(html.Div([
                html.Div(str(n_tut), className="metric-value", style={"color": "#10b981", "fontSize": "1.4rem"}),
                html.Div("Tut", className="metric-label"),
            ], className="metric-card"), width=2),
            dbc.Col(html.Div([
                html.Div(str(n_inc), className="metric-value", style={"color": "#f59e0b", "fontSize": "1.4rem"}),
                html.Div("İncele", className="metric-label"),
            ], className="metric-card"), width=2),
            dbc.Col(html.Div([
                html.Div(str(n_cik), className="metric-value", style={"color": "#ef4444", "fontSize": "1.4rem"}),
                html.Div("Çıkar", className="metric-label"),
            ], className="metric-card"), width=2),
            dbc.Col(html.Div([
                html.Div(str(len(summary)), className="metric-value", style={"fontSize": "1.4rem"}),
                html.Div("Toplam Değişken", className="metric-label"),
            ], className="metric-card"), width=3),
        ], className="mb-4"),
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
                "Korr Değeri":  {"value": "Başka bir değişkenle en yüksek **|r|** korelasyon değeri "
                                          "(Train WoE üzerinden).\n\n"
                                          "- **≥ 0.75** → İncele uyarısı", "type": "markdown"},
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
                                         "- **> 80%** → Çıkar\n- **20–80%** → İncele", "type": "markdown"},
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
    State("store-key", "data"),
    State("chk-varsummary-woe", "value"),
    prevent_initial_call=True,
)
def update_var_summary(config, n_clicks, expert_excluded, active_tab, key, woe_toggle):
    if active_tab != "tab-var-summary":
        return dash.no_update
    if not key or not config or not config.get("target_col"):
        return html.Div()

    seg_col      = config.get("segment_col")
    seg_val      = config.get("segment_val")
    use_woe      = "woe" in (woe_toggle or [])
    excluded_set = set(expert_excluded or [])

    # Cache'den oku — precompute tarafından hazırlanmış olabilir
    cache_key = f"{key}_varsummary_{seg_col}_{seg_val}"
    cached = _SERVER_STORE.get(cache_key)

    if cached is not None:
        summary = cached.copy()
        if excluded_set:
            summary = summary[~summary["Değişken"].isin(excluded_set)].reset_index(drop=True)
        return _render_var_summary(summary, use_woe)

    # Cache yoksa: 6 DataFrame yoksa oluştur, sonra hesapla
    _pfx = f"{key}_ds_{seg_col}_{seg_val}"
    if f"{_pfx}_train" not in _SERVER_STORE:
        df_orig = _get_df(key)
        if df_orig is None:
            return html.Div("Veri yüklenmemiş.", className="alert-info-custom")
        df_active = apply_segment_filter(df_orig, seg_col, seg_val)
        target = config["target_col"]
        date_col = config.get("date_col")

        df_train, df_test, df_oot = get_splits(df_active, config)
        _SERVER_STORE[f"{_pfx}_train"] = df_train
        _SERVER_STORE[f"{_pfx}_test"]  = df_test
        _SERVER_STORE[f"{_pfx}_oot"]   = df_oot

        var_list = [c for c in df_train.columns if c != target
                    and c != date_col and c != seg_col]
        woe_result = build_woe_datasets(df_train, df_test, df_oot, target, var_list)
        _SERVER_STORE[f"{_pfx}_train_woe"] = woe_result["train_woe"]
        _SERVER_STORE[f"{_pfx}_test_woe"]  = woe_result["test_woe"]
        _SERVER_STORE[f"{_pfx}_oot_woe"]   = woe_result["oot_woe"]
        _SERVER_STORE[f"{key}_iv_{seg_col}_{seg_val}"] = woe_result["iv_df"]
        _SERVER_STORE[f"{_pfx}_optb"]      = woe_result["optb_dict"]
        _SERVER_STORE[f"{_pfx}_bins"]      = woe_result["bins_dict"]
        _SERVER_STORE[f"{_pfx}_iv_tables"] = woe_result["iv_tables"]
        _SERVER_STORE[f"{_pfx}_failed"]    = woe_result["failed"]

    summary = compute_var_summary_table(config, key, seg_col, seg_val)

    if excluded_set:
        summary = summary[~summary["Değişken"].isin(excluded_set)].reset_index(drop=True)

    return _render_var_summary(summary, use_woe)
