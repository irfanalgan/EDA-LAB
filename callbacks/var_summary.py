from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd

from app_instance import app
from server_state import _SERVER_STORE, get_df as _get_df
from utils.helpers import apply_segment_filter
from utils.chart_helpers import _build_woe_dataset
from modules.deep_dive import compute_iv_ranking_optimal, compute_psi
from modules.correlation import compute_correlation_matrix, find_high_corr_pairs, compute_vif


# ── Callback: Değişken Özeti ───────────────────────────────────────────────────
@app.callback(
    Output("div-var-summary", "children"),
    Input("btn-var-summary", "n_clicks"),
    State("store-key", "data"),
    State("store-config", "data"),
    State("dd-segment-val", "value"),
    State("dd-segment-col", "value"),
    State("chk-varsummary-woe", "value"),
    prevent_initial_call=True,
)
def update_var_summary(n_clicks, key, config, seg_val, seg_col_input, woe_toggle):
    if not n_clicks or not key or not config or not config.get("target_col"):
        return html.Div()

    df_orig = _get_df(key)
    if df_orig is None:
        return html.Div("Veri yüklenmemiş.", className="alert-info-custom")

    target    = config["target_col"]
    date_col  = config.get("date_col")
    seg_col   = config.get("segment_col") or (seg_col_input or None)
    df_active = apply_segment_filter(df_orig, seg_col, seg_val)
    use_woe   = "woe" in (woe_toggle or [])

    # ── 1. IV + Eksik% — her zaman ham veriden ────────────────────────────────
    iv_cache_key = f"{key}_iv_{seg_col}_{seg_val}"
    if iv_cache_key in _SERVER_STORE:
        iv_df = _SERVER_STORE[iv_cache_key]
    else:
        iv_df = compute_iv_ranking_optimal(df_active, target)
        _SERVER_STORE[iv_cache_key] = iv_df

    summary = iv_df[["Değişken", "IV", "Eksik %", "Güç"]].copy()
    var_list = summary["Değişken"].tolist()

    # ── 2. WoE dataset (gerekiyorsa) ──────────────────────────────────────────
    if use_woe:
        woe_cache_key = f"{key}_woe_{seg_col}_{seg_val}"
        if woe_cache_key not in _SERVER_STORE:
            woe_df_enc, _ = _build_woe_dataset(df_active, target, var_list)
            _SERVER_STORE[woe_cache_key] = (woe_df_enc, _)
        else:
            woe_df_enc, _ = _SERVER_STORE[woe_cache_key]
        # Analiz için df_analysis: _woe kolonları + target
        woe_cols_present = [f"{v}_woe" for v in var_list if f"{v}_woe" in woe_df_enc.columns]
        df_analysis = woe_df_enc[woe_cols_present].copy()
        df_analysis[target] = df_active[target].values
        # Korelasyon/VIF/PSI için kolon adlarını orijinale eşle
        col_rename = {f"{v}_woe": v for v in var_list}
        df_analysis_renamed = df_analysis.rename(columns=col_rename)
    else:
        df_analysis_renamed = df_active

    # ── 3. Korelasyon: yüksek korelasyon flag ─────────────────────────────────
    corr_flag: dict[str, str] = {}
    if use_woe:
        # Taze hesapla — WoE kolonları üzerinden
        try:
            num_cols = [v for v in var_list if v in df_analysis_renamed.columns
                        and pd.api.types.is_numeric_dtype(df_analysis_renamed[v])]
            if len(num_cols) >= 2:
                corr_df_woe = compute_correlation_matrix(df_analysis_renamed, num_cols)
                high_pairs  = find_high_corr_pairs(corr_df_woe, threshold=0.75)
                for _, row in high_pairs.iterrows():
                    v1, v2 = row["Değişken 1"], row["Değişken 2"]
                    r = abs(row.get("Korelasyon", row.get(high_pairs.columns[2], 0)))
                    for v in (v1, v2):
                        if v not in corr_flag or r > float(corr_flag[v].split("r=")[-1].rstrip(")")):
                            corr_flag[v] = f"⚠ (r={r:.2f})"
        except Exception:
            pass
    else:
        # Cache'den al
        for k in _SERVER_STORE:
            if k.startswith(f"{key}_corr_{seg_col}_{seg_val}_"):
                try:
                    corr_df_found, _ = _SERVER_STORE[k]
                    high_pairs = find_high_corr_pairs(corr_df_found, threshold=0.75)
                    for _, row in high_pairs.iterrows():
                        v1, v2 = row["Değişken 1"], row["Değişken 2"]
                        r = abs(row.get("Korelasyon", row.get(high_pairs.columns[2], 0)))
                        for v in (v1, v2):
                            if v not in corr_flag or r > float(corr_flag[v].split("r=")[-1].rstrip(")")):
                                corr_flag[v] = f"⚠ (r={r:.2f})"
                except Exception:
                    pass
                break
    summary["Yüksek Korr."] = summary["Değişken"].map(lambda v: corr_flag.get(v, "—"))

    # ── 4. VIF ────────────────────────────────────────────────────────────────
    vif_map: dict[str, float] = {}
    if use_woe:
        try:
            num_cols_vif = [v for v in var_list if v in df_analysis_renamed.columns
                            and pd.api.types.is_numeric_dtype(df_analysis_renamed[v])]
            if len(num_cols_vif) >= 2:
                vif_res = compute_vif(df_analysis_renamed, num_cols_vif)
                if not vif_res.empty and "Değişken" in vif_res.columns:
                    for _, row in vif_res.iterrows():
                        vif_map[row["Değişken"]] = row["VIF"]
        except Exception:
            pass
    else:
        for k in _SERVER_STORE:
            if k.startswith(f"{key}_vif_{seg_col}_{seg_val}_"):
                vif_df_cached = _SERVER_STORE[k]
                if not vif_df_cached.empty and "Değişken" in vif_df_cached.columns and "VIF" in vif_df_cached.columns:
                    for _, row in vif_df_cached.iterrows():
                        vif_map[row["Değişken"]] = row["VIF"]
                break
    summary["VIF"] = summary["Değişken"].map(
        lambda v: round(vif_map[v], 1) if v in vif_map else "—"
    )

    # ── 5. PSI (batch) ────────────────────────────────────────────────────────
    psi_map: dict[str, float] = {}
    psi_label_map: dict[str, str] = {}
    if date_col:
        for var in var_list:
            try:
                if use_woe:
                    woe_col = f"{var}_woe"
                    if woe_col not in woe_df_enc.columns:
                        continue
                    tmp = woe_df_enc[[woe_col]].copy()
                    tmp.columns = [var]
                    tmp[target]   = df_active[target].values
                    tmp[date_col] = df_active[date_col].values
                    res = compute_psi(tmp, var, target, date_col=date_col)
                else:
                    res = compute_psi(df_active, var, target, date_col=date_col)
                if res.get("psi") is not None:
                    psi_map[var]       = res["psi"]
                    psi_label_map[var] = res["label"]
            except Exception:
                pass
    summary["PSI"] = summary["Değişken"].map(
        lambda v: round(psi_map[v], 4) if v in psi_map else ("—" if not date_col else "Hata")
    )
    summary["PSI Durum"] = summary["Değişken"].map(lambda v: psi_label_map.get(v, "—"))

    # ── 5. Öneri mantığı ──────────────────────────────────────────────────────
    def _recommend_with_reason(row):
        iv_val    = row["IV"]
        eksik_val = row["Eksik %"]
        psi_val   = row["PSI"] if isinstance(row["PSI"], (int, float)) else None
        high_corr = row["Yüksek Korr."] != "—"
        vif_val   = row["VIF"] if isinstance(row["VIF"], (int, float)) else None

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
        if high_corr:
            inc_reasons.append(f"Korr.{row['Yüksek Korr.']}")
        if vif_val is not None and vif_val > 5.0:
            inc_reasons.append(f"VIF={vif_val:.1f}>5")

        if inc_reasons:
            return "⚠️ İncele", "; ".join(inc_reasons)

        return "✅ Tut", "—"

    summary[["Öneri", "Sebep"]] = summary.apply(
        lambda r: pd.Series(_recommend_with_reason(r)), axis=1
    )

    # ── 6. Renk kodlaması ─────────────────────────────────────────────────────
    style_conditions = [
        {"if": {"filter_query": '{Öneri} = "✅ Tut"',    "column_id": "Öneri"}, "color": "#10b981", "fontWeight": "700"},
        {"if": {"filter_query": '{Öneri} = "⚠️ İncele"', "column_id": "Öneri"}, "color": "#f59e0b", "fontWeight": "600"},
        {"if": {"filter_query": '{Öneri} = "❌ Çıkar"',  "column_id": "Öneri"}, "color": "#ef4444", "fontWeight": "700"},
        {"if": {"filter_query": '{Öneri} = "⚠️ İncele"', "column_id": "Sebep"}, "color": "#f59e0b"},
        {"if": {"filter_query": '{Öneri} = "❌ Çıkar"',  "column_id": "Sebep"}, "color": "#ef4444"},
        {"if": {"filter_query": '{Güç} = "Güçlü"',      "column_id": "Güç"},   "color": "#10b981"},
        {"if": {"filter_query": '{Güç} = "Orta"',       "column_id": "Güç"},   "color": "#4F8EF7"},
        {"if": {"filter_query": '{Güç} = "Zayıf"',      "column_id": "Güç"},   "color": "#f59e0b"},
        {"if": {"filter_query": '{Güç} = "Çok Zayıf"',  "column_id": "Güç"},   "color": "#7e8fa4"},
        {"if": {"filter_query": '{Güç} = "Şüpheli"',    "column_id": "Güç"},   "color": "#ef4444"},
        {"if": {"filter_query": '{PSI Durum} = "Kritik Kayma"',  "column_id": "PSI Durum"}, "color": "#ef4444"},
        {"if": {"filter_query": '{PSI Durum} = "Hafif Kayma"',   "column_id": "PSI Durum"}, "color": "#f59e0b"},
        {"if": {"filter_query": '{PSI Durum} = "Stabil"',        "column_id": "PSI Durum"}, "color": "#10b981"},
        {"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"},
    ]

    # Öneri ve Sebep kolonlarını öne al
    col_order = ["Değişken", "Öneri", "Sebep", "IV", "Güç", "Eksik %",
                 "PSI", "PSI Durum", "Yüksek Korr.", "VIF"]
    summary = summary[[c for c in col_order if c in summary.columns]]

    # Tam özeti cache'e yaz — Playground önizlemesi buradan okur
    _SERVER_STORE[f"{key}_summary_{seg_col}_{seg_val}"] = summary.copy()

    n_cik   = (summary["Öneri"] == "❌ Çıkar").sum()
    n_inc   = (summary["Öneri"] == "⚠️ İncele").sum()
    n_tut   = (summary["Öneri"] == "✅ Tut").sum()

    tsv = summary.to_csv(sep="\t", index=False)

    woe_note = html.Div(
        "★ PSI · Korelasyon · VIF — WoE dönüştürülmüş değerler üzerinden hesaplandı.",
        style={"color": "#a78bfa", "fontSize": "0.75rem", "marginBottom": "0.75rem"},
    ) if use_woe else html.Div()

    return html.Div([
        woe_note,
        # Özet sayaçlar
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
        # Tablo başlığı + kopyala
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
                "IV":          {"value": "**Information Value** — target ile doğrusal olmayan ilişki gücü\n\n"
                                         "| Aralık | Güç |\n|---|---|\n"
                                         "| < 0.02 | Çok Zayıf |\n"
                                         "| 0.02–0.10 | Zayıf |\n"
                                         "| 0.10–0.30 | Orta |\n"
                                         "| 0.30–0.50 | Güçlü |\n"
                                         "| > 0.50 | Şüpheli (overfit riski) |", "type": "markdown"},
                "Güç":         {"value": "IV değerine göre değişken gücü kategorisi", "type": "markdown"},
                "Eksik %":     {"value": "Değişkendeki boş (null/NaN) değerlerin yüzdesi\n\n"
                                         "- **> 80%** → Çıkar\n- **20–80%** → İncele", "type": "markdown"},
                "PSI":         {"value": "**Population Stability Index** — veri dağılımının zaman içinde kayması\n\n"
                                         "| PSI | Durum |\n|---|---|\n"
                                         "| < 0.10 | Stabil |\n"
                                         "| 0.10–0.25 | Hafif Kayma |\n"
                                         "| > 0.25 | Kritik Kayma |", "type": "markdown"},
                "PSI Durum":   {"value": "PSI değerine göre stabilite etiketi", "type": "markdown"},
                "Yüksek Korr.": {"value": "Başka bir değişkenle **r ≥ 0.75** korelasyon varsa gösterir.\n"
                                          "Yüksek korelasyon çoklu doğrusallık sorununa yol açabilir.", "type": "markdown"},
                "VIF":         {"value": "**Variance Inflation Factor** — çoklu doğrusal bağlantı ölçüsü\n\n"
                                         "- **< 5** — Normal\n- **5–10** — Dikkat\n- **> 10** — Kritik", "type": "markdown"},
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
