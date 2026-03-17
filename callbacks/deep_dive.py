from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from app_instance import app
from server_state import _SERVER_STORE, get_df as _get_df
from utils.helpers import apply_segment_filter, get_splits
from utils.chart_helpers import _tab_info, _PLOT_LAYOUT, _AXIS_STYLE
from modules.deep_dive import get_variable_stats, get_woe_detail, compute_psi, compute_period_badrate


# ── Callback: Deep Dive — Değişken seçeneklerini doldur ──────────────────────
@app.callback(
    Output("tab-deep-dive", "children"),
    Input("store-config", "data"),
    Input("dd-segment-val", "value"),
    Input("store-expert-exclude", "data"),
    State("store-key", "data"),
    State("dd-segment-col", "value"),
)
def render_deep_dive_shell(config, seg_val, expert_excluded, key, seg_col_input):
    df = _get_df(key)
    if df is None or not config or not config.get("target_col"):
        return html.Div()

    expert_excluded = set(expert_excluded or [])
    screen_result = _SERVER_STORE.get(f"{key}_screen")
    if screen_result:
        passed_cols, _ = screen_result
        base_cols = [c for c in passed_cols if c != config["target_col"]]
    else:
        base_cols = [c for c in df.columns if c != config["target_col"]]
    cols = [c for c in base_cols if c not in expert_excluded]
    col_options = [{"label": f"{c}  [{df[c].dtype}]", "value": c} for c in cols]

    # PSI kesim tarihi — OOT date varsa otomatik, yoksa manuel seçim
    date_col = config.get("date_col")
    oot_date = config.get("oot_date")
    if oot_date:
        # OOT date config'de tanımlı → picker gizle, bilgi badge'i göster
        psi_date_col = dbc.Col([
            dbc.Label("PSI Kesim Tarihi", className="form-label"),
            html.Div(
                [html.Span("OOT: ", style={"color": "#7e8fa4", "fontSize": "0.72rem"}),
                 html.Span(f"≥ {oot_date}", style={"color": "#a78bfa", "fontSize": "0.80rem", "fontWeight": "700"})],
                className="form-hint",
            ),
            dbc.Select(id="dd-psi-split", options=[{"label": oot_date, "value": oot_date}],
                       value=oot_date, className="dark-select", disabled=True),
        ], width=3)
    elif date_col and date_col in df.columns:
        raw_dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        distinct  = sorted(raw_dates.dt.to_period("M").unique().astype(str))
        date_opts = [{"label": d, "value": d} for d in distinct]
        psi_date_col = dbc.Col([
            dbc.Label("PSI Kesim Tarihi", className="form-label"),
            html.Div("Öncesi = Baseline  ·  Sonrası = Karşılaştırma", className="form-hint"),
            dbc.Select(id="dd-psi-split", options=date_opts,
                       value=distinct[len(distinct)//2] if distinct else None,
                       className="dark-select"),
        ], width=3)
    else:
        psi_date_col = dbc.Col([
            dbc.Label("PSI Kesim Tarihi", className="form-label"),
            html.Div("\u00a0", className="form-hint"),
            dbc.Select(id="dd-psi-split", options=[], value=None,
                       className="dark-select", disabled=True,
                       placeholder="Tarih kolonu seçilmedi"),
        ], width=3)

    return html.Div([
        _tab_info("Değişken Analizi", "WoE · PSI · Bivariate Deep Dive",
                  "Seçilen değişken için Weight of Evidence (WoE) eğrisi, bin bazında bad rate, "
                  "PSI (Population Stability Index) ve hedefle ilişkiyi derinlemesine inceler. "
                  "PSI < 0.10 stabil, 0.10–0.25 dikkat, > 0.25 dağılım kayması var.",
                  "#4F8EF7"),
        html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Değişken Seç", className="form-label"),
                    html.Div("\u00a0", className="form-hint"),
                    dbc.Select(id="dd-deepdive-col", options=col_options,
                               value=cols[0] if cols else None,
                               className="dark-select"),
                    html.Div([
                        html.Span("Kolon tipi: ",
                                  style={"color": "#7e8fa4", "fontSize": "0.72rem",
                                         "marginRight": "6px"}),
                        dbc.RadioItems(
                            id="dd-dtype-override",
                            options=[
                                {"label": "Otomatik", "value": "auto"},
                                {"label": "Sayısal",  "value": "numerical"},
                                {"label": "Kategorik", "value": "categorical"},
                            ],
                            value="auto",
                            inline=True,
                            labelStyle={"fontSize": "0.72rem", "color": "#9aa5bc",
                                        "marginRight": "10px", "cursor": "pointer"},
                            inputStyle={"marginRight": "3px", "cursor": "pointer"},
                        ),
                    ], style={"marginTop": "6px", "display": "flex", "alignItems": "center"}),
                ], width=4),
                psi_date_col,
                dbc.Col([
                    dbc.Label("Max Bin Sayısı", className="form-label"),
                    html.Div("WOE & IV için", className="form-hint"),
                    dbc.Input(id="dd-max-bins", type="number", value=4,
                              min=2, max=20, step=1,
                              style={"backgroundColor": "#1a2035", "color": "#c8cdd8",
                                     "border": "1px solid #2d3a4f", "fontSize": "0.85rem"}),
                ], width=2),
            ], className="mb-4"),
        ]),
        dcc.Loading(html.Div(id="deep-dive-content"), type="dot", color="#4F8EF7", delay_show=300),
        # Config'i aşağıya ilet
        dcc.Store(id="store-dd-config", data={
            "target_col":     config["target_col"],
            "target_type":    config.get("target_type", "binary"),
            "date_col":       config.get("date_col"),
            "oot_date":       config.get("oot_date"),
            "has_test_split": config.get("has_test_split", False),
            "test_size":      config.get("test_size", 20),
            "seg_col":        config.get("segment_col") or (seg_col_input or None),
            "seg_val":        seg_val,
            "key":            key,
        }),
    ])


@app.callback(
    Output("dd-dtype-override", "value"),
    Input("dd-deepdive-col", "value"),
    prevent_initial_call=True,
)
def reset_dtype_override(_col):
    return "auto"


@app.callback(
    Output("deep-dive-content", "children"),
    Input("dd-deepdive-col", "value"),
    Input("dd-psi-split", "value"),
    Input("dd-dtype-override", "value"),
    State("store-dd-config", "data"),
    State("dd-max-bins", "value"),
    prevent_initial_call=False,
)
def render_deep_dive_content(col, psi_split, dtype_override, dd_config, max_n_bins):
    if not col or not dd_config:
        return html.Div()

    df_orig = _get_df(dd_config["key"])
    if df_orig is None:
        return html.Div()

    target      = dd_config["target_col"]
    target_type = dd_config.get("target_type", "binary")
    date_col    = dd_config.get("date_col")
    oot_date    = dd_config.get("oot_date")
    seg_col     = dd_config.get("seg_col")
    seg_val     = dd_config.get("seg_val")
    df_active = apply_segment_filter(df_orig, seg_col, seg_val)

    # Periyot bölünmesi: get_splits ile train/test/OOT
    df_train, df_test, df_oot = get_splits(df_active, dd_config)

    vstats = get_variable_stats(df_active, col, target)

    _max_bins    = int(max_n_bins) if max_n_bins and int(max_n_bins) >= 2 else 4
    _force_dtype = dtype_override if dtype_override and dtype_override != "auto" else None

    # WOE yalnızca binary target için anlamlı, sadece train üzerinden
    if target_type == "binary":
        woe_df, iv_total_dd, woe_bin_edges = get_woe_detail(
            df_train, col, target, _max_bins, force_dtype=_force_dtype)
    else:
        woe_df, iv_total_dd, woe_bin_edges = pd.DataFrame(), 0.0, None

    # PSI cutoff: OOT date öncelikli, yoksa manuel seçim
    cutoff_date = oot_date if oot_date else (psi_split if psi_split else None)
    psi_res = compute_psi(
        df_active, col, target,
        date_col=date_col if date_col else None,
        cutoff_date=cutoff_date,
        bin_edges=woe_bin_edges,
        force_dtype=_force_dtype,
    )

    is_num = (vstats["is_numeric"] if _force_dtype is None
              else (_force_dtype == "numerical"))

    # ── 1. Özet İstatistik Kartları ───────────────────────────────────────────
    def sc(val, lbl, color="#4F8EF7"):
        return dbc.Col(html.Div([
            html.Div(str(val), className="metric-value",
                     style={"fontSize": "1.1rem", "color": color}),
            html.Div(lbl, className="metric-label"),
        ], className="metric-card"))

    missing_color = "#ef4444" if vstats["missing_pct"] > 50 else "#f59e0b" if vstats["missing_pct"] > 5 else "#10b981"

    stat_cards = [
        sc(vstats["dtype"],                    "Tip"),
        sc(f"{vstats['missing']:,}",           f"Eksik  (%{vstats['missing_pct']})", missing_color),
        sc(f"{vstats['unique']:,}",            "Tekil Değer"),
    ]
    if is_num:
        skew_color = "#f59e0b" if abs(vstats.get("skewness") or 0) > 1 else "#10b981"
        stat_cards += [
            sc(vstats.get("skewness", "—"),    "Çarpıklık", skew_color),
            sc(vstats.get("kurtosis", "—"),    "Basıklık"),
            sc(f"{vstats.get('outlier_count', 0):,}  (%{vstats.get('outlier_pct', 0)})",
               "IQR Aykırı", "#f59e0b" if vstats.get("outlier_pct", 0) > 5 else "#556070"),
        ]

    stats_row = dbc.Row(stat_cards, className="g-3 mb-4")

    # Eksik vs Target kartı
    missing_target_card = html.Div()
    if vstats["missing"] > 0:
        if target_type == "binary" and vstats.get("missing_bad_rate") is not None:
            diff = vstats["missing_bad_rate"] - vstats["present_bad_rate"]
            diff_color = "#ef4444" if abs(diff) > 3 else "#f59e0b" if abs(diff) > 1 else "#10b981"
            missing_target_card = html.Div([
                html.P("Eksik Değer & Target İlişkisi", className="section-title"),
                dbc.Row([
                    sc(f"%{vstats['present_bad_rate']}", "Dolu → Bad Rate", "#4F8EF7"),
                    sc(f"%{vstats['missing_bad_rate']}", "Eksik → Bad Rate", "#f59e0b"),
                    sc(f"{'+' if diff > 0 else ''}{diff:.2f}pp", "Fark", diff_color),
                ], className="g-3 mb-4"),
            ])
        elif target_type in ("continuous", "multiclass"):
            # Binary dışı: bad rate yerine ortalama target farkını göster
            lc_m = df_active[[col, target]].copy()
            lc_m[target] = pd.to_numeric(lc_m[target], errors="coerce")
            present_mean = lc_m[lc_m[col].notna()][target].mean()
            missing_mean = lc_m[lc_m[col].isna()][target].mean()
            if pd.notna(present_mean) and pd.notna(missing_mean):
                diff_m = missing_mean - present_mean
                ref = abs(present_mean) or 1.0
                diff_color = "#ef4444" if abs(diff_m) > ref * 0.1 else "#f59e0b" if abs(diff_m) > ref * 0.03 else "#10b981"
                missing_target_card = html.Div([
                    html.P("Eksik Değer & Target İlişkisi", className="section-title"),
                    dbc.Row([
                        sc(f"{present_mean:.4f}", "Dolu → Ort. Target", "#4F8EF7"),
                        sc(f"{missing_mean:.4f}", "Eksik → Ort. Target", "#f59e0b"),
                        sc(f"{'+' if diff_m > 0 else ''}{diff_m:.4f}", "Fark", diff_color),
                    ], className="g-3 mb-4"),
                ])

    # ── 2. Dağılım Grafikleri ─────────────────────────────────────────────────
    if is_num:
        local = df_active[[col, target]].dropna(subset=[col, target]).copy()

        if target_type == "continuous":
            # Continuous target: basit histogram + scatter (col vs target)
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=local[col], name=col,
                marker_color="#4F8EF7", opacity=0.75, nbinsx=50,
                hovertemplate="Değer: %{x}<br>Sayı: %{y}<extra></extra>",
            ))
            if vstats.get("iqr_lower") is not None:
                for x_val, lbl, clr in [
                    (vstats["iqr_lower"], "IQR Alt", "#f59e0b"),
                    (vstats["iqr_upper"], "IQR Üst", "#f59e0b"),
                    (vstats["p1"],  "P1",  "#556070"),
                    (vstats["p99"], "P99", "#556070"),
                ]:
                    fig_dist.add_vline(x=x_val, line_dash="dot", line_color=clr,
                                       opacity=0.6, annotation_text=lbl,
                                       annotation_font_color=clr, annotation_font_size=9)
            fig_dist.update_layout(
                **_PLOT_LAYOUT,
                title=dict(text=f"{col} — Dağılım", font=dict(color="#E8EAF0", size=13)),
                xaxis=dict(**_AXIS_STYLE, title=col),
                yaxis=dict(**_AXIS_STYLE, title="Frekans"),
                height=320,
            )

            # Scatter: col vs target (korelasyon görselleştirmesi)
            sample = local.sample(min(2000, len(local)), random_state=42)
            fig_scatter = go.Figure(go.Scatter(
                x=sample[col], y=sample[target],
                mode="markers",
                marker=dict(color="#4F8EF7", opacity=0.4, size=4),
                hovertemplate=f"{col}: %{{x}}<br>{target}: %{{y}}<extra></extra>",
            ))
            fig_scatter.update_layout(
                **_PLOT_LAYOUT,
                title=dict(text=f"{col} vs {target} — Scatter", font=dict(color="#E8EAF0", size=13)),
                xaxis=dict(**_AXIS_STYLE, title=col),
                yaxis=dict(**_AXIS_STYLE, title=target),
                height=320,
            )

            dist_section = dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_dist,    config={"displayModeBar": False}), width=6),
                dbc.Col(dcc.Graph(figure=fig_scatter, config={"displayModeBar": False}), width=6),
            ], className="mb-4")

        else:
            # Binary / multiclass: her sınıf için ayrı histogram
            local[target] = local[target].astype(str).str.replace(r'\.0$', '', regex=True)

            fig_dist = go.Figure()
            colors = {"0": "#4F8EF7", "1": "#ef4444"}
            for t_val, grp in local.groupby(target)[col]:
                fig_dist.add_trace(go.Histogram(
                    x=grp, name=f"Target={t_val}",
                    marker_color=colors.get(str(t_val), "#8892a4"),
                    opacity=0.65, nbinsx=50,
                    hovertemplate=f"Target={t_val}<br>Değer: %{{x}}<br>Sayı: %{{y}}<extra></extra>",
                ))

            if vstats.get("iqr_lower") is not None:
                for x_val, lbl, clr in [
                    (vstats["iqr_lower"], "IQR Alt", "#f59e0b"),
                    (vstats["iqr_upper"], "IQR Üst", "#f59e0b"),
                    (vstats["p1"],  "P1",  "#556070"),
                    (vstats["p99"], "P99", "#556070"),
                ]:
                    fig_dist.add_vline(x=x_val, line_dash="dot", line_color=clr,
                                       opacity=0.6, annotation_text=lbl,
                                       annotation_font_color=clr, annotation_font_size=9)

            fig_dist.update_layout(
                **_PLOT_LAYOUT,
                barmode="overlay",
                title=dict(text=f"{col} — Dağılım (Target Kırılımı)", font=dict(color="#E8EAF0", size=13)),
                xaxis=dict(**_AXIS_STYLE, title=col),
                yaxis=dict(**_AXIS_STYLE, title="Frekans"),
                legend=dict(bgcolor="#161C27", bordercolor="#232d3f"),
                height=320,
            )

            # Target grubu istatistik karşılaştırma tablosu
            stat_rows = []
            local = local.dropna(subset=[target])
            grp_data = {str(int(float(tv))): g[col].dropna()
                        for tv, g in local.groupby(local[target])}
            for stat_name, fn in [
                ("Gözlem",  lambda s: f"{len(s):,}"),
                ("Ortalama", lambda s: f"{s.mean():.4f}"),
                ("Std",      lambda s: f"{s.std():.4f}"),
                ("Min",      lambda s: f"{s.min():.4f}"),
                ("P25",      lambda s: f"{s.quantile(.25):.4f}"),
                ("Medyan",   lambda s: f"{s.median():.4f}"),
                ("P75",      lambda s: f"{s.quantile(.75):.4f}"),
                ("P95",      lambda s: f"{s.quantile(.95):.4f}"),
                ("P99",      lambda s: f"{s.quantile(.99):.4f}"),
                ("Max",      lambda s: f"{s.max():.4f}"),
            ]:
                row = {"İstatistik": stat_name}
                for tv, g in grp_data.items():
                    row[f"Target={tv}"] = fn(g) if len(g) else "—"
                stat_rows.append(row)

            stat_tbl_cols = ["İstatistik"] + [f"Target={k}" for k in sorted(grp_data.keys())]
            stat_table = dash_table.DataTable(
                data=stat_rows,
                columns=[{"name": c, "id": c} for c in stat_tbl_cols],
                style_table={"overflowX": "auto"},
                style_header={"backgroundColor": "#111827", "color": "#a8b2c2",
                              "fontWeight": "700", "fontSize": "0.72rem",
                              "border": "1px solid #2d3a4f", "textTransform": "uppercase"},
                style_data={"backgroundColor": "#161C27", "color": "#c8cdd8",
                            "fontSize": "0.82rem", "border": "1px solid #232d3f"},
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"},
                    {"if": {"column_id": "Target=1"}, "color": "#ef4444"},
                    {"if": {"column_id": "Target=0"}, "color": "#4F8EF7"},
                ],
                style_cell={"padding": "0.4rem 0.7rem"},
                style_cell_conditional=[
                    {"if": {"column_id": "İstatistik"}, "fontWeight": "600",
                     "color": "#a8b2c2", "textAlign": "left"},
                ],
            )
            dist_section = dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_dist, config={"displayModeBar": False}), width=8),
                dbc.Col([
                    html.P("Good vs Bad İstatistikleri", className="section-title",
                           style={"marginTop": "0.5rem"}),
                    stat_table,
                ], width=4),
            ], className="mb-4")

    else:
        # Kategorik — bar chart
        local = df_active[[col, target]].copy()
        local[col] = local[col].fillna("Eksik").astype(str)
        top_cats = local[col].value_counts().head(20).index
        local = local[local[col].isin(top_cats)]

        fig_dist = go.Figure()
        if target_type == "continuous":
            # Continuous target: sadece frekans (target kırılımı anlamsız)
            vc = local[col].value_counts().reset_index()
            vc.columns = [col, "count"]
            fig_dist.add_trace(go.Bar(
                x=vc[col], y=vc["count"],
                marker_color="#4F8EF7",
                hovertemplate="%{x}<br>Sayı: %{y}<extra></extra>",
            ))
            bar_title = f"{col} — Değer Dağılımı (Top 20)"
        else:
            local[target] = pd.to_numeric(local[target], errors='coerce')
            local = local.dropna(subset=[target])
            vc = local.groupby([col, target]).size().reset_index(name="count")
            vc[target] = vc[target].astype(str).str.replace(r'\.0$', '', regex=True)
            colors = {"0": "#4F8EF7", "1": "#ef4444"}
            for t_val, grp in vc.groupby(target):
                fig_dist.add_trace(go.Bar(
                    x=grp[col], y=grp["count"],
                    name=f"Target={t_val}",
                    marker_color=colors.get(str(t_val), "#8892a4"),
                    hovertemplate="%{x}<br>Sayı: %{y}<extra></extra>",
                ))
            bar_title = f"{col} — Değer Dağılımı (Top 20)"

        fig_dist.update_layout(
            **_PLOT_LAYOUT,
            barmode="stack",
            title=dict(text=bar_title, font=dict(color="#E8EAF0", size=13)),
            xaxis=dict(**_AXIS_STYLE, title=col, tickangle=-30),
            yaxis=dict(**_AXIS_STYLE, title="Frekans"),
            legend=dict(bgcolor="#161C27", bordercolor="#232d3f"),
            height=340,
        )
        dist_section = html.Div([
            dcc.Graph(figure=fig_dist, config={"displayModeBar": False}),
        ], className="mb-4")

    # ── 3. WOE / Bad Rate Grafiği (Periyot Bazlı) ────────────────────────────
    woe_section = html.Div()
    if target_type != "binary":
        woe_section = html.Div(
            "ℹ WOE/IV analizi yalnızca binary (0/1) target için hesaplanır. "
            "Değişken gücü için Değişken Özeti sekmesindeki MI skoruna bakınız.",
            style={"color": "#7e8fa4", "fontSize": "0.80rem",
                   "padding": "0.6rem 0.8rem", "border": "1px solid #2d3a4f",
                   "borderRadius": "5px", "marginBottom": "1.5rem"},
        )
    elif not woe_df.empty:
        iv_total = iv_total_dd
        iv_label = ("Çok Zayıf" if iv_total < 0.02 else "Zayıf" if iv_total < 0.1
                    else "Orta" if iv_total < 0.3 else "Güçlü" if iv_total < 0.5 else "Şüpheli")
        iv_color = {"Çok Zayıf": "#4a5568", "Zayıf": "#f59e0b", "Orta": "#4F8EF7",
                    "Güçlü": "#10b981", "Şüpheli": "#ef4444"}.get(iv_label, "#4F8EF7")

        def _build_woe_period_panel(period_label, period_df, ref_woe_df,
                                    ref_bin_edges, accent, is_train=False):
            """Tek bir periyot için bad rate paneli oluşturur.
            X: bin etiketleri, bar: adet (y2), line: bad rate % (y1).
            Tüm periyotlar için tablo gösterilir.
            """
            if period_df is None or len(period_df) == 0:
                return None

            tsv_id = f"woe-tsv-{period_label.lower()}"

            if is_train:
                chart_df = ref_woe_df[ref_woe_df["Bin"] != "TOPLAM"].copy()
                table_df = ref_woe_df.copy()
            else:
                # Test / OOT: train bin edges ile bad rate hesapla
                p_df = compute_period_badrate(period_df, col, target,
                                              ref_woe_df, ref_bin_edges)
                if p_df.empty:
                    return None
                chart_df = p_df[p_df["Bin"] != "TOPLAM"].copy()
                # Tablo: period sonuçları + train WOE referansı
                table_df = p_df.copy()
                woe_ref = ref_woe_df.set_index("Bin")["WOE"].to_dict()
                table_df["WOE (Train)"] = table_df["Bin"].map(
                    lambda b: woe_ref.get(b, "")
                )

            n_period   = len(period_df)
            bad_n      = int(period_df[target].sum())
            br_overall = round(bad_n / n_period * 100, 2) if n_period > 0 else 0.0

            title_text = (f"{period_label}  ·  n={n_period:,}  ·  Bad Rate: {br_overall:.2f}%"
                          + (f"  |  IV: {iv_total:.4f}  [{iv_label}]" if is_train else ""))

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=chart_df["Bin"], y=chart_df["Toplam"],
                name="Adet", marker_color="#232d4f", opacity=0.75,
                yaxis="y2",
                hovertemplate="Bin: %{x}<br>Adet: %{y:,}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=chart_df["Bin"], y=chart_df["Bad Rate %"],
                name="Bad Rate %", mode="lines+markers",
                line=dict(color=accent, width=2),
                marker=dict(size=6),
                hovertemplate="Bin: %{x}<br>Bad Rate: %{y:.2f}%<extra></extra>",
            ))
            fig.update_layout(
                **_PLOT_LAYOUT,
                title=dict(text=title_text,
                           font=dict(color=iv_color if is_train else "#c8cdd8", size=12)),
                xaxis=dict(**_AXIS_STYLE, tickangle=-30),
                yaxis=dict(**_AXIS_STYLE, title="Bad Rate %", ticksuffix="%"),
                yaxis2=dict(title="Adet", overlaying="y", side="right", showgrid=False,
                            zeroline=False),
                legend=dict(bgcolor="#161C27", bordercolor="#232d3f", font=dict(size=10)),
                height=300,
            )

            # Monotonluk etiketi
            br_arr   = chart_df["Bad Rate %"].values
            br_diffs = np.diff(br_arr)
            if len(br_diffs) == 0:
                mono_txt, mono_clr = "—", "#6b7a99"
            elif all(d >= 0 for d in br_diffs):
                mono_txt, mono_clr = "↑ Monoton Artan", "#10b981"
            elif all(d <= 0 for d in br_diffs):
                mono_txt, mono_clr = "↓ Monoton Azalan", "#10b981"
            else:
                mono_txt, mono_clr = "✗ Monoton Değil", "#ef4444"

            table_tsv = table_df.to_csv(sep="\t", index=False)
            table_cols = [{"name": c, "id": c} for c in table_df.columns]

            children = [
                html.Div([
                    html.Span(f"{period_label}  ", style={"color": accent, "fontWeight": "700",
                                                           "fontSize": "0.80rem"}),
                    html.Span("Bad Rate ", style={"color": "#7e8fa4", "fontSize": "0.72rem"}),
                    html.Span(mono_txt, style={"color": mono_clr, "fontSize": "0.72rem",
                                               "fontWeight": "700"}),
                ], style={"marginBottom": "0.3rem"}),
                dcc.Graph(figure=fig, config={"displayModeBar": False}),
                html.Div([
                    html.Div(
                        dcc.Clipboard(target_id=tsv_id, title="Kopyala",
                                      style={"cursor": "pointer", "fontSize": "0.72rem",
                                             "color": "#a8b2c2", "padding": "0.2rem 0.55rem",
                                             "border": "1px solid #2d3a4f", "borderRadius": "4px",
                                             "backgroundColor": "#1a2035", "float": "right",
                                             "marginBottom": "0.4rem"}),
                        style={"overflow": "hidden"},
                    ),
                    html.Pre(table_tsv, id=tsv_id, style={"display": "none"}),
                    dash_table.DataTable(
                        data=table_df.to_dict("records"),
                        columns=table_cols,
                        style_table={"overflowX": "auto"},
                        style_header={"backgroundColor": "#111827", "color": "#a8b2c2",
                                      "fontWeight": "700", "fontSize": "0.72rem",
                                      "border": "1px solid #2d3a4f", "textTransform": "uppercase"},
                        style_data={"backgroundColor": "#161C27", "color": "#c8cdd8",
                                    "fontSize": "0.82rem", "border": "1px solid #232d3f"},
                        style_data_conditional=[
                            {"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"},
                            {"if": {"filter_query": '{Bin} = "TOPLAM"'},
                             "backgroundColor": "#1a3050", "fontWeight": "700",
                             "color": "#E8EAF0", "borderTop": "1px solid #4F8EF7"},
                        ],
                        style_cell={"padding": "0.4rem 0.65rem", "textAlign": "right"},
                        style_cell_conditional=[{"if": {"column_id": "Bin"}, "textAlign": "left"}],
                    ),
                ]),
            ]

            return html.Div(children, style={
                "border": "1px solid #1e2a3a", "borderRadius": "6px",
                "padding": "0.75rem 0.85rem", "marginBottom": "0.85rem",
                "backgroundColor": "#0d1520",
            })

        # Periyot listesi
        period_configs = [("Train", df_train, "#ef4444", True)]
        if df_test is not None:
            period_configs.append(("Test", df_test, "#f59e0b", False))
        if df_oot is not None:
            period_configs.append(("OOT", df_oot, "#a78bfa", False))

        period_panels = []
        for p_label, p_df, p_accent, is_train in period_configs:
            panel = _build_woe_period_panel(
                p_label, p_df, woe_df, woe_bin_edges, p_accent,
                is_train=is_train,
            )
            if panel:
                period_panels.append(panel)

        woe_section = html.Div([
            html.P("WOE & Bad Rate Analizi (Periyot Bazlı)", className="section-title"),
            html.Div(f"IV: {iv_total:.4f}  [{iv_label}]",
                     style={"color": iv_color, "fontWeight": "700",
                            "fontSize": "0.82rem", "marginBottom": "0.75rem"}),
            *period_panels,
        ], className="mb-4")

    # ── 4. PSI ────────────────────────────────────────────────────────────────
    psi_section = html.Div()
    if psi_res.get("psi") is not None:
        psi_val   = psi_res["psi"]
        psi_label = psi_res["label"]
        psi_color = "#10b981" if psi_val < 0.1 else "#f59e0b" if psi_val < 0.25 else "#ef4444"
        psi_df    = psi_res["detail_df"]

        fig_psi = go.Figure()
        fig_psi.add_trace(go.Bar(
            x=psi_df["Bin"], y=psi_df["Baseline %"],
            name=f"Baseline  {psi_res['split_label']}  (n={psi_res['n_baseline']:,})",
            marker_color="#4F8EF7", opacity=0.75,
        ))
        fig_psi.add_trace(go.Bar(
            x=psi_df["Bin"], y=psi_df["Karşılaştırma %"],
            name=f"Karşılaştırma  {psi_res['comp_label']}  (n={psi_res['n_compare']:,})",
            marker_color="#f59e0b", opacity=0.75,
        ))
        fig_psi.update_layout(
            **_PLOT_LAYOUT,
            barmode="group",
            title=dict(
                text=f"{col} — PSI: {psi_val:.4f}  [{psi_label}]",
                font=dict(color=psi_color, size=13),
            ),
            xaxis=dict(**_AXIS_STYLE, tickangle=-30),
            yaxis=dict(**_AXIS_STYLE, title="Dağılım %", ticksuffix="%"),
            legend=dict(bgcolor="#161C27", bordercolor="#232d3f"),
            height=300,
        )

        psi_section = html.Div([
            html.P("PSI — Popülasyon Stabilite İndeksi", className="section-title"),
            dbc.Row([
                sc(f"{psi_val:.4f}", "PSI", psi_color),
                sc(psi_label, "Değerlendirme", psi_color),
                sc(f"{psi_res['n_baseline']:,}", "Baseline N"),
                sc(f"{psi_res['n_compare']:,}", "Karşılaştırma N"),
            ], className="g-3 mb-3"),
            dcc.Graph(figure=fig_psi, config={"displayModeBar": False}),
            html.Div([
                html.Span("PSI Eşikleri: ", style={"color": "#7e8fa4", "fontSize": "0.73rem"}),
                html.Span("< 0.10 Stabil  · ", style={"color": "#10b981", "fontSize": "0.73rem"}),
                html.Span("0.10–0.25 Hafif Kayma  · ", style={"color": "#f59e0b", "fontSize": "0.73rem"}),
                html.Span("> 0.25 Kritik Kayma", style={"color": "#ef4444", "fontSize": "0.73rem"}),
            ], style={"marginTop": "0.5rem"}),
        ], className="mb-4")

    return html.Div([
        html.P("Özet İstatistikler", className="section-title"),
        stats_row,
        missing_target_card,
        html.P("Dağılım Analizi", className="section-title"),
        dist_section,
        woe_section,
        psi_section,
    ])
