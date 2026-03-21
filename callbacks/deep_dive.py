from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from app_instance import app
from server_state import _SERVER_STORE, get_df as _get_df
from utils.helpers import apply_segment_filter, get_splits
from utils.chart_helpers import _tab_info, _PLOT_LAYOUT, _AXIS_STYLE, calc_psi as _calc_psi, psi_label as _psi_label
from modules.deep_dive import get_variable_stats, get_woe_detail
from utils.anomaly_hints import (build_hint_section, check_iv, check_psi,
                                  check_variable_stats, check_train_size)


# ── Callback: Deep Dive — Değişken seçeneklerini doldur ──────────────────────
@app.callback(
    Output("tab-deep-dive", "children"),
    Input("store-config", "data"),
    Input("store-expert-exclude", "data"),
    State("store-key", "data"),
)
def render_deep_dive_shell(config, expert_excluded, key):
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

    # PSI kesim tarihi — OOT date config'den otomatik belirlenir
    date_col = config.get("date_col")
    oot_date = config.get("oot_date")
    _psi_val = oot_date or None
    if not _psi_val and date_col and date_col in df.columns:
        raw_dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        distinct  = sorted(raw_dates.dt.to_period("M").unique().astype(str))
        _psi_val  = distinct[len(distinct)//2] if distinct else None

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
                    dcc.Dropdown(id="dd-deepdive-col", options=col_options,
                                 value=cols[0] if cols else None,
                                 className="dark-select", searchable=True, placeholder="Kolon ara\u2026"),
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
                # Hidden — callback uyumluluğu için
                html.Div([
                    dcc.Dropdown(id="dd-psi-split", options=[], value=_psi_val,
                                 style={"display": "none"}),
                ], style={"display": "none"}),
            ], className="mb-4"),
        ]),
        # Ham / WoE tab ayrımı
        dbc.Tabs(
            id="dd-data-tab",
            active_tab="dd-tab-woe",
            children=[
                dbc.Tab(label="WoE Değerler", tab_id="dd-tab-woe",
                        tab_style={"fontSize": "0.78rem"},
                        active_label_style={"color": "#4F8EF7", "fontWeight": "700"}),
                dbc.Tab(label="Ham Değerler", tab_id="dd-tab-raw",
                        tab_style={"fontSize": "0.78rem"},
                        active_label_style={"color": "#10b981", "fontWeight": "700"}),
            ],
            className="mb-3",
        ),
        dcc.Loading(html.Div(id="deep-dive-content"), type="dot", color="#4F8EF7", delay_show=300),
        # Config'i aşağıya ilet
        dcc.Store(id="store-dd-config", data={
            "target_col":     config["target_col"],
            "date_col":       config.get("date_col"),
            "sort_col":       config.get("sort_col"),
            "oot_date":       config.get("oot_date"),
            "has_test_split": config.get("has_test_split", False),
            "test_size":      config.get("test_size", 20),
            "max_bins":       config.get("max_bins", 4),
            "seg_col":        config.get("segment_col"),
            "seg_val":        config.get("segment_val"),
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
    Input("dd-data-tab", "active_tab"),
    State("store-dd-config", "data"),
    prevent_initial_call=False,
)
def render_deep_dive_content(col, psi_split, dtype_override, active_data_tab, dd_config):
    if not col or not dd_config:
        return html.Div()

    df_orig = _get_df(dd_config["key"])
    if df_orig is None:
        return html.Div()

    target      = dd_config["target_col"]
    date_col    = dd_config.get("date_col")
    oot_date    = dd_config.get("oot_date")
    seg_col     = dd_config.get("seg_col")
    seg_val     = dd_config.get("seg_val")
    df_active = apply_segment_filter(df_orig, seg_col, seg_val)

    # Periyot bölünmesi: get_splits ile train/test/OOT
    df_train, df_test, df_oot = get_splits(df_active, dd_config)

    vstats = get_variable_stats(df_active, col, target)

    _max_bins = int(dd_config.get("max_bins", 4))
    _force_dtype = dtype_override if dtype_override and dtype_override != "auto" else None
    _is_woe_tab  = (active_data_tab != "dd-tab-raw")

    _pfx_bins = f"{dd_config['key']}_ds_{seg_col}_{seg_val}"

    # WOE — cache'den oku, yoksa veya dtype override varsa hesapla
    _woe_tables = _SERVER_STORE.get(f"{_pfx_bins}_woe_tables", {})
    _cached_entry = _woe_tables.get(col)
    if _cached_entry and not _force_dtype:
        woe_df = pd.DataFrame(_cached_entry["train_table"])
        iv_total_dd = _cached_entry.get("iv_train", 0.0)
        # bin_edges cache'den
        _bins_dict = _SERVER_STORE.get(f"{_pfx_bins}_bins", {})
        woe_bin_edges = _bins_dict.get(col)
    else:
        woe_df, iv_total_dd, woe_bin_edges, _ = get_woe_detail(
            df_train, col, target, _max_bins, force_dtype=_force_dtype)

    # PSI — TEK KAYNAK: calc_psi (train vs OOT)
    # WoE tab → discrete=True (her unique WoE değeri bir bin)
    # Ham tab → discrete=False, n_bins=10, np.linspace
    _pfx = f"{dd_config['key']}_ds_{seg_col}_{seg_val}"
    cutoff_date = oot_date if oot_date else (psi_split if psi_split else None)
    psi_res = {"psi": None}
    if df_oot is not None and not df_oot.empty and col in df_train.columns:
        if _is_woe_tab:
            _tr_woe = _SERVER_STORE.get(f"{_pfx}_train_woe")
            _oot_woe = _SERVER_STORE.get(f"{_pfx}_oot_woe")
            if _tr_woe is not None and _oot_woe is not None and col in _tr_woe.columns and col in _oot_woe.columns:
                _tv = _tr_woe[col].dropna().values
                _ov = _oot_woe[col].dropna().values
                if len(_tv) >= 2 and len(_ov) >= 2:
                    _d = _calc_psi(_tv, _ov, discrete=True, detail=True)
                    psi_res = {"psi": _d["psi"], "label": _psi_label(_d["psi"]),
                               "detail_df": pd.DataFrame(_d["rows"]),
                               "split_label": "Train", "comp_label": "OOT",
                               "n_baseline": len(_tv), "n_compare": len(_ov)}
        else:
            _tv = df_train[col].dropna().values
            _ov = df_oot[col].dropna().values
            if len(_tv) >= 2 and len(_ov) >= 2:
                try:
                    _d = _calc_psi(_tv, _ov, n_bins=10, discrete=False, detail=True)
                    psi_res = {"psi": _d["psi"], "label": _psi_label(_d["psi"]),
                               "detail_df": pd.DataFrame(_d["rows"]),
                               "split_label": "Train", "comp_label": "OOT",
                               "n_baseline": len(_tv), "n_compare": len(_ov)}
                except Exception:
                    pass

    is_num = (vstats["is_numeric"] if _force_dtype is None
              else (_force_dtype == "numerical"))

    # ── 0. Anomali / Tanı Bölümü ──────────────────────────────────────────────
    _hints = []
    _hints += check_train_size(len(df_train))
    _hints += check_variable_stats(vstats)
    _hints += check_iv(iv_total_dd, woe_df.empty)
    _hints += check_psi(psi_res.get("psi"), date_col, cutoff_date)
    hint_section = build_hint_section(_hints)

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
        sc(f"{iv_total_dd:.4f}",
           "IV",
           "#10b981" if iv_total_dd >= 0.1 else "#f59e0b" if iv_total_dd >= 0.02 else "#ef4444"),
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
    if vstats["missing"] > 0 and vstats.get("missing_bad_rate") is not None:
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

    # ── 2. Dağılım Grafikleri ─────────────────────────────────────────────────
    _has_date = bool(date_col and date_col in df_active.columns)

    def _build_temporal_fig(df_t, col_t, target_t, date_col_t, is_num_t):
        """Zaman bazlı dağılım: x=tarih periyodu, bar=değişken ort./sayı, line=DR."""
        tmp = df_t[[col_t, target_t, date_col_t]].copy()
        tmp[date_col_t] = pd.to_datetime(tmp[date_col_t], errors="coerce")
        tmp = tmp.dropna(subset=[date_col_t])
        tmp["_period"] = tmp[date_col_t].dt.to_period("M").astype(str)

        if is_num_t:
            tmp_c = tmp.dropna(subset=[col_t])
            bar_agg = (tmp_c.groupby("_period")[col_t]
                       .mean().reset_index().rename(columns={col_t: "_bar"}))
            bar_label = f"Ort. {col_t}"
            bar_hover = "Ort. Değer: %{y:.4f}"
        else:
            bar_agg = (tmp.groupby("_period")[col_t]
                       .count().reset_index().rename(columns={col_t: "_bar"}))
            bar_label = "Gözlem"
            bar_hover = "Gözlem: %{y:,}"

        tmp_dr = tmp.dropna(subset=[target_t]).copy()
        tmp_dr[target_t] = pd.to_numeric(tmp_dr[target_t], errors="coerce")
        tmp_dr = tmp_dr.dropna(subset=[target_t])
        dr_agg = (tmp_dr.groupby("_period")[target_t]
                  .mean().reset_index().rename(columns={target_t: "_dr"}))

        merged = bar_agg.merge(dr_agg, on="_period", how="left").sort_values("_period")

        merged["_dr"] = (merged["_dr"] * 100).round(2)
        dr_label, dr_suffix = "Default Rate %", "%"

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=merged["_period"], y=merged["_bar"],
            name=bar_label,
            marker_color="#4F8EF7", opacity=0.7,
            hovertemplate=f"Dönem: %{{x}}<br>{bar_hover}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=merged["_period"], y=merged["_dr"],
            name=dr_label,
            mode="lines+markers",
            line=dict(color="#ef4444", width=2),
            marker=dict(size=5),
            yaxis="y2",
            hovertemplate=f"Dönem: %{{x}}<br>{dr_label}: %{{y:.2f}}{dr_suffix}<extra></extra>",
        ))
        fig.update_layout(
            **_PLOT_LAYOUT,
            title=dict(text=f"{col_t} — Zaman Dağılımı", font=dict(color="#E8EAF0", size=13)),
            xaxis=dict(**_AXIS_STYLE, tickangle=-30),
            yaxis=dict(**_AXIS_STYLE, title=bar_label),
            yaxis2=dict(
                title=dr_label, overlaying="y", side="right",
                showgrid=False, zeroline=False,
                tickfont=dict(color="#ef4444", size=9),
                ticksuffix=dr_suffix,
            ),
            legend=dict(bgcolor="#161C27", bordercolor="#232d3f"),
            height=320,
        )
        return fig

    if is_num:
        local = df_active[[col, target]].dropna(subset=[col, target]).copy()

        if _has_date:
            fig_dist = _build_temporal_fig(df_active, col, target, date_col, True)
        else:
            # Binary — decile bad rate fallback (tarih yok)
            local_br = local.dropna(subset=[col, target]).copy()
            local_br[target] = pd.to_numeric(local_br[target], errors="coerce")
            local_br = local_br.dropna(subset=[target])
            try:
                local_br["_decile"] = pd.qcut(local_br[col], q=10, labels=False,
                                               duplicates="drop")
            except Exception:
                local_br["_decile"] = pd.cut(local_br[col], bins=10, labels=False)
            decile_agg = (
                local_br.groupby("_decile", observed=True)
                .agg(bad_rate=(target, "mean"), count=(target, "count"),
                     min_val=(col, "min"), max_val=(col, "max"))
                .reset_index().sort_values("_decile")
            )
            decile_agg["bad_rate_pct"] = (decile_agg["bad_rate"] * 100).round(2)
            decile_agg["bin_label"] = decile_agg.apply(
                lambda r: f"[{r['min_val']:.2f}, {r['max_val']:.2f}]", axis=1)
            overall_br = local_br[target].mean() * 100
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Bar(
                x=decile_agg["bin_label"], y=decile_agg["count"],
                name="Gözlem", marker_color="#4F8EF7", opacity=0.6, yaxis="y2",
                hovertemplate="Aralık: %{x}<br>Gözlem: %{y:,}<extra></extra>",
            ))
            fig_dist.add_trace(go.Scatter(
                x=decile_agg["bin_label"], y=decile_agg["bad_rate_pct"],
                name="Bad Rate %", mode="lines+markers",
                line=dict(color="#ef4444", width=2), marker=dict(size=7), yaxis="y",
                hovertemplate="Aralık: %{x}<br>Bad Rate: %{y:.2f}%<extra></extra>",
            ))
            fig_dist.add_hline(y=overall_br, line_dash="dot", line_color="#f59e0b",
                               annotation_text=f"Ortalama %{overall_br:.2f}",
                               annotation_font_color="#f59e0b", annotation_font_size=9)
            fig_dist.update_layout(
                **_PLOT_LAYOUT,
                title=dict(text=f"{col} — Bad Rate by Desil", font=dict(color="#E8EAF0", size=13)),
                xaxis=dict(**_AXIS_STYLE, tickangle=-30),
                yaxis=dict(**_AXIS_STYLE, title="Bad Rate %", ticksuffix="%"),
                yaxis2=dict(title="Gözlem", overlaying="y", side="right",
                            showgrid=False, zeroline=False,
                            tickfont=dict(color="#8892a4")),
                legend=dict(bgcolor="#161C27", bordercolor="#232d3f"),
                height=320,
            )

        # Binary — stat table on right
        local_tbl = local.dropna(subset=[target]).copy()
        local_tbl[target] = local_tbl[target].astype(str).str.replace(r'\.0$', '', regex=True)
        grp_data = {str(int(float(tv))): g[col].dropna()
                    for tv, g in local_tbl.groupby(local_tbl[target])}
        stat_rows = []
        for stat_name, fn in [
            ("Gözlem",   lambda s: f"{len(s):,}"),
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
        # Kategorik
        if _has_date:
            fig_dist = _build_temporal_fig(df_active, col, target, date_col, False)
        else:
            local = df_active[[col, target]].copy()
            local[col] = local[col].fillna("Eksik").astype(str)
            top_cats = local[col].value_counts().head(20).index
            local = local[local[col].isin(top_cats)]
            fig_dist = go.Figure()
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
    if not woe_df.empty:
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
                _period_iv = iv_total
            else:
                # Test / OOT: cache'den oku, yoksa get_woe_detail ile hesapla
                _wt = _SERVER_STORE.get(f"{_pfx_bins}_woe_tables", {}).get(col, {})
                _period_key = "test_table" if period_label == "Test" else "oot_table"
                _iv_key = "iv_test" if period_label == "Test" else "iv_oot"
                _cached_period = _wt.get(_period_key)
                if _cached_period and not _force_dtype:
                    p_df = pd.DataFrame(_cached_period)
                    _period_iv = _wt.get(_iv_key, 0.0)
                else:
                    _optb_dict = _SERVER_STORE.get(f"{_pfx_bins}_optb", {})
                    _optb = _optb_dict.get(col)
                    if _optb is not None:
                        _sp_woe = _wt.get("train_special_woe", {})
                        p_df, _period_iv, _, _ = get_woe_detail(
                            period_df, col, target, fitted_optb=_optb,
                            use_edges=True, train_special_woe=_sp_woe)
                    else:
                        p_df = pd.DataFrame()
                        _period_iv = 0.0
                if isinstance(p_df, pd.DataFrame) and p_df.empty:
                    return None
                chart_df = p_df[p_df["Bin"] != "TOPLAM"].copy()
                table_df = p_df.copy()

            n_period   = len(period_df)
            bad_n      = int(period_df[target].sum())
            br_overall = round(bad_n / n_period * 100, 2) if n_period > 0 else 0.0

            # IV label ve renk
            _p_iv_label = ("Çok Zayıf" if _period_iv < 0.02 else "Zayıf" if _period_iv < 0.1
                           else "Orta" if _period_iv < 0.3 else "Güçlü" if _period_iv < 0.5
                           else "Şüpheli")
            _p_iv_color = {"Çok Zayıf": "#4a5568", "Zayıf": "#f59e0b", "Orta": "#4F8EF7",
                           "Güçlü": "#10b981", "Şüpheli": "#ef4444"}.get(_p_iv_label, "#4F8EF7")

            title_text = (f"{period_label}  ·  n={n_period:,}  ·  Bad Rate: {br_overall:.2f}%"
                          f"  |  IV: {_period_iv:.4f}  [{_p_iv_label}]")

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
                           font=dict(color=_p_iv_color, size=12)),
                xaxis=dict(**_AXIS_STYLE, tickangle=-30),
                yaxis=dict(**_AXIS_STYLE, title="Bad Rate %", ticksuffix="%"),
                yaxis2=dict(title="Adet", overlaying="y", side="right", showgrid=False,
                            zeroline=False),
                legend=dict(bgcolor="#161C27", bordercolor="#232d3f", font=dict(size=10)),
                height=300,
            )

            # Monotonluk etiketi — Eksik/Special satırları hariç
            _mono_mask = ~chart_df["Bin"].isin(["Eksik", "TOPLAM"]) & \
                         ~chart_df["Bin"].str.startswith("Special", na=False)
            br_arr   = chart_df.loc[_mono_mask, "Bad Rate %"].values
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

    sections = [
        hint_section,
    ]
    sections.extend([
        html.P("Özet İstatistikler", className="section-title"),
        stats_row,
        missing_target_card,
        html.P("Dağılım Analizi", className="section-title"),
        dist_section,
    ])
    # WoE & Bad Rate — sadece WoE tab'da göster
    if _is_woe_tab:
        sections.append(woe_section)
    sections.append(psi_section)

    return html.Div(sections)


