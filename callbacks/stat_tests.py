from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

from app_instance import app
from server_state import _SERVER_STORE, get_df as _get_df
from utils.helpers import apply_segment_filter
from utils.chart_helpers import _PLOT_LAYOUT, _TABLE_STYLE
from modules.correlation import get_numeric_cols, compute_vif


# ── Callback: Test Paneli Göster/Gizle ───────────────────────────────────────
@app.callback(
    Output("stat-corr-panel",  "style"),
    Output("stat-chi-panel",   "style"),
    Output("stat-anova-panel", "style"),
    Output("stat-ks-panel",    "style"),
    Output("stat-vif-panel",   "style"),
    Input("stat-test-type", "value"),
)
def toggle_stat_panels(test_type):
    _show = {}
    _hide = {"display": "none"}
    return (
        _show if test_type == "correlation" else _hide,
        _show if test_type == "chi_square"  else _hide,
        _show if test_type == "anova"       else _hide,
        _show if test_type == "ks"          else _hide,
        _show if test_type == "vif_sandbox" else _hide,
    )


# ── Callback: Test Dropdown'larını Doldur ────────────────────────────────────
@app.callback(
    Output("chi-var1",  "options"), Output("chi-var1",  "value"),
    Output("chi-var2",  "options"), Output("chi-var2",  "value"),
    Output("anova-var", "options"), Output("anova-var", "value"),
    Output("ks-var",    "options"), Output("ks-var",    "value"),
    Input("store-config", "data"),
    State("store-key", "data"),
)
def populate_stat_dropdowns(config, key):
    empty = ([], None)
    if not config or not key:
        return empty + empty + empty + empty
    df_orig = _get_df(key)
    if df_orig is None:
        return empty + empty + empty + empty
    target   = config.get("target_col", "")
    date_col = config.get("date_col", "")
    excl     = {c for c in [target, date_col] if c}
    all_cols = [c for c in df_orig.columns if c not in excl]
    num_cols = [c for c in df_orig.select_dtypes(include=[np.number]).columns if c not in excl]
    all_opts = [{"label": c, "value": c} for c in all_cols]
    num_opts = [{"label": c, "value": c} for c in num_cols]
    chi_v1 = all_cols[0] if all_cols else None
    chi_v2 = all_cols[1] if len(all_cols) > 1 else chi_v1
    num_default = num_cols[0] if num_cols else None
    return (
        all_opts, chi_v1,
        all_opts, chi_v2,
        num_opts, num_default,
        num_opts, num_default,
    )


# ── Render: Chi-Square ───────────────────────────────────────────────────────
def _render_chi_square(df_active: pd.DataFrame, var1: str, var2: str, max_cats: int) -> html.Div:
    def _cap_categories(series: pd.Series, n: int) -> pd.Series:
        top = series.value_counts().nlargest(n).index
        return series.where(series.isin(top), other="Diğer")

    s1 = df_active[var1].fillna("(boş)").astype(str)
    s2 = df_active[var2].fillna("(boş)").astype(str)
    if s1.nunique() > max_cats:
        s1 = _cap_categories(df_active[var1].fillna("(boş)").astype(str), max_cats)
    if s2.nunique() > max_cats:
        s2 = _cap_categories(df_active[var2].fillna("(boş)").astype(str), max_cats)

    ctab = pd.crosstab(s1, s2)
    chi2, p, dof, _ = scipy_stats.chi2_contingency(ctab)
    n_total = ctab.values.sum()
    cramers_v = float(np.sqrt(chi2 / (n_total * (min(ctab.shape) - 1)))) if min(ctab.shape) > 1 else 0.0

    # p-value yorumu
    if p < 0.001:
        p_interp = "p < 0.001 — Çok güçlü bağımlılık kanıtı"
        p_color  = "#10b981"
    elif p < 0.05:
        p_interp = f"p = {p:.4f} — Anlamlı bağımlılık"
        p_color  = "#f59e0b"
    else:
        p_interp = f"p = {p:.4f} — Bağımsızlık reddedilemedi"
        p_color  = "#ef4444"

    # Cramér's V yorumu
    if cramers_v >= 0.5:
        v_label = "Güçlü"
    elif cramers_v >= 0.3:
        v_label = "Orta"
    elif cramers_v >= 0.1:
        v_label = "Zayıf"
    else:
        v_label = "Önemsiz"

    stat_cards = dbc.Row([
        dbc.Col(html.Div([
            html.Div("χ² İstatistiği", className="metric-label"),
            html.Div(f"{chi2:,.2f}", className="metric-value"),
        ], className="metric-card"), width=3),
        dbc.Col(html.Div([
            html.Div("p-değeri", className="metric-label"),
            html.Div(f"{p:.6f}", className="metric-value", style={"color": p_color}),
        ], className="metric-card"), width=3),
        dbc.Col(html.Div([
            html.Div("Serbestlik Derecesi", className="metric-label"),
            html.Div(str(dof), className="metric-value"),
        ], className="metric-card"), width=2),
        dbc.Col(html.Div([
            html.Div("Cramér's V", className="metric-label"),
            html.Div(f"{cramers_v:.4f}  ({v_label})", className="metric-value"),
        ], className="metric-card"), width=4),
    ], className="mb-3")

    # Contingency heatmap (normalize by row)
    ctab_norm = ctab.div(ctab.sum(axis=1), axis=0)
    fig_heat = go.Figure(go.Heatmap(
        z=ctab_norm.values,
        x=[str(c) for c in ctab_norm.columns],
        y=[str(r) for r in ctab_norm.index],
        colorscale="Blues",
        zmin=0, zmax=1,
        text=ctab.values,
        customdata=ctab_norm.values,
        hovertemplate="<b>%{y}</b> → <b>%{x}</b><br>Sayı: %{text:,}<br>Satır %: %{customdata:.1%}<extra></extra>",
        colorbar=dict(
            title=dict(text="Satır%", font=dict(color="#8892a4", size=10)),
            thickness=12, tickformat=".0%",
            tickfont=dict(color="#8892a4", size=10),
        ),
    ))
    fig_heat.update_layout(
        **_PLOT_LAYOUT,
        title=dict(text=f"Contingency Heatmap — {var1}  ×  {var2}",
                   font=dict(color="#E8EAF0", size=13)),
        xaxis=dict(title=var2, tickfont=dict(size=9, color="#8892a4"),
                   tickangle=-30, showgrid=False),
        yaxis=dict(title=var1, tickfont=dict(size=9, color="#8892a4"),
                   showgrid=False, autorange="reversed"),
        height=max(300, len(ctab) * 30 + 100),
    )
    fig_heat.update_layout(margin=dict(l=120, r=60, t=50, b=100))

    # Row totals table (top 20 rows × top 10 cols)
    ctab_show = ctab.iloc[:20, :10].copy()
    ctab_show["TOPLAM"] = ctab_show.sum(axis=1)
    tbl_data = ctab_show.reset_index()
    tbl_data.columns = [str(c) for c in tbl_data.columns]
    tbl_cols = [{"name": c, "id": c} for c in tbl_data.columns]

    return html.Div([
        html.P("Chi-Square (Ki-Kare) Bağımsızlık Testi", className="section-title"),
        html.Div(p_interp, style={"color": p_color, "fontSize": "0.82rem",
                                  "marginBottom": "1rem", "fontWeight": "600"}),
        stat_cards,
        dcc.Graph(figure=fig_heat, config={"displayModeBar": False}),
        html.P("Contingency Tablosu (ilk 20×10)", className="section-title",
               style={"marginTop": "1.5rem"}),
        dash_table.DataTable(
            data=tbl_data.to_dict("records"),
            columns=tbl_cols,
            sort_action="native",
            page_size=20,
            **_TABLE_STYLE,
        ),
        html.Div([
            html.Span("Not: ", style={"color": "#7e8fa4", "fontSize": "0.72rem"}),
            html.Span(f"Toplam {n_total:,} gözlem · {ctab.shape[0]} × {ctab.shape[1]} kontenjans tablosu",
                      style={"color": "#a8b2c2", "fontSize": "0.72rem"}),
        ], style={"marginTop": "0.5rem"}),
    ])


# ── Render: ANOVA ─────────────────────────────────────────────────────────────
def _render_anova(df_active: pd.DataFrame, var_col: str, target: str) -> html.Div:
    col_data = df_active[[var_col, target]].dropna()
    groups = col_data.groupby(target)[var_col]
    group_list = [grp.values for _, grp in groups]

    if len(group_list) < 2:
        return html.Div("En az 2 grup gerekli.", className="alert-info-custom")

    # Büyük veri: örnekle (her gruptan maks 200k satır)
    MAX_PER_GROUP = 200_000
    sampled = [g if len(g) <= MAX_PER_GROUP else np.random.default_rng(42).choice(g, MAX_PER_GROUP, replace=False)
               for g in group_list]
    f_stat, p_val = scipy_stats.f_oneway(*sampled)

    # Grup istatistikleri (tüm veri üzerinden)
    grp_stats = col_data.groupby(target)[var_col].agg(
        N="count", Ortalama="mean", Std="std", Min="min", Medyan="median", Maks="max"
    ).reset_index()
    grp_stats.columns = [str(c) for c in grp_stats.columns]
    for col in ["Ortalama", "Std", "Min", "Medyan", "Maks"]:
        grp_stats[col] = grp_stats[col].round(4)

    # p yorumu
    if p_val < 0.001:
        p_interp = "p < 0.001 — Gruplar arası fark istatistiksel olarak çok anlamlı"
        p_color  = "#10b981"
    elif p_val < 0.05:
        p_interp = f"p = {p_val:.4f} — Anlamlı fark"
        p_color  = "#f59e0b"
    else:
        p_interp = f"p = {p_val:.4f} — Anlamlı fark bulunamadı"
        p_color  = "#ef4444"

    stat_cards = dbc.Row([
        dbc.Col(html.Div([
            html.Div("F İstatistiği", className="metric-label"),
            html.Div(f"{f_stat:,.4f}", className="metric-value"),
        ], className="metric-card"), width=3),
        dbc.Col(html.Div([
            html.Div("p-değeri", className="metric-label"),
            html.Div(f"{p_val:.6f}", className="metric-value", style={"color": p_color}),
        ], className="metric-card"), width=3),
        dbc.Col(html.Div([
            html.Div("Grup Sayısı", className="metric-label"),
            html.Div(str(len(group_list)), className="metric-value"),
        ], className="metric-card"), width=2),
        dbc.Col(html.Div([
            html.Div("Toplam N", className="metric-label"),
            html.Div(f"{len(col_data):,}", className="metric-value"),
        ], className="metric-card"), width=4),
    ], className="mb-3")

    # Box plot — örnekle (görselleştirme için 50k yeterli)
    VIZ_MAX = 50_000
    plot_df = col_data if len(col_data) <= VIZ_MAX else col_data.sample(VIZ_MAX, random_state=42)
    fig_box = go.Figure()
    target_vals = sorted(plot_df[target].unique())
    colors = {0: "#4F8EF7", 1: "#ef4444"}
    labels = {0: "Good (0)", 1: "Bad (1)"}
    for tv in target_vals:
        subset = plot_df[plot_df[target] == tv][var_col]
        fig_box.add_trace(go.Box(
            y=subset, name=labels.get(tv, str(tv)),
            marker_color=colors.get(tv, "#8892a4"),
            boxmean=True,
            hovertemplate=f"<b>{labels.get(tv, str(tv))}</b><br>%{{y}}<extra></extra>",
        ))
    fig_box.update_layout(
        **_PLOT_LAYOUT,
        title=dict(text=f"ANOVA — {var_col}  ×  {target}",
                   font=dict(color="#E8EAF0", size=13)),
        yaxis=dict(title=var_col, gridcolor="#1e293b", tickfont=dict(color="#8892a4")),
        xaxis=dict(tickfont=dict(color="#8892a4")),
        height=420,
        showlegend=True,
    )
    fig_box.update_layout(margin=dict(l=60, r=40, t=50, b=60))

    return html.Div([
        html.P("ANOVA Testi (Target vs Sayısal Değişken)", className="section-title"),
        html.Div(p_interp, style={"color": p_color, "fontSize": "0.82rem",
                                  "marginBottom": "1rem", "fontWeight": "600"}),
        stat_cards,
        dcc.Graph(figure=fig_box, config={"displayModeBar": False}),
        html.P("Grup İstatistikleri", className="section-title", style={"marginTop": "1.5rem"}),
        dash_table.DataTable(
            data=grp_stats.to_dict("records"),
            columns=[{"name": c, "id": c} for c in grp_stats.columns],
            sort_action="native",
            **_TABLE_STYLE,
        ),
        html.Div(
            "Not: F-testi büyük veri için her gruptan en fazla 200.000 örnekle hesaplanmıştır. Grup istatistikleri tüm veri üzerinden alınmıştır.",
            style={"color": "#7e8fa4", "fontSize": "0.72rem", "marginTop": "0.75rem"},
        ),
    ])


# ── Render: Pearson & Spearman Korelasyon (continuous target) ────────────────
def _render_pearson_spearman(df_active: pd.DataFrame, var_col: str, target: str) -> html.Div:
    """Continuous target için sayısal değişken ~ target korelasyon testi."""
    data = df_active[[var_col, target]].copy()
    data[var_col] = pd.to_numeric(data[var_col], errors="coerce")
    data[target]  = pd.to_numeric(data[target],  errors="coerce")
    data = data.dropna()

    if len(data) < 5:
        return html.Div("Yeterli veri yok (en az 5 gözlem gerekli).",
                        className="alert-info-custom")

    pearson_r,  pearson_p  = scipy_stats.pearsonr(data[var_col], data[target])
    spearman_r, spearman_p = scipy_stats.spearmanr(data[var_col], data[target])

    def _p_color(p): return "#10b981" if p < 0.001 else "#f59e0b" if p < 0.05 else "#ef4444"
    def _r_label(r):
        a = abs(r)
        return "Güçlü" if a >= 0.5 else "Orta" if a >= 0.3 else "Zayıf" if a >= 0.1 else "Önemsiz"

    stat_cards = dbc.Row([
        dbc.Col(html.Div([
            html.Div("Pearson r", className="metric-label"),
            html.Div(f"{pearson_r:+.4f}  ({_r_label(pearson_r)})",
                     className="metric-value",
                     style={"color": "#10b981" if abs(pearson_r) >= 0.3 else "#f59e0b"}),
        ], className="metric-card"), width=3),
        dbc.Col(html.Div([
            html.Div("Pearson p", className="metric-label"),
            html.Div(f"{pearson_p:.6f}", className="metric-value",
                     style={"color": _p_color(pearson_p)}),
        ], className="metric-card"), width=3),
        dbc.Col(html.Div([
            html.Div("Spearman ρ", className="metric-label"),
            html.Div(f"{spearman_r:+.4f}  ({_r_label(spearman_r)})",
                     className="metric-value",
                     style={"color": "#10b981" if abs(spearman_r) >= 0.3 else "#f59e0b"}),
        ], className="metric-card"), width=3),
        dbc.Col(html.Div([
            html.Div("Spearman p", className="metric-label"),
            html.Div(f"{spearman_p:.6f}", className="metric-value",
                     style={"color": _p_color(spearman_p)}),
        ], className="metric-card"), width=3),
    ], className="mb-3")

    # Scatter + regresyon doğrusu (max 3000 örnek)
    sample = data.sample(min(3000, len(data)), random_state=42)
    coeffs  = np.polyfit(data[var_col], data[target], 1)
    x_range = np.linspace(float(data[var_col].min()), float(data[var_col].max()), 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sample[var_col], y=sample[target],
        mode="markers",
        marker=dict(color="#4F8EF7", opacity=0.35, size=4),
        name="Gözlemler",
        hovertemplate=f"{var_col}: %{{x:.3f}}<br>{target}: %{{y:.3f}}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x_range, y=np.polyval(coeffs, x_range),
        mode="lines", line=dict(color="#ef4444", width=2, dash="dash"),
        name=f"Regresyon Doğrusu (y={coeffs[0]:+.3f}x {'+' if coeffs[1]>=0 else ''}{coeffs[1]:.3f})",
    ))
    fig.update_layout(
        **_PLOT_LAYOUT,
        title=dict(text=f"{var_col}  ×  {target} — Scatter & Korelasyon",
                   font=dict(color="#E8EAF0", size=13)),
        xaxis=dict(title=var_col, gridcolor="#1e293b", tickfont=dict(color="#8892a4")),
        yaxis=dict(title=target,  gridcolor="#1e293b", tickfont=dict(color="#8892a4")),
        height=400, showlegend=True,
        legend=dict(font=dict(color="#c8cdd8", size=10), bgcolor="rgba(0,0,0,0)"),
    )

    p_interp_txt = (
        "p < 0.001 — İlişki istatistiksel olarak çok anlamlı" if pearson_p < 0.001 else
        f"p = {pearson_p:.4f} — Anlamlı ilişki" if pearson_p < 0.05 else
        f"p = {pearson_p:.4f} — Anlamlı ilişki bulunamadı"
    )
    return html.Div([
        html.P(f"Korelasyon Analizi — {var_col}  ×  {target}  (Continuous Target)",
               className="section-title"),
        html.Div(p_interp_txt, style={"color": _p_color(pearson_p), "fontSize": "0.82rem",
                                      "marginBottom": "1rem", "fontWeight": "600"}),
        stat_cards,
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        html.Div([
            html.Span("Not: ", style={"color": "#7e8fa4", "fontSize": "0.72rem"}),
            html.Span(
                f"Scatter için en fazla 3.000 örnek gösterilir. "
                f"Pearson doğrusal ilişkiyi, Spearman monoton ilişkiyi ölçer. "
                f"N = {len(data):,}",
                style={"color": "#a8b2c2", "fontSize": "0.72rem"},
            ),
        ], style={"marginTop": "0.5rem"}),
    ])


# ── Render: Kruskal-Wallis (multiclass / 3+ grup) ────────────────────────────
def _render_kruskal(df_active: pd.DataFrame, var_col: str, target: str) -> html.Div:
    """Non-parametrik ANOVA: 3+ sınıf için grup dağılımı karşılaştırması."""
    data = df_active[[var_col, target]].dropna()
    data[var_col] = pd.to_numeric(data[var_col], errors="coerce")
    data = data.dropna()
    groups = [grp[var_col].values for _, grp in data.groupby(target)
              if len(grp) >= 2]
    if len(groups) < 2:
        return html.Div("Yeterli grup yok (en az 2 sınıf gerekli).",
                        className="alert-info-custom")
    stat, p_val = scipy_stats.kruskal(*groups)
    p_color = "#10b981" if p_val < 0.001 else "#f59e0b" if p_val < 0.05 else "#ef4444"
    p_interp = (
        "p < 0.001 — Gruplar arası medyan farklılığı istatistiksel olarak anlamlı"
        if p_val < 0.001 else
        f"p = {p_val:.4f} — Anlamlı farklılık" if p_val < 0.05
        else f"p = {p_val:.4f} — Gruplar arası fark gözlenemedi (H₀ reddedilemedi)"
    )
    grp_stats = (data.groupby(target)[var_col]
                 .agg(N="count", Medyan="median", Ort="mean", Std="std")
                 .reset_index().round(4))
    fig = go.Figure()
    for cls, grp in data.groupby(target):
        fig.add_trace(go.Box(
            y=grp[var_col], name=str(int(float(cls))),
            marker_color="#4F8EF7", boxmean="sd", opacity=0.8,
        ))
    fig.update_layout(
        **_PLOT_LAYOUT,
        title=dict(text=f"Kruskal-Wallis — {var_col} × {target}",
                   font=dict(color="#E8EAF0", size=13)),
        yaxis=dict(**{k: v for k, v in {
            "gridcolor": "#1e293b", "tickfont": dict(color="#8892a4"),
            "title": var_col,
        }.items()}),
        xaxis=dict(tickfont=dict(color="#8892a4"), title=target),
        height=380,
    )
    return html.Div([
        html.P("Kruskal-Wallis Testi (Multiclass)", className="section-title"),
        html.Div(p_interp, style={"color": p_color, "fontSize": "0.82rem",
                                  "marginBottom": "1rem", "fontWeight": "600"}),
        dbc.Row([
            dbc.Col(html.Div([html.Div("H İstatistiği", className="metric-label"),
                              html.Div(f"{stat:.4f}", className="metric-value")],
                             className="metric-card"), width=3),
            dbc.Col(html.Div([html.Div("p-değeri", className="metric-label"),
                              html.Div(f"{p_val:.6f}", className="metric-value",
                                       style={"color": p_color})],
                             className="metric-card"), width=3),
            dbc.Col(html.Div([html.Div("Grup Sayısı", className="metric-label"),
                              html.Div(str(len(groups)), className="metric-value")],
                             className="metric-card"), width=3),
        ], className="mb-3"),
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        html.P("Grup Medyan İstatistikleri", className="section-title",
               style={"marginTop": "1.5rem"}),
        dash_table.DataTable(
            data=grp_stats.to_dict("records"),
            columns=[{"name": c, "id": c} for c in grp_stats.columns],
            sort_action="native", **_TABLE_STYLE,
        ),
    ])


# ── Render: KS Testi ─────────────────────────────────────────────────────────
def _render_ks(df_active: pd.DataFrame, var_col: str, target: str,
               target_type: str = "binary") -> html.Div:
    col_data = df_active[[var_col, target]].dropna()
    col_data = col_data[pd.api.types.is_numeric_dtype(col_data[var_col]) |
                        col_data[var_col].apply(lambda x: isinstance(x, (int, float)))]
    try:
        col_data[var_col] = pd.to_numeric(col_data[var_col], errors="coerce")
    except Exception:
        pass
    col_data = col_data.dropna()

    # Grubu target tipine göre belirle
    if target_type == "binary":
        good = col_data[col_data[target] == 0][var_col].values
        bad  = col_data[col_data[target] == 1][var_col].values
        grp_labels = ("Good (0)", "Bad (1)")
    elif target_type == "multiclass":
        # En büyük sınıf vs geri kalanlar
        dominant = col_data[target].mode()[0]
        good = col_data[col_data[target] == dominant][var_col].values
        bad  = col_data[col_data[target] != dominant][var_col].values
        grp_labels = (f"Dominant ({int(dominant)})", "Diğer Sınıflar")
    else:
        # Continuous/categorical: medyan split
        med  = col_data[target].median()
        good = col_data[col_data[target] <= med][var_col].values
        bad  = col_data[col_data[target] >  med][var_col].values
        grp_labels = (f"Düşük (≤ {med:.2f})", f"Yüksek (> {med:.2f})")

    if len(good) == 0 or len(bad) == 0:
        return html.Div("Yeterli veri yok — her iki grupta da gözlem gerekli.",
                        className="alert-info-custom")

    # KS stat (büyük veri: tüm veri, scipy optimize edilmiş)
    ks_stat, p_val = scipy_stats.ks_2samp(good, bad)

    # p yorumu
    if p_val < 0.001:
        p_interp = "p < 0.001 — Dağılımlar istatistiksel olarak farklı"
        p_color  = "#10b981"
    elif p_val < 0.05:
        p_interp = f"p = {p_val:.4f} — Anlamlı farklılık"
        p_color  = "#f59e0b"
    else:
        p_interp = f"p = {p_val:.4f} — Dağılımlar benzer (H₀ reddedilemedi)"
        p_color  = "#ef4444"

    stat_cards = dbc.Row([
        dbc.Col(html.Div([
            html.Div("KS İstatistiği", className="metric-label"),
            html.Div(f"{ks_stat:.6f}", className="metric-value"),
        ], className="metric-card"), width=3),
        dbc.Col(html.Div([
            html.Div("p-değeri", className="metric-label"),
            html.Div(f"{p_val:.6f}", className="metric-value", style={"color": p_color}),
        ], className="metric-card"), width=3),
        dbc.Col(html.Div([
            html.Div(f"{grp_labels[0]} N", className="metric-label"),
            html.Div(f"{len(good):,}", className="metric-value"),
        ], className="metric-card"), width=3),
        dbc.Col(html.Div([
            html.Div(f"{grp_labels[1]} N", className="metric-label"),
            html.Div(f"{len(bad):,}", className="metric-value"),
        ], className="metric-card"), width=3),
    ], className="mb-3")

    # Ampirik CDF — görselleştirme için örnekle
    VIZ_MAX = 20_000
    g_plot = np.sort(good[:VIZ_MAX] if len(good) > VIZ_MAX else good)
    b_plot = np.sort(bad[:VIZ_MAX] if len(bad) > VIZ_MAX else bad)
    g_cdf  = np.arange(1, len(g_plot) + 1) / len(g_plot)
    b_cdf  = np.arange(1, len(b_plot) + 1) / len(b_plot)

    # KS noktasını bul (en büyük fark)
    all_x   = np.union1d(g_plot, b_plot)
    g_ecdf_fn = scipy_stats.ecdf(g_plot).cdf.evaluate
    b_ecdf_fn = scipy_stats.ecdf(b_plot).cdf.evaluate
    g_all = g_ecdf_fn(all_x)
    b_all = b_ecdf_fn(all_x)
    diff  = np.abs(g_all - b_all)
    ks_idx = int(np.argmax(diff))
    ks_x   = float(all_x[ks_idx])
    ks_y1  = float(g_all[ks_idx])
    ks_y2  = float(b_all[ks_idx])

    fig_cdf = go.Figure()
    fig_cdf.add_trace(go.Scatter(
        x=g_plot, y=g_cdf, mode="lines",
        name=grp_labels[0], line=dict(color="#4F8EF7", width=2),
    ))
    fig_cdf.add_trace(go.Scatter(
        x=b_plot, y=b_cdf, mode="lines",
        name=grp_labels[1], line=dict(color="#ef4444", width=2),
    ))
    # KS mesafesi işareti
    fig_cdf.add_shape(
        type="line",
        x0=ks_x, x1=ks_x, y0=min(ks_y1, ks_y2), y1=max(ks_y1, ks_y2),
        line=dict(color="#f59e0b", width=2, dash="dot"),
    )
    fig_cdf.add_annotation(
        x=ks_x, y=(ks_y1 + ks_y2) / 2,
        text=f"KS = {ks_stat:.4f}",
        showarrow=True, arrowhead=2,
        font=dict(color="#f59e0b", size=11),
        bgcolor="#1a2035", bordercolor="#f59e0b",
        ax=40, ay=0,
    )
    fig_cdf.update_layout(
        **_PLOT_LAYOUT,
        title=dict(text=f"Ampirik CDF — {var_col}",
                   font=dict(color="#E8EAF0", size=13)),
        xaxis=dict(title=var_col, gridcolor="#1e293b", tickfont=dict(color="#8892a4")),
        yaxis=dict(title="CDF", gridcolor="#1e293b", tickfont=dict(color="#8892a4"),
                   tickformat=".1%"),
        height=420,
        legend=dict(font=dict(color="#c8cdd8"), bgcolor="rgba(0,0,0,0)"),
    )
    fig_cdf.update_layout(margin=dict(l=60, r=40, t=50, b=60))

    return html.Div([
        html.P("Kolmogorov-Smirnov (KS) Ayırıcılık Testi", className="section-title"),
        html.Div(p_interp, style={"color": p_color, "fontSize": "0.82rem",
                                  "marginBottom": "1rem", "fontWeight": "600"}),
        stat_cards,
        dcc.Graph(figure=fig_cdf, config={"displayModeBar": False}),
        html.Div(
            "Not: CDF grafiği görselleştirme için en fazla 20.000 örnek kullanır; KS istatistiği tüm veri üzerinden hesaplanmıştır.",
            style={"color": "#7e8fa4", "fontSize": "0.72rem", "marginTop": "0.5rem"},
        ),
    ])


# ── Render: VIF Sandbox ───────────────────────────────────────────────────────
def _render_vif_sandbox(df_active: pd.DataFrame, var_set: str, max_cols: int,
                        config: dict, expert_excluded: list, key: str,
                        seg_col: str, seg_val: str) -> html.Div:
    target   = config.get("target_col", "")
    date_col = config.get("date_col", "")
    excl     = {c for c in [target, date_col] if c}
    iv_cache_key = f"{key}_iv_{seg_col}_{seg_val}"

    all_num = get_numeric_cols(df_active, exclude=list(excl), max_cols=max_cols)
    screen_result = _SERVER_STORE.get(f"{key}_screen")
    if screen_result:
        passed_set = set(screen_result[0])
        all_num = [c for c in all_num if c in passed_set]
    expert_set = set(expert_excluded or [])
    all_num = [c for c in all_num if c not in expert_set]

    if var_set == "iv_filtered" and iv_cache_key in _SERVER_STORE:
        iv_df_c = _SERVER_STORE[iv_cache_key]
        iv_high = set(iv_df_c[iv_df_c["IV"] >= 0.10]["Değişken"].tolist())
        filtered = [c for c in all_num if c in iv_high]
        cols     = filtered if len(filtered) >= 2 else all_num
        iv_note  = f"IV ≥ 0.10 filtresi uygulandı ({len(cols)} değişken)"
    else:
        cols    = all_num
        iv_note = f"Tüm numerik değişkenler ({len(cols)} adet)"

    if len(cols) < 2:
        return html.Div("VIF için en az 2 değişken gerekli. Target & IV sekmesini önce açınız.",
                        className="alert-info-custom")

    vif_df = compute_vif(df_active, cols)
    if vif_df is None or vif_df.empty:
        return html.Div("VIF hesaplanamadı.", className="alert-info-custom")

    # "En Benzer" kolonu ekle
    try:
        corr_sub = df_active[cols].corr()
        en_benzer = []
        for var in vif_df["Değişken"]:
            if var not in corr_sub.columns:
                en_benzer.append("—")
                continue
            row = corr_sub[var].drop(var, errors="ignore").abs()
            top = row.idxmax()
            en_benzer.append(f"{top}  (r = {corr_sub[var][top]:+.3f})")
        vif_df = vif_df.copy()
        vif_df.insert(2, "En Benzer", en_benzer)
    except Exception:
        pass

    vif_cond = [
        {"if": {"filter_query": '{Uyarı} = "✗ Yüksek"', "column_id": "Uyarı"},
         "color": "#ef4444", "fontWeight": "700"},
        {"if": {"filter_query": '{Uyarı} = "⚠ Orta"',   "column_id": "Uyarı"},
         "color": "#f59e0b", "fontWeight": "600"},
        {"if": {"filter_query": '{Uyarı} = "✓ Normal"',  "column_id": "Uyarı"},
         "color": "#10b981"},
        {"if": {"filter_query": "{VIF} >= 10", "column_id": "VIF"},
         "color": "#ef4444", "fontWeight": "700"},
        {"if": {"filter_query": "{VIF} >= 5 && {VIF} < 10", "column_id": "VIF"},
         "color": "#f59e0b"},
        {"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"},
    ]

    n_high = int((vif_df["VIF"] >= 10).sum()) if "VIF" in vif_df.columns else 0
    n_mid  = int(((vif_df["VIF"] >= 5) & (vif_df["VIF"] < 10)).sum()) if "VIF" in vif_df.columns else 0

    summary_cards = dbc.Row([
        dbc.Col(html.Div([
            html.Div("Değişken Sayısı", className="metric-label"),
            html.Div(str(len(vif_df)), className="metric-value"),
        ], className="metric-card"), width=3),
        dbc.Col(html.Div([
            html.Div("VIF ≥ 10 (Yüksek)", className="metric-label"),
            html.Div(str(n_high), className="metric-value",
                     style={"color": "#ef4444" if n_high > 0 else "#c8cdd8"}),
        ], className="metric-card"), width=3),
        dbc.Col(html.Div([
            html.Div("VIF 5–10 (Orta)", className="metric-label"),
            html.Div(str(n_mid), className="metric-value",
                     style={"color": "#f59e0b" if n_mid > 0 else "#c8cdd8"}),
        ], className="metric-card"), width=3),
    ], className="mb-3")

    vif_tsv = vif_df.to_csv(sep="\t", index=False)
    return html.Div([
        html.P("VIF Kum Havuzu (Çoklu Doğrusallık)", className="section-title"),
        html.Div(iv_note, className="form-hint", style={"marginBottom": "1rem"}),
        summary_cards,
        html.Div([
            dcc.Clipboard(target_id="vif-sandbox-tsv", title="Kopyala",
                          style={"cursor": "pointer", "fontSize": "0.72rem",
                                 "color": "#a8b2c2", "padding": "0.2rem 0.55rem",
                                 "border": "1px solid #2d3a4f", "borderRadius": "4px",
                                 "backgroundColor": "#1a2035", "float": "right",
                                 "marginBottom": "0.4rem"}),
        ], style={"overflow": "hidden"}),
        html.Pre(vif_tsv, id="vif-sandbox-tsv", style={"display": "none"}),
        dash_table.DataTable(
            data=vif_df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in vif_df.columns],
            sort_action="native", filter_action="native",
            page_size=25,
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#111827", "color": "#a8b2c2",
                          "fontWeight": "700", "fontSize": "0.72rem",
                          "border": "1px solid #2d3a4f", "textTransform": "uppercase"},
            style_data={"backgroundColor": "#161C27", "color": "#c8cdd8",
                        "fontSize": "0.82rem", "border": "1px solid #232d3f"},
            style_data_conditional=vif_cond,
            style_cell={"padding": "0.4rem 0.65rem"},
            style_cell_conditional=[
                {"if": {"column_id": "VIF"}, "textAlign": "right"},
                {"if": {"column_id": "En Benzer"}, "color": "#a8b2c2", "fontSize": "0.78rem"},
            ],
            style_filter={"backgroundColor": "#0e1117", "color": "#c8cdd8",
                          "border": "1px solid #2d3a4f"},
        ),
        html.Div([
            html.Span("VIF Eşikleri: ", style={"color": "#7e8fa4", "fontSize": "0.73rem"}),
            html.Span("< 5 Normal  · ", style={"color": "#10b981", "fontSize": "0.73rem"}),
            html.Span("5–10 Orta  · ", style={"color": "#f59e0b", "fontSize": "0.73rem"}),
            html.Span("> 10 Yüksek", style={"color": "#ef4444", "fontSize": "0.73rem"}),
        ], style={"marginTop": "0.75rem"}),
    ])


# ── Callback: Chi-Square Hesapla ─────────────────────────────────────────────
@app.callback(
    Output("stat-chi-result", "children"),
    Input("btn-chi-compute", "n_clicks"),
    State("chi-var1", "value"),
    State("chi-var2", "value"),
    State("chi-max-cats", "value"),
    State("store-key", "data"),
    State("store-config", "data"),
    State("dd-segment-val", "value"),
    State("dd-segment-col", "value"),
    prevent_initial_call=True,
)
def compute_chi_square(n_clicks, var1, var2, max_cats_str, key, config, seg_val, seg_col_input):
    if not all([var1, var2, key, config]):
        return html.Div()
    df_orig = _get_df(key)
    if df_orig is None:
        return html.Div()
    seg_col     = config.get("segment_col") or (seg_col_input or None)
    target_type = config.get("target_type", "binary")
    target      = config.get("target_col", "")
    df_active   = apply_segment_filter(df_orig, seg_col, seg_val).copy()
    max_cats    = int(max_cats_str or 15)
    # Continuous target: auto-bin ile 5 gruba böl
    for col in [var1, var2]:
        if col == target and target_type == "continuous":
            try:
                df_active[col] = pd.qcut(
                    pd.to_numeric(df_active[col], errors="coerce"),
                    q=5, duplicates="drop"
                ).astype(str)
            except Exception:
                pass
    try:
        return _render_chi_square(df_active, var1, var2, max_cats)
    except Exception as exc:
        return html.Div(f"Hata: {exc}", style={"color": "#ef4444", "padding": "1rem"})


# ── Callback: ANOVA Hesapla ───────────────────────────────────────────────────
@app.callback(
    Output("stat-anova-result", "children"),
    Input("btn-anova-compute", "n_clicks"),
    State("anova-var", "value"),
    State("store-key", "data"),
    State("store-config", "data"),
    State("dd-segment-val", "value"),
    State("dd-segment-col", "value"),
    prevent_initial_call=True,
)
def compute_anova(n_clicks, var_col, key, config, seg_val, seg_col_input):
    if not all([var_col, key, config]):
        return html.Div()
    df_orig = _get_df(key)
    if df_orig is None:
        return html.Div()
    seg_col     = config.get("segment_col") or (seg_col_input or None)
    df_active   = apply_segment_filter(df_orig, seg_col, seg_val)
    target      = config.get("target_col")
    target_type = config.get("target_type", "binary")
    if not target:
        return html.Div("Config'de target kolonu tanımlanmamış.", className="alert-info-custom")
    try:
        # Continuous: Pearson/Spearman korelasyon testi
        if target_type == "continuous":
            return _render_pearson_spearman(df_active, var_col, target)
        # Multiclass ve 3+ sınıf varsa Kruskal-Wallis
        if target_type == "multiclass":
            n_cls = df_active[target].dropna().nunique()
            if n_cls >= 3:
                return _render_kruskal(df_active, var_col, target)
        # Binary / 2-sınıf multiclass: ANOVA
        return _render_anova(df_active, var_col, target)
    except Exception as exc:
        return html.Div(f"Hata: {exc}", style={"color": "#ef4444", "padding": "1rem"})


# ── Callback: KS Hesapla ──────────────────────────────────────────────────────
@app.callback(
    Output("stat-ks-result", "children"),
    Input("btn-ks-compute", "n_clicks"),
    State("ks-var", "value"),
    State("store-key", "data"),
    State("store-config", "data"),
    State("dd-segment-val", "value"),
    State("dd-segment-col", "value"),
    prevent_initial_call=True,
)
def compute_ks_test(n_clicks, var_col, key, config, seg_val, seg_col_input):
    if not all([var_col, key, config]):
        return html.Div()
    df_orig = _get_df(key)
    if df_orig is None:
        return html.Div()
    seg_col     = config.get("segment_col") or (seg_col_input or None)
    df_active   = apply_segment_filter(df_orig, seg_col, seg_val)
    target      = config.get("target_col")
    target_type = config.get("target_type", "binary")
    if not target:
        return html.Div("Config'de target kolonu tanımlanmamış.", className="alert-info-custom")
    try:
        return _render_ks(df_active, var_col, target, target_type)
    except Exception as exc:
        return html.Div(f"Hata: {exc}", style={"color": "#ef4444", "padding": "1rem"})


# ── Callback: VIF Sandbox Hesapla ─────────────────────────────────────────────
@app.callback(
    Output("stat-vif-result", "children"),
    Input("btn-vif-sandbox-compute", "n_clicks"),
    State("vif-var-set", "value"),
    State("vif-max-cols", "value"),
    State("store-key", "data"),
    State("store-config", "data"),
    State("store-expert-exclude", "data"),
    State("dd-segment-val", "value"),
    State("dd-segment-col", "value"),
    prevent_initial_call=True,
)
def compute_vif_sandbox(n_clicks, var_set, max_cols_str, key, config, expert_excluded, seg_val, seg_col_input):
    if not key or not config:
        return html.Div()
    df_orig = _get_df(key)
    if df_orig is None:
        return html.Div()
    seg_col   = config.get("segment_col") or (seg_col_input or None)
    df_active = apply_segment_filter(df_orig, seg_col, seg_val)
    max_cols  = int(max_cols_str or 20)
    try:
        return _render_vif_sandbox(df_active, var_set or "iv_filtered", max_cols,
                                   config, expert_excluded, key, seg_col, seg_val)
    except Exception as exc:
        return html.Div(f"Hata: {exc}", style={"color": "#ef4444", "padding": "1rem"})
