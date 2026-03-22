from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np



# ── Tab açıklama kutusu yardımcısı ────────────────────────────────────────────
def _tab_info(title: str, subtitle: str, body: str, color: str = "#4F8EF7") -> html.Div:
    """Her tab'ın üstünde gösterilen açıklama kartı."""
    return html.Div([
        html.Div([
            html.Span(title, style={"color": "#E8EAF0", "fontWeight": "700",
                                    "fontSize": "0.82rem"}),
            html.Span(f"  ·  {subtitle}",
                      style={"color": "#7e8fa4", "fontSize": "0.78rem"}),
        ], style={"marginBottom": "0.3rem"}),
        html.Div(body, style={"color": "#7e8fa4", "fontSize": "0.76rem",
                               "lineHeight": "1.55"}),
    ], style={
        "backgroundColor": "#111827",
        "borderLeft": f"3px solid {color}",
        "borderRadius": "4px",
        "padding": "0.6rem 1rem",
        "marginBottom": "1.1rem",
    })


# ── Ortak grafik teması ───────────────────────────────────────────────────────
_PLOT_LAYOUT = dict(
    paper_bgcolor="#161C27",
    plot_bgcolor="#0E1117",
    font=dict(family="Inter, Segoe UI, sans-serif", color="#8892a4", size=11),
    margin=dict(l=40, r=20, t=40, b=40),
)

_AXIS_STYLE = dict(gridcolor="#232d3f", linecolor="#232d3f", zerolinecolor="#232d3f")

# ── DataTable ortak stil sabiti ────────────────────────────────────────────────
_TABLE_STYLE = dict(
    style_table={"overflowX": "auto"},
    style_header={"backgroundColor": "#111827", "color": "#a8b2c2",
                  "fontWeight": "700", "fontSize": "0.72rem",
                  "border": "1px solid #2d3a4f", "textTransform": "uppercase"},
    style_data={"backgroundColor": "#161C27", "color": "#c8cdd8",
                "fontSize": "0.82rem", "border": "1px solid #232d3f"},
    style_cell={"padding": "0.4rem 0.65rem"},
    style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"}],
)


# ── PSI — Train WoE vs OOT WoE ───────────────────────────────────────────────
def calc_psi(base: np.ndarray, comp: np.ndarray,
             n_bins: int = 10, discrete: bool = True,
             detail: bool = False):
    """
    Population Stability Index — TEK KAYNAK.
    Tüm PSI hesaplamaları bu fonksiyondan geçer.

    discrete=True  → WoE değerleri (kesikli): her unique değere düşen oran karşılaştırılır.
    discrete=False → Ham değerler (sürekli): n_bins eşit aralıklı bin ile histogram karşılaştırması.
    detail=False   → float döndürür (toplam PSI).
    detail=True    → dict döndürür: {"psi": float, "rows": [{"Bin", "Baseline %", "Karşılaştırma %", "Δ (pp)", "PSI Katkı"}, ...]}
    """
    eps = 1e-4

    if discrete:
        # ── WoE: her unique değer bir bin ──────────────────────────────────
        all_vals = np.union1d(np.unique(base), np.unique(comp))
        if len(all_vals) < 2:
            return {"psi": 0.0, "rows": []} if detail else 0.0
        n_base, n_comp = len(base), len(comp)
        base_counts = {v: 0 for v in all_vals}
        comp_counts = {v: 0 for v in all_vals}
        for v in base:
            base_counts[v] = base_counts.get(v, 0) + 1
        for v in comp:
            comp_counts[v] = comp_counts.get(v, 0) + 1
        psi = 0.0
        rows = []
        for v in all_vals:
            b_pct = max(base_counts[v] / n_base, eps)
            c_pct = max(comp_counts[v] / n_comp, eps)
            contrib = (c_pct - b_pct) * np.log(c_pct / b_pct)
            psi += contrib
            if detail:
                rows.append({
                    "Bin": str(round(float(v), 4)),
                    "Baseline %": round(b_pct * 100, 2),
                    "Karşılaştırma %": round(c_pct * 100, 2),
                    "Δ (pp)": round((c_pct - b_pct) * 100, 2),
                    "PSI Katkı": round(float(contrib), 5),
                })
        if detail:
            return {"psi": float(psi), "rows": rows}
        return float(psi)
    else:
        # ── Ham değerler: quantile binning (çarpık veriye dayanıklı) ────────
        if len(base) == 0:
            return {"psi": 0.0, "rows": []} if detail else 0.0
        quantiles = np.linspace(0, 100, n_bins + 1)
        bins = np.unique(np.percentile(base, quantiles))
        if len(bins) < 2:
            return {"psi": 0.0, "rows": []} if detail else 0.0
        bins[0] = -np.inf
        bins[-1] = np.inf
        b_hist = np.histogram(base, bins=bins)[0]
        c_hist = np.histogram(comp, bins=bins)[0]
        b_pct = b_hist / len(base)
        c_pct = c_hist / len(comp)
        b_pct = np.where(b_pct < eps, eps, b_pct)
        c_pct = np.where(c_pct < eps, eps, c_pct)
        contribs = (c_pct - b_pct) * np.log(c_pct / b_pct)
        psi_total = float(np.sum(contribs))
        if not detail:
            return psi_total
        rows = []
        for i in range(len(contribs)):
            lo = f"{bins[i]:.2f}" if not np.isinf(bins[i]) else "-∞"
            hi = f"{bins[i+1]:.2f}" if not np.isinf(bins[i+1]) else "∞"
            rows.append({
                "Bin": f"({lo}, {hi}]",
                "Baseline %": round(float(b_pct[i]) * 100, 2),
                "Karşılaştırma %": round(float(c_pct[i]) * 100, 2),
                "Δ (pp)": round(float(c_pct[i] - b_pct[i]) * 100, 2),
                "PSI Katkı": round(float(contribs[i]), 5),
            })
        return {"psi": psi_total, "rows": rows}


def psi_label(psi_val: float) -> str:
    if psi_val < 0.10:
        return "Stabil"
    if psi_val < 0.25:
        return "Hafif Kayma"
    return "Kritik Kayma"





def mono_check(bt):
    """Bad Rate % üzerinden monotonluk kontrol et (Eksik/Special/TOPLAM hariç)."""
    m = bt[~bt["Bin"].astype(str).str.match(r"^(TOPLAM|Eksik|Special|special_)")]
    nums = [float(w) for w in m["Bad Rate %"].dropna().tolist()
            if isinstance(w, (int, float))]
    if len(nums) < 2:
        return "–"
    diffs = [nums[i+1] - nums[i] for i in range(len(nums)-1)]
    if all(d >= 0 for d in diffs):
        return "Artan ↑"
    if all(d <= 0 for d in diffs):
        return "Azalan ↓"
    return "Monoton Değil ✗"




# ── Yardımcı: r Badge ────────────────────────────────────────────────────────
def _make_r_badge(r):
    if np.isnan(r):
        return html.Div()
    r_str   = f"{r:+.4f}"
    r_color = "#10b981" if abs(r) < 0.4 else "#f59e0b" if abs(r) < 0.75 else "#ef4444"
    label   = "Zayıf" if abs(r) < 0.4 else "Orta" if abs(r) < 0.75 else "Yüksek" if abs(r) < 0.9 else "Çok Yüksek"
    return html.Div([
        html.Div("Korelasyon (r)", className="metric-label"),
        html.Div(r_str, className="metric-value",
                 style={"color": r_color, "fontSize": "1.35rem"}),
        html.Div(label, style={"fontSize": "0.65rem", "color": r_color,
                                "fontWeight": "700", "letterSpacing": "0.06em",
                                "textTransform": "uppercase", "marginTop": "0.15rem"}),
    ], className="metric-card", style={"padding": "0.55rem 1rem", "minWidth": "120px"})


# ── Yardımcı: Çift Analiz Grafikleri ─────────────────────────────────────────
def _make_pair_scatter(df_active, var1, var2, target):
    local = df_active[[var1, var2, target]].dropna(subset=[var1, var2]).copy()
    n_total = len(local)

    _no_data = html.Div("Yeterli çakışan veri yok.",
                        style={"color": "#7e8fa4", "fontSize": "0.8rem",
                               "padding": "1rem 0"})
    if n_total < 5:
        return _no_data

    is_num1 = pd.api.types.is_numeric_dtype(local[var1])
    is_num2 = pd.api.types.is_numeric_dtype(local[var2])

    def _trunc(s, n=24):
        return s[:n] + "…" if len(s) > n else s

    overall_br = float(local[target].mean() * 100)
    colors_t   = {0: "#4F8EF7", 1: "#ef4444"}

    # target'ın gerçek unique değerleri (NaN hariç)
    t_vals_raw = sorted(local[target].dropna().unique())
    if not t_vals_raw:
        return _no_data

    # ════════════════════════════════════════════════════════════════════════
    # NUMERIC × NUMERIC  —  Kantil Tabanlı Bad Rate Isı Haritası
    # Her iki değişkeni 10 kantile böler, hücre rengi = o kombinasyonun
    # kötü oranı (bad rate). Yeşil=düşük risk, Kırmızı=yüksek risk.
    # Sadece agregasyon yapıldığı için 5M+ satırda anlık çalışır.
    # ════════════════════════════════════════════════════════════════════════
    if is_num1 and is_num2:
        try:
            r = float(local[[var1, var2]].corr().iloc[0, 1])
        except Exception:
            r = float("nan")
        r_str = f"{r:+.4f}" if not np.isnan(r) else "—"

        N_BINS = 10
        work = local[[var1, var2, target]].dropna().copy()

        # Kantil tabanlı binleme — tekrar eden değerler için duplicates='drop'
        try:
            work["_b1"] = pd.qcut(work[var1], q=N_BINS, duplicates="drop", labels=False)
            work["_b2"] = pd.qcut(work[var2], q=N_BINS, duplicates="drop", labels=False)
        except Exception:
            work["_b1"] = pd.cut(work[var1], bins=N_BINS, labels=False)
            work["_b2"] = pd.cut(work[var2], bins=N_BINS, labels=False)

        grid = (
            work.groupby(["_b1", "_b2"], observed=True)
            .agg(bad_rate=(target, "mean"), n=(target, "count"))
            .reset_index()
        )
        if grid.empty:
            return _no_data

        # Bin orta noktaları — eksen etiketleri için
        b1_mid = work.groupby("_b1", observed=True)[var1].median().sort_index()
        b2_mid = work.groupby("_b2", observed=True)[var2].median().sort_index()

        pivot_br = grid.pivot(index="_b2", columns="_b1", values="bad_rate")
        pivot_n  = grid.pivot(index="_b2", columns="_b1", values="n")

        x_labels = [f"{b1_mid.get(c, c):.3g}" for c in pivot_br.columns]
        y_labels = [f"{b2_mid.get(i, i):.3g}" for i in pivot_br.index]

        hover = [
            [
                "Veri yok" if pd.isna(pivot_br.values[r][c])
                else (f"Bad Rate: %{pivot_br.values[r][c]*100:.1f}<br>"
                      f"n: {int(pivot_n.values[r][c]) if not pd.isna(pivot_n.values[r][c]) else 0:,}")
                for c in range(pivot_br.shape[1])
            ]
            for r in range(pivot_br.shape[0])
        ]

        fig = go.Figure(go.Heatmap(
            z=pivot_br.values * 100,
            x=x_labels,
            y=y_labels,
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            colorscale=[
                [0.0,  "#10b981"],   # yeşil  — düşük risk
                [0.5,  "#f59e0b"],   # sarı   — orta risk
                [1.0,  "#ef4444"],   # kırmızı — yüksek risk
            ],
            colorbar=dict(
                title=dict(text="Bad Rate %", font=dict(color="#8892a4", size=10)),
                tickfont=dict(color="#8892a4", size=9),
                ticksuffix="%",
                bgcolor="#161C27",
                bordercolor="#232d3f",
            ),
            zmid=overall_br,
        ))

        fig.update_layout(
            paper_bgcolor="#161C27",
            plot_bgcolor="#0E1117",
            font=dict(family="Inter, Segoe UI, sans-serif", color="#8892a4", size=11),
            title=dict(
                text=(f"Bad Rate Isı Haritası  ·  {_trunc(var1)} × {_trunc(var2)}"
                      f"  ·  r = {r_str}  ·  Genel Bad Rate %{overall_br:.1f}"),
                font=dict(color="#E8EAF0", size=12),
            ),
            xaxis=dict(**_AXIS_STYLE,
                       title=dict(text=f"{_trunc(var1)} (kantil medyanı)",
                                  font=dict(color="#8892a4", size=10)),
                       tickangle=-35),
            yaxis=dict(**_AXIS_STYLE,
                       title=dict(text=f"{_trunc(var2)} (kantil medyanı)",
                                  font=dict(color="#8892a4", size=10))),
            height=480,
            margin=dict(l=80, r=20, t=65, b=80),
        )

        n_note = (f"n = {n_total:,}  ·  {N_BINS}×{N_BINS} kantil hücre  "
                  f"·  Renk = hücre bad rate  ·  Genel ort. = %{overall_br:.1f}")

        return html.Div([
            html.P(n_note, style={"fontSize": "0.72rem", "color": "#7e8fa4",
                                   "marginBottom": "0.3rem"}),
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
        ])

    # ════════════════════════════════════════════════════════════════════════
    # CATEGORICAL × NUMERIC
    # Yan yana Box plot — kategori başına hedef kırılımı
    # ════════════════════════════════════════════════════════════════════════
    else:
        num_col = var1 if is_num1 else var2
        cat_col = var2 if is_num1 else var1

        local[cat_col] = local[cat_col].fillna("Eksik").astype(str)
        top_cats  = local[cat_col].value_counts().head(15).index.tolist()
        local_f   = local[local[cat_col].isin(top_cats)].copy()
        cat_order = (local_f.groupby(cat_col)[num_col]
                     .median().sort_values().index.tolist())

        fig = go.Figure()
        for tv in t_vals_raw:
            g    = local_f[local_f[target] == tv]
            clr  = colors_t.get(int(tv), "#8892a4")
            fill = f"rgba({'79,142,247' if int(tv)==0 else '239,68,68'}, 0.18)"
            fig.add_trace(go.Box(
                x=g[cat_col], y=g[num_col],
                name=f"Target = {int(tv)}",
                marker_color=clr,
                line=dict(color=clr, width=1.5),
                fillcolor=fill,
                boxmean=True,
                hovertemplate=(
                    f"Target = {int(tv)}<br>"
                    f"{_trunc(cat_col)}: %{{x}}<br>"
                    f"{_trunc(num_col)}: %{{y:.3f}}<extra></extra>"
                ),
            ))

        fig.update_layout(
            paper_bgcolor="#161C27", plot_bgcolor="#0E1117",
            font=dict(family="Inter, Segoe UI, sans-serif", color="#8892a4", size=11),
            boxmode="group",
            title=dict(
                text=f"{_trunc(cat_col)} × {_trunc(num_col)}  ·  Genel Bad Rate %{overall_br:.1f}",
                font=dict(color="#E8EAF0", size=12),
            ),
            xaxis=dict(**_AXIS_STYLE,
                       title=dict(text=cat_col, font=dict(color="#8892a4", size=10)),
                       categoryorder="array", categoryarray=cat_order, tickangle=-30),
            yaxis=dict(**_AXIS_STYLE,
                       title=dict(text=num_col, font=dict(color="#8892a4", size=10))),
            legend=dict(bgcolor="#161C27", bordercolor="#232d3f", font=dict(size=10)),
            height=430,
            margin=dict(l=60, r=20, t=50, b=80),
        )

        n_note = (f"n = {n_total:,}  ·  İlk 15 kategori  "
                  f"·  {_trunc(num_col, 20)} medyanına göre sıralı  "
                  f"·  — ortalama  ·  Mavi = Target 0  ·  Kırmızı = Target 1")

        return html.Div([
            html.P(n_note, style={"fontSize": "0.72rem", "color": "#7e8fa4",
                                   "marginBottom": "0.3rem"}),
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
        ])


def _safe_pair_scatter(df_active, var1, var2, target):
    try:
        return _make_pair_scatter(df_active, var1, var2, target)
    except Exception as exc:
        return html.Div(
            f"Grafik oluşturulamadı: {exc}",
            style={"color": "#ef4444", "padding": "1rem", "fontSize": "0.8rem"},
        )
