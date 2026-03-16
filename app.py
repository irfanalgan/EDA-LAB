import threading
import webbrowser
import uuid

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from data.loader import get_data_from_sql
from utils.helpers import apply_segment_filter
from modules.profiling import compute_profile, profile_summary
from modules.target_analysis import compute_target_stats, compute_target_over_time
from modules.deep_dive import get_variable_stats, get_woe_detail, compute_psi, compute_iv_ranking_optimal, get_woe_encoder
from modules.correlation import get_numeric_cols, compute_correlation_matrix, find_high_corr_pairs, compute_vif
from modules.screening import screen_columns

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from scipy import stats as scipy_stats

# ── Sunucu tarafı veri deposu ─────────────────────────────────────────────────
_SERVER_STORE: dict[str, pd.DataFrame] = {}

# ── WoE Dataset Builder ───────────────────────────────────────────────────────
def _build_woe_dataset(df: pd.DataFrame, target: str, cols: list) -> tuple:
    """
    Her kolon için WoE encode eder; kolon adı '{col}_woe' olarak kaydedilir.
    Returns: (woe_df, failed_cols)
    """
    result = {}
    failed = []
    for col in cols:
        try:
            woe_series, _, ok = get_woe_encoder(df, col, target)
            if ok:
                result[f"{col}_woe"] = woe_series.values
            else:
                failed.append(col)
        except Exception:
            failed.append(col)
    woe_df = pd.DataFrame(result, index=df.index)
    return woe_df, failed


# ── Sidebar geçiş sabitleri ────────────────────────────────────────────────────
_T = "max-width 0.3s ease-in-out, flex 0.3s ease-in-out, opacity 0.25s ease-in-out"
_COL_SIDEBAR_OPEN   = {"padding": "0", "position": "relative", "overflow": "visible",
                        "flex": "0 0 25%",  "maxWidth": "25%",  "opacity": "1",
                        "transition": _T}
_COL_SIDEBAR_CLOSED = {"padding": "0", "position": "relative", "overflow": "visible",
                        "flex": "0 0 36px", "maxWidth": "36px", "opacity": "1",
                        "transition": _T}
_COL_MAIN_OPEN      = {"padding": "0", "flex": "0 0 75%",               "maxWidth": "75%",
                        "transition": _T}
_COL_MAIN_CLOSED    = {"padding": "0", "flex": "0 0 calc(100% - 36px)", "maxWidth": "calc(100% - 36px)",
                        "transition": _T}
_SIDEBAR_OPEN_STYLE   = {"transition": "opacity 0.25s ease-in-out, max-width 0.3s ease-in-out"}
_SIDEBAR_CLOSED_STYLE = {"maxWidth": "0", "overflow": "hidden", "opacity": "0",
                          "padding": "0", "minHeight": "0",
                          "transition": "opacity 0.2s ease-in-out, max-width 0.3s ease-in-out"}

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="EDA Laboratuvarı",
    suppress_callback_exceptions=True,
)


def _get_df(key) -> pd.DataFrame | None:
    if not key:
        return None
    return _SERVER_STORE.get(key)


# ── Layout ────────────────────────────────────────────────────────────────────
def build_navbar():
    return dbc.Navbar(
        dbc.Container([
            html.Div([
                html.Span("EDA", className="navbar-logo-text"),
                html.Span("Laboratuvarı", className="navbar-brand-title"),
            ], style={"display": "flex", "alignItems": "baseline", "gap": "0.4rem"}),
            html.Div("Keşifsel Veri Analizi", className="navbar-subtitle"),
        ], fluid=True, style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}),
        color="#111827",
        dark=True,
        style={"borderBottom": "1px solid #232d3f", "padding": "0.6rem 0"},
    )


def build_sidebar():
    return html.Div([

        # ── Bölüm 1: Bağlantı ────────────────────────────────────────────────
        html.P("Veri Kaynağı", className="sidebar-section-title"),
        dbc.RadioItems(
            id="radio-source",
            options=[
                {"label": " SQL Server", "value": "sql"},
                {"label": " CSV Dosyası", "value": "csv"},
            ],
            value="sql",
            inline=True,
            className="mb-3",
            style={"color": "#c8cdd8", "fontSize": "0.82rem"},
            inputStyle={"marginRight": "4px"},
            labelStyle={"marginRight": "14px"},
        ),

        # SQL paneli
        html.Div(id="source-sql-div", children=[
            dbc.Label("Tablo Adı", className="form-label"),
            dbc.Input(
                id="input-table",
                value="dbo.MODEL_DATA",
                type="text",
                className="form-control mb-2",
                style={"fontSize": "0.85rem"},
            ),
            dbc.Button("Veriyi Yükle", id="btn-load", className="btn-load mb-1", n_clicks=0),
        ]),

        # CSV paneli
        html.Div(id="source-csv-div", style={"display": "none"}, children=[
            dcc.Upload(
                id="upload-csv",
                children=html.Div([
                    html.Span("CSV dosyasını buraya sürükleyin", style={"color": "#a8b2c2", "fontSize": "0.82rem"}),
                    html.Br(),
                    html.Span("veya tıklayın", style={"color": "#7e8fa4", "fontSize": "0.75rem"}),
                ]),
                accept=".csv",
                style={
                    "width": "100%", "borderWidth": "1px",
                    "borderStyle": "dashed", "borderRadius": "6px",
                    "borderColor": "#2d3a4f", "textAlign": "center",
                    "backgroundColor": "#0e1117", "padding": "1rem 0.5rem",
                    "cursor": "pointer", "marginBottom": "0.5rem",
                },
                style_active={
                    "borderColor": "#4F8EF7", "backgroundColor": "#111f35",
                },
            ),
            html.Div(id="csv-filename-display",
                     style={"color": "#7e8fa4", "fontSize": "0.72rem",
                            "marginBottom": "0.4rem", "fontStyle": "italic"}),
            dbc.Row([
                dbc.Col(
                    dbc.Select(
                        id="csv-separator",
                        options=[
                            {"label": "Virgül  (,)",    "value": ","},
                            {"label": "Noktalı virgül (;)", "value": ";"},
                            {"label": "Tab (\\t)",       "value": "\t"},
                            {"label": "Pipe  (|)",       "value": "|"},
                        ],
                        value=",",
                        className="dark-select",
                        style={"fontSize": "0.78rem"},
                    ), width=8,
                ),
                dbc.Col(
                    dbc.Button("Yükle", id="btn-load-csv",
                               color="primary", size="sm", n_clicks=0),
                    width=4, className="d-flex align-items-center",
                ),
            ], className="g-1"),
        ]),

        html.Div(id="load-status", style={"marginTop": "0.5rem", "fontSize": "0.8rem"}),

        html.Hr(className="sidebar-divider"),

        # ── Bölüm 2: Kolon Yapılandırması (veri yüklenince açılır) ───────────
        dbc.Collapse(
            html.Div([
                html.P("Kolon Yapılandırması", className="sidebar-section-title"),

                dbc.Label([
                    "Target Kolonu",
                    html.Span("*", style={"color": "#ef4444", "marginLeft": "3px"}),
                ], className="form-label"),
                dbc.Select(
                    id="dd-target-col",
                    options=[{"label": "Kolon seçiniz...", "value": "", "disabled": True}],
                    value="",
                    className="dark-select mb-3",
                ),

                dbc.Label("Tarih Kolonu", className="form-label"),
                html.Div("opsiyonel", className="form-hint"),
                dbc.Select(
                    id="dd-date-col",
                    options=[{"label": "—", "value": ""}],
                    value="",
                    className="dark-select mb-3",
                ),

                dbc.Label("Segment Kolonu", className="form-label"),
                html.Div("opsiyonel", className="form-hint"),
                dbc.Select(
                    id="dd-segment-col",
                    options=[{"label": "—", "value": ""}],
                    value="",
                    className="dark-select mb-3",
                ),

                dbc.Button(
                    "Yapılandırmayı Onayla",
                    id="btn-confirm",
                    className="btn-confirm",
                    n_clicks=0,
                ),
                html.Div(id="config-status", style={"marginTop": "0.5rem", "fontSize": "0.8rem"}),
            ]),
            id="collapse-config",
            is_open=False,
        ),

        # ── Bölüm 3: Aktif Segment Filtresi ──────────────────────────────────
        dbc.Collapse(
            html.Div([
                html.Hr(className="sidebar-divider"),
                html.P("Segment Filtresi", className="sidebar-section-title"),
                dbc.Label(id="segment-val-label", className="form-label"),
                dbc.Select(
                    id="dd-segment-val",
                    options=[],
                    value="Tümü",
                    className="dark-select",
                ),
                html.Div(id="segment-badge-area", style={"marginTop": "0.5rem"}),
            ]),
            id="collapse-segment",
            is_open=False,
        ),

    ], id="sidebar")


def build_main():
    return html.Div([
        html.Div(id="config-banner"),
        html.Div(id="metrics-row", style={"marginBottom": "1.5rem"}),
        dbc.Tabs([
            dbc.Tab(dcc.Loading(html.Div(id="data-preview"),   type="dot", color="#4F8EF7", delay_show=200), label="Önizleme",  tab_id="tab-preview",   className="tab-content-area"),
            dbc.Tab(dcc.Loading(html.Div(id="tab-profiling"), type="dot", color="#4F8EF7", delay_show=200), label="Profiling", tab_id="tab-profiling", className="tab-content-area"),
            dbc.Tab(dcc.Loading(html.Div(id="tab-target-iv"),  type="dot", color="#4F8EF7", delay_show=300), label="Target & IV",  tab_id="tab-target-iv",  className="tab-content-area"),
            dbc.Tab(dcc.Loading(html.Div(id="tab-deep-dive"), type="dot", color="#4F8EF7", delay_show=300), label="Değişken Analizi", tab_id="tab-deep-dive", className="tab-content-area"),
            dbc.Tab(html.Div([
                # ── Test Seçici ─────────────────────────────────────────────
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Test Türü", className="form-label"),
                        dbc.Select(
                            id="stat-test-type",
                            options=[
                                {"label": "Korelasyon",                                        "value": "correlation"},
                                {"label": "Chi-Square (Ki-Kare) Bağımsızlık Testi",            "value": "chi_square"},
                                {"label": "ANOVA (Target vs Sayısal Değişken)",                "value": "anova"},
                                {"label": "Kolmogorov-Smirnov (KS) Ayırıcılık Testi",         "value": "ks"},
                                {"label": "VIF Kum Havuzu (Çoklu Doğrusallık)",               "value": "vif_sandbox"},
                            ],
                            value="correlation",
                            className="dark-select",
                            style={"maxWidth": "460px"},
                        ),
                    ], width=6),
                ], className="mb-4"),

                # ── Korelasyon Paneli ─────────────────────────────────────
                html.Div(id="stat-corr-panel", children=[
                    html.Div([
                        html.Div([
                            html.Span("Korelasyon Analizi", style={"color": "#c8cdd8", "fontWeight": "700", "fontSize": "0.82rem"}),
                            html.Span("  ·  Pearson r", style={"color": "#7e8fa4", "fontSize": "0.78rem"}),
                        ], style={"marginBottom": "0.45rem"}),
                        html.Div([
                            html.Span("Amaç: ", style={"color": "#a8b2c2", "fontWeight": "600", "fontSize": "0.78rem"}),
                            html.Span("İki sayısal değişken arasındaki doğrusal ilişkinin yönünü ve şiddetini ölçer.", style={"color": "#6b7a94", "fontSize": "0.78rem"}),
                        ], style={"marginBottom": "0.25rem"}),
                        html.Div([
                            html.Span("Nasıl çalışır: ", style={"color": "#a8b2c2", "fontWeight": "600", "fontSize": "0.78rem"}),
                            html.Span("Pearson r katsayısı −1 ile +1 arasında değer alır. Değişken sayısı eşiği aşarsa en yüksek varyanslılar seçilir, ardından tam matris hesaplanır.", style={"color": "#6b7a94", "fontSize": "0.78rem"}),
                        ], style={"marginBottom": "0.25rem"}),
                        html.Div([
                            html.Span("Yorum: ", style={"color": "#a8b2c2", "fontWeight": "600", "fontSize": "0.78rem"}),
                            html.Span("|r| < 0.5 düşük  ·  0.5–0.75 orta  ·  > 0.75 yüksek korelasyon. Yüksek korelasyon çiftleri model için çoklu doğrusallık riski taşır — bu değişkenlerden biri modelden çıkarılabilir.", style={"color": "#6b7a94", "fontSize": "0.78rem"}),
                        ]),
                    ], style={
                        "backgroundColor": "#111827", "borderLeft": "3px solid #4F8EF7",
                        "borderRadius": "4px", "padding": "0.75rem 1rem",
                        "marginBottom": "1.25rem",
                    }),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Korelasyon Eşiği", className="form-label"),
                            html.Div("0 ile 1 arasında (önerilen: 0.75)", className="form-hint"),
                            dbc.Input(id="corr-threshold", type="number",
                                      min=0.0, max=1.0, step=0.01, value=0.75,
                                      className="form-control",
                                      style={"maxWidth": "140px"}),
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Maksimum Kolon Sayısı", className="form-label"),
                            html.Div("Fazla kolonda varyansa göre seçilir", className="form-hint"),
                            dbc.Select(id="corr-max-cols",
                                       options=[{"label": str(v), "value": str(v)}
                                                for v in [10, 15, 20, 30, 40]],
                                       value="20", className="dark-select",
                                       style={"maxWidth": "140px"}),
                        ], width=3),
                    ], className="mb-4"),
                    dcc.Loading(html.Div(id="corr-content"), type="dot", color="#4F8EF7", delay_show=300),
                ]),

                # ── Chi-Square Paneli ─────────────────────────────────────
                html.Div(id="stat-chi-panel", style={"display": "none"}, children=[
                    html.Div([
                        html.Div([
                            html.Span("Chi-Square (Ki-Kare) Bağımsızlık Testi", style={"color": "#c8cdd8", "fontWeight": "700", "fontSize": "0.82rem"}),
                            html.Span("  ·  χ²", style={"color": "#7e8fa4", "fontSize": "0.78rem"}),
                        ], style={"marginBottom": "0.45rem"}),
                        html.Div([
                            html.Span("Amaç: ", style={"color": "#a8b2c2", "fontWeight": "600", "fontSize": "0.78rem"}),
                            html.Span("İki değişkenin birbirinden istatistiksel olarak bağımsız olup olmadığını test eder. Kategorik × Kategorik ilişkiler için uygundur; sayısal kolonlar otomatik bin'lenir.", style={"color": "#6b7a94", "fontSize": "0.78rem"}),
                        ], style={"marginBottom": "0.25rem"}),
                        html.Div([
                            html.Span("Nasıl çalışır: ", style={"color": "#a8b2c2", "fontWeight": "600", "fontSize": "0.78rem"}),
                            html.Span("Gözlenen frekanslar ile bağımsızlık varsayımı altında beklenen frekanslar karşılaştırılır. 5M+ satır için önce pd.crosstab ile aggregate edilir, χ² scipy üzerinden kontenjans tablosu üzerinde hesaplanır.", style={"color": "#6b7a94", "fontSize": "0.78rem"}),
                        ], style={"marginBottom": "0.25rem"}),
                        html.Div([
                            html.Span("Yorum: ", style={"color": "#a8b2c2", "fontWeight": "600", "fontSize": "0.78rem"}),
                            html.Span("p < 0.05 → değişkenler bağımlı (ilişki var).  Cramér's V etki büyüklüğünü verir: < 0.10 önemsiz  ·  0.10–0.30 zayıf  ·  0.30–0.50 orta  ·  ≥ 0.50 güçlü ilişki.", style={"color": "#6b7a94", "fontSize": "0.78rem"}),
                        ]),
                    ], style={
                        "backgroundColor": "#111827", "borderLeft": "3px solid #a78bfa",
                        "borderRadius": "4px", "padding": "0.75rem 1rem",
                        "marginBottom": "1.25rem",
                    }),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Değişken 1", className="form-label"),
                            html.Div("Kategorik veya sayısal", className="form-hint"),
                            dbc.Select(id="chi-var1", className="dark-select"),
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Değişken 2", className="form-label"),
                            html.Div("Kategorik veya sayısal", className="form-hint"),
                            dbc.Select(id="chi-var2", className="dark-select"),
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Maks. Kategori", className="form-label"),
                            html.Div("Fazlası 'Diğer' olarak gruplandırılır", className="form-hint"),
                            dbc.Select(
                                id="chi-max-cats",
                                options=[{"label": str(v), "value": str(v)} for v in [10, 15, 20, 30]],
                                value="15", className="dark-select",
                                style={"maxWidth": "120px"},
                            ),
                        ], width=2),
                        dbc.Col([
                            dbc.Label("\u00a0", className="form-label"),
                            html.Div("\u00a0", className="form-hint"),
                            dbc.Button("Hesapla", id="btn-chi-compute", color="primary", size="sm"),
                        ], width=2),
                    ], className="mb-4"),
                    dcc.Loading(html.Div(id="stat-chi-result"), type="dot", color="#4F8EF7", delay_show=300),
                ]),

                # ── ANOVA Paneli ──────────────────────────────────────────
                html.Div(id="stat-anova-panel", style={"display": "none"}, children=[
                    html.Div([
                        html.Div([
                            html.Span("ANOVA — Varyans Analizi", style={"color": "#c8cdd8", "fontWeight": "700", "fontSize": "0.82rem"}),
                            html.Span("  ·  F-testi", style={"color": "#7e8fa4", "fontSize": "0.78rem"}),
                        ], style={"marginBottom": "0.45rem"}),
                        html.Div([
                            html.Span("Amaç: ", style={"color": "#a8b2c2", "fontWeight": "600", "fontSize": "0.78rem"}),
                            html.Span("Bir sayısal değişkenin target grupları (Good=0 / Bad=1) arasında istatistiksel olarak anlamlı farklılık gösterip göstermediğini test eder. Değişkenin ayırıcı güç taşıyıp taşımadığına dair güçlü bir ipucu verir.", style={"color": "#6b7a94", "fontSize": "0.78rem"}),
                        ], style={"marginBottom": "0.25rem"}),
                        html.Div([
                            html.Span("Nasıl çalışır: ", style={"color": "#a8b2c2", "fontWeight": "600", "fontSize": "0.78rem"}),
                            html.Span("Gruplar arası varyans ile gruplar içi varyans oranlanır (F istatistiği). Büyük veri için her gruptan en fazla 200.000 satır örneklenerek hesap yapılır; grup istatistikleri tüm veri üzerinden alınır.", style={"color": "#6b7a94", "fontSize": "0.78rem"}),
                        ], style={"marginBottom": "0.25rem"}),
                        html.Div([
                            html.Span("Yorum: ", style={"color": "#a8b2c2", "fontWeight": "600", "fontSize": "0.78rem"}),
                            html.Span("Yüksek F + düşük p (< 0.05) → grupların ortalamaları birbirinden anlamlı biçimde farklı, değişken ayırıcı güç taşıyor. p ≥ 0.05 → grup ortalamaları benzer, değişkenin tek başına katkısı sınırlı.", style={"color": "#6b7a94", "fontSize": "0.78rem"}),
                        ]),
                    ], style={
                        "backgroundColor": "#111827", "borderLeft": "3px solid #10b981",
                        "borderRadius": "4px", "padding": "0.75rem 1rem",
                        "marginBottom": "1.25rem",
                    }),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Sayısal Değişken", className="form-label"),
                            html.Div("Target gruplarına göre F-testi uygulanır", className="form-hint"),
                            dbc.Select(id="anova-var", className="dark-select"),
                        ], width=4),
                        dbc.Col([
                            dbc.Label("\u00a0", className="form-label"),
                            html.Div("\u00a0", className="form-hint"),
                            dbc.Button("Hesapla", id="btn-anova-compute", color="primary", size="sm"),
                        ], width=2),
                    ], className="mb-4"),
                    dcc.Loading(html.Div(id="stat-anova-result"), type="dot", color="#4F8EF7", delay_show=300),
                ]),

                # ── KS Paneli ─────────────────────────────────────────────
                html.Div(id="stat-ks-panel", style={"display": "none"}, children=[
                    html.Div([
                        html.Div([
                            html.Span("Kolmogorov-Smirnov (KS) Ayırıcılık Testi", style={"color": "#c8cdd8", "fontWeight": "700", "fontSize": "0.82rem"}),
                            html.Span("  ·  2 Örnekli", style={"color": "#7e8fa4", "fontSize": "0.78rem"}),
                        ], style={"marginBottom": "0.45rem"}),
                        html.Div([
                            html.Span("Amaç: ", style={"color": "#a8b2c2", "fontWeight": "600", "fontSize": "0.78rem"}),
                            html.Span("Good (target=0) ve Bad (target=1) gruplarının bir değişken üzerindeki kümülatif dağılımlarının ne kadar farklı olduğunu ölçer. Kredi risk modellemesinde temel ayırıcılık metriğidir.", style={"color": "#6b7a94", "fontSize": "0.78rem"}),
                        ], style={"marginBottom": "0.25rem"}),
                        html.Div([
                            html.Span("Nasıl çalışır: ", style={"color": "#a8b2c2", "fontWeight": "600", "fontSize": "0.78rem"}),
                            html.Span("İki grubun ampirik CDF'leri arasındaki maksimum dikey mesafe KS istatistiğini verir. İstatistik tüm veri üzerinden hesaplanır; CDF grafiği görselleştirme için 20.000 örnekle çizilir.", style={"color": "#6b7a94", "fontSize": "0.78rem"}),
                        ], style={"marginBottom": "0.25rem"}),
                        html.Div([
                            html.Span("Yorum: ", style={"color": "#a8b2c2", "fontWeight": "600", "fontSize": "0.78rem"}),
                            html.Span("KS < 0.20 zayıf  ·  0.20–0.30 orta  ·  0.30–0.50 iyi  ·  > 0.50 çok iyi ayırıcılık.  p < 0.05 → dağılımlar istatistiksel olarak farklı. Grafikteki sarı noktalı çizgi KS mesafesinin konumunu gösterir.", style={"color": "#6b7a94", "fontSize": "0.78rem"}),
                        ]),
                    ], style={
                        "backgroundColor": "#111827", "borderLeft": "3px solid #f59e0b",
                        "borderRadius": "4px", "padding": "0.75rem 1rem",
                        "marginBottom": "1.25rem",
                    }),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Değişken", className="form-label"),
                            html.Div("Target=0 ile Target=1 grupları karşılaştırılır", className="form-hint"),
                            dbc.Select(id="ks-var", className="dark-select"),
                        ], width=4),
                        dbc.Col([
                            dbc.Label("\u00a0", className="form-label"),
                            html.Div("\u00a0", className="form-hint"),
                            dbc.Button("Hesapla", id="btn-ks-compute", color="primary", size="sm"),
                        ], width=2),
                    ], className="mb-4"),
                    dcc.Loading(html.Div(id="stat-ks-result"), type="dot", color="#4F8EF7", delay_show=300),
                ]),

                # ── VIF Kum Havuzu Paneli ─────────────────────────────────
                html.Div(id="stat-vif-panel", style={"display": "none"}, children=[
                    html.Div([
                        html.Div([
                            html.Span("VIF — Varyans Şişme Faktörü", style={"color": "#c8cdd8", "fontWeight": "700", "fontSize": "0.82rem"}),
                            html.Span("  ·  Çoklu Doğrusallık", style={"color": "#7e8fa4", "fontSize": "0.78rem"}),
                        ], style={"marginBottom": "0.45rem"}),
                        html.Div([
                            html.Span("Amaç: ", style={"color": "#a8b2c2", "fontWeight": "600", "fontSize": "0.78rem"}),
                            html.Span("Bağımsız değişkenler arasındaki çoklu doğrusallığı (multicollinearity) saptar. Bir değişkenin diğerleri tarafından ne kadar açıklanabildiğini gösterir. Lojistik regresyon başta olmak üzere doğrusal modellerde kritik öneme sahiptir.", style={"color": "#6b7a94", "fontSize": "0.78rem"}),
                        ], style={"marginBottom": "0.25rem"}),
                        html.Div([
                            html.Span("Nasıl çalışır: ", style={"color": "#a8b2c2", "fontWeight": "600", "fontSize": "0.78rem"}),
                            html.Span("Her değişken diğerlerine regresse edilir; elde edilen R² üzerinden VIF = 1 / (1 − R²) formülüyle hesaplanır. IV ≥ 0.10 filtresi seçilirse önce bilgi değeri yüksek değişkenler seçilir.", style={"color": "#6b7a94", "fontSize": "0.78rem"}),
                        ], style={"marginBottom": "0.25rem"}),
                        html.Div([
                            html.Span("Yorum: ", style={"color": "#a8b2c2", "fontWeight": "600", "fontSize": "0.78rem"}),
                            html.Span("VIF < 5 → normal, sorun yok  ·  5–10 → orta risk, dikkat  ·  > 10 → yüksek çoklu doğrusallık, bu değişkenlerden biri veya birkaçı modelden çıkarılmalı ya da PCA/WoE dönüşümü uygulanmalı.", style={"color": "#6b7a94", "fontSize": "0.78rem"}),
                        ]),
                    ], style={
                        "backgroundColor": "#111827", "borderLeft": "3px solid #ef4444",
                        "borderRadius": "4px", "padding": "0.75rem 1rem",
                        "marginBottom": "1.25rem",
                    }),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Değişken Seti", className="form-label"),
                            html.Div("IV ≥ 0.10 filtresi varsayılan; tümü de seçilebilir", className="form-hint"),
                            dbc.Select(
                                id="vif-var-set",
                                options=[
                                    {"label": "IV ≥ 0.10 olan değişkenler", "value": "iv_filtered"},
                                    {"label": "Tüm numerik değişkenler",      "value": "all_numeric"},
                                ],
                                value="iv_filtered",
                                className="dark-select",
                                style={"maxWidth": "280px"},
                            ),
                        ], width=4),
                        dbc.Col([
                            dbc.Label("Maks. Kolon", className="form-label"),
                            html.Div("VIF hesabı için max değişken sayısı", className="form-hint"),
                            dbc.Select(
                                id="vif-max-cols",
                                options=[{"label": str(v), "value": str(v)} for v in [10, 15, 20, 30, 40]],
                                value="20", className="dark-select",
                                style={"maxWidth": "120px"},
                            ),
                        ], width=2),
                        dbc.Col([
                            dbc.Label("\u00a0", className="form-label"),
                            html.Div("\u00a0", className="form-hint"),
                            dbc.Button("Hesapla", id="btn-vif-sandbox-compute", color="primary", size="sm"),
                        ], width=2),
                    ], className="mb-4"),
                    dcc.Loading(html.Div(id="stat-vif-result"), type="dot", color="#4F8EF7", delay_show=300),
                ]),

            ]), label="İstatistiksel Testler", tab_id="tab-correlation", className="tab-content-area"),
            dbc.Tab(html.Div([
                html.Div("Tüm değişkenler için IV, Eksik%, PSI, Korelasyon ve VIF bilgilerini tek tabloda gösterir.",
                         className="form-hint", style={"marginBottom": "0.75rem"}),
                dbc.Row([
                    dbc.Col(
                        dbc.Checklist(
                            id="chk-varsummary-woe",
                            options=[{"label": "PSI · Korelasyon · VIF — WoE dönüştürülmüş değerler üzerinden hesapla",
                                      "value": "woe"}],
                            value=[],
                            inline=True,
                            style={"color": "#c8cdd8", "fontSize": "0.83rem"},
                        ), width=10,
                    ),
                    dbc.Col(
                        dbc.Button("Hesapla", id="btn-var-summary",
                                   color="primary", size="sm"),
                        width=2, className="d-flex justify-content-end",
                    ),
                ], align="center", className="mb-3"),
                dcc.Loading(html.Div(id="div-var-summary"), type="dot", color="#4F8EF7", delay_show=300),
            ]), label="Değişken Özeti", tab_id="tab-var-summary", className="tab-content-area"),
            dbc.Tab(html.Div([
                # ── Grafik Oluşturucu ─────────────────────────────────────────
                html.P("Grafik Oluşturucu", className="section-title"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("X Ekseni", className="form-label"),
                        html.Div("\u00a0", className="form-hint"),
                        dbc.Select(id="pg-x-col", className="dark-select"),
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Y Ekseni (Sol)", className="form-label"),
                        html.Div("\u00a0", className="form-hint"),
                        dbc.Select(id="pg-y-col", className="dark-select"),
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Y2 Ekseni (Sağ)", className="form-label"),
                        html.Div("Seçilirse sağ eksende çizgi olarak eklenir.",
                                 className="form-hint"),
                        dbc.Select(id="pg-y2-col", className="dark-select", value=""),
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Grafik Tipi", className="form-label"),
                        html.Div("\u00a0", className="form-hint"),
                        dbc.Select(id="pg-chart-type", className="dark-select",
                                   options=[
                                       {"label": "Bar",            "value": "bar"},
                                       {"label": "Line",           "value": "line"},
                                       {"label": "Bar + Line",     "value": "bar_line"},
                                       {"label": "Scatter",        "value": "scatter"},
                                       {"label": "Box",            "value": "box"},
                                       {"label": "Histogram",      "value": "histogram"},
                                   ], value="bar"),
                    ], width=3),
                ], className="mb-2"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Agregasyon (Y)", className="form-label"),
                        html.Div("\u00a0", className="form-hint"),
                        dbc.Select(id="pg-agg", className="dark-select",
                                   options=[
                                       {"label": "Ortalama", "value": "mean"},
                                       {"label": "Toplam",   "value": "sum"},
                                       {"label": "Adet",     "value": "count"},
                                   ], value="mean"),
                    ], width=2),
                    dbc.Col([
                        dbc.Label("Zaman Birimi", className="form-label"),
                        html.Div("X tarihse geçerlidir.",
                                 className="form-hint"),
                        dbc.Select(id="pg-time-unit", className="dark-select",
                                   options=[
                                       {"label": "Gün",    "value": "D"},
                                       {"label": "Ay",     "value": "M"},
                                       {"label": "Çeyrek", "value": "Q"},
                                       {"label": "Yıl",    "value": "Y"},
                                   ], value="M"),
                    ], width=2),
                    dbc.Col([
                        dbc.Label("Grupla (renk)", className="form-label"),
                        html.Div("Y2 seçiliyken devre dışıdır.",
                                 className="form-hint"),
                        dbc.Select(id="pg-color-col", className="dark-select", value=""),
                    ], width=3),
                    dbc.Col([
                        html.Div("\u00a0", className="form-label"),
                        html.Div("\u00a0", className="form-hint"),
                        dbc.Button("Çiz", id="btn-pg-chart", color="primary", size="sm"),
                    ], width="auto"),
                ], className="mb-3"),
                dcc.Loading(html.Div(id="pg-chart-output"),
                            type="dot", color="#4F8EF7", delay_show=200),

                html.Div(style={"borderTop": "1px solid #232d3f",
                                "margin": "2rem 0 1.5rem"}),

                # ── Hızlı Model ───────────────────────────────────────────────
                html.P("Hızlı Model", className="section-title"),

                # Değişken özeti önizleme
                html.Div([
                    html.Div("Değişken Özeti (IV önizleme — Değişken Özeti sekmesinden hesaplandıysa gösterilir)",
                             className="form-hint", style={"marginBottom": "0.4rem"}),
                    html.Div(id="pg-var-summary-preview",
                             style={"maxHeight": "200px", "overflowY": "auto",
                                    "border": "1px solid #2d3a4f", "borderRadius": "6px"}),
                ], className="mb-3"),

                dbc.Row([
                    # Sol — kaynak liste
                    dbc.Col([
                        dbc.Label("Mevcut Değişkenler", className="form-label"),
                        html.Div("Seç → ile model listesine taşı.",
                                 className="form-hint"),
                        html.Div(id="pg-source-container",
                                 style={"maxHeight": "280px", "overflowY": "auto",
                                        "backgroundColor": "#0e1117",
                                        "border": "1px solid #2d3a4f",
                                        "borderRadius": "6px",
                                        "padding": "0.5rem 0.75rem",
                                        "minHeight": "60px"}),
                    ], width=5),
                    # Orta — ok butonları
                    dbc.Col([
                        html.Div([
                            dbc.Button("►", id="btn-pg-add",
                                       color="primary", size="sm",
                                       className="mb-2",
                                       style={"width": "40px"}),
                            html.Br(),
                            dbc.Button("◄", id="btn-pg-remove",
                                       color="secondary", size="sm",
                                       outline=True,
                                       style={"width": "40px"}),
                        ], style={"textAlign": "center", "paddingTop": "2.5rem"}),
                    ], width=1,
                       className="d-flex align-items-center justify-content-center"),
                    # Sağ — model listesi
                    dbc.Col([
                        dbc.Label("Model Listesi", className="form-label"),
                        html.Div("Model kurulacak değişkenler.",
                                 className="form-hint"),
                        html.Div(id="pg-model-container",
                                 style={"maxHeight": "280px", "overflowY": "auto",
                                        "backgroundColor": "#0e1117",
                                        "border": "1px solid #2d3a4f",
                                        "borderRadius": "6px",
                                        "padding": "0.5rem 0.75rem",
                                        "minHeight": "60px"}),
                    ], width=5),
                ], className="mb-3"),

                # WoE toggle (gizli — arka planda her zaman her ikisi de hesaplanır)
                html.Div(dbc.Checklist(id="chk-use-woe", options=[], value=[]),
                         style={"display": "none"}),

                # Model parametreleri — satır 1: target + split
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Target Kolonu", className="form-label"),
                        html.Div("\u00a0", className="form-hint"),
                        dbc.Select(id="pg-target-col", className="dark-select"),
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Train/Test Bölünmesi", className="form-label"),
                        html.Div("\u00a0", className="form-hint"),
                        dbc.Select(id="pg-split-method", className="dark-select",
                                   options=[
                                       {"label": "Rastgele (%)",  "value": "random"},
                                       {"label": "Tarihe Göre",   "value": "date"},
                                   ], value="random"),
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Test Oranı (%)", className="form-label"),
                        html.Div("Rastgele bölünmede geçerlidir.",
                                 className="form-hint"),
                        dbc.Input(id="pg-test-size", type="number",
                                  value=30, min=10, max=50, step=5,
                                  style={"maxWidth": "110px"}),
                    ], width=2),
                    dbc.Col([
                        dbc.Label("Kesim Tarihi", className="form-label"),
                        html.Div("Öncesi = Train  ·  Sonrası = Test",
                                 className="form-hint"),
                        dbc.Select(id="pg-split-date", className="dark-select",
                                   placeholder="Tarihe Göre seçiliyse"),
                    ], width=3),
                ], className="mb-2"),

                # Model parametreleri — satır 2: model tipi + LR-C + kur
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Model", className="form-label"),
                        html.Div("\u00a0", className="form-hint"),
                        dbc.Select(id="pg-model-type", className="dark-select",
                                   options=[
                                       {"label": "Logistic Regression", "value": "lr"},
                                       {"label": "LightGBM",            "value": "lgbm"},
                                       {"label": "XGBoost",             "value": "xgb"},
                                       {"label": "Random Forest",       "value": "rf"},
                                   ],
                                   value="lr",
                                   style={"maxWidth": "220px"}),
                    ], width=3),
                    dbc.Col([
                        dbc.Label("C (Regülarizasyon)", className="form-label"),
                        html.Div("Yalnızca Logistic Regression için", className="form-hint"),
                        dbc.Input(id="pg-c-value", type="number",
                                  value=1.0, min=0.001, step=0.1,
                                  style={"maxWidth": "110px"}),
                    ], width=2),
                    dbc.Col([
                        html.Div("\u00a0", className="form-label"),
                        html.Div("\u00a0", className="form-hint"),
                        dbc.Button("Model Kur", id="btn-pg-build",
                                   color="success", size="sm"),
                    ], width=2),
                ], className="mb-3"),
                dcc.Loading(html.Div(id="pg-model-output"),
                            type="dot", color="#4F8EF7", delay_show=300),

                dcc.Store(id="store-pg-model-vars", storage_type="memory"),
            ]), label="Playground", tab_id="tab-playground",
               className="tab-content-area"),
        ], id="main-tabs", active_tab="tab-preview", className="main-tabs"),
    ], id="main-content")


app.layout = html.Div([
    build_navbar(),
    dbc.Row([
        dbc.Col(
            html.Div([
                build_sidebar(),
                html.Button("‹", id="btn-sidebar-toggle", className="sidebar-toggle"),
            ]),
            id="col-sidebar", width=3, style=_COL_SIDEBAR_OPEN,
        ),
        dbc.Col(build_main(), id="col-main", width=9, style=_COL_MAIN_OPEN),
    ], style={"margin": "0"}),
    dcc.Store(id="store-key", storage_type="memory"),
    dcc.Store(id="store-config", storage_type="memory"),
    dcc.Store(id="store-expert-exclude", storage_type="memory"),
])


# ── Callback: Kaynak Toggle ───────────────────────────────────────────────────
@app.callback(
    Output("source-sql-div", "style"),
    Output("source-csv-div", "style"),
    Input("radio-source", "value"),
)
def toggle_source(source):
    if source == "csv":
        return {"display": "none"}, {}
    return {}, {"display": "none"}


# ── Callback: CSV Dosya Adı Göster ────────────────────────────────────────────
@app.callback(
    Output("csv-filename-display", "children"),
    Input("upload-csv", "filename"),
    prevent_initial_call=True,
)
def show_csv_filename(filename):
    if filename:
        return f"📄 {filename}"
    return ""


# ── Callback: CSV Yükle ───────────────────────────────────────────────────────
@app.callback(
    Output("store-key",   "data",     allow_duplicate=True),
    Output("load-status", "children", allow_duplicate=True),
    Input("btn-load-csv", "n_clicks"),
    State("upload-csv",   "contents"),
    State("upload-csv",   "filename"),
    State("csv-separator", "value"),
    prevent_initial_call=True,
)
def load_csv(n_clicks, contents, filename, sep):
    import base64, io
    if not contents or not filename:
        return dash.no_update, dbc.Alert(
            "Önce bir CSV dosyası seçin.", color="warning",
            style={"padding": "0.4rem 0.75rem", "fontSize": "0.8rem"},
        )
    if not filename.lower().endswith(".csv"):
        return dash.no_update, dbc.Alert(
            "Yalnızca .csv uzantılı dosyalar desteklenir.", color="warning",
            style={"padding": "0.4rem 0.75rem", "fontSize": "0.8rem"},
        )
    try:
        _, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8", errors="replace")),
                         sep=sep, low_memory=False)
        key = str(uuid.uuid4())
        _SERVER_STORE[key] = df
        return key, dbc.Alert(
            [html.Strong(f"{len(df):,} satır"),
             f"  ·  {df.shape[1]} kolon  ·  {filename}"],
            color="success",
            style={"padding": "0.4rem 0.75rem", "fontSize": "0.8rem"},
        )
    except Exception as e:
        return dash.no_update, dbc.Alert(
            f"Okuma hatası: {e}", color="danger",
            style={"padding": "0.4rem 0.75rem", "fontSize": "0.8rem"},
        )


# ── Callback: Veriyi Yükle (SQL) ──────────────────────────────────────────────
@app.callback(
    Output("store-key", "data"),
    Output("load-status", "children"),
    Input("btn-load", "n_clicks"),
    State("input-table", "value"),
    prevent_initial_call=True,
)
def load_data(n_clicks, table_name):
    if not table_name or not table_name.strip():
        return dash.no_update, dbc.Alert(
            "Lütfen bir tablo adı girin.", color="warning",
            style={"padding": "0.4rem 0.75rem", "fontSize": "0.8rem"},
        )
    try:
        df = get_data_from_sql(table_name.strip())
        key = str(uuid.uuid4())
        _SERVER_STORE[key] = df
        return key, dbc.Alert(
            [
                html.Strong(f"{len(df):,} satır"),
                f"  ·  {df.shape[1]} kolon yüklendi.",
            ],
            color="success",
            style={"padding": "0.4rem 0.75rem", "fontSize": "0.8rem"},
        )
    except Exception as e:
        return dash.no_update, dbc.Alert(
            str(e), color="danger",
            style={"padding": "0.4rem 0.75rem", "fontSize": "0.8rem"},
        )


# ── Callback: Kolon Yapılandırması bölümünü aç, dropdown seçeneklerini doldur ─
@app.callback(
    Output("collapse-config", "is_open"),
    Output("dd-target-col", "options"),
    Output("dd-date-col", "options"),
    Output("dd-segment-col", "options"),
    Input("store-key", "data"),
)
def open_config_section(key):
    df = _get_df(key)
    if df is None:
        return False, [], [], []

    all_opts = (
        [{"label": "Kolon seçiniz...", "value": "", "disabled": True}]
        + [{"label": f"{c}  [{df[c].dtype}]", "value": c} for c in df.columns]
    )

    # Tarih kolonları — datetime veya isimde date/tarih geçenler başa alınır
    date_keywords = {"date", "tarih", "dt", "time", "zaman"}
    date_cols = sorted(
        df.columns,
        key=lambda c: (
            0 if pd.api.types.is_datetime64_any_dtype(df[c])
            else 1 if any(k in c.lower() for k in date_keywords)
            else 2
        ),
    )
    date_opts = (
        [{"label": "—", "value": ""}]
        + [{"label": f"{c}  [{df[c].dtype}]", "value": c} for c in date_cols]
    )

    # Segment kolonları — object veya düşük kardinalite
    seg_cols = [
        c for c in df.columns
        if df[c].dtype == object or df[c].nunique() <= 50
    ]
    seg_opts = (
        [{"label": "—", "value": ""}]
        + [{"label": f"{c}  [{df[c].dtype}]", "value": c} for c in seg_cols]
    )

    return True, all_opts, date_opts, seg_opts


# ── Callback: Yapılandırmayı Onayla ──────────────────────────────────────────
@app.callback(
    Output("store-config", "data"),
    Output("config-status", "children"),
    Input("btn-confirm", "n_clicks"),
    State("dd-target-col", "value"),
    State("dd-date-col", "value"),
    State("dd-segment-col", "value"),
    State("store-key", "data"),
    prevent_initial_call=True,
)
def confirm_config(n_clicks, target_col, date_col, segment_col, key):
    if not target_col or target_col == "":
        return dash.no_update, dbc.Alert(
            "Target kolonu zorunludur.", color="warning",
            style={"padding": "0.4rem 0.75rem", "fontSize": "0.8rem"},
        )

    config = {
        "target_col": target_col,
        "date_col": date_col or None,
        "segment_col": segment_col or None,
    }

    # ── Ön Eleme ──────────────────────────────────────────────────────────────
    df = _get_df(key)
    if df is not None:
        passed, screen_report = screen_columns(
            df, target_col, date_col or None, segment_col or None
        )
        _SERVER_STORE[f"{key}_screen"] = (passed, screen_report)

    parts = [html.Strong("✓ Onaylandı")]
    if date_col:
        parts += [f"  ·  Tarih: {date_col}"]
    if segment_col:
        parts += [f"  ·  Segment: {segment_col}"]

    return config, dbc.Alert(
        parts, color="success",
        style={"padding": "0.4rem 0.75rem", "fontSize": "0.8rem"},
    )


# ── Callback: Segment değer dropdown'ını aç ───────────────────────────────────
@app.callback(
    Output("collapse-segment", "is_open"),
    Output("dd-segment-val", "options"),
    Output("dd-segment-val", "value"),
    Output("segment-val-label", "children"),
    Input("store-config", "data"),
    Input("dd-segment-col", "value"),
    State("store-key", "data"),
)
def open_segment_filter(config, seg_col_input, key):
    # Kolon: onaylanmış config'den veya seçili dropdown'dan
    seg_col = (config or {}).get("segment_col") or (seg_col_input or None)
    if not seg_col:
        return False, [], None, "Segment Değeri"
    df = _get_df(key)
    if df is None or seg_col not in df.columns:
        return False, [], None, "Segment Değeri"

    unique_vals = sorted(df[seg_col].dropna().astype(str).unique().tolist())
    options = [{"label": "Tümü", "value": "Tümü"}] + [
        {"label": v, "value": v} for v in unique_vals
    ]
    label = f"{seg_col}  ({len(unique_vals)} değer)"
    return True, options, "Tümü", label


# ── Callback: Config Banner (onaylandıktan sonra üstte özetle) ─────────────────
@app.callback(
    Output("config-banner", "children"),
    Input("store-config", "data"),
    Input("dd-segment-val", "value"),
    State("dd-segment-col", "value"),
)
def update_config_banner(config, seg_val, seg_col_input):
    if not config or not config.get("target_col"):
        return html.Div()

    def badge(label, value, color="#4F8EF7"):
        return html.Span([
            html.Span(label, className="banner-badge-label"),
            html.Span(value, className="banner-badge-value"),
        ], className="banner-badge", style={"borderColor": color})

    items = [badge("TARGET", config["target_col"])]
    if config.get("date_col"):
        items.append(badge("TARİH", config["date_col"], "#10b981"))

    seg_col = config.get("segment_col") or (seg_col_input or None)
    if seg_col:
        seg_display = seg_val if (seg_val and seg_val != "Tümü") else seg_col
        items.append(badge("SEGMENT", seg_display, "#f59e0b"))

    return html.Div(items, className="config-banner")


# ── Callback: Metrikler ───────────────────────────────────────────────────────
@app.callback(
    Output("metrics-row", "children"),
    Input("store-config", "data"),
    Input("dd-segment-val", "value"),
    State("store-key", "data"),
    State("dd-segment-col", "value"),
)
def update_metrics(config, seg_val, key, seg_col_input):
    df_orig = _get_df(key)

    if df_orig is None:
        return html.Div(
            "Sol menüden tablo adını girin ve Veriyi Yükle butonuna tıklayın.",
            className="alert-info-custom",
        )
    if not config or not config.get("target_col"):
        return html.Div(
            "Kolon yapılandırmasını tamamlayın ve onaylayın.",
            className="alert-info-custom",
        )

    seg_col   = config.get("segment_col") or (seg_col_input or None)
    date_col  = config.get("date_col")
    df_active = apply_segment_filter(df_orig, seg_col, seg_val)
    active_rows = len(df_active)

    target      = config["target_col"]
    target_rate = df_active[target].mean() * 100 if pd.api.types.is_numeric_dtype(df_active[target]) else None

    def card(value, label, accent="#4F8EF7", tooltip=None):
        return dbc.Col(html.Div([
            html.Div(value, className="metric-value", style={"color": accent}),
            html.Div(label, className="metric-label"),
        ], className="metric-card",
           title=tooltip,
           style={"cursor": "help"} if tooltip else {}),
        width=3)

    # Target 0/1 sayıları
    target_tooltip = None
    if target_rate is not None:
        n_bad  = int(df_active[target].sum())
        n_good = int((df_active[target] == 0).sum())
        target_tooltip = f"1 (Bad):  {n_bad:,}\n0 (Good): {n_good:,}\nToplam: {active_rows:,}"

    # Tarih aralığı kartı
    if date_col and date_col in df_active.columns:
        try:
            dates   = pd.to_datetime(df_active[date_col], errors="coerce").dropna()
            d_min   = dates.min().strftime("%Y-%m")
            d_max   = dates.max().strftime("%Y-%m")
            date_card = card(f"{d_min} – {d_max}", f"Tarih Aralığı  ({date_col})",
                             "#7e8fa4")
        except Exception:
            date_card = card("—", f"Tarih Aralığı  ({date_col})", "#7e8fa4")
    else:
        date_card = card("—", "Tarih Aralığı", "#7e8fa4")

    cards = [
        card(f"{active_rows:,}", "Aktif Satır"),
        card(f"{df_active.shape[1]}", "Kolon Sayısı"),
        date_card,
        card(f"%{active_rows / len(df_orig) * 100:.1f}", "Segment Kapsamı", "#f59e0b")
        if seg_col else card(
            f"%{target_rate:.2f}" if target_rate is not None else "—",
            f"Target Oranı  ({target})",
            "#ef4444",
            tooltip=target_tooltip,
        ),
    ]

    return dbc.Row(cards, className="g-3 mb-0")


# ── Callback: Veri Önizleme ───────────────────────────────────────────────────
@app.callback(
    Output("data-preview", "children"),
    Input("store-config", "data"),
    Input("dd-segment-val", "value"),
    Input("store-expert-exclude", "data"),
    State("store-key", "data"),
    State("dd-segment-col", "value"),
)
def update_preview(config, seg_val, expert_excluded, key, seg_col_input):
    df_orig = _get_df(key)
    if df_orig is None or not config or not config.get("target_col"):
        return html.Div()
    expert_excluded = expert_excluded or []

    seg_col = config.get("segment_col") or (seg_col_input or None)
    df_active = apply_segment_filter(df_orig, seg_col, seg_val)
    preview = df_active.head(50)

    # ── Uzman görüşü paneli ───────────────────────────────────────────────────
    cfg_cols = {c for c in [config.get("target_col"), config.get("date_col"),
                             config.get("segment_col")] if c}
    excluded_set = set(expert_excluded)
    available = sorted([c for c in df_orig.columns if c not in cfg_cols and c not in excluded_set])
    chk_options = [{"label": c, "value": c} for c in available]

    current_exclusion_display = html.Div([
        html.Span("El ile elinen değişkenler: ",
                  style={"color": "#7e8fa4", "fontSize": "0.73rem", "marginRight": "0.4rem"}),
        *[html.Span(c, style={"color": "#a78bfa", "fontSize": "0.73rem",
                               "marginRight": "0.5rem", "fontWeight": "600"})
          for c in expert_excluded],
    ]) if expert_excluded else html.Div(
        "Henüz el ile elinen değişken yok.", className="form-hint",
    )

    expert_panel = html.Div([
        html.Div(style={"borderTop": "1px solid #232d3f",
                        "marginTop": "2rem", "marginBottom": "1.25rem"}),
        html.P("Uzman Görüşü: El ile Eleme", className="section-title"),
        html.Div("Analiz dışında tutmak istediğiniz değişkenleri seçip "
                 "\"Listeye Ekle\" butonuna tıklayın. Seçilen değişkenler "
                 "Ön Eleme Raporu'nda ve tüm sekmelerde (Deep Dive, Korelasyon vb.) görünmez.",
                 className="form-hint", style={"marginBottom": "0.75rem"}),
        html.Div(
            dbc.Checklist(
                id="chk-expert-cols",
                options=chk_options,
                value=[],
                inline=True,
                className="expert-checklist",
            ),
            style={"maxHeight": "220px", "overflowY": "auto",
                   "backgroundColor": "#0e1117", "borderRadius": "6px",
                   "border": "1px solid #2d3a4f", "padding": "0.5rem 0.75rem",
                   "marginBottom": "0.5rem"},
        ),
        dbc.Row([
            dbc.Col(dbc.Button("Listeye Ekle", id="btn-expert-add",
                               color="warning", size="sm"), width="auto"),
            dbc.Col(dbc.Button("Listeyi Temizle", id="btn-expert-clear",
                               color="secondary", size="sm", outline=True,
                               disabled=len(expert_excluded) == 0), width="auto"),
        ], className="mb-3 mt-2"),
        current_exclusion_display,
    ])

    preview_tsv = preview.to_csv(sep="\t", index=False)
    return html.Div([
        html.P("Veri Önizleme", className="section-title"),
        html.Div(
            dcc.Clipboard(target_id="preview-tsv", title="Kopyala",
                          style={"cursor": "pointer", "fontSize": "0.72rem",
                                 "color": "#a8b2c2", "padding": "0.2rem 0.55rem",
                                 "border": "1px solid #2d3a4f", "borderRadius": "4px",
                                 "backgroundColor": "#1a2035", "float": "right",
                                 "marginBottom": "0.4rem", "letterSpacing": "0.04em"}),
            style={"overflow": "hidden"},
        ),
        html.Pre(preview_tsv, id="preview-tsv", style={"display": "none"}),
        dash_table.DataTable(
            data=preview.to_dict("records"),
            columns=[{"name": c, "id": c} for c in preview.columns],
            page_size=20,
            page_action="native",
            sort_action="native",
            filter_action="native",
            style_table={"overflowX": "auto"},
            style_header={
                "backgroundColor": "#1a2035",
                "color": "#E8EAF0",
                "fontWeight": "600",
                "fontSize": "0.78rem",
                "border": "1px solid #2d3a4f",
                "textTransform": "uppercase",
                "letterSpacing": "0.05em",
            },
            style_data={
                "backgroundColor": "#161C27",
                "color": "#c8cdd8",
                "fontSize": "0.83rem",
                "border": "1px solid #232d3f",
            },
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"},
                {"if": {"state": "selected"}, "backgroundColor": "#1a3a6e", "border": "1px solid #4F8EF7"},
            ],
            style_filter={
                "backgroundColor": "#0e1117",
                "color": "#c8cdd8",
                "border": "1px solid #2d3a4f",
            },
            css=[{"selector": ".dash-filter input", "rule": "color: #c8cdd8 !important;"}],
        ),
        html.P(
            f"İlk 50 satır gösteriliyor  ·  Toplam aktif kayıt: {len(df_active):,}",
            style={"fontSize": "0.75rem", "color": "#7e8fa4", "marginTop": "0.5rem"},
        ),
        expert_panel,
        _build_screen_report(key, df_active, config, expert_excluded),
    ])


# ── Callback: Profiling Sekmesi ───────────────────────────────────────────────
@app.callback(
    Output("tab-profiling", "children"),
    Input("store-config", "data"),
    Input("dd-segment-val", "value"),
    State("store-key", "data"),
    State("dd-segment-col", "value"),
)
def update_profiling(config, seg_val, key, seg_col_input):
    df_orig = _get_df(key)
    if df_orig is None or not config or not config.get("target_col"):
        return html.Div()

    seg_col  = config.get("segment_col") or (seg_col_input or None)
    df_active = apply_segment_filter(df_orig, seg_col, seg_val)

    profile = compute_profile(df_active)   # local kopya üzerinde çalışır
    summary = profile_summary(profile, len(df_active))

    # ── Özet kartları ─────────────────────────────────────────────────────────
    def scard(value, label, color="#4F8EF7"):
        return dbc.Col(html.Div([
            html.Div(str(value), className="metric-value", style={"fontSize": "1.25rem", "color": color}),
            html.Div(label, className="metric-label"),
        ], className="metric-card"), width=2)

    summary_row = dbc.Row([
        scard(summary["total_cols"],      "Toplam Kolon"),
        scard(summary["numeric_cols"],    "Nümerik"),
        scard(summary["categorical_cols"],"Kategorik"),
        scard(summary["full_cols"],       "Tam Dolu",      "#10b981"),
        scard(summary["mid_missing"],     "Orta Eksik\n(5–50%)", "#f59e0b"),
        scard(summary["high_missing"],    "Yüksek Eksik\n(>50%)", "#ef4444"),
    ], className="g-3 mb-4")

    # ── Koşullu renklendirme ──────────────────────────────────────────────────
    cond_style = [
        {"if": {"filter_query": "{Eksik %} > 50", "column_id": "Eksik %"},
         "backgroundColor": "rgba(239,68,68,0.15)", "color": "#ef4444", "fontWeight": "700"},
        {"if": {"filter_query": "{Eksik %} > 5 && {Eksik %} <= 50", "column_id": "Eksik %"},
         "backgroundColor": "rgba(245,158,11,0.12)", "color": "#f59e0b", "fontWeight": "600"},
        {"if": {"filter_query": "{Eksik %} = 0", "column_id": "Eksik %"},
         "color": "#10b981"},
        {"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"},
    ]

    profile_tsv = profile.to_csv(sep="\t", index=False)
    profile_table = html.Div([
        html.Div(
            dcc.Clipboard(target_id="profile-tsv", title="Kopyala",
                          style={"cursor": "pointer", "fontSize": "0.72rem",
                                 "color": "#a8b2c2", "padding": "0.2rem 0.55rem",
                                 "border": "1px solid #2d3a4f", "borderRadius": "4px",
                                 "backgroundColor": "#1a2035", "float": "right",
                                 "marginBottom": "0.4rem", "letterSpacing": "0.04em"}),
            style={"overflow": "hidden"},
        ),
        html.Pre(profile_tsv, id="profile-tsv", style={"display": "none"}),
        dash_table.DataTable(
            data=profile.to_dict("records"),
            columns=[{"name": c, "id": c} for c in profile.columns],
            sort_action="native",
            filter_action="native",
            page_size=25,
            page_action="native",
            fixed_columns={"headers": True, "data": 1},
            style_table={"overflowX": "auto", "minWidth": "100%"},
            style_header={
                "backgroundColor": "#111827",
                "color": "#a8b2c2",
                "fontWeight": "700",
                "fontSize": "0.73rem",
                "border": "1px solid #2d3a4f",
                "textTransform": "uppercase",
                "letterSpacing": "0.06em",
            },
            style_data={
                "backgroundColor": "#161C27",
                "color": "#c8cdd8",
                "fontSize": "0.83rem",
                "border": "1px solid #232d3f",
            },
            style_data_conditional=cond_style,
            style_cell={"padding": "0.45rem 0.75rem", "textAlign": "left"},
            style_cell_conditional=[
                {"if": {"column_id": ["Dolu Sayı", "Eksik Sayı", "Eksik %",
                                      "Tekil Değer", "En Sık %",
                                      "Ortalama", "Std", "Min",
                                      "P1", "P5", "P10", "P25", "Medyan",
                                      "P75", "P90", "P95", "P99", "Max"]},
                 "textAlign": "right"},
            ],
            style_filter={
                "backgroundColor": "#0e1117",
                "color": "#c8cdd8",
                "border": "1px solid #2d3a4f",
            },
            css=[{"selector": ".dash-filter input", "rule": "color: #c8cdd8 !important;"}],
        ),
    ])

    return html.Div([
        html.P("Veri Profiling", className="section-title"),
        summary_row,
        profile_table,
        html.P(
            f"Renk kodları — Kırmızı: Eksik > %50  ·  Sarı: Eksik %5–50  ·  Yeşil: Tam Dolu",
            style={"fontSize": "0.73rem", "color": "#7e8fa4", "marginTop": "0.6rem"},
        ),
    ])


# ── Yardımcı: Ön Eleme Raporu ────────────────────────────────────────────────
def _build_screen_report(key, df_active, config, expert_excluded=None):
    # Aktif (segment filtrelenmiş) veri üzerinde canlı hesapla
    target_col  = config.get("target_col")
    date_col    = config.get("date_col")
    segment_col = config.get("segment_col")
    passed, report = screen_columns(
        df_active, target_col, date_col, segment_col
    )

    # Uzman görüşü elemeleri ekle
    if expert_excluded:
        already = set(report["Kolon"].tolist()) if not report.empty else set()
        new_rows = [
            {"Kolon": c, "Kural": "Uzman Görüşü", "Detay": "El ile elindi"}
            for c in expert_excluded if c not in already and c in df_active.columns
        ]
        if new_rows:
            report = pd.concat([report, pd.DataFrame(new_rows)], ignore_index=True)
        passed = [c for c in passed if c not in set(expert_excluded)]

    # Sonuçları cache'e yaz — diğer sekmeler (Deep Dive, Korelasyon) bu listeyi kullanır
    _SERVER_STORE[f"{key}_screen"] = (passed, report)

    # Konfigürasyon dışı tutulan kolonlar (target / tarih / segment)
    cfg_names = {c: lbl for c, lbl in [
        (target_col,  "Target"),
        (date_col,    "Tarih"),
        (segment_col, "Segment"),
    ] if c}
    n_config     = len(cfg_names)
    n_total      = df_active.shape[1]
    n_eliminated = len(report)
    n_kept      = len(passed)

    # ── Metrik bantı ─────────────────────────────────────────────────────────
    def badge(value, label, color="#4F8EF7"):
        return html.Div([
            html.Span(str(value),
                      style={"fontSize": "1.4rem", "fontWeight": "700",
                             "color": color, "display": "block",
                             "lineHeight": "1.1"}),
            html.Span(label,
                      style={"fontSize": "0.63rem", "fontWeight": "700",
                             "letterSpacing": "0.1em", "textTransform": "uppercase",
                             "color": "#7e8fa4", "marginTop": "0.2rem",
                             "display": "block"}),
        ], className="metric-card",
           style={"padding": "0.65rem 1rem", "minWidth": "90px",
                  "textAlign": "center"})

    metrics = html.Div([
        badge(n_total,      "Toplam"),
        badge(n_config,     "Konfigürasyon", "#4F8EF7"),
        badge(n_eliminated, "Elenen",
              "#ef4444" if n_eliminated else "#556070"),
        badge(n_kept,       "Analize Giren", "#10b981"),
    ], style={"display": "flex", "gap": "0.75rem",
              "flexWrap": "wrap", "marginBottom": "1rem"})

    # ── Konfigürasyon kolonu açıklaması ───────────────────────────────────────
    cfg_badges = html.Div([
        html.Span("Analiz dışı tutulanlar: ",
                  style={"color": "#7e8fa4", "fontSize": "0.73rem",
                         "marginRight": "0.3rem"}),
        *[html.Span(
            f"{lbl}: {col}",
            style={"color": "#4F8EF7", "fontSize": "0.73rem",
                   "marginRight": "0.75rem", "fontWeight": "600"},
        ) for col, lbl in cfg_names.items()],
    ], style={"marginBottom": "0.75rem"})

    # ── Eleme tablosu ─────────────────────────────────────────────────────────
    if report.empty:
        body = html.Div(
            "Kalite kontrolünden geçemeyen değişken bulunamadı.",
            className="form-hint",
            style={"padding": "0.6rem 0", "fontStyle": "normal",
                   "color": "#10b981"},
        )
    else:
        rule_cond = [
            {"if": {"filter_query": '{Kural} = "Yüksek Eksik"',
                    "column_id": "Kural"},
             "color": "#f59e0b", "fontWeight": "600"},
            {"if": {"filter_query": '{Kural} = "Sabit Değişken"',
                    "column_id": "Kural"},
             "color": "#ef4444", "fontWeight": "600"},
            {"if": {"filter_query": '{Kural} = "Uzman Görüşü"',
                    "column_id": "Kural"},
             "color": "#a78bfa", "fontWeight": "600"},
            {"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"},
        ]
        body = dash_table.DataTable(
            data=report.to_dict("records"),
            columns=[{"name": c, "id": c} for c in report.columns],
            sort_action="native",
            filter_action="native",
            page_size=20,
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#111827", "color": "#a8b2c2",
                          "fontWeight": "700", "fontSize": "0.72rem",
                          "border": "1px solid #2d3a4f",
                          "textTransform": "uppercase",
                          "letterSpacing": "0.06em"},
            style_data={"backgroundColor": "#161C27", "color": "#c8cdd8",
                        "fontSize": "0.82rem", "border": "1px solid #232d3f"},
            style_data_conditional=rule_cond,
            style_cell={"padding": "0.4rem 0.75rem", "textAlign": "left"},
            style_filter={"backgroundColor": "#0e1117", "color": "#c8cdd8",
                          "border": "1px solid #2d3a4f"},
            css=[{"selector": ".dash-filter input",
                  "rule": "color: #c8cdd8 !important;"}],
        )

    criteria = html.Div([
        html.Span("Elenme kriterleri — ",
                  style={"color": "#7e8fa4", "fontSize": "0.72rem"}),
        html.Span("> %80 eksik",
                  style={"color": "#f59e0b", "fontSize": "0.72rem",
                         "fontWeight": "600"}),
        html.Span("  ·  sabit değişken (1 tekil değer)",
                  style={"color": "#ef4444", "fontSize": "0.72rem",
                         "fontWeight": "600"}),
    ], style={"marginTop": "0.5rem"})

    return html.Div([
        html.Div(style={"borderTop": "1px solid #232d3f",
                        "marginTop": "2rem", "marginBottom": "1.25rem"}),
        html.P("Ön Eleme Raporu", className="section-title"),
        metrics,
        cfg_badges,
        body,
        criteria,
    ])


# ── Ortak grafik teması ───────────────────────────────────────────────────────
_PLOT_LAYOUT = dict(
    paper_bgcolor="#161C27",
    plot_bgcolor="#0E1117",
    font=dict(family="Inter, Segoe UI, sans-serif", color="#8892a4", size=11),
    margin=dict(l=40, r=20, t=40, b=40),
)

_AXIS_STYLE = dict(gridcolor="#232d3f", linecolor="#232d3f", zerolinecolor="#232d3f")


# ── Callback: Target & IV Sekmesi ─────────────────────────────────────────────
@app.callback(
    Output("tab-target-iv", "children"),
    Input("store-config", "data"),
    Input("dd-segment-val", "value"),
    State("store-key", "data"),
    State("dd-segment-col", "value"),
)
def update_target_iv(config, seg_val, key, seg_col_input):
    df_orig = _get_df(key)
    if df_orig is None or not config or not config.get("target_col"):
        return html.Div()

    target    = config["target_col"]
    date_col  = config.get("date_col")
    seg_col   = config.get("segment_col") or (seg_col_input or None)
    df_active = apply_segment_filter(df_orig, seg_col, seg_val)

    # ── IV cache: aynı key+segment için yeniden hesaplama yapma ──────────────
    cache_key = f"{key}_iv_{seg_col}_{seg_val}"
    if cache_key in _SERVER_STORE:
        iv_df = _SERVER_STORE[cache_key]
    else:
        iv_df = compute_iv_ranking_optimal(df_active, target)
        _SERVER_STORE[cache_key] = iv_df

    stats = compute_target_stats(df_active, target)

    # ── 1. Target İstatistik Kartları ─────────────────────────────────────────
    def tcard(value, label, color="#4F8EF7"):
        return dbc.Col(html.Div([
            html.Div(str(value), className="metric-value", style={"color": color}),
            html.Div(label, className="metric-label"),
        ], className="metric-card"), width=2)

    imbalance_color = "#ef4444" if stats["bad_rate"] < 5 or stats["bad_rate"] > 50 else "#f59e0b" if stats["bad_rate"] < 15 else "#10b981"

    stats_row = dbc.Row([
        tcard(f"{stats['valid']:,}",         "Geçerli Kayıt"),
        tcard(f"{stats['bad']:,}",           "Bad (1)",   "#ef4444"),
        tcard(f"{stats['good']:,}",          "Good (0)",  "#10b981"),
        tcard(f"%{stats['bad_rate']:.2f}",   "Bad Rate",  imbalance_color),
        tcard(f"{stats['ratio']:.1f}x",      "Good/Bad Oran"),
        tcard(f"{stats['missing']:,}",       "Target Eksik", "#f59e0b" if stats["missing"] > 0 else "#556070"),
    ], className="g-3 mb-4")

    # ── 2. Bad Rate Over Time (date_col varsa) ────────────────────────────────
    time_chart = html.Div()
    if date_col and date_col in df_active.columns:
        time_df = compute_target_over_time(df_active, target, date_col)
        if len(time_df) > 1:
            fig_time = go.Figure()
            fig_time.add_trace(go.Bar(
                x=time_df["Tarih"], y=time_df["total_count"],
                name="Toplam", marker_color="#232d4f", yaxis="y2", opacity=0.6,
            ))
            fig_time.add_trace(go.Scatter(
                x=time_df["Tarih"], y=time_df["bad_rate"],
                name="Bad Rate %", mode="lines+markers",
                line=dict(color="#ef4444", width=2),
                marker=dict(size=5),
            ))
            fig_time.update_layout(
                **_PLOT_LAYOUT,
                title=dict(text="Bad Rate Zaman Serisi", font=dict(color="#E8EAF0", size=13)),
                xaxis={**_AXIS_STYLE},
                yaxis=dict(**_AXIS_STYLE, title="Bad Rate %", ticksuffix="%"),
                yaxis2=dict(title="Kayıt Sayısı", overlaying="y", side="right",
                            showgrid=False),
                legend=dict(bgcolor="#161C27", bordercolor="#232d3f"),
                hovermode="x unified",
                height=320,
            )
            time_chart = html.Div([
                html.P("Bad Rate Trendi", className="section-title"),
                dcc.Graph(figure=fig_time, config={"displayModeBar": False}),
            ], style={"marginBottom": "2rem"})

    # ── 3. IV Ranking ─────────────────────────────────────────────────────────

    iv_color_map = {
        "Çok Zayıf": "#4a5568",
        "Zayıf":     "#f59e0b",
        "Orta":      "#4F8EF7",
        "Güçlü":     "#10b981",
        "Şüpheli":   "#ef4444",
    }

    # IV Bar chart (top 25)
    top_iv = iv_df.head(25).iloc[::-1]  # ters sıra — en yüksek üstte
    bar_colors = [iv_color_map.get(g, "#4F8EF7") for g in top_iv["Güç"]]

    fig_iv = go.Figure(go.Bar(
        x=top_iv["IV"],
        y=top_iv["Değişken"],
        orientation="h",
        marker_color=bar_colors,
        text=top_iv["IV"].apply(lambda x: f"{x:.4f}"),
        textposition="outside",
        textfont=dict(size=10, color="#8892a4"),
        hovertemplate="<b>%{y}</b><br>IV: %{x:.4f}<extra></extra>",
    ))
    fig_iv.update_layout(
        **_PLOT_LAYOUT,
        title=dict(text="IV Liderlik Tablosu (Top 25)", font=dict(color="#E8EAF0", size=13)),
        xaxis=dict(**_AXIS_STYLE, title="Information Value"),
        yaxis=dict(**_AXIS_STYLE, tickfont=dict(size=10)),
        height=max(400, len(top_iv) * 26),
        showlegend=False,
    )
    # Eşik çizgileri
    for thresh, label, color in [(0.02, "Zayıf", "#4a5568"), (0.10, "Orta", "#f59e0b"),
                                  (0.30, "Güçlü", "#10b981"), (0.50, "Şüpheli", "#ef4444")]:
        fig_iv.add_vline(x=thresh, line_dash="dot", line_color=color, opacity=0.5,
                         annotation_text=label, annotation_font_color=color,
                         annotation_font_size=9)

    # IV Tablo
    iv_cond = [
        {"if": {"filter_query": '{Güç} = "Güçlü"',    "column_id": "Güç"}, "color": "#10b981", "fontWeight": "700"},
        {"if": {"filter_query": '{Güç} = "Orta"',     "column_id": "Güç"}, "color": "#4F8EF7", "fontWeight": "600"},
        {"if": {"filter_query": '{Güç} = "Zayıf"',    "column_id": "Güç"}, "color": "#f59e0b"},
        {"if": {"filter_query": '{Güç} = "Şüpheli"',  "column_id": "Güç"}, "color": "#ef4444", "fontWeight": "700"},
        {"if": {"filter_query": '{Güç} = "Çok Zayıf"',"column_id": "Güç"}, "color": "#7e8fa4"},
        {"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"},
    ]

    iv_display = iv_df

    iv_tsv = iv_display.to_csv(sep="\t", index=False)
    iv_table = html.Div([
        html.Div(
            dcc.Clipboard(target_id="iv-tsv", title="Kopyala",
                          style={"cursor": "pointer", "fontSize": "0.72rem",
                                 "color": "#a8b2c2", "padding": "0.2rem 0.55rem",
                                 "border": "1px solid #2d3a4f", "borderRadius": "4px",
                                 "backgroundColor": "#1a2035", "float": "right",
                                 "marginBottom": "0.4rem", "letterSpacing": "0.04em"}),
            style={"overflow": "hidden"},
        ),
        html.Pre(iv_tsv, id="iv-tsv", style={"display": "none"}),
        dash_table.DataTable(
            data=iv_display.to_dict("records"),
            columns=[{"name": c, "id": c} for c in iv_display.columns],
            sort_action="native",
            filter_action="native",
            page_size=20,
            style_table={"overflowX": "auto"},
            style_header={
                "backgroundColor": "#111827", "color": "#a8b2c2",
                "fontWeight": "700", "fontSize": "0.73rem",
                "border": "1px solid #2d3a4f",
                "textTransform": "uppercase", "letterSpacing": "0.06em",
            },
            style_data={
                "backgroundColor": "#161C27", "color": "#c8cdd8",
                "fontSize": "0.83rem", "border": "1px solid #232d3f",
            },
            style_data_conditional=iv_cond,
            style_cell={"padding": "0.45rem 0.75rem"},
            style_cell_conditional=[
                {"if": {"column_id": ["IV", "Bin Sayısı", "Eksik %"]}, "textAlign": "right"},
            ],
            style_filter={"backgroundColor": "#0e1117", "color": "#c8cdd8", "border": "1px solid #2d3a4f"},
            css=[{"selector": ".dash-filter input", "rule": "color: #c8cdd8 !important;"}],
        ),
    ])

    return html.Div([
        html.P("Target Dağılımı", className="section-title"),
        stats_row,
        time_chart,
        dbc.Row([
            dbc.Col([
                html.P("IV Sıralaması", className="section-title"),
                dcc.Graph(figure=fig_iv, config={"displayModeBar": False}),
            ], width=6),
            dbc.Col([
                html.P("IV Tablosu", className="section-title"),
                iv_table,
            ], width=6),
        ]),
        html.Div([
            html.Span("IV Eşikleri: ", style={"color": "#7e8fa4", "fontSize": "0.73rem"}),
            html.Span("< 0.02 Çok Zayıf  · ", style={"color": "#7e8fa4", "fontSize": "0.73rem"}),
            html.Span("0.02–0.10 Zayıf  · ", style={"color": "#f59e0b", "fontSize": "0.73rem"}),
            html.Span("0.10–0.30 Orta  · ", style={"color": "#4F8EF7", "fontSize": "0.73rem"}),
            html.Span("0.30–0.50 Güçlü  · ", style={"color": "#10b981", "fontSize": "0.73rem"}),
            html.Span("> 0.50 Şüpheli", style={"color": "#ef4444", "fontSize": "0.73rem"}),
        ], style={"marginTop": "0.75rem"}),
    ])


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

    # PSI için date_col'dan distinct tarihler
    date_col = config.get("date_col")
    psi_date_col = html.Div()
    if date_col and date_col in df.columns:
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
        html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Değişken Seç", className="form-label"),
                    html.Div("\u00a0", className="form-hint"),
                    dbc.Select(id="dd-deepdive-col", options=col_options,
                               value=cols[0] if cols else None,
                               className="dark-select"),
                ], width=4),
                psi_date_col,
            ], className="mb-4"),
        ]),
        dcc.Loading(html.Div(id="deep-dive-content"), type="dot", color="#4F8EF7", delay_show=300),
        # Config'i aşağıya ilet
        dcc.Store(id="store-dd-config", data={
            "target_col": config["target_col"],
            "date_col":   config.get("date_col"),
            "seg_col":    config.get("segment_col") or (seg_col_input or None),
            "seg_val":    seg_val,
            "key":        key,
        }),
    ])


@app.callback(
    Output("deep-dive-content", "children"),
    Input("dd-deepdive-col", "value"),
    Input("dd-psi-split", "value"),
    State("store-dd-config", "data"),
    prevent_initial_call=False,
)
def render_deep_dive_content(col, psi_split, dd_config):
    if not col or not dd_config:
        return html.Div()

    df_orig = _get_df(dd_config["key"])
    if df_orig is None:
        return html.Div()

    target   = dd_config["target_col"]
    date_col = dd_config.get("date_col")
    seg_col  = dd_config.get("seg_col")
    seg_val  = dd_config.get("seg_val")
    df_active = apply_segment_filter(df_orig, seg_col, seg_val)

    vstats   = get_variable_stats(df_active, col, target)
    woe_df, iv_total_dd, _ = get_woe_detail(df_active, col, target)
    cutoff_date = psi_split if psi_split else None
    psi_res = compute_psi(
        df_active, col, target,
        date_col=date_col if date_col else None,
        cutoff_date=cutoff_date,
    )

    is_num = vstats["is_numeric"]

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
    if vstats["missing"] > 0 and vstats["missing_bad_rate"] is not None:
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
    if is_num:
        # Histogram — target sınıflarına göre renkli
        local = df_active[[col, target]].dropna(subset=[col]).copy()
        local[target] = local[target].astype(str)

        fig_dist = go.Figure()
        colors = {"0": "#4F8EF7", "1": "#ef4444"}
        for t_val, grp in local.groupby(target)[col]:
            fig_dist.add_trace(go.Histogram(
                x=grp, name=f"Target={t_val}",
                marker_color=colors.get(str(t_val), "#8892a4"),
                opacity=0.65, nbinsx=50,
                hovertemplate=f"Target={t_val}<br>Değer: %{{x}}<br>Sayı: %{{y}}<extra></extra>",
            ))

        # IQR sınır çizgileri
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
        # Kategorik — bar chart (value counts, target rengi)
        local = df_active[[col, target]].copy()
        local[col]    = local[col].fillna("Eksik").astype(str)
        local[target] = local[target].astype(float)
        top_cats = local[col].value_counts().head(20).index
        local = local[local[col].isin(top_cats)]
        vc = local.groupby([col, target]).size().reset_index(name="count")
        vc[target] = vc[target].astype(str)
        total = vc.groupby(col)["count"].transform("sum")
        vc["pct"] = (vc["count"] / total * 100).round(1)

        fig_dist = go.Figure()
        colors = {"0.0": "#4F8EF7", "1.0": "#ef4444"}
        for t_val, grp in vc.groupby(target):
            fig_dist.add_trace(go.Bar(
                x=grp[col], y=grp["count"],
                name=f"Target={t_val}",
                marker_color=colors.get(str(t_val), "#8892a4"),
                hovertemplate="%{x}<br>Sayı: %{y}<extra></extra>",
            ))
        fig_dist.update_layout(
            **_PLOT_LAYOUT,
            barmode="stack",
            title=dict(text=f"{col} — Değer Dağılımı (Top 20)", font=dict(color="#E8EAF0", size=13)),
            xaxis=dict(**_AXIS_STYLE, title=col, tickangle=-30),
            yaxis=dict(**_AXIS_STYLE, title="Frekans"),
            legend=dict(bgcolor="#161C27", bordercolor="#232d3f"),
            height=340,
        )
        dist_section = html.Div([
            dcc.Graph(figure=fig_dist, config={"displayModeBar": False}),
        ], className="mb-4")

    # ── 3. WOE / Bad Rate Grafiği ─────────────────────────────────────────────
    woe_section = html.Div()
    if not woe_df.empty:
        iv_total = iv_total_dd
        iv_label = ("Çok Zayıf" if iv_total < 0.02 else "Zayıf" if iv_total < 0.1
                    else "Orta" if iv_total < 0.3 else "Güçlü" if iv_total < 0.5 else "Şüpheli")
        iv_color = {"Çok Zayıf": "#4a5568", "Zayıf": "#f59e0b", "Orta": "#4F8EF7",
                    "Güçlü": "#10b981", "Şüpheli": "#ef4444"}.get(iv_label, "#4F8EF7")

        woe_chart_df = woe_df[woe_df["Bin"] != "TOPLAM"]

        # Monotonluk kontrolü
        _woe_vals = woe_chart_df["WOE"].values
        _br_vals  = woe_chart_df["Bad Rate %"].values
        def _monotone_label(arr):
            diffs = [arr[i+1] - arr[i] for i in range(len(arr)-1)]
            if all(d >= 0 for d in diffs): return "↑ Monoton Artan", "#10b981"
            if all(d <= 0 for d in diffs): return "↓ Monoton Azalan", "#10b981"
            return "✗ Monoton Değil", "#ef4444"
        woe_mono_txt, woe_mono_clr = _monotone_label(_woe_vals)
        br_mono_txt,  br_mono_clr  = _monotone_label(_br_vals)

        fig_woe = go.Figure()
        fig_woe.add_trace(go.Bar(
            x=woe_chart_df["Bin"], y=woe_chart_df["Bad Rate %"],
            name="Bad Rate %", marker_color="#ef4444", opacity=0.75,
            hovertemplate="Bin: %{x}<br>Bad Rate: %{y:.2f}%<extra></extra>",
        ))
        fig_woe.add_trace(go.Scatter(
            x=woe_chart_df["Bin"], y=woe_chart_df["WOE"],
            name="WOE", mode="lines+markers",
            line=dict(color="#4F8EF7", width=2), marker=dict(size=6),
            yaxis="y2",
            hovertemplate="Bin: %{x}<br>WOE: %{y:.4f}<extra></extra>",
        ))
        fig_woe.add_hline(y=0, line_dash="dot", line_color="#556070", opacity=0.5, yref="y2")
        fig_woe.update_layout(
            **_PLOT_LAYOUT,
            title=dict(
                text=f"{col} — WOE & Bad Rate  |  IV: {iv_total:.4f}  [{iv_label}]",
                font=dict(color=iv_color, size=13),
            ),
            xaxis=dict(**_AXIS_STYLE, tickangle=-30),
            yaxis=dict(**_AXIS_STYLE, title="Bad Rate %", ticksuffix="%"),
            yaxis2=dict(title="WOE", overlaying="y", side="right", showgrid=False,
                        zeroline=True, zerolinecolor="#556070"),
            legend=dict(bgcolor="#161C27", bordercolor="#232d3f"),
            height=340,
        )

        woe_tsv = woe_df.to_csv(sep="\t", index=False)
        woe_table = html.Div([
            html.Div(
                dcc.Clipboard(target_id="woe-tsv", title="Kopyala",
                              style={"cursor": "pointer", "fontSize": "0.72rem",
                                     "color": "#a8b2c2", "padding": "0.2rem 0.55rem",
                                     "border": "1px solid #2d3a4f", "borderRadius": "4px",
                                     "backgroundColor": "#1a2035", "float": "right",
                                     "marginBottom": "0.4rem", "letterSpacing": "0.04em"}),
                style={"overflow": "hidden"},
            ),
            html.Pre(woe_tsv, id="woe-tsv", style={"display": "none"}),
            dash_table.DataTable(
                data=woe_df.to_dict("records"),
                columns=[{"name": c, "id": c} for c in woe_df.columns],
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
        ])

        mono_badges = html.Div([
            html.Span("WOE ", style={"color": "#7e8fa4", "fontSize": "0.72rem"}),
            html.Span(woe_mono_txt, style={"color": woe_mono_clr, "fontSize": "0.72rem", "fontWeight": "700", "marginRight": "1rem"}),
            html.Span("Bad Rate ", style={"color": "#7e8fa4", "fontSize": "0.72rem"}),
            html.Span(br_mono_txt,  style={"color": br_mono_clr,  "fontSize": "0.72rem", "fontWeight": "700"}),
        ], style={"marginBottom": "0.6rem"})

        woe_section = html.Div([
            html.P("WOE & Bad Rate Analizi", className="section-title"),
            mono_badges,
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_woe, config={"displayModeBar": False}), width=8),
                dbc.Col(woe_table, width=4),
            ]),
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


# ── Callback: Korelasyon ─────────────────────────────────────────────────────
@app.callback(
    Output("corr-content", "children"),
    Input("store-config", "data"),
    Input("dd-segment-val", "value"),
    Input("corr-threshold", "value"),
    Input("corr-max-cols", "value"),
    Input("store-expert-exclude", "data"),
    State("store-key", "data"),
    State("dd-segment-col", "value"),
)
def render_correlation_content(config, seg_val, threshold, max_cols_str, expert_excluded, key, seg_col_input):
    if not key or not config or not config.get("target_col"):
        return html.Div()

    df_orig = _get_df(key)
    if df_orig is None:
        return html.Div()

    threshold  = float(threshold or 0.75)
    max_cols   = int(max_cols_str or 20)
    target     = config["target_col"]
    seg_col    = config.get("segment_col") or (seg_col_input or None)
    df_active  = apply_segment_filter(df_orig, seg_col, seg_val)
    expert_excluded_set = set(expert_excluded or [])

    # ── Cache: Korelasyon Matrisi ─────────────────────────────────────────────
    cache_key = f"{key}_corr_{seg_col}_{seg_val}_{max_cols}"
    if cache_key in _SERVER_STORE:
        corr_df, cols = _SERVER_STORE[cache_key]
        cols = [c for c in cols if c not in expert_excluded_set]
        corr_df = corr_df.loc[cols, cols] if cols else corr_df
    else:
        excl = [c for c in [target, config.get("date_col")] if c]
        cols = get_numeric_cols(df_active, exclude=excl, max_cols=max_cols)
        screen_result = _SERVER_STORE.get(f"{key}_screen")
        if screen_result:
            passed_set = set(screen_result[0])
            cols = [c for c in cols if c in passed_set]
        cols = [c for c in cols if c not in expert_excluded_set]
        if len(cols) < 2:
            return html.Div("Yeterli sayıda numerik kolon bulunamadı (en az 2 gerekli).",
                            className="alert-info-custom")
        corr_df = compute_correlation_matrix(df_active, cols)
        _SERVER_STORE[cache_key] = (corr_df, cols)

    # ── VIF: sadece IV ≥ 0.10 olan değişkenler ───────────────────────────────
    iv_cache_key = f"{key}_iv_{seg_col}_{seg_val}"
    vif_cols = cols
    iv_filtered = False
    if iv_cache_key in _SERVER_STORE:
        iv_df_cached = _SERVER_STORE[iv_cache_key]
        iv_high = set(iv_df_cached[iv_df_cached["IV"] >= 0.10]["Değişken"].tolist())
        filtered = [c for c in cols if c in iv_high]
        if len(filtered) >= 2:
            vif_cols = filtered
            iv_filtered = True

    vif_cache_key = f"{key}_vif_{seg_col}_{seg_val}_{max_cols}"
    if vif_cache_key in _SERVER_STORE:
        vif_df = _SERVER_STORE[vif_cache_key]
        # iv_filtered durumunu cache'den türet
        iv_filtered = "En Benzer" in vif_df.columns if not vif_df.empty else iv_filtered
    else:
        vif_df = compute_vif(df_active, vif_cols) if len(vif_cols) >= 2 else pd.DataFrame()

        # "En Benzer" kolonu: hangi değişkenle en yüksek korelasyon
        if not vif_df.empty:
            try:
                sub = corr_df.loc[
                    [v for v in vif_cols if v in corr_df.index],
                    [v for v in vif_cols if v in corr_df.columns],
                ]
                en_benzer = []
                for var in vif_df["Değişken"]:
                    if var not in sub.columns:
                        en_benzer.append("—")
                        continue
                    row = sub[var].drop(var, errors="ignore")
                    if row.empty:
                        en_benzer.append("—")
                        continue
                    top = row.abs().idxmax()
                    en_benzer.append(f"{top}  (r = {row[top]:+.3f})")
                vif_df = vif_df.copy()
                vif_df.insert(2, "En Benzer", en_benzer)
            except Exception:
                pass

        _SERVER_STORE[vif_cache_key] = vif_df

    # ── 1. Heatmap ────────────────────────────────────────────────────────────
    show_text = len(cols) <= 18
    fig_heat = go.Figure(go.Heatmap(
        z=corr_df.values,
        x=cols, y=cols,
        colorscale="RdBu",
        zmid=0, zmin=-1, zmax=1,
        text=corr_df.round(2).values if show_text else None,
        texttemplate="%{text}" if show_text else None,
        textfont=dict(size=9),
        colorbar=dict(
            title=dict(text="r", font=dict(color="#8892a4", size=11)),
            thickness=12, len=0.8,
            tickfont=dict(color="#8892a4", size=10),
        ),
        hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>r = %{z:.4f}<extra></extra>",
    ))
    cell_px = max(18, min(40, 600 // max(len(cols), 1)))
    fig_heat.update_layout(
        **_PLOT_LAYOUT,
        title=dict(text=f"Korelasyon Matrisi  ({len(cols)} değişken)",
                   font=dict(color="#E8EAF0", size=13)),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9, color="#8892a4"),
                   showgrid=False, linecolor="#232d3f"),
        yaxis=dict(tickfont=dict(size=9, color="#8892a4"),
                   showgrid=False, linecolor="#232d3f", autorange="reversed"),
        height=max(400, len(cols) * cell_px + 100),
    )
    fig_heat.update_layout(margin=dict(l=120, r=40, t=50, b=120))

    # ── 2. Tüm çiftler tablosu (|r| büyükten küçüğe) ─────────────────────────
    all_pairs = find_high_corr_pairs(corr_df, threshold=0.0)

    # ── 3. Korelasyon Çifti dropdownları ──────────────────────────────────────
    var_opts = [{"label": c, "value": c} for c in cols]
    default2 = cols[1] if len(cols) > 1 else cols[0]
    try:
        init_r = float(df_active[[cols[0], default2]].corr().iloc[0, 1])
    except Exception:
        init_r = float("nan")

    # ── 4. VIF Tablosu ────────────────────────────────────────────────────────
    vif_section = html.Div()
    if vif_df is not None and not vif_df.empty:
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
        vif_tsv = vif_df.to_csv(sep="\t", index=False)
        iv_note = (
            html.Div("IV ≥ 0.10 olan değişkenler üzerinden hesaplandı",
                     className="form-hint", style={"marginBottom": "0.4rem"})
            if iv_filtered else html.Div()
        )
        vif_section = html.Div([
            html.Div(
                dcc.Clipboard(target_id="vif-tsv", title="Kopyala",
                              style={"cursor": "pointer", "fontSize": "0.72rem",
                                     "color": "#a8b2c2", "padding": "0.2rem 0.55rem",
                                     "border": "1px solid #2d3a4f", "borderRadius": "4px",
                                     "backgroundColor": "#1a2035", "float": "right",
                                     "marginBottom": "0.4rem"}),
                style={"overflow": "hidden"},
            ),
            html.Pre(vif_tsv, id="vif-tsv", style={"display": "none"}),
            iv_note,
            dash_table.DataTable(
                data=vif_df.to_dict("records"),
                columns=[{"name": c, "id": c} for c in vif_df.columns],
                sort_action="native",
                page_size=20,
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
                    {"if": {"column_id": "En Benzer"},
                     "color": "#a8b2c2", "fontSize": "0.78rem"},
                ],
            ),
        ])
    elif iv_filtered is False and not (vif_df is not None and not vif_df.empty):
        vif_section = html.Div(
            "Target & IV sekmesini açarak IV hesaplatın — VIF, IV ≥ 0.10 değişkenler için otomatik filtrelenir.",
            className="form-hint",
        )

    # ── Legend notu ───────────────────────────────────────────────────────────
    legend = html.Div([
        html.Span("Korelasyon: ", style={"color": "#7e8fa4", "fontSize": "0.73rem"}),
        html.Span("Mavi = Pozitif  · ", style={"color": "#4F8EF7", "fontSize": "0.73rem"}),
        html.Span("Kırmızı = Negatif  · ", style={"color": "#ef4444", "fontSize": "0.73rem"}),
        html.Span("VIF Eşikleri: ", style={"color": "#7e8fa4", "fontSize": "0.73rem",
                                           "marginLeft": "1rem"}),
        html.Span("< 5 Normal  · ", style={"color": "#10b981", "fontSize": "0.73rem"}),
        html.Span("5–10 Orta  · ", style={"color": "#f59e0b", "fontSize": "0.73rem"}),
        html.Span("> 10 Yüksek", style={"color": "#ef4444", "fontSize": "0.73rem"}),
    ], style={"marginTop": "0.5rem", "marginBottom": "1.5rem"})

    # ── Pairs tablosu koşullu renk ────────────────────────────────────────────
    pairs_cond = [
        {"if": {"filter_query": "{|Korelasyon|} >= 0.9",  "column_id": "|Korelasyon|"},
         "color": "#ef4444", "fontWeight": "700"},
        {"if": {"filter_query": "{|Korelasyon|} >= 0.75 && {|Korelasyon|} < 0.9",
                "column_id": "|Korelasyon|"},
         "color": "#f59e0b", "fontWeight": "600"},
        {"if": {"filter_query": "{|Korelasyon|} >= 0.5 && {|Korelasyon|} < 0.75",
                "column_id": "|Korelasyon|"},
         "color": "#4F8EF7"},
        {"if": {"filter_query": "{Korelasyon} < 0", "column_id": "Korelasyon"},
         "color": "#ef4444"},
        {"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"},
    ]
    pairs_tsv = all_pairs.to_csv(sep="\t", index=False)

    return html.Div([
        html.P("Korelasyon Matrisi", className="section-title"),
        dcc.Graph(figure=fig_heat, config={"displayModeBar": False}),
        legend,

        # ── Çiftler tablosu + VIF ──────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                html.P("Korelasyon Çiftleri", className="section-title"),
                html.Div(
                    dcc.Clipboard(target_id="pairs-tsv", title="Kopyala",
                                  style={"cursor": "pointer", "fontSize": "0.72rem",
                                         "color": "#a8b2c2", "padding": "0.2rem 0.55rem",
                                         "border": "1px solid #2d3a4f", "borderRadius": "4px",
                                         "backgroundColor": "#1a2035", "float": "right",
                                         "marginBottom": "0.4rem"}),
                    style={"overflow": "hidden"},
                ),
                html.Pre(pairs_tsv, id="pairs-tsv", style={"display": "none"}),
                dash_table.DataTable(
                    data=all_pairs.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in all_pairs.columns],
                    sort_action="native",
                    page_size=15,
                    style_table={"overflowX": "auto"},
                    style_header={"backgroundColor": "#111827", "color": "#a8b2c2",
                                  "fontWeight": "700", "fontSize": "0.72rem",
                                  "border": "1px solid #2d3a4f",
                                  "textTransform": "uppercase"},
                    style_data={"backgroundColor": "#161C27", "color": "#c8cdd8",
                                "fontSize": "0.82rem", "border": "1px solid #232d3f"},
                    style_data_conditional=pairs_cond,
                    style_cell={"padding": "0.4rem 0.65rem"},
                    style_cell_conditional=[
                        {"if": {"column_id": ["Korelasyon", "|Korelasyon|"]},
                         "textAlign": "right"},
                    ],
                ),
                html.Div([
                    html.Span("|r| renk kodları — ",
                              style={"color": "#7e8fa4", "fontSize": "0.72rem"}),
                    html.Span("≥ 0.90 ", style={"color": "#ef4444", "fontSize": "0.72rem", "fontWeight": "700"}),
                    html.Span("≥ 0.75 ", style={"color": "#f59e0b", "fontSize": "0.72rem", "fontWeight": "600"}),
                    html.Span("≥ 0.50 ", style={"color": "#4F8EF7", "fontSize": "0.72rem"}),
                    html.Span("  ·  Negatif r kırmızı",
                              style={"color": "#ef4444", "fontSize": "0.72rem",
                                     "marginLeft": "0.5rem"}),
                ], style={"marginTop": "0.4rem"}),
            ], width=6),
            dbc.Col([
                html.P("VIF — Çoklu Doğrusallık", className="section-title"),
                vif_section,
            ], width=6),
        ], className="mb-4"),

        # ── İkili Analiz ───────────────────────────────────────────────────────
        html.P("İkili Değişken Analizi", className="section-title"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Değişken 1", className="form-label"),
                dbc.Select(id="corr-var1", options=var_opts, value=cols[0],
                           className="dark-select"),
            ], width=5),
            dbc.Col([
                dbc.Label("Değişken 2", className="form-label"),
                dbc.Select(id="corr-var2", options=var_opts, value=default2,
                           className="dark-select"),
            ], width=5),
            dbc.Col([
                dbc.Label("\u00a0", className="form-label"),
                html.Div(id="corr-r-badge", children=_make_r_badge(init_r)),
            ], width=2, style={"display": "flex", "alignItems": "flex-end"}),
        ], className="mb-3", align="end"),
        dcc.Loading(
            html.Div(id="pair-scatter",
                     children=_safe_pair_scatter(df_active, cols[0], default2, target)),
            type="dot", color="#4F8EF7", delay_show=250,
        ),
    ])


# ── Callback: Çift Scatter ────────────────────────────────────────────────────
@app.callback(
    Output("pair-scatter", "children"),
    Output("corr-r-badge", "children"),
    Input("corr-var1", "value"),
    Input("corr-var2", "value"),
    State("store-key", "data"),
    State("store-config", "data"),
    State("dd-segment-val", "value"),
    prevent_initial_call=True,
)
def render_pair_scatter(var1, var2, key, config, seg_val):
    if not var1 or not var2 or not key or not config:
        return html.Div(), html.Div()
    df_orig = _get_df(key)
    if df_orig is None:
        return html.Div(), html.Div()
    seg_col   = config.get("segment_col")
    target    = config["target_col"]
    df_active = apply_segment_filter(df_orig, seg_col, seg_val)

    is_num1 = pd.api.types.is_numeric_dtype(df_active[var1]) if var1 in df_active.columns else False
    is_num2 = pd.api.types.is_numeric_dtype(df_active[var2]) if var2 in df_active.columns else False

    if is_num1 and is_num2:
        try:
            r = float(df_active[[var1, var2]].corr().iloc[0, 1])
        except Exception:
            r = float("nan")
        r_badge = _make_r_badge(r)
    else:
        r_badge = html.Div(
            html.Div("Kategorik × Sayısal",
                     style={"fontSize": "0.72rem", "color": "#7e8fa4",
                            "fontWeight": "600", "letterSpacing": "0.06em",
                            "textTransform": "uppercase"}),
            className="metric-card",
            style={"padding": "0.55rem 1rem", "minWidth": "120px",
                   "textAlign": "center"},
        )

    try:
        pair_chart = _make_pair_scatter(df_active, var1, var2, target)
    except Exception as exc:
        pair_chart = html.Div(
            f"Grafik oluşturulamadı: {exc}",
            style={"color": "#ef4444", "padding": "1rem", "fontSize": "0.8rem"},
        )
    return pair_chart, r_badge


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


# ── Render: KS Testi ─────────────────────────────────────────────────────────
def _render_ks(df_active: pd.DataFrame, var_col: str, target: str) -> html.Div:
    col_data = df_active[[var_col, target]].dropna()
    col_data = col_data[pd.api.types.is_numeric_dtype(col_data[var_col]) |
                        col_data[var_col].apply(lambda x: isinstance(x, (int, float)))]
    try:
        col_data[var_col] = pd.to_numeric(col_data[var_col], errors="coerce")
    except Exception:
        pass
    col_data = col_data.dropna()

    good = col_data[col_data[target] == 0][var_col].values
    bad  = col_data[col_data[target] == 1][var_col].values

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
            html.Div("Good (0) N", className="metric-label"),
            html.Div(f"{len(good):,}", className="metric-value"),
        ], className="metric-card"), width=3),
        dbc.Col(html.Div([
            html.Div("Bad (1) N", className="metric-label"),
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
        name="Good (0)", line=dict(color="#4F8EF7", width=2),
    ))
    fig_cdf.add_trace(go.Scatter(
        x=b_plot, y=b_cdf, mode="lines",
        name="Bad (1)", line=dict(color="#ef4444", width=2),
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
    seg_col   = config.get("segment_col") or (seg_col_input or None)
    df_active = apply_segment_filter(df_orig, seg_col, seg_val)
    max_cats  = int(max_cats_str or 15)
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
    seg_col   = config.get("segment_col") or (seg_col_input or None)
    df_active = apply_segment_filter(df_orig, seg_col, seg_val)
    target    = config.get("target_col")
    if not target:
        return html.Div("Config'de target kolonu tanımlanmamış.", className="alert-info-custom")
    try:
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
    seg_col   = config.get("segment_col") or (seg_col_input or None)
    df_active = apply_segment_filter(df_orig, seg_col, seg_val)
    target    = config.get("target_col")
    if not target:
        return html.Div("Config'de target kolonu tanımlanmamış.", className="alert-info-custom")
    try:
        return _render_ks(df_active, var_col, target)
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


# ── Callback: Sidebar Toggle ──────────────────────────────────────────────────
@app.callback(
    Output("col-sidebar", "style"),
    Output("col-main", "style"),
    Output("sidebar", "style"),
    Output("btn-sidebar-toggle", "children"),
    Input("btn-sidebar-toggle", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_sidebar(n):
    if n and n % 2 == 1:   # tek tıkla kapat
        return _COL_SIDEBAR_CLOSED, _COL_MAIN_CLOSED, _SIDEBAR_CLOSED_STYLE, "›"
    return _COL_SIDEBAR_OPEN, _COL_MAIN_OPEN, _SIDEBAR_OPEN_STYLE, "‹"


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


# ── Callback: Uzman Görüşü — Listeye Ekle ─────────────────────────────────────
@app.callback(
    Output("store-expert-exclude", "data"),
    Input("btn-expert-add", "n_clicks"),
    State("chk-expert-cols", "value"),
    State("store-expert-exclude", "data"),
    prevent_initial_call=True,
)
def add_expert_exclusions(n_clicks, selected, current):
    if not selected:
        return dash.no_update
    current = current or []
    new = [c for c in selected if c not in set(current)]
    return current + new


# ── Callback: Uzman Görüşü — Listeyi Temizle ──────────────────────────────────
@app.callback(
    Output("store-expert-exclude", "data", allow_duplicate=True),
    Input("btn-expert-clear", "n_clicks"),
    prevent_initial_call=True,
)
def clear_expert_exclusions(_):
    return []


# ── Callback: Segment Badge ────────────────────────────────────────────────────
@app.callback(
    Output("segment-badge-area", "children"),
    Input("dd-segment-val", "value"),
    State("store-config", "data"),
)
def update_segment_badge(val, config):
    if config and config.get("segment_col") and val and val != "Tümü":
        return html.Span(
            f"{config['segment_col']}: {val}",
            className="segment-badge",
        )
    return html.Div()


# ── Playground: Kolon seçeneklerini doldur ────────────────────────────────────
@app.callback(
    Output("pg-x-col",     "options"),
    Output("pg-x-col",     "value"),
    Output("pg-y-col",     "options"),
    Output("pg-y-col",     "value"),
    Output("pg-y2-col",    "options"),
    Output("pg-y2-col",    "value"),
    Output("pg-color-col", "options"),
    Output("pg-color-col", "value"),
    Input("store-config", "data"),
    Input("store-expert-exclude", "data"),
    State("store-key", "data"),
)
def populate_pg_cols(config, expert_excluded, key):
    df = _get_df(key)
    if df is None or not config:
        empty = []
        return empty, None, empty, None, empty, "", empty, ""
    # Tüm kolonlar dahil (tarih, target, segment dahil)
    all_cols   = list(df.columns)
    opts       = [{"label": c, "value": c} for c in all_cols]
    y2_opts    = [{"label": "—", "value": ""}] + opts
    color_opts = [{"label": "—", "value": ""}] + opts

    date_col   = config.get("date_col")
    target_col = config.get("target_col")
    numeric_cols = [c for c in all_cols
                    if pd.api.types.is_numeric_dtype(df[c])
                    and c != target_col and c != date_col]

    # Akıllı defaults: X=tarih, Y=ilk numerik değişken, Y2=target
    x_val  = date_col  if date_col  in all_cols else (all_cols[0] if all_cols else None)
    y_val  = numeric_cols[0] if numeric_cols else (all_cols[1] if len(all_cols) > 1 else x_val)
    y2_val = target_col if target_col in all_cols else ""
    return opts, x_val, opts, y_val, y2_opts, y2_val, color_opts, ""


# ── Playground: Target kolonu ve kesim tarihi doldur ──────────────────────────
@app.callback(
    Output("pg-target-col", "options"),
    Output("pg-target-col", "value"),
    Output("pg-split-date", "options"),
    Output("pg-split-date", "value"),
    Input("store-config", "data"),
    State("store-key", "data"),
)
def populate_pg_model_params(config, key):
    empty = ([], None, [], None)
    if not config or not key:
        return empty
    target_col = config.get("target_col")
    date_col   = config.get("date_col")
    df = _get_df(key)

    # Target seçenekleri: numerik kolonlar
    if df is not None:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        target_opts = [{"label": c, "value": c} for c in num_cols]
        target_val  = target_col if target_col in num_cols else (num_cols[0] if num_cols else None)
    else:
        target_opts = [{"label": target_col, "value": target_col}] if target_col else []
        target_val  = target_col

    # Kesim tarihi seçenekleri: tarih kolonundan aylık distinct değerler
    split_opts = []
    split_val  = None
    if df is not None and date_col and date_col in df.columns:
        raw_dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        distinct  = sorted(raw_dates.dt.to_period("M").unique().astype(str))
        split_opts = [{"label": d, "value": d} for d in distinct]
        # Varsayılan: ortadaki ay (son %30 test)
        split_val  = distinct[int(len(distinct) * 0.7)] if distinct else None

    return target_opts, target_val, split_opts, split_val


# ── Playground: Grafik çiz ────────────────────────────────────────────────────
@app.callback(
    Output("pg-chart-output", "children"),
    Input("btn-pg-chart", "n_clicks"),
    State("pg-x-col",       "value"),
    State("pg-y-col",       "value"),
    State("pg-chart-type",  "value"),
    State("pg-agg",         "value"),
    State("pg-color-col",   "value"),
    State("pg-y2-col",      "value"),
    State("pg-time-unit",   "value"),
    State("store-key",      "data"),
    State("store-config",   "data"),
    State("dd-segment-val", "value"),
    State("dd-segment-col", "value"),
    prevent_initial_call=True,
)
def _render_pg_chart(n, x_col, y_col, chart_type, agg, color_col,
                     y2_col, time_unit, key, config, seg_val, seg_col_input):
    if not x_col or not key or not config:
        return html.Div()
    df_orig = _get_df(key)
    if df_orig is None:
        return html.Div()
    seg_col = config.get("segment_col") or (seg_col_input or None)
    df = apply_segment_filter(df_orig, seg_col, seg_val).copy()
    color = color_col if color_col else None
    y2    = y2_col    if y2_col    else None

    # ── Tarih gruplama: X tarih kolonu ise period'a dönüştür ─────────────────
    _DATE_UNIT_FMT = {"D": "%Y-%m-%d", "M": "%Y-%m", "Q": None, "Y": "%Y"}
    _DATE_UNIT_LABEL = {"D": "Gün", "M": "Ay", "Q": "Çeyrek", "Y": "Yıl"}
    x_is_date = False
    x_display_col = x_col   # gruplanmış kolon adı (df'ye eklenir)
    if x_col in df.columns:
        try:
            parsed = pd.to_datetime(df[x_col], errors="coerce")
            if parsed.notna().mean() > 0.5:          # çoğunluğu tarih
                unit = time_unit or "M"
                if unit == "Q":
                    df["__x_period__"] = parsed.dt.to_period("Q").astype(str)
                else:
                    df["__x_period__"] = parsed.dt.strftime(_DATE_UNIT_FMT[unit])
                x_display_col = "__x_period__"
                x_is_date = True
        except Exception:
            pass

    def _agg_series(frame, grp_cols, col):
        is_numeric = pd.api.types.is_numeric_dtype(frame[col]) if col in frame.columns else False
        effective_agg = agg if (agg == "count" or is_numeric) else "count"
        if effective_agg == "count":
            res = frame.groupby(grp_cols, observed=True).size().reset_index(name=col)
        elif effective_agg == "sum":
            res = frame.groupby(grp_cols, observed=True)[col].sum().reset_index()
        else:
            res = frame.groupby(grp_cols, observed=True)[col].mean().reset_index()
        return res

    def _agg_bad_rate(frame, grp_cols, col):
        """Y2 için özel: kolon binary (0/1) ise bad rate (%), değilse mean."""
        is_bin = (pd.api.types.is_numeric_dtype(frame[col])
                  and frame[col].dropna().isin([0, 1]).all())
        if is_bin:
            res = frame.groupby(grp_cols, observed=True)[col].mean().reset_index()
            res[col] = (res[col] * 100).round(2)   # 0-100 ölçeği
            return res, True
        else:
            return _agg_series(frame, grp_cols, col), False

    time_note  = f" — {_DATE_UNIT_LABEL.get(time_unit or 'M', '')} bazında" if x_is_date else ""
    tick_angle = -35 if x_is_date else 0

    def _y2_trace(frame, x_c, col):
        """Y2 için sağ eksende çizgi izi döndür."""
        d, is_br = _agg_bad_rate(frame, [x_c], col)
        label    = f"{col} (Bad Rate %)" if is_br else col
        y_title  = "Bad Rate %" if is_br else col
        trace = go.Scatter(
            x=d[x_c], y=d[col],
            name=label, mode="lines+markers",
            line=dict(color="#f59e0b", width=2),
            marker=dict(size=5),
        )
        return trace, y_title

    def _dual_layout(fig, y1_title, y2_title, title_txt):
        fig.update_layout(
            **_PLOT_LAYOUT,
            title=dict(text=title_txt, font=dict(color="#E8EAF0", size=13)),
            height=440,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8892a4", size=10)),
        )
        fig.update_xaxes(**_AXIS_STYLE, tickangle=tick_angle)
        fig.update_yaxes(**_AXIS_STYLE, title_text=y1_title, secondary_y=False)
        fig.update_yaxes(**_AXIS_STYLE, title_text=y2_title, secondary_y=True)

    try:
        # ── Scatter / Histogram / Box: Y2 desteklenmiyor, tek eksen ──────────
        if chart_type == "scatter":
            fig = px.scatter(df, x=x_display_col, y=y_col, color=color, opacity=0.6,
                             color_discrete_sequence=px.colors.qualitative.Set2)
        elif chart_type == "histogram":
            fig = px.histogram(df, x=x_display_col, color=color, nbins=40,
                               color_discrete_sequence=px.colors.qualitative.Set2)
        elif chart_type == "box":
            fig = px.box(df, x=color or x_display_col, y=y_col,
                         color_discrete_sequence=px.colors.qualitative.Set2)

        # ── Bar+Line: Y1=bar(sol), Y2=çizgi(sağ) — her zaman ─────────────────
        elif chart_type == "bar_line":
            grp   = [x_display_col] + ([color] if color and not y2 else [])
            y1_df = _agg_series(df, grp, y_col)
            fig   = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Bar(x=y1_df[x_display_col], y=y1_df[y_col],
                       name=y_col, marker_color="#4F8EF7", opacity=0.85),
                secondary_y=False)
            y2_title = ""
            if y2:
                tr, y2_title = _y2_trace(df, x_display_col, y2)
                fig.add_trace(tr, secondary_y=True)
            title_txt = f"{x_col}{time_note}  ·  {y_col}" + (f"  /  {y2}" if y2 else "")
            _dual_layout(fig, y_col, y2_title, title_txt)
            return dcc.Graph(figure=fig, config={"displayModeBar": False})

        # ── Bar veya Line: Y2 varsa → secondary Y axis, yoksa → tek eksen ────
        elif chart_type in ("bar", "line") and y2:
            y1_df = _agg_series(df, [x_display_col], y_col)
            fig   = make_subplots(specs=[[{"secondary_y": True}]])
            if chart_type == "bar":
                fig.add_trace(
                    go.Bar(x=y1_df[x_display_col], y=y1_df[y_col],
                           name=y_col, marker_color="#4F8EF7", opacity=0.85),
                    secondary_y=False)
            else:
                fig.add_trace(
                    go.Scatter(x=y1_df[x_display_col], y=y1_df[y_col],
                               name=y_col, mode="lines+markers",
                               line=dict(color="#4F8EF7", width=2),
                               marker=dict(size=5)),
                    secondary_y=False)
            tr, y2_title = _y2_trace(df, x_display_col, y2)
            fig.add_trace(tr, secondary_y=True)
            title_txt = f"{x_col}{time_note}  ·  {y_col}  /  {y2}"
            _dual_layout(fig, y_col, y2_title, title_txt)
            return dcc.Graph(figure=fig, config={"displayModeBar": False})

        # ── Bar veya Line: Y2 yok → renk/grup destekli tek eksen ─────────────
        else:
            grp_cols = [x_display_col] + ([color] if color else [])
            agg_df   = _agg_series(df, grp_cols, y_col)
            if chart_type == "bar":
                fig = px.bar(agg_df, x=x_display_col, y=y_col, color=color, barmode="group",
                             color_discrete_sequence=px.colors.qualitative.Set2)
            else:
                fig = px.line(agg_df, x=x_display_col, y=y_col, color=color, markers=True,
                              color_discrete_sequence=px.colors.qualitative.Set2)

        x_label = x_col + (f" ({_DATE_UNIT_LABEL.get(time_unit or 'M','')})" if x_is_date else "")
        title   = x_label + (f"  ×  {y_col}" if chart_type not in ("histogram", "box") else "")
        fig.update_layout(
            **_PLOT_LAYOUT,
            title=dict(text=title, font=dict(color="#E8EAF0", size=13)),
            xaxis=dict(**_AXIS_STYLE, tickangle=tick_angle),
            yaxis=dict(**_AXIS_STYLE),
            height=440,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8892a4", size=10)),
        )
        return dcc.Graph(figure=fig, config={"displayModeBar": False})
    except Exception as e:
        return html.Div(f"Grafik oluşturulamadı: {e}", className="alert-info-custom")


# ── Playground: Değişken özeti önizleme ───────────────────────────────────────
@app.callback(
    Output("pg-var-summary-preview", "children"),
    Input("store-config", "data"),
    Input("store-expert-exclude", "data"),
    Input("main-tabs", "active_tab"),
    State("store-key", "data"),
    State("dd-segment-val", "value"),
    State("dd-segment-col", "value"),
)
def render_pg_var_summary_preview(config, expert_excluded, active_tab, key, seg_val, seg_col_input):
    if not key or not config or not config.get("target_col"):
        return html.Div()
    seg_col = config.get("segment_col") or (seg_col_input or None)

    # Önce tam özet cache'ine bak (Değişken Özeti → Hesapla sonrası dolu olur)
    full_summary = _SERVER_STORE.get(f"{key}_summary_{seg_col}_{seg_val}")
    iv_df        = _SERVER_STORE.get(f"{key}_iv_{seg_col}_{seg_val}")

    if full_summary is not None:
        disp = full_summary.copy()
        source_note = None
    elif iv_df is not None:
        # Sadece IV varsa — sınırlı görünüm
        disp = iv_df[["Değişken", "IV", "Güç", "Eksik %"]].copy()
        source_note = html.Div(
            "Tam özet için 'Değişken Özeti' sekmesinde 'Hesapla' butonuna basın "
            "(PSI · Korelasyon · VIF · Öneri sütunları eklenecek).",
            className="form-hint",
            style={"padding": "0.3rem 0.5rem", "marginBottom": "0.4rem"})
    else:
        return html.Div(
            "Özet henüz hesaplanmadı. Önce 'Target & IV' veya 'Değişken Özeti' sekmesini açın.",
            className="form-hint", style={"padding": "0.5rem 0.75rem"})

    excluded = set(expert_excluded or [])
    disp = disp[~disp["Değişken"].isin(excluded)].copy()

    cond = [
        {"if": {"filter_query": '{Güç} = "Güçlü"',       "column_id": "Güç"},   "color": "#10b981"},
        {"if": {"filter_query": '{Güç} = "Orta"',         "column_id": "Güç"},   "color": "#4F8EF7"},
        {"if": {"filter_query": '{Güç} = "Zayıf"',        "column_id": "Güç"},   "color": "#f59e0b"},
        {"if": {"filter_query": '{Güç} = "Çok Zayıf"',    "column_id": "Güç"},   "color": "#7e8fa4"},
        {"if": {"filter_query": '{Güç} = "Şüpheli"',      "column_id": "Güç"},   "color": "#ef4444"},
        {"if": {"filter_query": '{Öneri} = "✅ Tut"',      "column_id": "Öneri"}, "color": "#10b981", "fontWeight": "700"},
        {"if": {"filter_query": '{Öneri} = "⚠️ İncele"',  "column_id": "Öneri"}, "color": "#f59e0b"},
        {"if": {"filter_query": '{Öneri} = "❌ Çıkar"',   "column_id": "Öneri"}, "color": "#ef4444", "fontWeight": "700"},
        {"if": {"filter_query": '{Öneri} = "⚠️ İncele"',  "column_id": "Sebep"},  "color": "#f59e0b"},
        {"if": {"filter_query": '{Öneri} = "❌ Çıkar"',   "column_id": "Sebep"},  "color": "#ef4444"},
        {"if": {"filter_query": '{PSI Durum} = "Kritik Kayma"', "column_id": "PSI Durum"}, "color": "#ef4444"},
        {"if": {"filter_query": '{PSI Durum} = "Hafif Kayma"',  "column_id": "PSI Durum"}, "color": "#f59e0b"},
        {"if": {"filter_query": '{PSI Durum} = "Stabil"',       "column_id": "PSI Durum"}, "color": "#10b981"},
        {"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"},
    ]
    tbl = dash_table.DataTable(
        data=disp.to_dict("records"),
        columns=[{"name": c, "id": c} for c in disp.columns],
        sort_action="native",
        filter_action="native",
        page_size=100,
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#161d2e", "color": "#a8b2c2",
                      "fontWeight": "600", "fontSize": "0.7rem",
                      "border": "1px solid #2d3a4f", "textTransform": "uppercase"},
        style_cell={"backgroundColor": "#111827", "color": "#d1d5db",
                    "fontSize": "0.78rem", "border": "1px solid #1f2a3c",
                    "padding": "4px 8px", "textAlign": "left"},
        style_data_conditional=cond,
    )
    return html.Div([source_note, tbl] if source_note else [tbl])


# ── Playground: Kaynak listeyi doldur ─────────────────────────────────────────
@app.callback(
    Output("pg-source-container", "children"),
    Input("store-config",       "data"),
    Input("store-expert-exclude", "data"),
    Input("store-pg-model-vars", "data"),
    State("store-key", "data"),
)
def render_pg_source(config, expert_excluded, model_vars, key):
    df = _get_df(key)
    if df is None or not config or not config.get("target_col"):
        return html.Div("Önce veri yükleyin.", className="form-hint")
    excluded = set(expert_excluded or [])
    in_model = set(model_vars or [])
    screen_result = _SERVER_STORE.get(f"{key}_screen")
    if screen_result:
        base = [c for c in screen_result[0]
                if c != config["target_col"] and c not in excluded and c not in in_model]
    else:
        cfg = {c for c in [config.get("target_col"), config.get("date_col"),
                            config.get("segment_col")] if c}
        base = [c for c in df.columns
                if c not in cfg and c not in excluded and c not in in_model]
    if not base:
        return html.Div("Tüm değişkenler model listesinde.", className="form-hint")
    return dbc.Checklist(
        id="chk-pg-source",
        options=[{"label": c, "value": c} for c in base],
        value=[],
        inline=True,
        className="expert-checklist",
    )


# ── Playground: Model listesini doldur ────────────────────────────────────────
@app.callback(
    Output("pg-model-container", "children"),
    Input("store-pg-model-vars", "data"),
)
def render_pg_model_list(model_vars):
    if not model_vars:
        return html.Div("Model listesi boş.", className="form-hint")
    return dbc.Checklist(
        id="chk-pg-model",
        options=[{"label": c, "value": c} for c in model_vars],
        value=[],
        inline=True,
        className="expert-checklist",
    )


# ── Playground: Değişken ekle ──────────────────────────────────────────────────
@app.callback(
    Output("store-pg-model-vars", "data"),
    Input("btn-pg-add", "n_clicks"),
    State("chk-pg-source",       "value"),
    State("store-pg-model-vars", "data"),
    prevent_initial_call=True,
)
def pg_add_vars(_, selected, current):
    if not selected:
        return dash.no_update
    current = current or []
    new = [c for c in selected if c not in set(current)]
    return current + new


# ── Playground: Değişken kaldır ───────────────────────────────────────────────
@app.callback(
    Output("store-pg-model-vars", "data", allow_duplicate=True),
    Input("btn-pg-remove", "n_clicks"),
    State("chk-pg-model",        "value"),
    State("store-pg-model-vars", "data"),
    prevent_initial_call=True,
)
def pg_remove_vars(_, selected, current):
    if not selected or not current:
        return dash.no_update
    remove_set = set(selected)
    return [c for c in current if c not in remove_set]


# ── Playground: Model kur ─────────────────────────────────────────────────────
@app.callback(
    Output("pg-model-output", "children"),
    Input("btn-pg-build", "n_clicks"),
    State("store-pg-model-vars", "data"),
    State("chk-use-woe",         "value"),
    State("pg-test-size",        "value"),
    State("pg-c-value",          "value"),
    State("pg-model-type",       "value"),
    State("pg-target-col",       "value"),
    State("pg-split-method",     "value"),
    State("pg-split-date",       "value"),
    State("store-key",           "data"),
    State("store-config",        "data"),
    State("dd-segment-val",      "value"),
    State("dd-segment-col",      "value"),
    prevent_initial_call=True,
)
def build_pg_model(_, model_vars, use_woe, test_size_pct, c_val, model_type,
                   target_sel, split_method, split_date,
                   key, config, seg_val, seg_col_input):
    if not model_vars or not key or not config:
        return html.Div("Model listesi boş veya konfigürasyon eksik.",
                        className="alert-info-custom")
    df_orig = _get_df(key)
    if df_orig is None:
        return html.Div("Veri yüklenmemiş.", className="alert-info-custom")

    target    = target_sel or config["target_col"]
    seg_col   = config.get("segment_col") or (seg_col_input or None)
    date_col  = config.get("date_col")
    df_active = apply_segment_filter(df_orig, seg_col, seg_val)
    test_size = float(test_size_pct or 30) / 100
    C         = float(c_val or 1.0)

    if target not in df_active.columns:
        return html.Div(f"Target kolonu '{target}' veri setinde bulunamadı.",
                        className="alert-info-custom")
    y_all = df_active[target].astype(float)

    # ── Train/Test split maskesi ───────────────────────────────────────────────
    if split_method == "date" and date_col and split_date and date_col in df_active.columns:
        dates = pd.to_datetime(df_active[date_col], errors="coerce")
        cutoff = pd.to_datetime(split_date)          # "2024-05" → 2024-05-01
        train_mask = (dates < cutoff).values
        test_mask  = ~train_mask
        split_info = (f"Tarih kesimi: {split_date}  ·  "
                      f"Train: {train_mask.sum():,}  /  Test: {test_mask.sum():,}")
        if train_mask.sum() < 30 or test_mask.sum() < 30:
            return html.Div(
                f"Yetersiz veri: Train={train_mask.sum():,}, Test={test_mask.sum():,}. "
                "Farklı bir kesim tarihi seçin.",
                className="alert-info-custom")
    else:
        indices = np.arange(len(df_active))
        try:
            tr_idx, te_idx = train_test_split(
                indices, test_size=test_size, random_state=42, stratify=y_all.values)
        except Exception as e:
            return html.Div(f"Split hatası: {e}", className="alert-info-custom")
        train_mask = np.zeros(len(df_active), dtype=bool)
        train_mask[tr_idx] = True
        test_mask = ~train_mask
        split_info = (f"Rastgele split  ·  "
                      f"Train: {train_mask.sum():,}  /  Test: {test_mask.sum():,}")

    # ── Base parametreler ─────────────────────────────────────────────────────
    _MODEL_PARAMS = {
        "lr":   {},   # C dışarıdan alınıyor
        "lgbm": dict(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            random_state=42, n_jobs=-1, verbose=-1,
        ),
        "xgb":  dict(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            random_state=42, n_jobs=-1, eval_metric="auc",
        ),
        "rf":   dict(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            max_features="sqrt", random_state=42, n_jobs=-1,
        ),
    }
    algo = model_type or "lr"

    def _fit_and_render(X_df, disp_names, label, accent):
        """Modeli kur, train+test sonuçlarını döndür."""
        X = X_df.copy()
        for col in X.columns:
            if X[col].isna().any():
                fill = X[col].median() if pd.api.types.is_numeric_dtype(X[col]) \
                       else X[col].mode().iloc[0]
                X[col] = X[col].fillna(fill)
        X_tr = X.iloc[train_mask]
        X_te = X.iloc[test_mask]
        y_tr = y_all.iloc[train_mask]
        y_te = y_all.iloc[test_mask]
        if len(X_tr) == 0 or len(X_te) == 0:
            return html.Div("Split sonrası boş küme oluştu.", className="alert-info-custom")

        # Scaling: sadece LR için anlamlı; tree modeller scale'e duyarsız
        is_tree = algo in ("lgbm", "xgb", "rf")
        if not is_tree:
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
        else:
            X_tr_s = X_tr.values
            X_te_s = X_te.values

        try:
            if algo == "lr":
                mdl = LogisticRegression(C=C, max_iter=1000, random_state=42)
            elif algo == "lgbm":
                mdl = lgb.LGBMClassifier(**_MODEL_PARAMS["lgbm"])
            elif algo == "xgb":
                pos_w = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1)
                mdl = xgb.XGBClassifier(**_MODEL_PARAMS["xgb"],
                                        scale_pos_weight=pos_w)
            else:  # rf
                mdl = RandomForestClassifier(**_MODEL_PARAMS["rf"])
            mdl.fit(X_tr_s, y_tr)
        except Exception as e:
            return html.Div(f"Model kurulamadı: {e}", className="alert-info-custom")

        def _metrics(y_true, y_prob_arr):
            y_pred_arr = (y_prob_arr >= 0.5).astype(int)
            auc_  = roc_auc_score(y_true, y_prob_arr)
            gini_ = 2 * auc_ - 1
            fpr_, tpr_, _ = roc_curve(y_true, y_prob_arr)
            ks_   = float(np.max(tpr_ - fpr_))
            f1_   = f1_score(y_true, y_pred_arr, zero_division=0)
            prec_ = precision_score(y_true, y_pred_arr, zero_division=0)
            rec_  = recall_score(y_true, y_pred_arr, zero_division=0)
            cm__  = confusion_matrix(y_true, y_pred_arr)
            return dict(auc=auc_, gini=gini_, ks=ks_, f1=f1_,
                        prec=prec_, rec=rec_, cm=cm__,
                        fpr=fpr_, tpr=tpr_, n=len(y_true))

        tr_prob = mdl.predict_proba(X_tr_s)[:, 1]
        te_prob = mdl.predict_proba(X_te_s)[:, 1]
        tr_m = _metrics(y_tr, tr_prob)
        te_m = _metrics(y_te, te_prob)

        # ── Metrik rengi ──────────────────────────────────────────────────────
        def _gc(g): return "#10b981" if g >= 0.4 else "#f59e0b" if g >= 0.2 else "#ef4444"

        # ── Metrik kartı ──────────────────────────────────────────────────────
        def mc(v, l, c="#4F8EF7", w=2):
            return dbc.Col(html.Div([
                html.Div(str(v), className="metric-value",
                         style={"color": c, "fontSize": "1.25rem"}),
                html.Div(l, className="metric-label"),
            ], className="metric-card"), width=w)

        def _metric_row(m, title, bg):
            gc = _gc(m["gini"])
            return html.Div([
                html.Div(title, style={"color": "#a8b2c2", "fontSize": "0.72rem",
                                       "fontWeight": "600", "letterSpacing": "0.06em",
                                       "textTransform": "uppercase",
                                       "marginBottom": "0.4rem", "paddingLeft": "0.25rem"}),
                dbc.Row([
                    mc(f"{m['gini']:.4f}", "Gini",      gc),
                    mc(f"{m['auc']:.4f}",  "AUC",       gc),
                    mc(f"{m['ks']:.4f}",   "KS",        "#4F8EF7"),
                    mc(f"{m['f1']:.4f}",   "F1",        "#a78bfa"),
                    mc(f"{m['prec']:.4f}", "Precision", "#a78bfa"),
                    mc(f"{m['rec']:.4f}",  "Recall",    "#a78bfa"),
                    mc(f"{m['n']:,}",      "N",         "#556070"),
                ], className="g-2"),
            ], style={"backgroundColor": bg, "borderRadius": "6px",
                      "padding": "0.6rem 0.5rem", "marginBottom": "0.5rem"})

        metric_panel = html.Div([
            _metric_row(tr_m, "Train", "#0d1520"),
            _metric_row(te_m, "Test",  "#0e1624"),
        ], className="mb-3")

        # ── ROC (train + test) ────────────────────────────────────────────────
        accent_tr = "#556070"
        fig_roc = go.Figure()
        for m, col_, nm in [
            (tr_m, accent_tr, f"Train (AUC={tr_m['auc']:.3f})"),
            (te_m, accent,    f"Test  (AUC={te_m['auc']:.3f})"),
        ]:
            fig_roc.add_trace(go.Scatter(
                x=m["fpr"], y=m["tpr"], mode="lines",
                line=dict(color=col_, width=2), name=nm,
                fill="tozeroy" if col_ == accent else None,
                fillcolor=f"rgba({','.join(str(int(accent.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.06)",
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode="lines",
            line=dict(color="#4a5568", dash="dash", width=1), showlegend=False))
        fig_roc.update_layout(
            **_PLOT_LAYOUT,
            title=dict(text="ROC Eğrisi", font=dict(color="#E8EAF0", size=13)),
            xaxis=dict(**_AXIS_STYLE, title="FPR"),
            yaxis=dict(**_AXIS_STYLE, title="TPR"),
            height=300, showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8892a4", size=10)),
        )

        # ── Confusion matrix (test) ───────────────────────────────────────────
        tn, fp, fn, tp_ = te_m["cm"].ravel()
        fig_cm = go.Figure(go.Heatmap(
            z=[[tn, fp], [fn, tp_]],
            x=["Pred: 0", "Pred: 1"], y=["Actual: 0", "Actual: 1"],
            text=[[str(tn), str(fp)], [str(fn), str(tp_)]],
            texttemplate="%{text}",
            colorscale=[[0, "#0e1117"], [1, accent]], showscale=False,
        ))
        fig_cm.update_layout(
            **_PLOT_LAYOUT,
            title=dict(text="Confusion Matrix — Test (thr=0.5)",
                       font=dict(color="#E8EAF0", size=13)),
            height=260, xaxis=dict(side="top"),
        )

        # ── Önem tablosu: LR → katsayı; tree → feature importance ───────────
        _tbl_style = dict(
            sort_action="native", page_size=15,
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#161d2e", "color": "#a8b2c2",
                          "fontWeight": "600", "fontSize": "0.7rem",
                          "border": "1px solid #2d3a4f",
                          "textTransform": "uppercase"},
            style_cell={"backgroundColor": "#111827", "color": "#d1d5db",
                        "fontSize": "0.78rem", "border": "1px solid #1f2a3c",
                        "padding": "5px 8px", "textAlign": "left"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"}
            ],
        )
        if not is_tree:
            # Logistic Regression: katsayı + odds ratio
            coef_rows = [{"Değişken": disp_names.get(c, c),
                          "Katsayı":    round(float(v), 4),
                          "Odds Ratio": round(float(np.exp(v)), 4),
                          "Etki": "↑ Bad" if v > 0 else "↓ Bad"}
                         for c, v in zip(X.columns, mdl.coef_[0])]
            imp_df = pd.DataFrame(coef_rows).sort_values("Katsayı", key=abs, ascending=False)
            imp_df_cond = [
                {"if": {"filter_query": '{Etki} = "↑ Bad"', "column_id": "Etki"},
                 "color": "#ef4444", "fontWeight": "600"},
                {"if": {"filter_query": '{Etki} = "↓ Bad"', "column_id": "Etki"},
                 "color": "#10b981", "fontWeight": "600"},
                {"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"},
            ]
            tbl_title = "Katsayı Tablosu"
            imp_table = dash_table.DataTable(
                data=imp_df.to_dict("records"),
                columns=[{"name": c, "id": c} for c in imp_df.columns],
                style_data_conditional=imp_df_cond,
                **{k: v for k, v in _tbl_style.items() if k != "style_data_conditional"},
            )
        else:
            # Tree modeller: feature importance (gain / split)
            raw_imp = mdl.feature_importances_
            total   = raw_imp.sum() or 1.0
            imp_rows = [
                {"Değişken": disp_names.get(c, c),
                 "Önem (%)": round(float(v / total * 100), 2),
                 "Önem (ham)": round(float(v), 4)}
                for c, v in zip(X.columns, raw_imp)
            ]
            imp_df = pd.DataFrame(imp_rows).sort_values("Önem (%)", ascending=False)
            # Renk: top %25 yeşil, orta sarı, alt kırmızı
            top_thr = float(imp_df["Önem (%)"].quantile(0.75))
            mid_thr = float(imp_df["Önem (%)"].quantile(0.25))
            imp_df_cond = [
                {"if": {"filter_query": f"{{Önem (%)}} >= {top_thr:.2f}",
                        "column_id": "Önem (%)"},
                 "color": "#10b981", "fontWeight": "700"},
                {"if": {"filter_query": f"{{Önem (%)}} >= {mid_thr:.2f} && {{Önem (%)}} < {top_thr:.2f}",
                        "column_id": "Önem (%)"},
                 "color": "#f59e0b"},
                {"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"},
            ]
            imp_label = {"lgbm": "gain (normalize)", "xgb": "gain (normalize)",
                         "rf": "mean decrease impurity"}.get(algo, "")
            tbl_title = f"Feature Importance  ·  {imp_label}"
            imp_table = dash_table.DataTable(
                data=imp_df.to_dict("records"),
                columns=[{"name": c, "id": c} for c in imp_df.columns],
                style_data_conditional=imp_df_cond,
                **{k: v for k, v in _tbl_style.items() if k != "style_data_conditional"},
            )

        return html.Div([
            html.Div(split_info,
                     style={"color": "#7e8fa4", "fontSize": "0.72rem",
                            "marginBottom": "0.6rem", "fontStyle": "italic"}),
            metric_panel,
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_roc, config={"displayModeBar": False}), width=5),
                dbc.Col([
                    html.P(tbl_title, className="section-title",
                           style={"marginBottom": "0.4rem"}),
                    imp_table,
                ], width=4),
                dbc.Col(dcc.Graph(figure=fig_cm, config={"displayModeBar": False}), width=3),
            ]),
        ])

    # ── Ham model ─────────────────────────────────────────────────────────────
    raw_cols = [v for v in model_vars if v in df_active.columns]
    X_raw    = pd.get_dummies(df_active[raw_cols].copy(), drop_first=True)
    raw_disp = {c: c for c in X_raw.columns}
    raw_tab  = _fit_and_render(X_raw, raw_disp, "Ham", "#4F8EF7")

    # ── WoE model ─────────────────────────────────────────────────────────────
    woe_cache_key = f"{key}_woe_{seg_col}_{seg_val}"
    if woe_cache_key not in _SERVER_STORE:
        woe_df_enc, _ = _build_woe_dataset(df_active, target, model_vars)
        _SERVER_STORE[woe_cache_key] = (woe_df_enc, _)
    else:
        woe_df_enc, _ = _SERVER_STORE[woe_cache_key]
        new_vars = [v for v in model_vars if f"{v}_woe" not in woe_df_enc.columns]
        if new_vars:
            extra, ef = _build_woe_dataset(df_active, target, new_vars)
            woe_df_enc = pd.concat([woe_df_enc, extra], axis=1)
            _SERVER_STORE[woe_cache_key] = (woe_df_enc, _ + ef)

    woe_feat_cols = [f"{v}_woe" for v in model_vars if f"{v}_woe" in woe_df_enc.columns]
    if woe_feat_cols:
        X_woe     = woe_df_enc[woe_feat_cols].copy()
        woe_disp  = {f"{v}_woe": v for v in model_vars}
        woe_tab   = _fit_and_render(X_woe, woe_disp, "WoE", "#a78bfa")
        failed_woe = [v for v in model_vars if f"{v}_woe" not in woe_df_enc.columns]
        note_txt  = f"★ WoE — {len(woe_feat_cols)}/{len(model_vars)} değişken encode edildi"
        if failed_woe:
            note_txt += f"  |  encode edilemeyen: {', '.join(failed_woe)}"
        woe_note  = html.Div(note_txt,
            style={"color": "#a78bfa", "fontSize": "0.73rem", "marginBottom": "0.4rem"})
        woe_content = html.Div([woe_note, woe_tab])
    else:
        failed_woe = model_vars
        woe_content = html.Div(
            f"WoE encode edilebilen değişken bulunamadı. "
            f"({len(model_vars)} değişken denendi: {', '.join(model_vars[:5])}{'...' if len(model_vars)>5 else ''})",
            className="alert-info-custom")

    return dbc.Tabs([
        dbc.Tab(raw_tab,     label="Ham Değerler",        tab_id="res-raw",
                className="tab-content-area"),
        dbc.Tab(woe_content, label="WoE Dönüştürülmüş",  tab_id="res-woe",
                className="tab-content-area"),
    ], active_tab="res-raw")


# ── Başlat ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    threading.Timer(1.2, lambda: webbrowser.open("http://localhost:8050")).start()
    app.run(debug=False, port=8050)
