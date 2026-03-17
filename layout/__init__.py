from dash import dcc, html
import dash_bootstrap_components as dbc

from utils.chart_helpers import _tab_info

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


def build_navbar():
    return dbc.Navbar(
        dbc.Container([
            html.Div([
                html.Span("EDA", className="navbar-logo-text"),
                html.Span("LAB", className="navbar-brand-title"),
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
            dbc.Tab(dcc.Loading(html.Div(id="tab-outlier"),   type="dot", color="#4F8EF7", delay_show=300), label="Outlier Analizi", tab_id="tab-outlier",   className="tab-content-area"),
            dbc.Tab(dcc.Loading(html.Div(id="tab-deep-dive"), type="dot", color="#4F8EF7", delay_show=300), label="Değişken Analizi", tab_id="tab-deep-dive", className="tab-content-area"),
            dbc.Tab(html.Div([
                _tab_info("İstatistiksel Testler", "Korelasyon · Chi-Square · ANOVA · KS · VIF",
                          "Beş farklı istatistiksel test arasından seçim yapın. Her test için "
                          "amaç, yöntem ve çıktı yorumu test seçimi altında açıklanır. "
                          "Segmentasyon aktifse tüm testler yalnızca aktif segment üzerinde çalışır.",
                          "#a78bfa"),
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
                _tab_info("Değişken Özeti", "IV · Eksik% · PSI · Korelasyon · VIF",
                          "Tüm değişkenleri tek tabloda karşılaştırır. IV ayırıcı gücü, "
                          "eksik oranı, PSI dağılım stabilitesi, max korelasyon ve VIF çoklu "
                          "doğrusallık bilgilerini yan yana görerek değişken eleme kararlarını "
                          "tek bakışta verebilirsiniz.",
                          "#10b981"),
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
                _tab_info("Playground", "Grafik Oluşturucu · Hızlı Model · SHAP",
                          "İki bölümden oluşur: Grafik Oluşturucu ile serbest keşif yapabilir, "
                          "Hızlı Model ile LR / LightGBM / XGBoost / Random Forest modellerini "
                          "hızlıca eğitip AUC, Gini, KS metriklerini ve SHAP beeswarm grafiğini "
                          "görebilirsiniz. WoE encode ve ham encode sonuçlar yan yana karşılaştırılır.",
                          "#f59e0b"),
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
                        html.Div([
                            dbc.Label("Mevcut Değişkenler", className="form-label",
                                      style={"marginBottom": "0"}),
                            html.Span(id="pg-source-count",
                                      className="form-hint",
                                      style={"marginLeft": "0.5rem"}),
                        ], style={"display": "flex", "alignItems": "baseline",
                                  "marginBottom": "0.2rem"}),
                        dbc.Input(
                            id="pg-source-search",
                            placeholder="Filtrele…",
                            debounce=False, size="sm",
                            style={"backgroundColor": "#0e1117", "color": "#c8cdd8",
                                   "border": "1px solid #2d3a4f", "borderRadius": "4px",
                                   "marginBottom": "0.35rem", "fontSize": "0.78rem"},
                        ),
                        html.Div(id="pg-source-container",
                                 style={"maxHeight": "260px", "overflowY": "auto",
                                        "backgroundColor": "#0e1117",
                                        "border": "1px solid #2d3a4f",
                                        "borderRadius": "6px",
                                        "padding": "0.4rem 0.6rem",
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
                        ], style={"textAlign": "center", "paddingTop": "2rem"}),
                    ], width=1,
                       className="d-flex align-items-center justify-content-center"),
                    # Sağ — model listesi
                    dbc.Col([
                        html.Div([
                            dbc.Label("Model Listesi", className="form-label",
                                      style={"marginBottom": "0"}),
                            html.Span(id="pg-model-count",
                                      className="form-hint",
                                      style={"marginLeft": "0.5rem"}),
                        ], style={"display": "flex", "alignItems": "baseline",
                                  "marginBottom": "0.2rem"}),
                        html.Div(style={"height": "2rem"}),  # arama kutusu yüksekliği kadar boşluk
                        html.Div(id="pg-model-container",
                                 style={"maxHeight": "260px", "overflowY": "auto",
                                        "backgroundColor": "#0e1117",
                                        "border": "1px solid #2d3a4f",
                                        "borderRadius": "6px",
                                        "padding": "0.4rem 0.6rem",
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

                # Model parametreleri — satır 2: model tipi + LR-C + eşik + kur
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
                        dbc.Label("Karar Eşiği", className="form-label"),
                        html.Div("Sınıflandırma kesim noktası", className="form-hint"),
                        dbc.Select(id="pg-threshold-method", className="dark-select",
                                   options=[
                                       {"label": "Sabit  (0.50)",  "value": "fixed"},
                                       {"label": "F1 Maks.",       "value": "f1"},
                                       {"label": "KS Noktası",     "value": "ks"},
                                       {"label": "Özel",           "value": "custom"},
                                   ],
                                   value="fixed",
                                   style={"maxWidth": "180px"}),
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Özel Eşik", className="form-label"),
                        html.Div("'Özel' seçiliyse geçerlidir", className="form-hint"),
                        dbc.Input(id="pg-threshold-val", type="number",
                                  value=0.50, min=0.01, max=0.99, step=0.01,
                                  style={"maxWidth": "100px"}),
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


def build_layout():
    return html.Div([
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
        dcc.Store(id="store-precompute-state", storage_type="memory"),
        dcc.Interval(id="interval-precompute", interval=300, disabled=True, n_intervals=0),

        # ── Precompute Modal ──────────────────────────────────────────────────────
        dbc.Modal([
            dbc.ModalHeader(
                html.Div([
                    html.Span("Sistem Hazırlanıyor", style={
                        "fontWeight": "700", "fontSize": "1.05rem", "color": "#c8cdd8"
                    }),
                    html.Span(" — sekmeler yüklendikten sonra bekleme olmayacak", style={
                        "fontSize": "0.8rem", "color": "#6b7a99", "marginLeft": "8px"
                    }),
                ]),
                close_button=False,
                style={"backgroundColor": "#0e1117", "borderBottom": "1px solid #1e2a3a"},
            ),
            dbc.ModalBody(
                id="precompute-modal-body",
                style={"backgroundColor": "#0e1117", "padding": "1.5rem 2rem"},
            ),
            dbc.ModalFooter(
                id="precompute-modal-footer",
                children=html.Div([
                    dbc.Button("Başlat", id="btn-precompute-start", color="primary",
                               style={"fontSize": "0.85rem", "padding": "0.4rem 1.2rem",
                                      "marginRight": "8px", "display": "none"}),
                    dbc.Button("Atla", id="btn-precompute-close", color="secondary", outline=True,
                               style={"fontSize": "0.85rem", "padding": "0.4rem 1.0rem",
                                      "display": "none"}),
                    dbc.Button("Tamam — Sekmelere Git", id="btn-precompute-done", color="primary",
                               style={"fontSize": "0.85rem", "padding": "0.4rem 1.2rem",
                                      "display": "none"}),
                ]),
                style={"backgroundColor": "#0e1117", "borderTop": "1px solid #1e2a3a"},
            ),
        ],
            id="modal-precompute",
            is_open=False,
            backdrop="static",
            keyboard=False,
            size="lg",
            style={"zIndex": 9999},
        ),
    ])
