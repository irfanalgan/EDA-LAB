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
            html.Div([
                html.Span("Keşifsel Veri Analizi", className="navbar-subtitle"),
                html.Button(
                    html.I(className="bi bi-question-circle"),
                    id="btn-help-open",
                    n_clicks=0,
                    style={
                        "marginLeft": "1rem",
                        "background": "none",
                        "border": "1px solid rgba(79,142,247,0.4)",
                        "borderRadius": "50%",
                        "color": "#4F8EF7",
                        "cursor": "pointer",
                        "fontSize": "0.85rem",
                        "width": "26px",
                        "height": "26px",
                        "lineHeight": "1",
                        "padding": "0",
                        "verticalAlign": "middle",
                        "display": "inline-flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                        "transition": "all 0.15s ease",
                    },
                    title="Yardım & Referans",
                ),
            ], style={"display": "flex", "alignItems": "center"}),
        ], fluid=True, style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}),
        color="#111827",
        dark=True,
        style={"borderBottom": "1px solid #232d3f", "padding": "0.6rem 0"},
    )


def _help_card(title: str, rows: list[tuple], note: str = "") -> "dbc.Card":
    """Başlık + satır listesi içeren küçük referans kartı."""
    return dbc.Card([
        dbc.CardHeader(title, style={"fontSize": "0.78rem", "fontWeight": "700",
                                     "color": "#c8cdd8", "backgroundColor": "#161C27",
                                     "padding": "0.4rem 0.75rem",
                                     "borderBottom": "1px solid #2d3a4f"}),
        dbc.CardBody([
            html.Table([
                html.Tbody([
                    html.Tr([
                        html.Td(k, style={"color": "#a8b2c2", "fontSize": "0.74rem",
                                          "paddingRight": "1rem", "whiteSpace": "nowrap",
                                          "fontWeight": "600"}),
                        html.Td(v, style={"color": "#c8cdd8", "fontSize": "0.74rem"}),
                    ]) for k, v in rows
                ])
            ], style={"width": "100%", "borderCollapse": "collapse"}),
            html.P(note, style={"fontSize": "0.72rem", "color": "#556070",
                                "marginTop": "0.5rem", "marginBottom": "0"}) if note else None,
        ], style={"padding": "0.5rem 0.75rem"}),
    ], style={"backgroundColor": "#111827", "border": "1px solid #2d3a4f",
              "borderRadius": "6px", "marginBottom": "0.75rem"})


def _faq_item(q: str, bullets: list[str], level: str = "warning") -> "html.Div":
    colors = {"error": "#ef4444", "warning": "#f59e0b", "info": "#4F8EF7"}
    c = colors.get(level, "#f59e0b")
    return html.Div([
        html.Div(q, style={"fontSize": "0.78rem", "fontWeight": "600",
                           "color": c, "marginBottom": "4px"}),
        html.Ul([html.Li(b, style={"fontSize": "0.74rem", "color": "#9aa5bc",
                                   "lineHeight": "1.7"}) for b in bullets],
                style={"paddingLeft": "1.2rem", "margin": "0 0 0.75rem 0"}),
    ])


def _build_help_tab() -> "html.Div":
    return html.Div([
        html.Div([
            html.Span("EDA Lab", style={"fontWeight": "700", "color": "#4F8EF7"}),
            html.Span("  ·  Yardım & Referans", style={"color": "#7e8fa4"}),
        ], style={"fontSize": "0.9rem", "marginBottom": "1.5rem",
                  "borderBottom": "1px solid #2d3a4f", "paddingBottom": "0.75rem"}),

        dbc.Row([
            # ── Sol: Metrik referans kartları ─────────────────────────────
            dbc.Col([
                html.P("Metrik Referansı", style={"fontSize": "0.7rem", "color": "#556070",
                                                   "textTransform": "uppercase",
                                                   "letterSpacing": "0.08em", "marginBottom": "0.5rem"}),
                _help_card("IV (Information Value)", [
                    ("< 0.02",  "Çok zayıf — anlamsız"),
                    ("0.02–0.1", "Zayıf"),
                    ("0.1–0.3",  "Orta"),
                    ("0.3–0.5",  "Güçlü"),
                    ("> 0.5",    "Şüpheli — overfitting riski"),
                ]),
                _help_card("PSI (Population Stability Index)", [
                    ("< 0.10",   "Stabil — model güvenilir"),
                    ("0.10–0.25","Dikkat — izle"),
                    ("> 0.25",   "Kritik kayma — model yenilenmeli"),
                ]),
                _help_card("Gini / AUC", [
                    ("AUC < 0.60", "Zayıf ayrım gücü"),
                    ("AUC 0.60–0.70", "Kabul edilebilir"),
                    ("AUC 0.70–0.80", "İyi"),
                    ("AUC > 0.80",    "Çok iyi"),
                ]),
                _help_card("p-value (Logit katsayısı)", [
                    ("< 0.05",  "İstatistiksel olarak anlamlı ✓"),
                    ("0.05–0.10","Sınırda anlamlı"),
                    ("> 0.10",  "Anlamlı değil — modelde tutma/çıkar"),
                ]),
            ], width=4),

            # ── Sağ: Sık karşılaşılan sorunlar ───────────────────────────
            dbc.Col([
                html.P("Sık Karşılaşılan Sorunlar", style={
                    "fontSize": "0.7rem", "color": "#556070",
                    "textTransform": "uppercase", "letterSpacing": "0.08em",
                    "marginBottom": "0.5rem"}),

                _faq_item("IV = 0 geliyor", [
                    "Train setinde yeterli event veya non-event yok (< 5 adet).",
                    "Değişken bu segmentte sabit (zero variance) — tüm değerler aynı.",
                    "sklearn ≥ 1.6 + eski optbinning uyumsuzluğu: modules/deep_dive.py "
                    "başındaki monkey-patch bloğunun çalıştığından emin olun.",
                    "Çözüm: Değişken tipini 'Kategorik' olarak zorla ve tekrar dene. "
                    "Ya da train/OOT kesim tarihini kaydırarak train setini büyüt.",
                ], "error"),

                _faq_item("PSI hesaplanamıyor (—)", [
                    "Tarih kolonu seçilmemiş → sol panelden tarih kolonu seç.",
                    "OOT kesim tarihi tanımlanmamış → 'PSI Kesim Tarihi' dropdown'ını doldur.",
                    "Seçilen kesim tarihinde o değişken tamamen eksik.",
                    "Baseline veya OOT periyodunda değişken sabit → PSI matematiksel olarak 0/inf üretir.",
                ], "warning"),

                _faq_item("WOE tablosu boş geliyor", [
                    "Target binary değil — WOE yalnızca 0/1 target için hesaplanır.",
                    "Train setinde çok az gözlem var.",
                    "Max Bin Sayısı değerini düşür (4 → 2) ve tekrar dene.",
                ], "warning"),

                _faq_item("Korelasyon matrisinde NaN / boş hücreler", [
                    "İki değişkenden biri sabit (zero variance) → korelasyon tanımsız.",
                    "Değişkende yüksek eksik veri varsa korelasyon çifti düşüyor.",
                    "Kategorik kolonlar otomatik çıkarılır; encode edilmiş hali dahil edilmez.",
                ], "info"),

                _faq_item("Çoklu tablo join sonrası beklenmedik kolon sayısı", [
                    "Join key dışında her iki tabloda da ortak kolon varsa (ör. Model_Segment) "
                    "otomatik olarak tablo 1'den alınır, tablo 2/3'ten düşürülür.",
                    "Beklenen kolon sayısı: T1 + T2 + T3 − (join_key × 3) − ortak_kolon × (n-1).",
                    "Kontrol: yükleme sonrası 'Önizleme' sekmesinde kolon listesine bak.",
                ], "info"),

                _faq_item("Logistic Regression modeli kurulamıyor (BFGS hatası)", [
                    "WOE sütunlarında NaN/Inf var — değişken seçiminde sorunlu sütunları çıkar.",
                    "Target tüm train setinde aynı değer (all-0 veya all-1) → segment filtresi genişlet.",
                    "Çok fazla değişken az gözlem ile birleşince matris tekil (singular) olabilir — "
                    "değişken sayısını azalt.",
                ], "error"),

                _faq_item("Segment seçince metrikler değişmiyor", [
                    "Segment kolonu ve değeri doğru seçildi mi? Sol panel → 'Segment Değeri'.",
                    "Seçilen segmentte çok az satır varsa bazı hesaplamalar boş dönebilir.",
                    "Bazı hesaplamalar df_active (segment filtreli), bazıları df_train kullanır — "
                    "OOT/train split segment filtresini de etkiler.",
                ], "info"),

            ], width=8),
        ]),

        # ── Alt: Versiyon / env notu ──────────────────────────────────────────
        html.Hr(style={"borderColor": "#2d3a4f", "marginTop": "1.5rem"}),
        html.Div([
            html.Span("Bağımlılıklar: ", style={"color": "#556070", "fontSize": "0.72rem"}),
            html.Span("optbinning · scikit-learn · statsmodels · lightgbm · xgboost · shap",
                      style={"color": "#7e8fa4", "fontSize": "0.72rem"}),
            html.Br(),
            html.Span("sklearn ≥ 1.6 monkey-patch: ", style={"color": "#556070", "fontSize": "0.72rem"}),
            html.Span("modules/deep_dive.py — başında otomatik uygulanır.",
                      style={"color": "#7e8fa4", "fontSize": "0.72rem"}),
        ]),
    ], style={"padding": "1.5rem"})


def build_sidebar():
    return html.Div([

        # ── Bölüm 0: Kayıtlı Profil ─────────────────────────────────────────
        html.P("Kayıtlı Profil", className="sidebar-section-title"),
        dbc.Row([
            dbc.Col(
                dcc.Dropdown(
                    id="dd-profile",
                    options=[],
                    value=None,
                    placeholder="Profil seçin…",
                    searchable=False,
                    className="dark-dd",
                    style={"fontSize": "0.78rem"},
                ),
                width=9, style={"paddingRight": "4px"},
            ),
            dbc.Col(
                dbc.Button("Yükle", id="btn-profile-load", color="primary",
                           size="sm", style={"width": "100%", "fontSize": "0.78rem",
                                             "height": "36px", "padding": "0"}),
                width=3, style={"paddingLeft": "4px", "display": "flex", "alignItems": "stretch"},
            ),
        ], className="g-0 mb-1"),
        html.Div(id="profile-status", style={"fontSize": "0.75rem", "marginBottom": "0.5rem"}),
        html.Hr(className="sidebar-divider"),

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
            # Bağlantı bilgileri
            dbc.Row([
                dbc.Col([
                    dbc.Label("Server", className="form-label"),
                    dbc.Input(id="input-sql-server", type="text",
                              placeholder="SERVERNAME",
                              className="form-control",
                              style={"fontSize": "0.82rem"}),
                ], width=12, className="mb-2"),
                dbc.Col([
                    dbc.Label("Database", className="form-label"),
                    dbc.Input(id="input-sql-database", type="text",
                              placeholder="DatabaseName",
                              className="form-control",
                              style={"fontSize": "0.82rem"}),
                ], width=12, className="mb-2"),
                dbc.Col([
                    dbc.Label("Driver", className="form-label"),
                    dbc.Select(
                        id="dd-sql-driver",
                        options=[
                            {"label": "ODBC Driver 18", "value": "ODBC Driver 18 for SQL Server"},
                            {"label": "ODBC Driver 17", "value": "ODBC Driver 17 for SQL Server"},
                            {"label": "ODBC Driver 13", "value": "ODBC Driver 13 for SQL Server"},
                        ],
                        className="dark-select",
                        style={"fontSize": "0.82rem"},
                    ),
                ], width=12, className="mb-3"),
            ], className="g-0"),

            # Tablo listesi
            dbc.Label("Tablolar", className="form-label"),
            # Tablo 1 — master, her zaman görünür
            dbc.Input(id="input-table-1", type="text", placeholder="dbo.TABLO1",
                      className="form-control mb-1",
                      style={"fontSize": "0.82rem"}),
            html.Div(id="div-sql-jk-1", style={"display": "none"}, children=[
                dbc.Input(id="input-sql-jk-1", type="text",
                          placeholder="Join key: must_no, sube_kodu",
                          className="form-control mb-1",
                          style={"fontSize": "0.75rem", "color": "#a78bfa"}),
            ]),
            # Tablo 2 — gizli başlar
            html.Div(id="sql-table-row-2", style={"display": "none"}, children=[
                dbc.InputGroup([
                    dbc.Input(id="input-table-2", type="text", placeholder="dbo.TABLO2",
                              className="form-control",
                              style={"fontSize": "0.82rem"}),
                    dbc.RadioItems(
                        id="radio-sql-join-2",
                        options=[{"label": "LEFT", "value": "left"},
                                 {"label": "INNER", "value": "inner"}],
                        value="left", inline=True,
                        className="join-type-toggle",
                    ),
                    dbc.Button("×", id="btn-remove-sql-2", color="link",
                               style={"color": "#ef4444", "fontSize": "1rem",
                                      "padding": "0 0.5rem"}),
                ], className="mb-1"),
                dbc.Input(id="input-sql-jk-2", type="text",
                          placeholder="Join key: bd_must_no, sube_kodu",
                          className="form-control mb-1",
                          style={"fontSize": "0.75rem", "color": "#a78bfa"}),
            ]),
            # Tablo 3 — gizli başlar
            html.Div(id="sql-table-row-3", style={"display": "none"}, children=[
                dbc.InputGroup([
                    dbc.Input(id="input-table-3", type="text", placeholder="dbo.TABLO3",
                              className="form-control",
                              style={"fontSize": "0.82rem"}),
                    dbc.RadioItems(
                        id="radio-sql-join-3",
                        options=[{"label": "LEFT", "value": "left"},
                                 {"label": "INNER", "value": "inner"}],
                        value="left", inline=True,
                        className="join-type-toggle",
                    ),
                    dbc.Button("×", id="btn-remove-sql-3", color="link",
                               style={"color": "#ef4444", "fontSize": "1rem",
                                      "padding": "0 0.5rem"}),
                ], className="mb-1"),
                dbc.Input(id="input-sql-jk-3", type="text",
                          placeholder="Join key: musteri_id, sube_kodu",
                          className="form-control mb-1",
                          style={"fontSize": "0.75rem", "color": "#a78bfa"}),
            ]),
            dbc.Button("+ Tablo Ekle", id="btn-add-sql-table", size="sm",
                       color="link", n_clicks=0,
                       style={"fontSize": "0.75rem", "color": "#4F8EF7",
                              "padding": "0", "marginBottom": "0.5rem"}),

            dbc.Checklist(
                id="chk-sql-top1000",
                options=[{"label": " İlk 1000 satır (test için)", "value": "top1000"}],
                value=[],
                inline=True,
                style={"fontSize": "0.73rem", "color": "#7e8fa4",
                       "marginBottom": "0.4rem"},
            ),
            dbc.Button("Veriyi Yükle", id="btn-load", className="btn-load mb-1", n_clicks=0),

            # State: kaç tablo görünür
            dcc.Store(id="store-sql-table-count", data=1),
        ]),

        # CSV paneli
        html.Div(id="source-csv-div", style={"display": "none"}, children=[
            # Dosya 1 — master, her zaman görünür
            dbc.Label("Dosyalar", className="form-label"),
            dcc.Upload(
                id="upload-csv",
                children=html.Div([
                    html.Span("Dosya 1 — sürükle veya tıkla",
                              style={"color": "#a8b2c2", "fontSize": "0.8rem"}),
                ]),
                accept=".csv",
                style={"width": "100%", "borderWidth": "1px", "borderStyle": "dashed",
                       "borderRadius": "6px", "borderColor": "#2d3a4f",
                       "textAlign": "center", "backgroundColor": "#0e1117",
                       "padding": "0.6rem 0.5rem", "cursor": "pointer",
                       "marginBottom": "0.25rem"},
                style_active={"borderColor": "#4F8EF7", "backgroundColor": "#111f35"},
            ),
            html.Div(id="csv-filename-display",
                     style={"color": "#a78bfa", "fontSize": "0.72rem",
                            "marginBottom": "0.4rem", "fontStyle": "italic"}),
            html.Div(id="div-csv-jk-1", style={"display": "none"}, children=[
                dbc.Input(id="input-csv-jk-1", type="text",
                          placeholder="Join key: must_no, sube_kodu",
                          className="form-control mb-1",
                          style={"fontSize": "0.75rem", "color": "#a78bfa"}),
            ]),

            # Dosya 2 — gizli başlar
            html.Div(id="csv-file-row-2", style={"display": "none"}, children=[
                dcc.Upload(
                    id="upload-csv-2",
                    children=html.Div([
                        html.Span("Dosya 2 — sürükle veya tıkla",
                                  style={"color": "#a8b2c2", "fontSize": "0.8rem"}),
                    ]),
                    accept=".csv",
                    style={"width": "100%", "borderWidth": "1px", "borderStyle": "dashed",
                           "borderRadius": "6px", "borderColor": "#2d3a4f",
                           "textAlign": "center", "backgroundColor": "#0e1117",
                           "padding": "0.6rem 0.5rem", "cursor": "pointer",
                           "marginBottom": "0.25rem"},
                    style_active={"borderColor": "#4F8EF7", "backgroundColor": "#111f35"},
                ),
                dbc.Row([
                    dbc.Col(html.Div(id="csv-filename-display-2",
                                    style={"color": "#a78bfa", "fontSize": "0.72rem",
                                           "fontStyle": "italic"}), width=6),
                    dbc.Col(dbc.RadioItems(
                        id="radio-csv-join-2",
                        options=[{"label": "LEFT", "value": "left"},
                                 {"label": "INNER", "value": "inner"}],
                        value="left", inline=True,
                        className="join-type-toggle",
                    ), width=4),
                    dbc.Col(dbc.Button("×", id="btn-remove-csv-2", color="link",
                                       style={"color": "#ef4444", "fontSize": "1rem",
                                              "padding": "0"}), width=2),
                ], className="mb-1 g-0"),
                dbc.Input(id="input-csv-jk-2", type="text",
                          placeholder="Join key: bd_must_no, sube_kodu",
                          className="form-control mb-1",
                          style={"fontSize": "0.75rem", "color": "#a78bfa"}),
            ]),

            # Dosya 3 — gizli başlar
            html.Div(id="csv-file-row-3", style={"display": "none"}, children=[
                dcc.Upload(
                    id="upload-csv-3",
                    children=html.Div([
                        html.Span("Dosya 3 — sürükle veya tıkla",
                                  style={"color": "#a8b2c2", "fontSize": "0.8rem"}),
                    ]),
                    accept=".csv",
                    style={"width": "100%", "borderWidth": "1px", "borderStyle": "dashed",
                           "borderRadius": "6px", "borderColor": "#2d3a4f",
                           "textAlign": "center", "backgroundColor": "#0e1117",
                           "padding": "0.6rem 0.5rem", "cursor": "pointer",
                           "marginBottom": "0.25rem"},
                    style_active={"borderColor": "#4F8EF7", "backgroundColor": "#111f35"},
                ),
                dbc.Row([
                    dbc.Col(html.Div(id="csv-filename-display-3",
                                    style={"color": "#a78bfa", "fontSize": "0.72rem",
                                           "fontStyle": "italic"}), width=6),
                    dbc.Col(dbc.RadioItems(
                        id="radio-csv-join-3",
                        options=[{"label": "LEFT", "value": "left"},
                                 {"label": "INNER", "value": "inner"}],
                        value="left", inline=True,
                        className="join-type-toggle",
                    ), width=4),
                    dbc.Col(dbc.Button("×", id="btn-remove-csv-3", color="link",
                                       style={"color": "#ef4444", "fontSize": "1rem",
                                              "padding": "0"}), width=2),
                ], className="mb-1 g-0"),
                dbc.Input(id="input-csv-jk-3", type="text",
                          placeholder="Join key: musteri_id, sube_kodu",
                          className="form-control mb-1",
                          style={"fontSize": "0.75rem", "color": "#a78bfa"}),
            ]),

            dbc.Button("+ Dosya Ekle", id="btn-add-csv-file", size="sm",
                       color="link", n_clicks=0,
                       style={"fontSize": "0.75rem", "color": "#4F8EF7",
                              "padding": "0", "marginBottom": "0.5rem"}),

            dbc.Row([
                dbc.Col(
                    dbc.Select(
                        id="csv-separator",
                        options=[
                            {"label": "Virgül  (,)",        "value": ","},
                            {"label": "Noktalı virgül (;)", "value": ";"},
                            {"label": "Tab (\\t)",          "value": "\t"},
                            {"label": "Pipe  (|)",          "value": "|"},
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

            # State: kaç dosya görünür
            dcc.Store(id="store-csv-file-count", data=1),
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
                dcc.Dropdown(
                    id="dd-target-col",
                    options=[],
                    value=None,
                    placeholder="Kolon ara…",
                    searchable=True,
                    className="dark-dd mb-3",
                ),

                dbc.Label("Tarih Kolonu", className="form-label"),
                html.Div("opsiyonel", className="form-hint"),
                dcc.Dropdown(
                    id="dd-date-col",
                    options=[],
                    value=None,
                    placeholder="Kolon ara…",
                    searchable=True,
                    className="dark-dd mb-3",
                ),

                # OOT tarihi — tarih kolonu seçilince açılır
                dbc.Collapse(
                    html.Div([
                        dbc.Label("OOT Başlangıç Tarihi", className="form-label"),
                        html.Div("Train = öncesi  ·  OOT = sonrası", className="form-hint"),
                        dbc.Select(
                            id="dd-oot-date",
                            options=[{"label": "— opsiyonel —", "value": ""}],
                            value="",
                            className="dark-select mb-2",
                        ),
                        dbc.Checklist(
                            id="chk-train-test-split",
                            options=[{"label": " Train / Test bölünmesi", "value": "split"}],
                            value=[],
                            className="mb-1",
                            style={"color": "#c8cdd8", "fontSize": "0.82rem"},
                            inputStyle={"marginRight": "5px"},
                        ),
                        dbc.Collapse(
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Test Oranı (%)", className="form-label",
                                              style={"fontSize": "0.78rem", "marginBottom": "2px"}),
                                    dbc.Input(id="input-test-size", type="number",
                                              value=20, min=10, max=50, step=5,
                                              style={"maxWidth": "90px", "fontSize": "0.82rem"}),
                                ]),
                            ], className="mb-2"),
                            id="collapse-test-size-cfg",
                            is_open=False,
                        ),
                    ]),
                    id="collapse-oot-date",
                    is_open=False,
                ),

                dbc.Label("Segment Kolonu", className="form-label"),
                html.Div("opsiyonel", className="form-hint"),
                dcc.Dropdown(
                    id="dd-segment-col",
                    options=[],
                    value=None,
                    placeholder="Kolon ara…",
                    searchable=True,
                    className="dark-dd mb-1",
                ),

                # Segment değer seçimi — segment kolonu seçilince görünür
                dbc.Collapse(
                    html.Div([
                        dbc.Label(id="segment-val-label", className="form-label"),
                        dcc.Dropdown(
                            id="dd-segment-val",
                            options=[],
                            value=["Tümü"],
                            multi=True,
                            searchable=False,
                            placeholder="Segment seçin…",
                            className="dark-dd mb-3",
                        ),
                    ]),
                    id="collapse-segment",
                    is_open=False,
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

        # ── Bölüm 4: Profil Kaydet / Sil ────────────────────────────────────
        dbc.Collapse(
            html.Div([
                html.Hr(className="sidebar-divider"),
                dbc.Row([
                    dbc.Col(
                        dbc.Button("Profil Kaydet", id="btn-profile-save", color="success",
                                   size="sm", outline=True,
                                   style={"width": "100%", "fontSize": "0.72rem"}),
                        width=6, style={"paddingRight": "4px"},
                    ),
                    dbc.Col(
                        dbc.Button("Profil Sil", id="btn-profile-delete", color="danger",
                                   size="sm", outline=True,
                                   style={"width": "100%", "fontSize": "0.72rem"}),
                        width=6, style={"paddingLeft": "4px"},
                    ),
                ], className="g-0"),
            ]),
            id="collapse-profile-actions",
            is_open=False,
        ),

        # ── Profil Kaydet Modal ──────────────────────────────────────────────
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Profil Kaydet",
                                           style={"fontSize": "0.95rem"})),
            dbc.ModalBody([
                dbc.Label("Profil Adı", className="form-label"),
                dbc.Input(id="input-profile-name", type="text",
                          placeholder="ör. İsim_segment",
                          className="form-control",
                          style={"fontSize": "0.82rem"}),
            ]),
            dbc.ModalFooter(
                dbc.Button("Kaydet", id="btn-profile-save-confirm",
                           color="success", size="sm"),
            ),
        ], id="modal-profile-save", is_open=False, centered=True),

        # ── Profil Sil Modal ──────────────────────────────────────────────────
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Profil Sil",
                                           style={"fontSize": "0.95rem"})),
            dbc.ModalBody([
                dbc.Label("Silinecek Profil", className="form-label"),
                dcc.Dropdown(
                    id="dd-profile-delete",
                    options=[],
                    value=None,
                    placeholder="Profil seçin…",
                    searchable=False,
                    className="dark-dd",
                    style={"fontSize": "0.78rem"},
                ),
                html.Div(id="delete-confirm-area", style={"marginTop": "0.75rem"}),
            ]),
            dbc.ModalFooter(
                dbc.Button("Evet, Sil", id="btn-profile-delete-confirm",
                           color="danger", size="sm", style={"display": "none"}),
            ),
        ], id="modal-profile-delete", is_open=False, centered=True),

        # ── Profil Kaydet Başarı Toast ───────────────────────────────────────
        dbc.Toast(
            id="toast-profile-saved",
            header="Profil Kaydedildi",
            is_open=False,
            duration=4000,
            icon="success",
            style={"position": "fixed", "top": 20, "right": 20, "zIndex": 9999,
                   "minWidth": "280px"},
        ),

    ], id="sidebar")


def build_main():
    return html.Div([
        # ── Yardım Overlay ────────────────────────────────────────────────────
        html.Div(
            id="help-overlay",
            style={"display": "none", "position": "absolute", "inset": "0",
                   "zIndex": "500", "backgroundColor": "#0E1117",
                   "overflowY": "auto", "padding": "0"},
            children=[
                html.Div([
                    html.Div([
                        html.Button(
                            "✕",
                            id="btn-help-close",
                            n_clicks=0,
                            style={
                                "background": "none", "border": "none",
                                "color": "#8892a4", "cursor": "pointer",
                                "fontSize": "1.2rem", "lineHeight": "1",
                                "padding": "0.25rem 0.5rem",
                            },
                        ),
                    ], style={"display": "flex", "justifyContent": "flex-end",
                              "padding": "0.75rem 1rem 0"}),
                    _build_help_tab(),
                ]),
            ],
        ),
        html.Div(id="config-banner"),
        html.Div(id="metrics-row", style={"marginBottom": "1.5rem"}),
        dbc.Tabs([
            dbc.Tab(dcc.Loading(html.Div(id="data-preview"),   type="dot", color="#4F8EF7", delay_show=200), label="Önizleme",  tab_id="tab-preview",   className="tab-content-area"),
            dbc.Tab(dcc.Loading(html.Div(id="tab-profiling"), type="dot", color="#4F8EF7", delay_show=200), label="Describe", tab_id="tab-profiling", className="tab-content-area"),
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

                # ── Ham / WoE Veri Kaynağı Seçimi ──────────────────────
                dbc.Tabs(
                    id="stat-data-tab",
                    active_tab="stat-tab-raw",
                    children=[
                        dbc.Tab(label="Ham Değerler", tab_id="stat-tab-raw",
                                tab_style={"fontSize": "0.78rem"},
                                active_label_style={"color": "#10b981", "fontWeight": "700"}),
                        dbc.Tab(label="WoE Değerler", tab_id="stat-tab-woe",
                                tab_style={"fontSize": "0.78rem"},
                                active_label_style={"color": "#4F8EF7", "fontWeight": "700"}),
                    ],
                    className="mb-3",
                ),

                # ── Korelasyon Paneli ─────────────────────────────────────
                html.Div(id="stat-corr-panel", children=[
                    html.Div([
                        html.Div([
                            html.Span("Korelasyon Analizi", style={"color": "#c8cdd8", "fontWeight": "700", "fontSize": "0.82rem"}),
                            html.Span("  ·  Pearson r", style={"color": "#7e8fa4", "fontSize": "0.78rem"}),
                        ], style={"marginBottom": "0.45rem"}),
                        html.Div([
                            html.Span("Amaç: ", style={"color": "#a8b2c2", "fontWeight": "600", "fontSize": "0.78rem"}),
                            html.Span("İki sayısal değişken arasındaki doğrusal ilişkinin yönünü ve şiddetini ölçer. Modele birlikte girecek değişkenlerin birbirini tekrar edip etmediğini anlamak için kullanılır.", style={"color": "#6b7a94", "fontSize": "0.78rem"}),
                        ], style={"marginBottom": "0.25rem"}),
                        html.Div([
                            html.Span("Nasıl çalışır: ", style={"color": "#a8b2c2", "fontWeight": "600", "fontSize": "0.78rem"}),
                            html.Span("Pearson r katsayısı −1 ile +1 arasında değer alır. +1 tam pozitif, −1 tam negatif doğrusal ilişki, 0 ilişki yok demektir. Heatmap'te koyu renkler güçlü ilişkiyi gösterir. Değişken sayısı eşiği aşarsa en yüksek varyanslılar seçilir.", style={"color": "#6b7a94", "fontSize": "0.78rem"}),
                        ], style={"marginBottom": "0.25rem"}),
                        html.Div([
                            html.Span("Çıktıyı nasıl okurum: ", style={"color": "#a8b2c2", "fontWeight": "600", "fontSize": "0.78rem"}),
                            html.Span("|r| < 0.50 → düşük (sorun yok)  ·  0.50–0.75 → orta (dikkat)  ·  > 0.75 → yüksek (çoklu doğrusallık riski). Eşiği aşan çiftler tabloda listelenir — bu çiftlerden IV'ü düşük olanı modelden çıkarın.", style={"color": "#6b7a94", "fontSize": "0.78rem"}),
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
                            html.Span("İki kategorik değişken arasında istatistiksel olarak anlamlı bir ilişki olup olmadığını test eder. Örneğin \"meslek\" ile \"temerrüt\" arasında bağımlılık var mı? Sayısal kolonlar otomatik bin'lenir.", style={"color": "#6b7a94", "fontSize": "0.78rem"}),
                        ], style={"marginBottom": "0.25rem"}),
                        html.Div([
                            html.Span("Nasıl çalışır: ", style={"color": "#a8b2c2", "fontWeight": "600", "fontSize": "0.78rem"}),
                            html.Span("İki değişkenin çapraz tablosu (crosstab) oluşturulur. Her hücredeki gözlenen frekans, \"iki değişken bağımsız olsaydı beklenen frekans\" ile karşılaştırılır. Fark büyükse χ² yükselir ve p-değeri düşer.", style={"color": "#6b7a94", "fontSize": "0.78rem"}),
                        ], style={"marginBottom": "0.25rem"}),
                        html.Div([
                            html.Span("Çıktıyı nasıl okurum: ", style={"color": "#a8b2c2", "fontWeight": "600", "fontSize": "0.78rem"}),
                            html.Span("p < 0.05 → değişkenler arasında anlamlı ilişki var (bağımsız değil). p ≥ 0.05 → ilişki istatistiksel olarak kanıtlanamadı. Cramér's V ilişkinin gücünü verir: < 0.10 önemsiz · 0.10–0.30 zayıf · 0.30–0.50 orta · ≥ 0.50 güçlü.", style={"color": "#6b7a94", "fontSize": "0.78rem"}),
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
                            dcc.Dropdown(id="chi-var1", className="dark-select", searchable=True, placeholder="Kolon ara\u2026"),
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Değişken 2", className="form-label"),
                            html.Div("Kategorik veya sayısal", className="form-hint"),
                            dcc.Dropdown(id="chi-var2", className="dark-select", searchable=True, placeholder="Kolon ara\u2026"),
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
                            html.Span("Bir sayısal değişkenin Good (target=0) ve Bad (target=1) grupları arasında gerçekten farklı davranıp davranmadığını test eder. Örneğin \"gelir\" değişkeni bad müşterilerde anlamlı şekilde düşük mü?", style={"color": "#6b7a94", "fontSize": "0.78rem"}),
                        ], style={"marginBottom": "0.25rem"}),
                        html.Div([
                            html.Span("Nasıl çalışır: ", style={"color": "#a8b2c2", "fontWeight": "600", "fontSize": "0.78rem"}),
                            html.Span("Gruplar arası varyans (ortalamalar arası fark) ile grup içi varyans (her grubun kendi dağılımı) oranlanır → F istatistiği. F büyükse gruplar arası fark, grup içi değişkenlikten belirgin şekilde fazladır.", style={"color": "#6b7a94", "fontSize": "0.78rem"}),
                        ], style={"marginBottom": "0.25rem"}),
                        html.Div([
                            html.Span("Çıktıyı nasıl okurum: ", style={"color": "#a8b2c2", "fontWeight": "600", "fontSize": "0.78rem"}),
                            html.Span("F yüksek + p < 0.05 → grupların ortalamaları anlamlı şekilde farklı, değişken ayırıcı güç taşıyor. p ≥ 0.05 → ortalamalar benzer, değişken tek başına grupları ayıramıyor. Grup istatistikleri tablosunda ortalama ve standart sapmayı karşılaştırın.", style={"color": "#6b7a94", "fontSize": "0.78rem"}),
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
                            dcc.Dropdown(id="anova-var", className="dark-select", searchable=True, placeholder="Kolon ara\u2026"),
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
                            html.Span("Bir değişkenin Good ve Bad gruplarını ne kadar iyi ayırabildiğini ölçer. Kredi risk skorlamalarında modelin ayırıcılık gücünü değerlendiren temel metriktir.", style={"color": "#6b7a94", "fontSize": "0.78rem"}),
                        ], style={"marginBottom": "0.25rem"}),
                        html.Div([
                            html.Span("Nasıl çalışır: ", style={"color": "#a8b2c2", "fontWeight": "600", "fontSize": "0.78rem"}),
                            html.Span("Good ve Bad gruplarının kümülatif dağılım eğrileri (CDF) ayrı ayrı çizilir. İki eğri arasındaki en büyük dikey mesafe KS istatistiğini verir. Mesafe büyükse değişken iki grubu net şekilde ayırıyor demektir.", style={"color": "#6b7a94", "fontSize": "0.78rem"}),
                        ], style={"marginBottom": "0.25rem"}),
                        html.Div([
                            html.Span("Çıktıyı nasıl okurum: ", style={"color": "#a8b2c2", "fontWeight": "600", "fontSize": "0.78rem"}),
                            html.Span("KS < 0.20 → zayıf ayırıcılık · 0.20–0.30 → orta · 0.30–0.50 → iyi · > 0.50 → çok iyi. p < 0.05 → iki grubun dağılımı istatistiksel olarak farklı. Grafikteki sarı noktalı çizgi en büyük ayrışma noktasını gösterir — değişkenin en etkili olduğu bölgedir.", style={"color": "#6b7a94", "fontSize": "0.78rem"}),
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
                            dcc.Dropdown(id="ks-var", className="dark-select", searchable=True, placeholder="Kolon ara\u2026"),
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
                            html.Span("Modeldeki bağımsız değişkenlerin birbirini ne kadar tekrar ettiğini ölçer. Yüksek VIF, bir değişkenin diğerlerinin kombinasyonuyla neredeyse tamamen açıklanabildiğini gösterir — bu durumda katsayılar güvenilmez olur.", style={"color": "#6b7a94", "fontSize": "0.78rem"}),
                        ], style={"marginBottom": "0.25rem"}),
                        html.Div([
                            html.Span("Nasıl çalışır: ", style={"color": "#a8b2c2", "fontWeight": "600", "fontSize": "0.78rem"}),
                            html.Span("Her değişken sırayla diğer tüm değişkenlere regresse edilir. Elde edilen R²'den VIF = 1/(1−R²) hesaplanır. R² yüksekse (değişken diğerlerince açıklanıyorsa) VIF büyük çıkar.", style={"color": "#6b7a94", "fontSize": "0.78rem"}),
                        ], style={"marginBottom": "0.25rem"}),
                        html.Div([
                            html.Span("Çıktıyı nasıl okurum: ", style={"color": "#a8b2c2", "fontWeight": "600", "fontSize": "0.78rem"}),
                            html.Span("VIF < 5 → sorun yok · 5–10 → dikkat, korelasyon tablosunu kontrol edin · > 10 → bu değişken diğerleriyle çok ilişkili, modelden çıkarılmalı veya WoE dönüşümü uygulanmalı. En yüksek VIF'li değişkeni çıkarıp tekrar hesaplayın — diğer VIF'ler de düşecektir.", style={"color": "#6b7a94", "fontSize": "0.78rem"}),
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
                # Hidden — callback uyumluluğu için (eski checkbox/buton kaldırıldı)
                dbc.Checklist(id="chk-varsummary-woe", options=[], value=["woe"],
                              style={"display": "none"}),
                html.Div(id="btn-var-summary", style={"display": "none"}),
                # Ham / WoE tab ayrımı
                dbc.Tabs(
                    id="varsummary-data-tab",
                    active_tab="vs-tab-woe",
                    children=[
                        dbc.Tab(label="WoE Değerler", tab_id="vs-tab-woe",
                                tab_style={"fontSize": "0.78rem"},
                                active_label_style={"color": "#4F8EF7", "fontWeight": "700"}),
                        dbc.Tab(label="Ham Değerler", tab_id="vs-tab-raw",
                                tab_style={"fontSize": "0.78rem"},
                                active_label_style={"color": "#10b981", "fontWeight": "700"}),
                    ],
                    className="mb-3",
                ),
                dcc.Loading(html.Div(id="div-var-summary"), type="dot", color="#4F8EF7", delay_show=300),
            ]), label="Değişken Özeti", tab_id="tab-var-summary", className="tab-content-area"),
            dbc.Tab(html.Div([
                _tab_info("Modelleme", "Grafik Oluşturucu · Hızlı Model · SHAP",
                          "İki bölümden oluşur: Grafik Oluşturucu ile serbest keşif yapabilir, "
                          "Hızlı Model ile LR / LightGBM / XGBoost modellerini "
                          "hızlıca eğitip AUC, Gini, KS metriklerini ve SHAP beeswarm grafiğini "
                          "görebilirsiniz. WoE encode ve ham encode sonuçlar yan yana karşılaştırılır.",
                          "#f59e0b"),
                # ── Grafik Oluşturucu ─────────────────────────────────────────
                html.P("Grafik Oluşturucu", className="section-title"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("X Ekseni", className="form-label"),
                        html.Div("\u00a0", className="form-hint"),
                        dcc.Dropdown(id="pg-x-col", className="dark-select", searchable=True, placeholder="Kolon ara\u2026"),
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Y Ekseni (Sol)", className="form-label"),
                        html.Div("\u00a0", className="form-hint"),
                        dcc.Dropdown(id="pg-y-col", className="dark-select", searchable=True, placeholder="Kolon ara\u2026"),
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Y2 Ekseni (Sağ)", className="form-label"),
                        html.Div("Seçilirse sağ eksende çizgi olarak eklenir.",
                                 className="form-hint"),
                        dcc.Dropdown(id="pg-y2-col", className="dark-select", searchable=True, placeholder="Kolon ara\u2026", value=None),
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
                        dcc.Dropdown(id="pg-color-col", className="dark-select", searchable=True, placeholder="Kolon ara\u2026", value=None),
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
                    dbc.Col([
                        html.Div([
                            dbc.Label("Model Değişkenleri", className="form-label",
                                      style={"marginBottom": "0"}),
                            html.Span(id="pg-model-count",
                                      className="form-hint",
                                      style={"marginLeft": "0.5rem"}),
                        ], style={"display": "flex", "alignItems": "baseline",
                                  "marginBottom": "0.35rem"}),
                        dcc.Dropdown(
                            id="pg-var-dropdown",
                            multi=True,
                            searchable=True,
                            placeholder="Değişken ara ve seç…",
                            className="dark-dd",
                            style={"minHeight": "42px"},
                        ),
                    ], width=10),
                    dbc.Col([
                        html.Div([
                            dbc.Button("Tümünü Ekle", id="btn-pg-add-all",
                                       color="primary", size="sm",
                                       className="mb-2",
                                       style={"width": "100%", "fontSize": "0.75rem"}),
                            dbc.Button("Temizle", id="btn-pg-remove-all",
                                       color="secondary", size="sm",
                                       outline=True,
                                       style={"width": "100%", "fontSize": "0.75rem"}),
                        ], style={"paddingTop": "1.5rem"}),
                    ], width=2),
                ], className="mb-3"),

                # Eski callback'ler için gizli elemanlar (ID uyumluluğu)
                html.Div([
                    html.Div(id="pg-source-container"),
                    html.Span(id="pg-source-count"),
                    dbc.Input(id="pg-source-search", style={"display": "none"}),
                    html.Div(id="pg-model-container"),
                    dbc.Button(id="btn-pg-add", style={"display": "none"}),
                    dbc.Button(id="btn-pg-remove", style={"display": "none"}),
                ], style={"display": "none"}),

                # WoE toggle (gizli — arka planda her zaman her ikisi de hesaplanır)
                html.Div(dbc.Checklist(id="chk-use-woe", options=[], value=[]),
                         style={"display": "none"}),

                # ── Kayıtlı Modeller ──────────────────────────────────────
                dbc.Collapse(
                    html.Div([
                        dbc.Label("Kayıtlı Modeller", className="form-label"),
                        dcc.Dropdown(
                            id="dd-saved-models",
                            options=[],
                            value=None,
                            placeholder="Model seçin…",
                            searchable=False,
                            className="dark-dd mb-2",
                        ),
                        html.Div([
                            dbc.Button("Yükle", id="btn-model-load",
                                       color="primary", size="sm",
                                       style={"fontSize": "0.75rem"}),
                            dbc.Button("Kaydet", id="btn-model-save",
                                       color="success", size="sm", className="ms-1",
                                       style={"fontSize": "0.75rem"}),
                            dbc.Button("Sil", id="btn-model-delete",
                                       color="danger", size="sm", outline=True,
                                       className="ms-1",
                                       style={"fontSize": "0.75rem"}),
                        ], className="d-flex mb-2"),
                        html.Div(id="model-save-status"),
                        dbc.Button("Evet, Sil", id="btn-model-delete-confirm",
                                   color="danger", size="sm",
                                   style={"display": "none", "fontSize": "0.75rem"},
                                   className="mt-1"),
                    ], style={"marginBottom": "0.75rem",
                              "padding": "0.5rem 0.6rem",
                              "backgroundColor": "#0d1520",
                              "borderRadius": "6px",
                              "border": "1px solid #1e2a3a"}),
                    id="collapse-model-actions",
                    is_open=False,
                ),

                # Üstüne kaydetme onay modalı
                dbc.Modal([
                    dbc.ModalHeader(
                        html.Span("Model Üstüne Kaydet", style={
                            "fontWeight": "700", "color": "#c8cdd8"}),
                        close_button=True,
                        style={"backgroundColor": "#111827",
                               "borderBottom": "1px solid #1e2a3a"}),
                    dbc.ModalBody(
                        html.Div(id="modal-overwrite-body",
                                 style={"color": "#d1d5db", "fontSize": "0.85rem"}),
                        style={"backgroundColor": "#111827"}),
                    dbc.ModalFooter([
                        dbc.Button("Evet, Üstüne Kaydet", id="btn-model-overwrite-confirm",
                                   color="warning", size="sm",
                                   style={"fontSize": "0.78rem"}),
                        dbc.Button("Vazgeç", id="btn-model-overwrite-cancel",
                                   color="secondary", size="sm", outline=True,
                                   style={"fontSize": "0.78rem"}),
                    ], style={"backgroundColor": "#111827",
                              "borderTop": "1px solid #1e2a3a"}),
                ], id="modal-model-overwrite", is_open=False, centered=True,
                   backdrop="static", style={"--bs-modal-bg": "#111827"}),

                # Model parametreleri — satır 1: target + test oranı
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Target Kolonu", className="form-label"),
                        html.Div("\u00a0", className="form-hint"),
                        dcc.Dropdown(id="pg-target-col", className="dark-select", searchable=True, placeholder="Kolon ara\u2026"),
                    ], width=3),
                    # Gizli — callback uyumluluğu için korunur
                    html.Div([
                        dbc.Input(id="pg-test-size", type="number", value=30),
                        dbc.Select(id="pg-split-method", value="random",
                                   options=[{"label": "Rastgele", "value": "random"}]),
                        dbc.Select(id="pg-split-date", value=""),
                    ], style={"display": "none"}),
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
                                   ],
                                   value="lr",
                                   style={"maxWidth": "220px"}),
                    ], width=3),
                    html.Div(
                        dbc.Input(id="pg-null-strategy", type="hidden", value="median"),
                        style={"display": "none"},
                    ),
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
                    ], id="pg-col-threshold", width=3),
                    dbc.Col([
                        dbc.Label("Özel Eşik", className="form-label"),
                        html.Div("'Özel' seçiliyse geçerlidir", className="form-hint"),
                        dbc.Input(id="pg-threshold-val", type="number",
                                  value=0.50, min=0.01, max=0.99, step=0.01,
                                  style={"maxWidth": "100px"}),
                    ], id="pg-col-threshold-val", width=2),
                    dbc.Col([
                        html.Div("\u00a0", className="form-label"),
                        html.Div("\u00a0", className="form-hint"),
                        dbc.Button("Model Kur", id="btn-pg-build",
                                   color="success", size="sm"),
                    ], width=2),
                ], className="mb-3"),
                html.Div(id="pg-null-review-panel"),
                dcc.Loading(html.Div(id="pg-model-output"),
                            type="dot", color="#4F8EF7", delay_show=300),

                dcc.Store(id="store-pg-model-vars", storage_type="memory"),
                dcc.Store(id="store-pg-null-strategies", storage_type="memory"),
            ]), label="Modelleme", tab_id="tab-playground",
               className="tab-content-area"),

            # ── Sonuç (Final Rapor) ────────────────────────────────────────
            dbc.Tab(
                dcc.Loading(html.Div(id="tab-results"), type="dot",
                            color="#4F8EF7", delay_show=300),
                label="Sonuç", tab_id="tab-results",
                className="tab-content-area"),
        ], id="main-tabs", active_tab="tab-preview", className="main-tabs"),
    ], id="main-content")


def _build_slideshow_modal():
    """Veri yüklenirken gösterilen eğitim slayt gösterisi modalı."""
    slides = [
        # Slide 0 — EDA Lab nedir?
        html.Div([
            html.H4("EDA Lab Nedir?", className="slide-title"),
            html.P(
                "EDA Lab, veri bilimciler ve analistler için tasarlanmış "
                "interaktif bir keşifsel veri analizi platformudur. "
                "Verilerinizi yükleyin, otomatik profilleme ile hızlıca tanıyın "
                "ve model kurmadan önce değişkenlerinizi derinlemesine inceleyin.",
                className="slide-text",
            ),
            html.Div([
                html.Span("Otomatik Profilleme", className="slide-chip"),
                html.Span("IV / WoE Analizi", className="slide-chip"),
                html.Span("Hızlı Model Kurma", className="slide-chip"),
            ], className="slide-chips"),
        ], className="slide-content"),

        # Slide 1 — Veri Yükleme
        html.Div([
            html.H4("Veri Yükleme & Yapılandırma", className="slide-title"),
            html.P(
                "SQL Server veya CSV dosyalarından veri çekin. "
                "Birden fazla tabloyu join key ile birleştirin. "
                "Target, tarih ve segment kolonlarını belirleyin — "
                "sistem geri kalanını otomatik yapılandırsın.",
                className="slide-text",
            ),
            html.Div([
                html.Div([
                    html.Span("1", className="slide-step-num"),
                    html.Span("Veri Kaynağı Seç", className="slide-step-label"),
                ], className="slide-step"),
                html.Div([
                    html.Span("2", className="slide-step-num"),
                    html.Span("Kolonları Yapılandır", className="slide-step-label"),
                ], className="slide-step"),
                html.Div([
                    html.Span("3", className="slide-step-num"),
                    html.Span("Analiz Et", className="slide-step-label"),
                ], className="slide-step"),
            ], className="slide-steps"),
        ], className="slide-content"),

        # Slide 2 — Önizleme & Screening
        html.Div([
            html.H4("Önizleme & Ön Eleme", className="slide-title"),
            html.P(
                "Verinin genel profilini görün: satır/kolon sayısı, eksik veri oranları, "
                "veri tipleri ve temel istatistikler. Yüksek eksiklik veya düşük varyans "
                "gösteren kolonları tespit edin, gerekirse eleyin.",
                className="slide-text",
            ),
            html.Div([
                html.Span("Eksik Veri Haritası", className="slide-chip"),
                html.Span("Tip Dağılımı", className="slide-chip"),
                html.Span("Otomatik Dönüşüm", className="slide-chip"),
            ], className="slide-chips"),
        ], className="slide-content"),

        # Slide 3 — Target & IV
        html.Div([
            html.H4("Target & IV Analizi", className="slide-title"),
            html.P(
                "Target değişkeninin dağılımını inceleyin. "
                "Her bağımsız değişkenin Information Value (IV) skorunu görün — "
                "değişken seçiminin ilk adımı. Düşük IV'li değişkenleri "
                "eşik değeriyle otomatik eleyin.",
                className="slide-text",
            ),
            html.Div("IV > 0.02 → Zayıf  |  IV > 0.1 → Orta  |  IV > 0.3 → Güçlü",
                      className="slide-highlight"),
        ], className="slide-content"),

        # Slide 4 — Deep Dive & WoE
        html.Div([
            html.H4("Değişken Analizi & WoE", className="slide-title"),
            html.P(
                "Her değişkeni tek tek inceleyin: dağılım, kutu grafik, "
                "bad rate eğrisi ve WoE (Weight of Evidence) grafikleri. "
                "Binning detaylarını görün, outlier tespiti yapın "
                "ve değişken bazında karar verin.",
                className="slide-text",
            ),
            html.Div([
                html.Span("Histogram", className="slide-chip"),
                html.Span("WoE Grafiği", className="slide-chip"),
                html.Span("Bad Rate", className="slide-chip"),
                html.Span("Outlier Tespiti", className="slide-chip"),
            ], className="slide-chips"),
        ], className="slide-content"),

        # Slide 5 — İstatistiksel Testler
        html.Div([
            html.H4("İstatistiksel Testler", className="slide-title"),
            html.P(
                "Korelasyon matrisi, Chi-kare bağımsızlık testi, "
                "ANOVA (grup farklılığı), KS testi (dağılım karşılaştırma) "
                "ve VIF (çoklu doğrusal bağımlılık) analizlerini tek tıkla çalıştırın.",
                className="slide-text",
            ),
            html.Div([
                html.Span("Korelasyon", className="slide-chip"),
                html.Span("Chi²", className="slide-chip"),
                html.Span("ANOVA", className="slide-chip"),
                html.Span("KS Testi", className="slide-chip"),
                html.Span("VIF", className="slide-chip"),
            ], className="slide-chips"),
        ], className="slide-content"),

        # Slide 6 — Değişken Özeti
        html.Div([
            html.H4("Değişken Özeti — Tek Bakışta Karar", className="slide-title"),
            html.P(
                "Tüm değişkenlerin IV, eksiklik, korelasyon ve "
                "istatistiksel test sonuçlarını tek bir tabloda görün. "
                "Hangi değişkeni modele alacağınıza bu özet tabloyla karar verin.",
                className="slide-text",
            ),
            html.Div("Her değişken için ✓ / ✗ karar kolonu ile hızlı seçim",
                      className="slide-highlight"),
        ], className="slide-content"),

        # Slide 7 — Playground
        html.Div([
            html.H4("Modelleme — Hızlı Model", className="slide-title"),
            html.P(
                "Seçtiğiniz değişkenlerle hızlıca Logistic Regression, "
                "LightGBM veya XGBoost modeli kurun. "
                "Gini, AUC ve katsayıları anında görün. "
                "Train / Test / OOT performansını karşılaştırın.",
                className="slide-text",
            ),
            html.Div([
                html.Span("Logistic Reg.", className="slide-chip"),
                html.Span("LightGBM", className="slide-chip"),
                html.Span("XGBoost", className="slide-chip"),
            ], className="slide-chips"),
        ], className="slide-content"),
    ]

    # Her slayta index attribute ekle, sadece ilki görünür başlasın
    slide_divs = []
    for i, slide in enumerate(slides):
        slide_divs.append(
            html.Div(
                slide,
                className="slideshow-slide" + (" slide-active" if i == 0 else ""),
                id=f"slide-{i}",
                style={"display": "block" if i == 0 else "none"},
            )
        )

    n_slides = len(slides)

    # Navigasyon noktaları
    dots = html.Div(
        [html.Span(
            "",
            className="slide-dot" + (" dot-active" if i == 0 else ""),
            id=f"slide-dot-{i}",
            n_clicks=0,
        ) for i in range(n_slides)],
        className="slide-dots",
    )

    # Progress bar
    progress = html.Div(
        html.Div(className="slide-progress-fill", id="slide-progress-fill",
                 style={"width": f"{100 / n_slides}%"}),
        className="slide-progress-bar",
    )

    return dbc.Modal([
        dbc.ModalHeader(
            html.Div([
                html.Div([
                    html.Span("Veri Yükleniyor", style={
                        "fontWeight": "700", "fontSize": "1.05rem", "color": "#c8cdd8",
                    }),
                    html.Span(" — ", style={"color": "#3B4A63"}),
                    html.Span(id="slideshow-elapsed", children="0:00", style={
                        "fontSize": "0.85rem", "color": "#6b7a99", "fontVariantNumeric": "tabular-nums",
                    }),
                ], style={"flex": "1"}),
                dbc.Button("✕", id="btn-slideshow-close", size="sm", outline=True,
                           color="secondary",
                           style={"padding": "0.15rem 0.5rem", "fontSize": "0.85rem",
                                  "lineHeight": "1", "borderColor": "#3B4A63",
                                  "color": "#6b7a99"}),
            ], style={"display": "flex", "alignItems": "center", "width": "100%"}),
            close_button=False,
            style={"backgroundColor": "#0e1117", "borderBottom": "1px solid #1e2a3a"},
        ),
        dbc.ModalBody([
            html.Div(slide_divs, className="slideshow-container"),
            dots,
            progress,
        ], style={"backgroundColor": "#0e1117", "padding": "1.5rem 2rem 1rem"}),
    ],
        id="modal-slideshow",
        is_open=False,
        backdrop="static",
        keyboard=False,
        centered=True,
        size="lg",
        style={"zIndex": 9998},
    )


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
            dbc.Col(build_main(), id="col-main", width=9,
                    style={**_COL_MAIN_OPEN, "position": "relative"}),
        ], style={"margin": "0"}),
        dcc.Store(id="store-key", storage_type="memory"),
        dcc.Store(id="store-config", storage_type="memory"),
        dcc.Store(id="store-expert-exclude", storage_type="memory"),
        dcc.Store(id="store-expert-thresholds", storage_type="memory"),
        dcc.Store(id="store-precompute-state", storage_type="memory"),
        dcc.Store(id="store-model-signal", storage_type="memory"),
        dcc.Store(id="store-profile-loaded", storage_type="memory"),
        dcc.Store(id="store-pending-note", storage_type="memory"),
        dcc.Store(id="store-loaded-model-index", storage_type="memory"),
        dcc.Interval(id="interval-precompute", interval=300, disabled=True, n_intervals=0),

        # ── Loading Slideshow ────────────────────────────────────────────────────
        dcc.Store(id="store-slide-index", data=0),
        dcc.Interval(id="interval-slideshow", interval=8000, disabled=True, n_intervals=0),
        _build_slideshow_modal(),

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
