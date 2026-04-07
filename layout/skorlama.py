"""Skorlama layout — Tekli + Toplu skorlama tab'lari."""

from dash import dcc, html
import dash_bootstrap_components as dbc

from utils.config import get_database_config

_DB = get_database_config()

# ── Ortak stiller ─────────────────────────────────────────────────────────────
_CARD = {
    "backgroundColor": "#111827", "border": "1px solid #2d3a4f",
    "borderRadius": "10px", "padding": "1.25rem", "marginBottom": "1.25rem",
}
_LABEL = {"color": "#a8b2c2", "fontSize": "0.8rem", "fontWeight": "600",
          "marginBottom": "0.25rem"}
_UPLOAD = {
    "width": "100%", "borderWidth": "1px", "borderStyle": "dashed",
    "borderRadius": "6px", "borderColor": "#2d3a4f", "textAlign": "center",
    "backgroundColor": "#0e1117", "padding": "0.6rem 0.5rem", "cursor": "pointer",
    "marginBottom": "0.25rem",
}
_UPLOAD_ACTIVE = {"borderColor": "#4F8EF7", "backgroundColor": "#111f35"}
_INPUT = {"fontSize": "0.82rem"}
_BADGE = {
    "display": "inline-flex", "alignItems": "center", "justifyContent": "center",
    "width": "22px", "height": "22px", "borderRadius": "50%",
    "backgroundColor": "#4F8EF7", "color": "#fff", "fontSize": "0.72rem",
    "fontWeight": "700", "marginRight": "0.5rem",
}
_DRIVER_OPTS = [
    {"label": "ODBC Driver 18", "value": "ODBC Driver 18 for SQL Server"},
    {"label": "ODBC Driver 17", "value": "ODBC Driver 17 for SQL Server"},
    {"label": "ODBC Driver 13", "value": "ODBC Driver 13 for SQL Server"},
]


def _section_title(number, text):
    return html.Div([
        html.Span(str(number), style=_BADGE),
        html.Span(text, style={"color": "#E8EAF0", "fontWeight": "600",
                                "fontSize": "0.9rem"}),
    ], style={"marginBottom": "0.75rem"})


# ═════════════════════════════════════════════════════════════════════════════
#  Tab 1 — Tekli Skorlama
# ═════════════════════════════════════════════════════════════════════════════
def _build_tab_tekli():
    return html.Div(id="skorlama-tab-tekli", children=[

        # ── 1. Veri Yukle ─────────────────────────────────────────────────
        html.Div([
            _section_title(1, "Veri Yukle"),
            dbc.RadioItems(
                id="skorlama-radio-source",
                options=[
                    {"label": " SQL Server", "value": "sql"},
                    {"label": " CSV Dosyasi", "value": "csv"},
                ],
                value="sql", inline=True, className="mb-3",
                style={"color": "#c8cdd8", "fontSize": "0.82rem"},
                inputStyle={"marginRight": "4px"},
                labelStyle={"marginRight": "14px"},
            ),
            # SQL paneli
            html.Div(id="skorlama-source-sql-div", children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Server", style=_LABEL),
                        dbc.Input(id="skorlama-sql-src-server", type="text",
                                  value=_DB.get("server", ""),
                                  placeholder="SERVERNAME",
                                  className="form-control", style=_INPUT),
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Database", style=_LABEL),
                        dbc.Input(id="skorlama-sql-src-database", type="text",
                                  value=_DB.get("database", ""),
                                  placeholder="DatabaseName",
                                  className="form-control", style=_INPUT),
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Driver", style=_LABEL),
                        dbc.Select(id="skorlama-sql-src-driver", options=_DRIVER_OPTS,
                                   value=_DB.get("driver", "ODBC Driver 18 for SQL Server"),
                                   className="dark-select", style=_INPUT),
                    ], width=4),
                ], className="g-2 mb-2"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Tablo", style=_LABEL),
                        dbc.Input(id="skorlama-sql-src-table", type="text",
                                  placeholder="dbo.SKOR_VERI",
                                  className="form-control", style=_INPUT),
                    ], width=8),
                    dbc.Col([
                        dbc.Checklist(
                            id="skorlama-chk-top1000",
                            options=[{"label": " Top 1000", "value": "top1000"}],
                            value=[], inline=True,
                            style={"fontSize": "0.73rem", "color": "#7e8fa4",
                                   "marginTop": "1.6rem"},
                        ),
                    ], width=4),
                ], className="g-2 mb-2"),
                dbc.Button("Veriyi Yukle", id="skorlama-btn-load-sql",
                           className="btn-load", n_clicks=0),
            ]),
            # CSV paneli
            html.Div(id="skorlama-source-csv-div", style={"display": "none"}, children=[
                dcc.Upload(
                    id="skorlama-upload-data",
                    children=html.Div([
                        html.Span("CSV dosyasi — surukle veya tikla",
                                  style={"color": "#a8b2c2", "fontSize": "0.8rem"}),
                    ]),
                    accept=".csv",
                    style=_UPLOAD, style_active=_UPLOAD_ACTIVE,
                ),
                dbc.Row([
                    dbc.Col(
                        dbc.Select(
                            id="skorlama-csv-separator",
                            options=[
                                {"label": "Virgul (,)", "value": ","},
                                {"label": "Noktali virgul (;)", "value": ";"},
                                {"label": "Tab (\\t)", "value": "\t"},
                                {"label": "Pipe (|)", "value": "|"},
                            ],
                            value=",", className="dark-select",
                            style={"fontSize": "0.78rem"},
                        ), width=8,
                    ),
                    dbc.Col(
                        dbc.Button("Yukle", id="skorlama-btn-load-csv",
                                   color="primary", size="sm", n_clicks=0),
                        width=4, className="d-flex align-items-center",
                    ),
                ], className="g-1"),
            ]),
            html.Div(id="skorlama-load-status",
                     style={"marginTop": "0.5rem", "fontSize": "0.8rem"}),
        ], style=_CARD),

        # ── 2. Model Yukle ────────────────────────────────────────────────
        html.Div([
            _section_title(2, "Model Yukle"),
            dbc.Label("Model Pickle (zorunlu)", style=_LABEL),
            dcc.Upload(
                id="skorlama-upload-model",
                children=html.Div([
                    html.Span(".pkl dosyasi — surukle veya tikla",
                              style={"color": "#a8b2c2", "fontSize": "0.8rem"}),
                ]),
                accept=".pkl,.pickle",
                style=_UPLOAD, style_active=_UPLOAD_ACTIVE,
            ),
            html.Div(id="skorlama-model-status",
                     style={"fontSize": "0.78rem", "marginBottom": "0.75rem"}),
            dbc.Label("OPT Pickle (opsiyonel — WoE modeli icin)", style=_LABEL),
            dcc.Upload(
                id="skorlama-upload-opt",
                children=html.Div([
                    html.Span(".pkl dosyasi — surukle veya tikla",
                              style={"color": "#a8b2c2", "fontSize": "0.8rem"}),
                ]),
                accept=".pkl,.pickle",
                style=_UPLOAD, style_active=_UPLOAD_ACTIVE,
            ),
            html.Div(id="skorlama-opt-status", style={"fontSize": "0.78rem"}),
        ], style=_CARD),

        # ── 3. Haric Tutulacak Kolonlar ───────────────────────────────────
        html.Div([
            _section_title(3, "Haric Tutulacak Kolonlar"),
            html.P("Model degiskenleri disindaki kolonlari secin (ornek: musteri_no, tarih).",
                   style={"color": "#6B7A99", "fontSize": "0.78rem",
                          "marginBottom": "0.5rem"}),
            dcc.Dropdown(id="skorlama-dd-exclude-cols", multi=True, searchable=True,
                         placeholder="Kolon sec...", className="dark-dd",
                         style={"minHeight": "42px"}),
            html.Div(id="skorlama-validation-msg",
                     style={"marginTop": "0.5rem", "fontSize": "0.78rem"}),
        ], style=_CARD),

        # ── 4. Skorla ────────────────────────────────────────────────────
        html.Div([
            _section_title(4, "Skorla"),
            dbc.Button("Skorla", id="skorlama-btn-score",
                       className="btn-load", n_clicks=0),
            html.Div(id="skorlama-score-status",
                     style={"marginTop": "0.5rem", "fontSize": "0.8rem"}),
            html.Div(id="skorlama-preview-table", style={"marginTop": "1rem"}),
        ], style=_CARD),

        # ── 5. SQL'e Yaz ─────────────────────────────────────────────────
        html.Div([
            _section_title(5, "SQL'e Yaz"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Server", style=_LABEL),
                    dbc.Input(id="skorlama-sql-server", type="text",
                              value=_DB.get("server", ""),
                              className="form-control", style=_INPUT),
                ], width=3),
                dbc.Col([
                    dbc.Label("Database", style=_LABEL),
                    dbc.Input(id="skorlama-sql-database", type="text",
                              value=_DB.get("database", ""),
                              className="form-control", style=_INPUT),
                ], width=3),
                dbc.Col([
                    dbc.Label("Driver", style=_LABEL),
                    dbc.Select(id="skorlama-sql-driver", options=_DRIVER_OPTS,
                               value=_DB.get("driver", "ODBC Driver 18 for SQL Server"),
                               className="dark-select", style=_INPUT),
                ], width=3),
                dbc.Col([
                    dbc.Label("Tablo Adi", style=_LABEL),
                    dbc.Input(id="skorlama-sql-table-name", type="text",
                              placeholder="dbo.SKOR_SONUC",
                              className="form-control", style=_INPUT),
                ], width=3),
            ], className="g-2 mb-3"),
            dbc.Button("SQL'e Yukle", id="skorlama-btn-push-sql",
                       className="btn-confirm", n_clicks=0, disabled=True),
            html.Div(id="skorlama-sql-status",
                     style={"marginTop": "0.5rem", "fontSize": "0.8rem"}),
        ], style=_CARD),

        # Hidden stores
        dcc.Store(id="skorlama-store-session-key", storage_type="memory"),

    ], style={"padding": "2rem 3rem", "maxWidth": "1100px", "margin": "0 auto"})


# ═════════════════════════════════════════════════════════════════════════════
#  Tab 2 — Toplu Skorlama
# ═════════════════════════════════════════════════════════════════════════════
def _build_batch_progress_modal():
    return dbc.Modal([
        dbc.ModalBody([
            html.Div([
                html.Div(className="spinner-border text-primary", role="status",
                         style={"width": "2.5rem", "height": "2.5rem",
                                "marginBottom": "1rem"}),
                html.H5("Toplu Skorlama",
                         style={"color": "#c8cdd8", "marginBottom": "0.5rem"}),
                html.P(id="skorlama-batch-modal-step",
                       children="Baslatiliyor...",
                       style={"color": "#7e8fa4", "fontSize": "0.82rem",
                              "marginBottom": "0.75rem"}),
                dbc.Progress(
                    id="skorlama-batch-modal-progress",
                    value=0, max=100, striped=True, animated=True,
                    style={"height": "6px", "backgroundColor": "#1a2332",
                           "width": "80%", "margin": "0 auto"},
                    color="primary",
                ),
                html.Div(id="skorlama-batch-modal-elapsed", children="0:00",
                         style={"color": "#6b7a99", "fontSize": "0.78rem",
                                "marginTop": "0.75rem",
                                "fontVariantNumeric": "tabular-nums"}),
            ], style={"textAlign": "center", "padding": "2rem 1rem"}),
        ], style={"backgroundColor": "#0e1117"}),
    ],
        id="skorlama-batch-modal",
        is_open=False, backdrop="static", keyboard=False,
        centered=True, size="md",
        style={"zIndex": 9999},
    )


def _build_batch_guide_modal():
    steps = [
        ("1", "Model Yukle",
         "Egitilmis model pickle dosyanizi yukleyin. WoE modeli kullaniyorsaniz "
         "OPT pickle'i da ekleyin."),
        ("2", "Veri Kaynagi Sec",
         "SQL Server: Buyuk tablonuzu secin ve chunk boyutunu belirleyin (varsayilan 100K satir). "
         "Sistem tabloyu parcalara bolup tek tek isler.\n"
         "Dosya: Birden fazla CSV veya Parquet dosyasi yukleyin. Her dosya ayri bir parca olarak islenir."),
        ("3", "Hedef SQL Ayarla",
         "Skorlanan verinin yazilacagi hedef tablo bilgilerini girin. "
         "Ilk parca tabloyu olusturur, sonrakiler eklenir."),
        ("4", "Baglan & Onizle",
         "Kaynak veriden sadece kolon bilgisi cekilir (veri yuklenmez). "
         "Model degiskenleri ile eslestirme yapilir, gerekirse haric tutulacak kolonlari secin."),
        ("5", "Toplu Skorla",
         "Tek tikla tum islem baslar. Her parca sirayla: oku → skorla → SQL'e yaz. "
         "Bellekte ayni anda sadece 1 parca bulunur, diger tab'lar etkilenmez."),
    ]
    return dbc.Modal([
        dbc.ModalHeader(
            dbc.ModalTitle("Toplu Skorlama Nasil Calisir?",
                           style={"fontSize": "1rem", "color": "#E8EAF0"}),
            close_button=True,
            style={"backgroundColor": "#0e1117", "borderBottom": "1px solid #1f2a3c"},
        ),
        dbc.ModalBody([
            html.Div([
                html.Div([
                    html.Span(s[0], style={**_BADGE, "flexShrink": "0"}),
                    html.Div([
                        html.Strong(s[1], style={"color": "#E8EAF0", "fontSize": "0.85rem"}),
                        html.P(s[2], style={"color": "#7e8fa4", "fontSize": "0.78rem",
                                            "marginBottom": "0", "marginTop": "0.2rem",
                                            "whiteSpace": "pre-line"}),
                    ]),
                ], style={"display": "flex", "gap": "0.6rem", "alignItems": "flex-start",
                          "marginBottom": "1rem"})
                for s in steps
            ]),
            html.Hr(style={"borderColor": "#1f2a3c", "margin": "0.75rem 0"}),
            html.P([
                html.I(className="bi bi-info-circle", style={"marginRight": "0.4rem"}),
                "5 milyon satirlik bir tablo ~100K chunk ile yaklasik 50 parcada islenir. "
                "Islem bittiginde bellekte hicbir veri kalmaz.",
            ], style={"color": "#6b7a99", "fontSize": "0.76rem", "marginBottom": "0"}),
        ], style={"backgroundColor": "#0e1117"}),
    ],
        id="skorlama-batch-guide-modal",
        is_open=False, centered=True, size="lg",
        style={"zIndex": 9998},
    )


def _build_tab_toplu():
    return html.Div(id="skorlama-tab-toplu", style={"display": "none"},
                    children=[html.Div(children=[

        # ── Rehber butonu ─────────────────────────────────────────────────
        html.Div([
            html.Button([
                html.I(className="bi bi-question-circle",
                       style={"marginRight": "0.35rem"}),
                "Nasil calisir?",
            ], id="skorlama-batch-btn-guide", n_clicks=0,
               style={"background": "none", "border": "1px solid #2d3a4f",
                      "borderRadius": "6px", "color": "#7e8fa4",
                      "fontSize": "0.76rem", "padding": "0.3rem 0.7rem",
                      "cursor": "pointer"}),
        ], style={"marginBottom": "1rem"}),
        _build_batch_guide_modal(),

        # ── 1. Model Yukle ────────────────────────────────────────────────
        html.Div([
            _section_title(1, "Model Yukle"),
            dbc.Label("Model Pickle (zorunlu)", style=_LABEL),
            dcc.Upload(
                id="skorlama-batch-upload-model",
                children=html.Div([
                    html.Span(".pkl dosyasi — surukle veya tikla",
                              style={"color": "#a8b2c2", "fontSize": "0.8rem"}),
                ]),
                accept=".pkl,.pickle",
                style=_UPLOAD, style_active=_UPLOAD_ACTIVE,
            ),
            html.Div(id="skorlama-batch-model-status",
                     style={"fontSize": "0.78rem", "marginBottom": "0.75rem"}),
            dbc.Label("OPT Pickle (opsiyonel — WoE modeli icin)", style=_LABEL),
            dcc.Upload(
                id="skorlama-batch-upload-opt",
                children=html.Div([
                    html.Span(".pkl dosyasi — surukle veya tikla",
                              style={"color": "#a8b2c2", "fontSize": "0.8rem"}),
                ]),
                accept=".pkl,.pickle",
                style=_UPLOAD, style_active=_UPLOAD_ACTIVE,
            ),
            html.Div(id="skorlama-batch-opt-status", style={"fontSize": "0.78rem"}),
        ], style=_CARD),

        # ── 2. Veri Kaynagi ──────────────────────────────────────────────
        html.Div([
            _section_title(2, "Veri Kaynagi"),
            dbc.RadioItems(
                id="skorlama-batch-radio-source",
                options=[
                    {"label": " SQL Server", "value": "sql"},
                    {"label": " Dosya (CSV / Parquet)", "value": "files"},
                ],
                value="sql", inline=True, className="mb-3",
                style={"color": "#c8cdd8", "fontSize": "0.82rem"},
                inputStyle={"marginRight": "4px"},
                labelStyle={"marginRight": "14px"},
            ),
            # SQL paneli
            html.Div(id="skorlama-batch-source-sql-div", children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Server", style=_LABEL),
                        dbc.Input(id="skorlama-batch-sql-src-server", type="text",
                                  value=_DB.get("server", ""),
                                  placeholder="SERVERNAME",
                                  className="form-control", style=_INPUT),
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Database", style=_LABEL),
                        dbc.Input(id="skorlama-batch-sql-src-database", type="text",
                                  value=_DB.get("database", ""),
                                  placeholder="DatabaseName",
                                  className="form-control", style=_INPUT),
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Driver", style=_LABEL),
                        dbc.Select(id="skorlama-batch-sql-src-driver", options=_DRIVER_OPTS,
                                   value=_DB.get("driver", "ODBC Driver 18 for SQL Server"),
                                   className="dark-select", style=_INPUT),
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Tablo", style=_LABEL),
                        dbc.Input(id="skorlama-batch-sql-src-table", type="text",
                                  placeholder="dbo.BUYUK_VERI",
                                  className="form-control", style=_INPUT),
                    ], width=3),
                ], className="g-2 mb-2"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Chunk Boyutu (satir)", style=_LABEL),
                        dbc.Input(id="skorlama-batch-chunk-size", type="number",
                                  value=100000, min=1000, step=10000,
                                  className="form-control", style=_INPUT),
                    ], width=4),
                ], className="g-2"),
            ]),
            # Dosya paneli
            html.Div(id="skorlama-batch-source-files-div",
                     style={"display": "none"}, children=[
                dcc.Upload(
                    id="skorlama-batch-upload-files",
                    children=html.Div([
                        html.Span("CSV / Parquet dosyalari — surukle veya tikla",
                                  style={"color": "#a8b2c2", "fontSize": "0.8rem"}),
                    ]),
                    accept=".csv,.parquet,.parq",
                    multiple=True,
                    style=_UPLOAD, style_active=_UPLOAD_ACTIVE,
                ),
                dbc.Row([
                    dbc.Col(
                        dbc.Select(
                            id="skorlama-batch-csv-separator",
                            options=[
                                {"label": "Virgul (,)", "value": ","},
                                {"label": "Noktali virgul (;)", "value": ";"},
                                {"label": "Tab (\\t)", "value": "\t"},
                                {"label": "Pipe (|)", "value": "|"},
                            ],
                            value=",", className="dark-select",
                            style={"fontSize": "0.78rem"},
                        ), width=6,
                    ),
                ], className="g-1 mt-2"),
                html.Div(id="skorlama-batch-files-list",
                         style={"marginTop": "0.5rem", "fontSize": "0.78rem",
                                "color": "#7e8fa4"}),
            ]),
        ], style=_CARD),

        # ── 3. Hedef SQL ─────────────────────────────────────────────────
        html.Div([
            _section_title(3, "Hedef SQL"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Server", style=_LABEL),
                    dbc.Input(id="skorlama-batch-tgt-server", type="text",
                              value=_DB.get("server", ""),
                              className="form-control", style=_INPUT),
                ], width=3),
                dbc.Col([
                    dbc.Label("Database", style=_LABEL),
                    dbc.Input(id="skorlama-batch-tgt-database", type="text",
                              value=_DB.get("database", ""),
                              className="form-control", style=_INPUT),
                ], width=3),
                dbc.Col([
                    dbc.Label("Driver", style=_LABEL),
                    dbc.Select(id="skorlama-batch-tgt-driver", options=_DRIVER_OPTS,
                               value=_DB.get("driver", "ODBC Driver 18 for SQL Server"),
                               className="dark-select", style=_INPUT),
                ], width=3),
                dbc.Col([
                    dbc.Label("Tablo Adi", style=_LABEL),
                    dbc.Input(id="skorlama-batch-tgt-table", type="text",
                              placeholder="dbo.SKOR_SONUC",
                              className="form-control", style=_INPUT),
                ], width=3),
            ], className="g-2"),
        ], style=_CARD),

        # ── 4. Baglan & Onizle ───────────────────────────────────────────
        html.Div([
            _section_title(4, "Baglan & Onizle"),
            dbc.Button("Baglan & Onizle", id="skorlama-batch-btn-preview",
                       className="btn-load", n_clicks=0),
            html.Div(id="skorlama-batch-preview-status",
                     style={"marginTop": "0.5rem", "fontSize": "0.8rem"}),
            html.Div([
                dbc.Label("Haric Tutulacak Kolonlar", style={**_LABEL, "marginTop": "0.75rem"}),
                dcc.Dropdown(id="skorlama-batch-dd-exclude", multi=True, searchable=True,
                             placeholder="Kolon sec...", className="dark-dd",
                             style={"minHeight": "42px"}),
            ], id="skorlama-batch-exclude-wrap", style={"display": "none"}),
            html.Div(id="skorlama-batch-validation-msg",
                     style={"marginTop": "0.5rem", "fontSize": "0.78rem"}),
        ], style=_CARD),

        # ── 5. Basla ─────────────────────────────────────────────────────
        html.Div([
            _section_title(5, "Toplu Skorla"),
            html.Div(id="skorlama-batch-estimate",
                     style={"marginBottom": "0.6rem", "fontSize": "0.78rem"}),
            dbc.Button("Toplu Skorla", id="skorlama-batch-btn-start",
                       className="btn-confirm", n_clicks=0, disabled=True),
            html.Div(id="skorlama-batch-result-status",
                     style={"marginTop": "0.5rem", "fontSize": "0.8rem"}),
        ], style=_CARD),

        # ── Modal + Hidden ───────────────────────────────────────────────
        _build_batch_progress_modal(),
        dcc.Store(id="skorlama-batch-store-key", storage_type="memory"),
        dcc.Store(id="skorlama-batch-compute-state", storage_type="memory"),
        dcc.Interval(id="skorlama-batch-interval", interval=500,
                     disabled=True, n_intervals=0),

    ], style={"padding": "2rem 3rem", "maxWidth": "1100px", "margin": "0 auto"})])


# ═════════════════════════════════════════════════════════════════════════════
#  Ana container
# ═════════════════════════════════════════════════════════════════════════════
def build_skorlama_container():
    """Skorlama ust duzey container — iki tab: Tekli + Toplu."""
    return html.Div(
        id="container-skorlama",
        style={"display": "none"},
        children=[
            # Baslik + Tab butonlari — ortalanmis
            html.Div([
                html.H4("Skorlama", style={"fontWeight": "700", "color": "#E8EAF0",
                                            "marginBottom": "0.25rem"}),
                html.P("Veri yukleyin, model ile skorlayin ve sonucu SQL'e yazin.",
                       style={"color": "#6B7A99", "fontSize": "0.82rem",
                              "marginBottom": "0"}),
                html.Div([
                    html.Button("Tekli Skorlama", id="skorlama-btn-tab-tekli",
                                n_clicks=0, className="top-nav-link active"),
                    html.Button("Toplu Skorlama", id="skorlama-btn-tab-toplu",
                                n_clicks=0, className="top-nav-link"),
                ], style={"display": "flex", "gap": "0.2rem",
                          "marginTop": "0.75rem",
                          "borderBottom": "1px solid #1E293B",
                          "paddingBottom": "0.5rem"}),
            ], style={"padding": "1.5rem 0 0 0",
                      "maxWidth": "1100px", "margin": "0 auto",
                      "paddingLeft": "3rem", "paddingRight": "3rem"}),

            # Tab icerikleri
            _build_tab_tekli(),
            _build_tab_toplu(),
        ],
    )
