"""İzleme (Monitoring) tab layout — bağımsız sidebar + main area.

Tüm component ID'leri `mon-` prefix'lidir. Geliştirme tarafıyla
hiçbir state, store veya component ID paylaşılmaz.
"""

from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc


# ── Sidebar geçiş sabitleri (Geliştirme ile aynı değerler, farklı ID'ler) ────
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


def _build_format_modal():
    """İzleme format modal — tablonun beklenen yapısını açıklar."""
    _label = {"fontSize": "0.82rem", "color": "#c8cdd8", "lineHeight": "1.7"}
    _hint  = {"fontSize": "0.76rem", "color": "#7e8fa4", "marginLeft": "1.2rem",
              "marginBottom": "0.4rem"}

    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Veri Formatı Bilgisi",
                                        style={"fontSize": "0.95rem"})),
        dbc.ModalBody([
            html.P("Referans ve İzleme tabloları aynı kolon yapısına sahip olmalıdır. "
                   "Aşağıdaki kolonlar beklenir:",
                    style={"fontSize": "0.85rem", "marginBottom": "0.75rem"}),

            # ── Target ──
            html.Div([
                html.Span("1. ", style={"color": "#4F8EF7", "fontWeight": "bold"}),
                html.Strong("Target Kolonu"),
                html.Span(" (zorunlu)", style={"color": "#ef4444", "fontSize": "0.75rem"}),
            ], style=_label),
            html.P("İkili (binary) 0/1 değer. 1 = default gerçekleşmiş, 0 = default yok. "
                   "Henüz olgunlaşmamış gözlemler için 0 yazılabilir.",
                   style=_hint),

            # ── Tarih ──
            html.Div([
                html.Span("2. ", style={"color": "#4F8EF7", "fontWeight": "bold"}),
                html.Strong("Tarih Kolonu"),
                html.Span(" (zorunlu)", style={"color": "#ef4444", "fontSize": "0.75rem"}),
            ], style=_label),
            html.P("Gözlemin hangi döneme ait olduğunu belirler. "
                   "Pandas'ın tanıyacağı herhangi bir tarih formatı (YYYY-MM-DD, DD/MM/YYYY, vb.). "
                   "Sistem bu kolonu aylık veya çeyreklik dönemlere böler.",
                   style=_hint),

            # ── PD / Rating ──
            html.Div([
                html.Span("3. ", style={"color": "#4F8EF7", "fontWeight": "bold"}),
                html.Strong("PD / Rating Kolonu"),
                html.Span(" (zorunlu)", style={"color": "#ef4444", "fontSize": "0.75rem"}),
            ], style=_label),
            html.P([
                "Modelin ürettiği skor. İki formattan birini kabul eder:",
                html.Br(),
                html.Span("  a) PD (olasılık): ", style={"color": "#a78bfa"}),
                "0 ile 1 arasında ondalık değer (ör. 0.0342). Sistem otomatik olarak "
                "1–25 rating skalasına dönüştürür.",
                html.Br(),
                html.Span("  b) Rating (tam sayı): ", style={"color": "#a78bfa"}),
                "Doğrudan 1–25 arası tam sayı. Sistem bunu algılar ve dönüşüm yapmadan kullanır.",
            ], style=_hint),

            # ── ID ──
            html.Div([
                html.Span("4. ", style={"color": "#4F8EF7", "fontWeight": "bold"}),
                html.Strong("ID Kolonu"),
                html.Span(" (opsiyonel)", style={"color": "#7e8fa4", "fontSize": "0.75rem"}),
            ], style=_label),
            html.P("Müşteri/kredi numarası. Yalnızca Göç Matrisi hesaplaması için gereklidir — "
                   "referans ve izleme verisindeki aynı müşterileri eşleştirmek için kullanılır. "
                   "Belirtilmezse Göç Matrisi tabı devre dışı kalır.",
                   style=_hint),

            # ── Değişkenler ──
            html.Div([
                html.Span("5. ", style={"color": "#4F8EF7", "fontWeight": "bold"}),
                html.Strong("Model Değişkenleri"),
            ], style=_label),
            html.P("Modelde kullanılan girdi değişkenleri. Target, Tarih, PD ve ID dışındaki "
                   "tüm sayısal kolonlar otomatik olarak model değişkeni kabul edilir. "
                   "Değişken PSI ve IV hesaplamaları bu kolonlar üzerinden yapılır.",
                   style=_hint),

            html.Hr(style={"borderColor": "#2d3a4f"}),

            # ── Referans vs İzleme ──
            html.P("Veri Setleri", style={"fontSize": "0.85rem", "color": "#f59e0b",
                                          "fontWeight": "bold", "marginBottom": "0.4rem"}),
            html.Div([
                html.P([
                    html.Span("Referans: ", style={"color": "#10b981", "fontWeight": "bold"}),
                    "Modelin geliştirildiği sample. PSI, KS, Gini gibi tüm metriklerin "
                    "karşılaştırma tabanını oluşturur. Sabit kalır, bir kere yüklenir.",
                ], style={"fontSize": "0.80rem", "color": "#a8b2c2",
                          "marginBottom": "0.25rem"}),
                html.P([
                    html.Span("İzleme: ", style={"color": "#3b82f6", "fontWeight": "bold"}),
                    "Canlı ortam verisi. Modelin üretim ortamındaki performansını temsil eder. "
                    "Sürekli büyüyen tablo — yeni dönemler eklendikçe sistem sadece yeni "
                    "dönemlerin özetlerini hesaplar.",
                ], style={"fontSize": "0.80rem", "color": "#a8b2c2",
                          "marginBottom": "0.25rem"}),
            ]),

            html.Hr(style={"borderColor": "#2d3a4f"}),

            # ── WoE ──
            html.P("WoE Dönüşümü (opsiyonel)", style={"fontSize": "0.85rem",
                   "color": "#a78bfa", "fontWeight": "bold", "marginBottom": "0.4rem"}),
            html.P([
                "Eğer model WoE (Weight of Evidence) dönüşümü kullanıyorsa, ",
                html.Strong("opt pickle"),
                " dosyasını yükleyebilirsiniz. Bu durumda değişken PSI hesaplaması "
                "WoE dönüştürülmüş değerler üzerinden yapılır. "
                "Diğer metrikler (KS, Gini, Bad Rate, vb.) WoE'den etkilenmez — "
                "her zaman PD/Rating kolonundan hesaplanır.",
            ], style={"fontSize": "0.80rem", "color": "#a8b2c2"}),

            html.Hr(style={"borderColor": "#2d3a4f"}),

            # ── Örnek ──
            html.P("Örnek Tablo Yapısı", style={"fontSize": "0.85rem",
                   "color": "#c8cdd8", "fontWeight": "bold", "marginBottom": "0.4rem"}),
            dash_table.DataTable(
                columns=[{"name": c, "id": c} for c in
                         ["musteri_no", "tarih", "pd", "target", "var1", "var2"]],
                data=[
                    {"musteri_no": "1001", "tarih": "2024-01-15", "pd": "0.0342",
                     "target": "0", "var1": "2.15", "var2": "0.87"},
                    {"musteri_no": "1002", "tarih": "2024-01-20", "pd": "0.1580",
                     "target": "1", "var1": "5.40", "var2": "1.23"},
                    {"musteri_no": "1003", "tarih": "2024-02-10", "pd": "0.0025",
                     "target": "0", "var1": "0.90", "var2": "0.45"},
                ],
                style_header={"backgroundColor": "#1a2332", "color": "#c8cdd8",
                              "fontWeight": "bold", "fontSize": "0.72rem"},
                style_cell={"backgroundColor": "#0e1117", "color": "#a8b2c2",
                            "fontSize": "0.72rem", "border": "1px solid #2d3a4f",
                            "padding": "3px 6px"},
                style_table={"overflowX": "auto"},
            ),
            html.P("veya Rating kolonu ile:", style={"fontSize": "0.76rem",
                   "color": "#7e8fa4", "marginTop": "0.3rem", "marginBottom": "0.3rem"}),
            dash_table.DataTable(
                columns=[{"name": c, "id": c} for c in
                         ["musteri_no", "tarih", "rating", "target", "var1", "var2"]],
                data=[
                    {"musteri_no": "1001", "tarih": "2024-01-15", "rating": "14",
                     "target": "0", "var1": "2.15", "var2": "0.87"},
                    {"musteri_no": "1002", "tarih": "2024-01-20", "rating": "20",
                     "target": "1", "var1": "5.40", "var2": "1.23"},
                    {"musteri_no": "1003", "tarih": "2024-02-10", "rating": "5",
                     "target": "0", "var1": "0.90", "var2": "0.45"},
                ],
                style_header={"backgroundColor": "#1a2332", "color": "#c8cdd8",
                              "fontWeight": "bold", "fontSize": "0.72rem"},
                style_cell={"backgroundColor": "#0e1117", "color": "#a8b2c2",
                            "fontSize": "0.72rem", "border": "1px solid #2d3a4f",
                            "padding": "3px 6px"},
                style_table={"overflowX": "auto"},
            ),
        ]),
        dbc.ModalFooter(
            dbc.Button("Anladım", id="mon-btn-format-modal-close",
                       color="primary", size="sm"),
        ),
    ], id="mon-modal-format", is_open=False, centered=True, size="lg")


def build_izleme_sidebar():
    """İzleme sidebar'ı — Geliştirme ile aynı görünüm, bağımsız ID'ler."""
    return html.Div([

        # ── Format Modal ────────────────────────────────────────────────────
        _build_format_modal(),

        # ── Bölüm 0: Kayıtlı Profil ─────────────────────────────────────────
        html.P("Kayıtlı Profil", className="sidebar-section-title"),
        dbc.Row([
            dbc.Col(
                dcc.Dropdown(
                    id="mon-dd-profile",
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
                dbc.Button("Yükle", id="mon-btn-profile-load", color="primary",
                           size="sm", style={"width": "100%", "fontSize": "0.78rem",
                                             "height": "36px", "padding": "0"}),
                width=3, style={"paddingLeft": "4px", "display": "flex", "alignItems": "stretch"},
            ),
        ], className="g-0 mb-1"),
        html.Div(id="mon-profile-status", style={"fontSize": "0.75rem", "marginBottom": "0.5rem"}),
        html.Hr(className="sidebar-divider"),

        # ── Bölüm 0.5: Referans / İzleme Toggle ───────────────────────────
        html.P("Veri Türü", className="sidebar-section-title"),
        dbc.ButtonGroup([
            dbc.Button("Referans", id="mon-btn-toggle-ref", color="primary",
                       outline=False, size="sm", active=True,
                       style={"fontSize": "0.75rem", "flex": "1"}),
            dbc.Button("İzleme", id="mon-btn-toggle-mon", color="primary",
                       outline=True, size="sm", active=False,
                       style={"fontSize": "0.75rem", "flex": "1"}),
        ], style={"width": "100%", "marginBottom": "0.5rem"}),
        # Durum göstergeleri
        html.Div(id="mon-ref-status",
                 style={"fontSize": "0.72rem", "color": "#7e8fa4",
                        "marginBottom": "0.15rem"}),
        html.Div(id="mon-mon-status",
                 style={"fontSize": "0.72rem", "color": "#7e8fa4",
                        "marginBottom": "0.5rem"}),

        # ── WoE Dönüşüm ──────────────────────────────────────────────────
        dbc.Checklist(
            id="mon-chk-woe",
            options=[{"label": " WoE dönüşümü uygula", "value": "woe"}],
            value=[],
            inline=True,
            style={"fontSize": "0.75rem", "color": "#c8cdd8",
                   "marginBottom": "0.35rem"},
        ),
        dbc.Collapse(
            html.Div([
                dcc.Upload(
                    id="mon-upload-opt-pickle",
                    children=html.Div([
                        html.Span("opt pickle — sürükle veya tıkla",
                                  style={"color": "#a8b2c2", "fontSize": "0.78rem"}),
                    ]),
                    accept=".pkl,.pickle",
                    style={"width": "100%", "borderWidth": "1px",
                           "borderStyle": "dashed", "borderRadius": "6px",
                           "borderColor": "#2d3a4f", "textAlign": "center",
                           "backgroundColor": "#0e1117",
                           "padding": "0.5rem 0.5rem", "cursor": "pointer",
                           "marginBottom": "0.25rem"},
                    style_active={"borderColor": "#a78bfa",
                                  "backgroundColor": "#111f35"},
                ),
                html.Div(id="mon-opt-pickle-status",
                         style={"fontSize": "0.72rem", "color": "#7e8fa4",
                                "marginBottom": "0.3rem"}),
            ]),
            id="mon-collapse-woe-upload",
            is_open=False,
        ),

        # Kolon uyumsuzluğu toast
        dbc.Toast(
            id="mon-toast-column-mismatch",
            header="Kolon Uyumsuzluğu",
            is_open=False, duration=6000, icon="warning",
            style={"position": "fixed", "top": 20, "right": 20, "zIndex": 9999,
                   "minWidth": "320px"},
        ),

        # ── Bölüm 1: Veri Kaynağı ───────────────────────────────────────────
        html.P("Veri Kaynağı", className="sidebar-section-title"),
        dbc.RadioItems(
            id="mon-radio-source",
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
        html.Div(id="mon-source-sql-div", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Label("Server", className="form-label"),
                    dbc.Input(id="mon-input-sql-server", type="text",
                              placeholder="SERVERNAME",
                              className="form-control",
                              style={"fontSize": "0.82rem"}),
                ], width=12, className="mb-2"),
                dbc.Col([
                    dbc.Label("Database", className="form-label"),
                    dbc.Input(id="mon-input-sql-database", type="text",
                              placeholder="DatabaseName",
                              className="form-control",
                              style={"fontSize": "0.82rem"}),
                ], width=12, className="mb-2"),
                dbc.Col([
                    dbc.Label("Driver", className="form-label"),
                    dbc.Select(
                        id="mon-dd-sql-driver",
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
            dbc.Input(id="mon-input-table-1", type="text", placeholder="dbo.TABLO1",
                      className="form-control mb-1", style={"fontSize": "0.82rem"}),
            html.Div(id="mon-div-sql-jk-1", style={"display": "none"}, children=[
                dbc.Input(id="mon-input-sql-jk-1", type="text",
                          placeholder="Join key: must_no, sube_kodu",
                          className="form-control mb-1",
                          style={"fontSize": "0.75rem", "color": "#a78bfa"}),
            ]),
            # Tablo 2
            html.Div(id="mon-sql-table-row-2", style={"display": "none"}, children=[
                dbc.InputGroup([
                    dbc.Input(id="mon-input-table-2", type="text", placeholder="dbo.TABLO2",
                              className="form-control", style={"fontSize": "0.82rem"}),
                    dbc.RadioItems(
                        id="mon-radio-sql-join-2",
                        options=[{"label": "LEFT", "value": "left"},
                                 {"label": "INNER", "value": "inner"}],
                        value="left", inline=True, className="join-type-toggle",
                    ),
                    dbc.Button("×", id="mon-btn-remove-sql-2", color="link",
                               style={"color": "#ef4444", "fontSize": "1rem",
                                      "padding": "0 0.5rem"}),
                ], className="mb-1"),
                dbc.Input(id="mon-input-sql-jk-2", type="text",
                          placeholder="Join key: bd_must_no, sube_kodu",
                          className="form-control mb-1",
                          style={"fontSize": "0.75rem", "color": "#a78bfa"}),
            ]),
            # Tablo 3
            html.Div(id="mon-sql-table-row-3", style={"display": "none"}, children=[
                dbc.InputGroup([
                    dbc.Input(id="mon-input-table-3", type="text", placeholder="dbo.TABLO3",
                              className="form-control", style={"fontSize": "0.82rem"}),
                    dbc.RadioItems(
                        id="mon-radio-sql-join-3",
                        options=[{"label": "LEFT", "value": "left"},
                                 {"label": "INNER", "value": "inner"}],
                        value="left", inline=True, className="join-type-toggle",
                    ),
                    dbc.Button("×", id="mon-btn-remove-sql-3", color="link",
                               style={"color": "#ef4444", "fontSize": "1rem",
                                      "padding": "0 0.5rem"}),
                ], className="mb-1"),
                dbc.Input(id="mon-input-sql-jk-3", type="text",
                          placeholder="Join key: musteri_id, sube_kodu",
                          className="form-control mb-1",
                          style={"fontSize": "0.75rem", "color": "#a78bfa"}),
            ]),
            dbc.Button("+ Tablo Ekle", id="mon-btn-add-sql-table", size="sm",
                       color="link", n_clicks=0,
                       style={"fontSize": "0.75rem", "color": "#4F8EF7",
                              "padding": "0", "marginBottom": "0.5rem"}),

            dbc.Checklist(
                id="mon-chk-sql-top1000",
                options=[{"label": " İlk 1000 satır (test için)", "value": "top1000"}],
                value=[],
                inline=True,
                style={"fontSize": "0.73rem", "color": "#7e8fa4",
                       "marginBottom": "0.4rem"},
            ),
            dbc.Button("Veriyi Yükle", id="mon-btn-load", className="btn-load mb-1", n_clicks=0),

            dcc.Store(id="mon-store-sql-table-count", data=1),
        ]),

        # CSV paneli
        html.Div(id="mon-source-csv-div", style={"display": "none"}, children=[
            dbc.Label("Dosyalar", className="form-label"),
            dcc.Upload(
                id="mon-upload-csv",
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
            html.Div(id="mon-csv-filename-display",
                     style={"color": "#a78bfa", "fontSize": "0.72rem",
                            "marginBottom": "0.4rem", "fontStyle": "italic"}),
            html.Div(id="mon-div-csv-jk-1", style={"display": "none"}, children=[
                dbc.Input(id="mon-input-csv-jk-1", type="text",
                          placeholder="Join key: must_no, sube_kodu",
                          className="form-control mb-1",
                          style={"fontSize": "0.75rem", "color": "#a78bfa"}),
            ]),
            # Dosya 2
            html.Div(id="mon-csv-file-row-2", style={"display": "none"}, children=[
                dcc.Upload(
                    id="mon-upload-csv-2",
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
                    dbc.Col(html.Div(id="mon-csv-filename-display-2",
                                    style={"color": "#a78bfa", "fontSize": "0.72rem",
                                           "fontStyle": "italic"}), width=6),
                    dbc.Col(dbc.RadioItems(
                        id="mon-radio-csv-join-2",
                        options=[{"label": "LEFT", "value": "left"},
                                 {"label": "INNER", "value": "inner"}],
                        value="left", inline=True, className="join-type-toggle",
                    ), width=4),
                    dbc.Col(dbc.Button("×", id="mon-btn-remove-csv-2", color="link",
                                       style={"color": "#ef4444", "fontSize": "1rem",
                                              "padding": "0"}), width=2),
                ], className="mb-1 g-0"),
                dbc.Input(id="mon-input-csv-jk-2", type="text",
                          placeholder="Join key: bd_must_no, sube_kodu",
                          className="form-control mb-1",
                          style={"fontSize": "0.75rem", "color": "#a78bfa"}),
            ]),
            # Dosya 3
            html.Div(id="mon-csv-file-row-3", style={"display": "none"}, children=[
                dcc.Upload(
                    id="mon-upload-csv-3",
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
                    dbc.Col(html.Div(id="mon-csv-filename-display-3",
                                    style={"color": "#a78bfa", "fontSize": "0.72rem",
                                           "fontStyle": "italic"}), width=6),
                    dbc.Col(dbc.RadioItems(
                        id="mon-radio-csv-join-3",
                        options=[{"label": "LEFT", "value": "left"},
                                 {"label": "INNER", "value": "inner"}],
                        value="left", inline=True, className="join-type-toggle",
                    ), width=4),
                    dbc.Col(dbc.Button("×", id="mon-btn-remove-csv-3", color="link",
                                       style={"color": "#ef4444", "fontSize": "1rem",
                                              "padding": "0"}), width=2),
                ], className="mb-1 g-0"),
                dbc.Input(id="mon-input-csv-jk-3", type="text",
                          placeholder="Join key: musteri_id, sube_kodu",
                          className="form-control mb-1",
                          style={"fontSize": "0.75rem", "color": "#a78bfa"}),
            ]),

            dbc.Button("+ Dosya Ekle", id="mon-btn-add-csv-file", size="sm",
                       color="link", n_clicks=0,
                       style={"fontSize": "0.75rem", "color": "#4F8EF7",
                              "padding": "0", "marginBottom": "0.5rem"}),

            dbc.Row([
                dbc.Col(
                    dbc.Select(
                        id="mon-csv-separator",
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
                    dbc.Button("Yükle", id="mon-btn-load-csv",
                               color="primary", size="sm", n_clicks=0),
                    width=4, className="d-flex align-items-center",
                ),
            ], className="g-1"),

            dcc.Store(id="mon-store-csv-file-count", data=1),
        ]),

        html.Div(id="mon-load-status", style={"marginTop": "0.5rem", "fontSize": "0.8rem"}),

        html.Hr(className="sidebar-divider"),

        # ── Bölüm 2: Kolon Yapılandırması ────────────────────────────────────
        dbc.Collapse(
            html.Div([
                html.P("Kolon Yapılandırması", className="sidebar-section-title"),

                dbc.Label([
                    "Target Kolonu",
                    html.Span("*", style={"color": "#ef4444", "marginLeft": "3px"}),
                ], className="form-label"),
                dcc.Dropdown(
                    id="mon-dd-target-col",
                    options=[], value=None,
                    placeholder="Kolon ara…", searchable=True,
                    className="dark-dd mb-3",
                ),

                dbc.Label([
                    "Tarih Kolonu",
                    html.Span("*", style={"color": "#ef4444", "marginLeft": "3px"}),
                ], className="form-label"),
                dcc.Dropdown(
                    id="mon-dd-date-col",
                    options=[], value=None,
                    placeholder="Kolon ara…", searchable=True,
                    className="dark-dd mb-3",
                ),

                dbc.Label([
                    "PD Kolonu",
                    html.Span("*", style={"color": "#ef4444", "marginLeft": "3px"}),
                ], className="form-label"),
                html.Div("Model skoru (olasılık değeri)", className="form-hint"),
                dcc.Dropdown(
                    id="mon-dd-pd-col",
                    options=[], value=None,
                    placeholder="Kolon ara…", searchable=True,
                    className="dark-dd mb-3",
                ),

                dbc.Label("ID Kolonu", className="form-label"),
                html.Div("opsiyonel — Göç Matrisi için gerekli",
                         className="form-hint"),
                dcc.Dropdown(
                    id="mon-dd-id-col",
                    options=[], value=None,
                    placeholder="Kolon ara…", searchable=True,
                    className="dark-dd mb-3",
                ),

                dbc.Label("Olgunlaşma Süresi (ay)", className="form-label"),
                html.Div("Sonuç bağımlı metrikler için bekleme süresi",
                         className="form-hint"),
                dbc.Input(
                    id="mon-input-maturity",
                    type="number", value=12, min=1, max=60, step=1,
                    className="form-control mb-3",
                    style={"fontSize": "0.82rem", "width": "100px"},
                ),

                dbc.Label("Dönem Frekansı", className="form-label"),
                dbc.RadioItems(
                    id="mon-radio-period-freq",
                    options=[
                        {"label": " Aylık", "value": "M"},
                        {"label": " Çeyreklik", "value": "Q"},
                    ],
                    value="M",
                    inline=True,
                    className="mb-3",
                    style={"color": "#c8cdd8", "fontSize": "0.82rem"},
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"marginRight": "14px"},
                ),

                dbc.Button(
                    "Yapılandırmayı Onayla",
                    id="mon-btn-confirm",
                    className="btn-confirm",
                    n_clicks=0,
                ),
                html.Div(id="mon-config-status",
                         style={"marginTop": "0.5rem", "fontSize": "0.8rem"}),
            ]),
            id="mon-collapse-config",
            is_open=False,
        ),

        # ── Bölüm 4: Profil Kaydet / Sil ────────────────────────────────────
        dbc.Collapse(
            html.Div([
                html.Hr(className="sidebar-divider"),
                dbc.Row([
                    dbc.Col(
                        dbc.Button("Profil Kaydet", id="mon-btn-profile-save", color="success",
                                   size="sm", outline=True,
                                   style={"width": "100%", "fontSize": "0.72rem"}),
                        width=6, style={"paddingRight": "4px"},
                    ),
                    dbc.Col(
                        dbc.Button("Profil Sil", id="mon-btn-profile-delete", color="danger",
                                   size="sm", outline=True,
                                   style={"width": "100%", "fontSize": "0.72rem"}),
                        width=6, style={"paddingLeft": "4px"},
                    ),
                ], className="g-0"),
            ]),
            id="mon-collapse-profile-actions",
            is_open=False,
        ),

        # ── Profil Kaydet Modal ──────────────────────────────────────────────
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Profil Kaydet",
                                           style={"fontSize": "0.95rem"})),
            dbc.ModalBody([
                dbc.Label("Profil Adı", className="form-label"),
                dbc.Input(id="mon-input-profile-name", type="text",
                          placeholder="ör. İsim_segment",
                          className="form-control",
                          style={"fontSize": "0.82rem"}),
            ]),
            dbc.ModalFooter(
                dbc.Button("Kaydet", id="mon-btn-profile-save-confirm",
                           color="success", size="sm"),
            ),
        ], id="mon-modal-profile-save", is_open=False, centered=True),

        # ── Profil Sil Modal ─────────────────────────────────────────────────
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Profil Sil",
                                           style={"fontSize": "0.95rem"})),
            dbc.ModalBody([
                dbc.Label("Silinecek Profil", className="form-label"),
                dcc.Dropdown(
                    id="mon-dd-profile-delete",
                    options=[], value=None,
                    placeholder="Profil seçin…",
                    searchable=False, className="dark-dd",
                    style={"fontSize": "0.78rem"},
                ),
                html.Div(id="mon-delete-confirm-area", style={"marginTop": "0.75rem"}),
            ]),
            dbc.ModalFooter(
                dbc.Button("Evet, Sil", id="mon-btn-profile-delete-confirm",
                           color="danger", size="sm", style={"display": "none"}),
            ),
        ], id="mon-modal-profile-delete", is_open=False, centered=True),

        # ── Profil Toast ─────────────────────────────────────────────────────
        dbc.Toast(
            id="mon-toast-profile-saved",
            header="Profil Kaydedildi",
            is_open=False, duration=4000, icon="success",
            style={"position": "fixed", "top": 20, "right": 20, "zIndex": 9999,
                   "minWidth": "280px"},
        ),

    ], id="mon-sidebar")


def _placeholder_tab_content(msg="Geliştirilecektir"):
    """Placeholder içerik — henüz geliştirilmemiş tab'lar için."""
    return html.Div(
        html.P(msg, style={"color": "#7e8fa4", "fontSize": "0.9rem",
                           "textAlign": "center", "padding": "3rem 0"}),
    )


def _metric_sub_tabs(metric_key):
    """Trend + Kümülatif sub-tab yapısı üretir.
    metric_key: 'psi', 'disc', 'badrate', 'hhi', 'migration', 'backtesting'
    """
    return dbc.Tabs([
        dbc.Tab(
            html.Div([
                dbc.Row([
                    dbc.Col(
                        dcc.Dropdown(
                            id=f"mon-{metric_key}-trend-dd",
                            options=[], value=None,
                            placeholder="Dönem seçin…",
                            searchable=False, className="dark-dd",
                            style={"fontSize": "0.78rem"},
                        ),
                        width=4,
                    ),
                ], className="mb-3 mt-2"),
                html.Div(id=f"mon-{metric_key}-trend-detail"),
                html.Div(id=f"mon-{metric_key}-trend-chart",
                         style={"marginTop": "1rem"}),
            ], id=f"mon-{metric_key}-trend-content"),
            label="Trend",
            tab_id=f"tab-mon-{metric_key}-trend",
        ),
        dbc.Tab(
            html.Div(id=f"mon-{metric_key}-cum-content"),
            label="Kümülatif",
            tab_id=f"tab-mon-{metric_key}-cum",
        ),
    ], id=f"mon-{metric_key}-subtabs",
       active_tab=f"tab-mon-{metric_key}-trend",
       className="sub-tabs mt-2")


def build_izleme_main():
    """İzleme ana içerik alanı — config banner, metric kartları, tab'lar."""
    return html.Div([
        html.Div(id="mon-config-banner"),
        html.Div(id="mon-metrics-row", style={"marginBottom": "1.5rem"}),
        dbc.Tabs([
            dbc.Tab(
                dcc.Loading(
                    html.Div(id="mon-data-preview"),
                    type="dot", color="#4F8EF7", delay_show=200,
                ),
                label="Önizleme",
                tab_id="tab-mon-preview",
                className="tab-content-area",
            ),
            dbc.Tab(
                html.Div(id="mon-tab-psi-content",
                         children=_metric_sub_tabs("psi")),
                label="PSI",
                tab_id="tab-mon-psi",
                className="tab-content-area",
            ),
            dbc.Tab(
                html.Div(id="mon-tab-disc-content",
                         children=_metric_sub_tabs("disc")),
                label="Gini/KS",
                tab_id="tab-mon-discrimination",
                className="tab-content-area",
            ),
            dbc.Tab(
                html.Div(id="mon-tab-badrate-content",
                         children=_metric_sub_tabs("badrate")),
                label="Bad Rate",
                tab_id="tab-mon-badrate",
                className="tab-content-area",
            ),
            dbc.Tab(
                html.Div(id="mon-tab-hhi-content",
                         children=_metric_sub_tabs("hhi")),
                label="HHI",
                tab_id="tab-mon-hhi",
                className="tab-content-area",
            ),
            dbc.Tab(
                html.Div(id="mon-tab-migration-content",
                         children=_metric_sub_tabs("migration")),
                label="Göç Matrisi",
                tab_id="tab-mon-migration",
                className="tab-content-area",
            ),
            dbc.Tab(
                html.Div(id="mon-tab-backtesting-content",
                         children=_metric_sub_tabs("backtesting")),
                label="Backtesting",
                tab_id="tab-mon-backtesting",
                className="tab-content-area",
            ),
        ], id="mon-tabs", active_tab="tab-mon-preview", className="main-tabs"),
    ], style={"padding": "1.5rem"})


_MON_N_SLIDES = 6


def _build_mon_slideshow_modal():
    """İzleme veri yüklenirken gösterilen eğitim slayt gösterisi."""
    slides = [
        # Slide 0 — İzleme Sistemi
        html.Div([
            html.H4("Model İzleme Sistemi", className="slide-title"),
            html.P(
                "Canlıya alınmış modellerin performansını dönemsel olarak takip edin. "
                "Referans (geliştirme sample'ı) ile İzleme (canlı veri) karşılaştırılarak "
                "modelin stabilitesi, ayrım gücü ve kalibrasyon kalitesi ölçülür.",
                className="slide-text",
            ),
            html.Div([
                html.Span("Dönemsel Analiz", className="slide-chip"),
                html.Span("Trend & Kümülatif", className="slide-chip"),
                html.Span("Otomatik Raporlama", className="slide-chip"),
            ], className="slide-chips"),
        ], className="slide-content"),

        # Slide 1 — PSI
        html.Div([
            html.H4("PSI — Population Stability Index", className="slide-title"),
            html.P(
                "Modelin girdi değişkenlerinin ve rating dağılımının zaman içinde "
                "ne kadar değiştiğini ölçer. Referans ile izleme dönemleri karşılaştırılır.",
                className="slide-text",
            ),
            html.Div("PSI < 0.10 → Stabil  |  0.10–0.25 → Orta  |  > 0.25 → Yüksek Kayma",
                      className="slide-highlight"),
            html.P("Her değişken ve rating bazında ayrı PSI hesaplanır. "
                   "Trend grafiğiyle zaman içindeki değişimi izleyin.",
                   className="slide-text",
                   style={"marginTop": "0.5rem", "fontSize": "0.78rem"}),
        ], className="slide-content"),

        # Slide 2 — Gini/KS
        html.Div([
            html.H4("Gini/KS", className="slide-title"),
            html.P(
                "Modelin iyi/kötü müşterileri ayırt etme gücünü ölçer. "
                "KS (Kolmogorov-Smirnov) kümülatif dağılım farkını, "
                "Gini/AR ise CAP eğrisi altında kalan alanı hesaplar.",
                className="slide-text",
            ),
            html.Div([
                html.Div([
                    html.Span("KS", className="slide-step-num"),
                    html.Span("Max |CumBad% - CumGood%|", className="slide-step-label"),
                ], className="slide-step"),
                html.Div([
                    html.Span("Gini", className="slide-step-num"),
                    html.Span("2 × AUC − 1", className="slide-step-label"),
                ], className="slide-step"),
                html.Div([
                    html.Span("AR", className="slide-step-num"),
                    html.Span("Accuracy Ratio (CAP)", className="slide-step-label"),
                ], className="slide-step"),
            ], className="slide-steps"),
        ], className="slide-content"),

        # Slide 3 — Backtesting
        html.Div([
            html.H4("Backtesting — Binomial Test", className="slide-title"),
            html.P(
                "Her rating için gerçekleşen default oranını (DR) beklenen PD ile "
                "karşılaştırır. Binomial güven aralıkları ile DR'nin beklenen "
                "sınırlar içinde olup olmadığını test eder.",
                className="slide-text",
            ),
            html.Div([
                html.Span("Conservatism", className="slide-chip"),
                html.Span("Monotonicity", className="slide-chip"),
                html.Span("Güven Sınırları", className="slide-chip"),
            ], className="slide-chips"),
        ], className="slide-content"),

        # Slide 4 — HHI & Bad Rate
        html.Div([
            html.H4("HHI & Bad Rate", className="slide-title"),
            html.P(
                "HHI (Herfindahl-Hirschman Index) rating dağılımının "
                "konsantrasyonunu ölçer — düşük HHI dengeli dağılım demektir. "
                "Bad Rate ise dönemsel default oranını takip eder.",
                className="slide-text",
            ),
            html.Div("HHI < 0.06 → Dengeli  |  0.06–0.10 → Orta  |  > 0.10 → Yoğun",
                      className="slide-highlight"),
        ], className="slide-content"),

        # Slide 5 — Göç Matrisi
        html.Div([
            html.H4("Göç Matrisi — Rating Geçişleri", className="slide-title"),
            html.P(
                "Aynı müşterilerin referanstaki ve izlemedeki rating'lerini karşılaştırır. "
                "Köşegen üzerindeki yoğunluk yüksekse model kararlıdır. "
                "ID kolonu seçildiğinde aktif olur.",
                className="slide-text",
            ),
            html.Div([
                html.Span("Kararlılık Oranı", className="slide-chip"),
                html.Span("25×25 Heatmap", className="slide-chip"),
                html.Span("ID Eşleştirme", className="slide-chip"),
            ], className="slide-chips"),
        ], className="slide-content"),
    ]

    slide_divs = []
    for i, slide in enumerate(slides):
        slide_divs.append(
            html.Div(
                slide,
                className="slideshow-slide" + (" slide-active" if i == 0 else ""),
                id=f"mon-slide-{i}",
                style={"display": "block" if i == 0 else "none"},
            )
        )

    dots = html.Div(
        [html.Span(
            "",
            className="slide-dot" + (" dot-active" if i == 0 else ""),
            id=f"mon-slide-dot-{i}",
            n_clicks=0,
        ) for i in range(_MON_N_SLIDES)],
        className="slide-dots",
    )

    progress = html.Div(
        html.Div(className="slide-progress-fill", id="mon-slide-progress-fill",
                 style={"width": f"{100 / _MON_N_SLIDES}%"}),
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
                    html.Span(id="mon-slideshow-elapsed", children="0:00", style={
                        "fontSize": "0.85rem", "color": "#6b7a99",
                        "fontVariantNumeric": "tabular-nums",
                    }),
                ], style={"flex": "1"}),
                dbc.Button("✕", id="mon-btn-slideshow-close", size="sm", outline=True,
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
        id="mon-modal-slideshow",
        is_open=False,
        backdrop="static",
        keyboard=False,
        centered=True,
        size="lg",
        style={"zIndex": 9998},
    )


def _build_mon_compute_modal():
    """Sistem Hazırlanıyor modalı — hesaplama sırasında kullanıcıyı bekletir."""
    return dbc.Modal([
        dbc.ModalBody([
            html.Div([
                html.Div(
                    className="spinner-border text-primary",
                    role="status",
                    style={"width": "2.5rem", "height": "2.5rem",
                           "marginBottom": "1rem"},
                ),
                html.H5("Sistem Hazırlanıyor",
                         style={"color": "#c8cdd8", "marginBottom": "0.5rem"}),
                html.P(id="mon-compute-modal-step",
                       children="Hesaplama başlatılıyor…",
                       style={"color": "#7e8fa4", "fontSize": "0.82rem",
                              "marginBottom": "0.75rem"}),
                dbc.Progress(
                    id="mon-compute-modal-progress",
                    value=0, max=100,
                    striped=True, animated=True,
                    style={"height": "6px", "backgroundColor": "#1a2332",
                           "width": "80%", "margin": "0 auto"},
                    color="primary",
                ),
                html.Div(
                    id="mon-compute-modal-elapsed",
                    children="0:00",
                    style={"color": "#6b7a99", "fontSize": "0.78rem",
                           "marginTop": "0.75rem",
                           "fontVariantNumeric": "tabular-nums"},
                ),
            ], style={"textAlign": "center", "padding": "2rem 1rem"}),
        ], style={"backgroundColor": "#0e1117"}),
    ],
        id="mon-modal-compute",
        is_open=False,
        backdrop="static",
        keyboard=False,
        centered=True,
        size="md",
        style={"zIndex": 9999},
    )


def build_izleme_container():
    """İzleme üst düzey container — sidebar + main, tam bağımsız."""
    return html.Div(
        id="container-izleme",
        style={"display": "none"},
        children=[
            _build_mon_slideshow_modal(),
            _build_mon_compute_modal(),
            dbc.Row([
                dbc.Col(
                    html.Div([
                        build_izleme_sidebar(),
                        html.Button("‹", id="mon-btn-sidebar-toggle",
                                    className="sidebar-toggle"),
                    ]),
                    id="col-mon-sidebar", width=3, style=_COL_SIDEBAR_OPEN,
                ),
                dbc.Col(
                    build_izleme_main(),
                    id="col-mon-main", width=9,
                    style={**_COL_MAIN_OPEN, "position": "relative"},
                ),
            ], style={"margin": "0"}),
        ],
    )
