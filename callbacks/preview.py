import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd

from app_instance import app
from server_state import _SERVER_STORE, get_df as _get_df
from utils.helpers import apply_segment_filter
from utils.chart_helpers import _tab_info
from modules.screening import screen_columns
from layout import (
    _COL_SIDEBAR_OPEN, _COL_SIDEBAR_CLOSED,
    _COL_MAIN_OPEN, _COL_MAIN_CLOSED,
    _SIDEBAR_OPEN_STYLE, _SIDEBAR_CLOSED_STYLE,
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
        _tab_info("Önizleme", "Ham Veri & Uzman Eleme",
                  "Yüklenen verinin ilk satırlarını gösterir. Uzman bilgisiyle belirli değişkenleri "
                  "analiz kapsamından çıkarabilirsiniz; bu seçim tüm sekmelere yansır ve orijinal "
                  "veriyi değiştirmez.",
                  "#4F8EF7"),
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
