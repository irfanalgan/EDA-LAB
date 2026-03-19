"""
Anomali tespiti ve inline hint bileşenleri.
Her check fonksiyonu bir (seviye, başlık, açıklama_listesi) tuple döndürür.
build_hint_section() bunları tek bir Dash bileşenine dönüştürür.
"""
from dash import html
import dash_bootstrap_components as dbc

# seviye → renk / ikon
_LEVEL = {
    "error":   ("#ef4444", "✕"),
    "warning": ("#f59e0b", "⚠"),
    "info":    ("#4F8EF7", "ℹ"),
}


def _hint_card(level: str, title: str, bullets: list[str]) -> html.Div:
    color, icon = _LEVEL.get(level, _LEVEL["info"])
    return html.Div([
        html.Div(
            [html.Span(icon, style={"marginRight": "6px", "fontWeight": "700"}),
             html.Span(title, style={"fontWeight": "600"})],
            style={"color": color, "fontSize": "0.78rem", "marginBottom": "4px"},
        ),
        html.Ul([
            html.Li(b, style={"fontSize": "0.74rem", "color": "#9aa5bc",
                              "lineHeight": "1.6"})
            for b in bullets
        ], style={"paddingLeft": "1.2rem", "margin": "0"}),
    ], style={
        "backgroundColor": "#0e1520",
        "border": f"1px solid {color}44",
        "borderLeft": f"3px solid {color}",
        "borderRadius": "5px",
        "padding": "0.5rem 0.7rem",
        "marginBottom": "0.4rem",
    })


def build_hint_section(checks: list[tuple]) -> html.Div:
    """
    checks: [(level, title, [bullet, ...]), ...]
    Boş liste gelirse html.Div() döner.
    """
    if not checks:
        return html.Div()
    cards = [_hint_card(lvl, title, bullets) for lvl, title, bullets in checks]
    return dbc.Collapse(
        html.Div([
            html.P("Tanı", style={"fontSize": "0.7rem", "color": "#556070",
                                  "textTransform": "uppercase", "letterSpacing": "0.08em",
                                  "marginBottom": "0.4rem", "marginTop": "0"}),
            *cards,
        ], style={"marginBottom": "1rem"}),
        is_open=True,
    )


# ── Kural kütüphanesi ─────────────────────────────────────────────────────────

def check_iv(iv_total: float, woe_df_empty: bool) -> list[tuple]:
    hints = []
    if woe_df_empty or iv_total == 0.0:
        hints.append(("error", "IV = 0 — WOE hesaplanamadı", [
            "Train setinde yeterli event veya non-event yok (genellikle < 5)",
            "Değişken sabit / zero variance (tüm değerler aynı)",
            "sklearn ≥ 1.6 ile optbinning uyumsuzluğu — monkey-patch eksik olabilir",
            "Değişken tipi yanlış algılandı; 'Kategorik' olarak zorla ve tekrar dene",
        ]))
    elif iv_total < 0.02:
        hints.append(("warning", f"IV çok düşük ({iv_total:.4f}) — prediktif güç zayıf", [
            "Bu değişken modele katkı sağlamayabilir",
            "Segment filtresi aktifse o segmentte anlamlı olmayabilir",
        ]))
    return hints


def check_psi(psi_val, date_col: str | None, cutoff_date: str | None) -> list[tuple]:
    hints = []
    if psi_val is None:
        if not date_col:
            hints.append(("info", "PSI hesaplanmadı — tarih kolonu seçilmemiş", [
                "Sol panelden bir tarih kolonu seçin",
                "Tarih kolonu yoksa PSI bu veri setiyle hesaplanamaz",
            ]))
        elif not cutoff_date:
            hints.append(("info", "PSI hesaplanmadı — kesim tarihi belirlenmemiş", [
                "Üstteki 'PSI Kesim Tarihi' dropdown'ından bir ay seçin",
                "OOT tarihi config'de tanımlanmışsa otomatik dolmalıdır",
            ]))
        else:
            hints.append(("warning", "PSI hesaplanamadı", [
                "Seçilen kesim tarihinde yeterli veri olmayabilir",
                "Değişken seçilen periyotta tamamen eksik olabilir",
                "Baseline veya karşılaştırma setinde bu değişken sabit olabilir",
            ]))
    elif psi_val > 0.25:
        hints.append(("error", f"PSI = {psi_val:.4f} — ciddi dağılım kayması", [
            "OOT ve train dağılımları birbirinden çok farklı",
            "Veri üretim süreci değişmiş olabilir (segment, dönem, kapsam)",
            "Model bu değişkenle birlikte yeniden eğitilmeli",
        ]))
    elif psi_val > 0.10:
        hints.append(("warning", f"PSI = {psi_val:.4f} — orta düzey kayma", [
            "Dağılım dikkat gerektiriyor; izlemeye devam edin",
            "Mevsimsellik veya kısa vadeli trend olabilir",
        ]))
    return hints


def check_variable_stats(vstats: dict) -> list[tuple]:
    hints = []
    if vstats.get("unique", 99) <= 1:
        hints.append(("error", "Zero variance — değişken sabit", [
            "Tüm satırlar aynı değeri taşıyor",
            "Bu değişken modele dahil edilmemeli",
            "Segment filtresi aktifse sadece o segmentte sabit olabilir",
        ]))
    if vstats.get("missing_pct", 0) > 50:
        hints.append(("warning", f"Yüksek eksik veri (%{vstats['missing_pct']})", [
            "Verinin yarısından fazlası eksik — WOE ve PSI güvenilirliği azalır",
            "Eksik değer imputation stratejisi belirlenebilir",
            "Eksik → Bad Rate farkı aşağıdaki 'Eksik Değer & Target İlişkisi' kartında görünür",
        ]))
    return hints


def check_train_size(n_train: int) -> list[tuple]:
    hints = []
    if n_train < 100:
        hints.append(("warning", f"Train seti küçük (n={n_train:,})", [
            "WOE ve IV tahminleri güvenilir olmayabilir",
            "Segment filtresi aktifse daha geniş bir segment deneyin",
            "OOT kesim tarihini ileriye çekerek train'i büyütebilirsiniz",
        ]))
    return hints
