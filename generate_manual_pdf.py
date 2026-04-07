"""
EDA Lab — Kullanıcı Manueli PDF Üretici  (fpdf2)
Çalıştır:  python generate_manual_pdf.py
Çıktı:     EDA_Lab_Kullanici_Manueli.pdf
"""
from fpdf import FPDF, XPos, YPos
import pathlib

OUTPUT = pathlib.Path(__file__).parent / "EDA_Lab_Kullanici_Manueli.pdf"

# ── Renkler ──────────────────────────────────────────────────────────────────
BLUE   = (30, 58, 95)       # başlık
ACCENT = (79, 142, 247)     # vurgu
BLACK  = (26, 26, 26)
GRAY   = (100, 100, 100)
WHITE  = (255, 255, 255)
TH_BG  = (232, 238, 245)    # tablo başlık arka plan
ROW_BG = (245, 247, 250)    # çift satır arka plan
INFO_BG   = (238, 244, 255)
INFO_BD   = (79, 142, 247)
WARN_BG   = (255, 248, 230)
WARN_BD   = (245, 158, 11)
TIP_BG    = (230, 255, 240)
TIP_BD    = (16, 185, 129)

LH = 6  # satır yüksekliği


class ManualPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("ArialTR", "",  "C:/Windows/Fonts/arial.ttf")
        self.add_font("ArialTR", "B", "C:/Windows/Fonts/arialbd.ttf")
        self.add_font("ArialTR", "I", "C:/Windows/Fonts/ariali.ttf")
        self.add_font("ArialTR", "BI", "C:/Windows/Fonts/arialbi.ttf")
        self.set_auto_page_break(auto=True, margin=20)

    # ── Footer ──
    def footer(self):
        self.set_y(-15)
        self.set_font("ArialTR", "I", 7)
        self.set_text_color(*GRAY)
        self.cell(0, 10, f"EDA Lab — Kullanıcı Manueli — Sayfa {self.page_no()}/{{nb}}",
                  align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ── Yardımcılar ──
    def h1(self, text):
        self.set_font("ArialTR", "B", 20)
        self.set_text_color(*BLUE)
        self.cell(0, 12, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*ACCENT)
        self.set_line_width(0.8)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(6)

    def h2(self, text):
        self.ln(4)
        self.set_font("ArialTR", "B", 15)
        self.set_text_color(*BLUE)
        self.cell(0, 10, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(192, 208, 224)
        self.set_line_width(0.4)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def h3(self, text):
        self.ln(3)
        self.set_font("ArialTR", "B", 12)
        self.set_text_color(42, 80, 128)
        self.cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def h4(self, text):
        self.ln(2)
        self.set_font("ArialTR", "B", 10)
        self.set_text_color(58, 96, 144)
        self.cell(0, 7, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def body(self, text):
        self.set_font("ArialTR", "", 9.5)
        self.set_text_color(*BLACK)
        self.multi_cell(0, LH, text)
        self.ln(1)

    def bold_body(self, text):
        self.set_font("ArialTR", "B", 9.5)
        self.set_text_color(*BLACK)
        self.multi_cell(0, LH, text)
        self.ln(1)

    def bullet(self, text):
        self.set_font("ArialTR", "", 9.5)
        self.set_text_color(*BLACK)
        x0 = self.get_x()
        self.cell(6, LH, "  •  ")
        w = self.w - self.r_margin - self.get_x()
        self.multi_cell(w, LH, text)
        self.set_x(x0)

    def step(self, num, text):
        self.set_font("ArialTR", "B", 9.5)
        self.set_text_color(*ACCENT)
        x0 = self.get_x()
        self.cell(8, LH, f"{num}.")
        self.set_font("ArialTR", "", 9.5)
        self.set_text_color(*BLACK)
        w = self.w - self.r_margin - self.get_x()
        self.multi_cell(w, LH, text)
        self.set_x(x0)

    def info_box(self, text):
        self._color_box(INFO_BG, INFO_BD, text)

    def warn_box(self, text):
        self._color_box(WARN_BG, WARN_BD, text)

    def tip_box(self, text):
        self._color_box(TIP_BG, TIP_BD, text)

    def _color_box(self, bg, bd, text):
        self.ln(2)
        x, y = self.get_x(), self.get_y()
        w = self.w - self.l_margin - self.r_margin
        self.set_fill_color(*bg)
        self.set_font("ArialTR", "", 9)
        self.set_text_color(*BLACK)
        # calc height
        lines = self.multi_cell(w - 10, LH, text, dry_run=True, output="LINES")
        h = len(lines) * LH + 8
        if self.get_y() + h > self.h - 20:
            self.add_page()
            y = self.get_y()
        self.rect(x, y, w, h, style="F")
        self.set_draw_color(*bd)
        self.set_line_width(1)
        self.line(x, y, x, y + h)
        self.set_xy(x + 5, y + 4)
        self.multi_cell(w - 10, LH, text)
        self.set_y(y + h + 3)

    def table(self, headers, rows, col_widths=None):
        """Basit tablo çizer."""
        w_total = self.w - self.l_margin - self.r_margin
        if col_widths is None:
            cw = [w_total / len(headers)] * len(headers)
        else:
            cw = col_widths
        # header
        self.set_font("ArialTR", "B", 8.5)
        self.set_fill_color(*TH_BG)
        self.set_text_color(*BLUE)
        self.set_draw_color(192, 200, 208)
        for i, h in enumerate(headers):
            self.cell(cw[i], 7, h, border=1, fill=True, align="L")
        self.ln()
        # rows
        self.set_font("ArialTR", "", 8.5)
        self.set_text_color(*BLACK)
        for ri, row in enumerate(rows):
            if ri % 2 == 1:
                self.set_fill_color(*ROW_BG)
                fill = True
            else:
                self.set_fill_color(*WHITE)
                fill = True
            max_lines = 1
            row_cells = []
            for ci, cell_text in enumerate(row):
                lines = self.multi_cell(cw[ci], LH, str(cell_text), dry_run=True, output="LINES")
                max_lines = max(max_lines, len(lines))
                row_cells.append(str(cell_text))
            rh = max(7, max_lines * LH)
            if self.get_y() + rh > self.h - 20:
                self.add_page()
            for ci, cell_text in enumerate(row_cells):
                x, y = self.get_x(), self.get_y()
                self.rect(x, y, cw[ci], rh, style="DF" if fill else "D")
                self.set_xy(x + 1, y + 1)
                self.multi_cell(cw[ci] - 2, LH, cell_text)
                self.set_xy(x + cw[ci], y)
            self.set_y(y + rh)
        self.ln(3)


def build():
    pdf = ManualPDF()
    pdf.alias_nb_pages()
    pw = pdf.w - pdf.l_margin - pdf.r_margin

    # ══════════════════════════════════════════════════════════════════════
    # KAPAK
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.ln(70)
    pdf.set_font("ArialTR", "B", 40)
    pdf.set_text_color(*ACCENT)
    pdf.cell(0, 16, "EDA LAB", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    pdf.set_font("ArialTR", "", 16)
    pdf.set_text_color(*GRAY)
    pdf.cell(0, 10, "Kullanıcı Manueli", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(6)
    pdf.set_font("ArialTR", "", 12)
    pdf.set_text_color(68, 68, 68)
    pdf.cell(0, 8, "Keşifsel Veri Analizi ve Kredi Riski Modelleme Platformu", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(30)
    pdf.set_font("ArialTR", "", 10)
    pdf.set_text_color(*GRAY)
    pdf.cell(0, 7, "Sürüm 1.0 — Nisan 2026", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 7, "localhost:8060", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ══════════════════════════════════════════════════════════════════════
    # İÇİNDEKİLER
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.h1("İçindekiler")
    toc = [
        ("1. Giriş ve Genel Bakış", 0),
        ("1.1 EDA Lab Nedir?", 1),
        ("1.2 Temel Kavramlar", 1),
        ("1.3 Uygulamayı Başlatma", 1),
        ("2. Geliştirme Ekranı", 0),
        ("2.1 Veri Yükleme (SQL / CSV)", 1),
        ("2.2 Kolon Yapılandırması", 1),
        ("2.3 Ön İşlem Pipeline", 1),
        ("2.4 Önizleme", 1),
        ("2.5 Describe (Profilleme)", 1),
        ("2.6 Target & IV", 1),
        ("2.7 Outlier Analizi", 1),
        ("2.8 Değişken Analizi (Deep Dive)", 1),
        ("2.9 İstatistiksel Testler", 1),
        ("2.10 Değişken Özeti", 1),
        ("2.11 Modelleme (Playground)", 1),
        ("2.12 Sonuç", 1),
        ("2.13 Profil Yönetimi", 1),
        ("3. İzleme Ekranı", 0),
        ("3.1 Referans ve İzleme Verisi Yükleme", 1),
        ("3.2 Yapılandırma", 1),
        ("3.3 Önizleme", 1),
        ("3.4 PSI (Population Stability Index)", 1),
        ("3.5 Gini / KS (Ayırıcılık Gücü)", 1),
        ("3.6 Temerrüt Oranı (Bad Rate)", 1),
        ("3.7 HHI (Herfindahl-Hirschman Endeksi)", 1),
        ("3.8 Göç Matrisi (Migration Matrix)", 1),
        ("3.9 Backtesting", 1),
        ("4. Metrik Referans Tablosu", 0),
        ("5. Sık Karşılaşılan Sorunlar ve İpuçları", 0),
    ]
    for title, level in toc:
        indent = 4 + level * 10
        bold = "B" if level == 0 else ""
        size = 10 if level == 0 else 9
        pdf.set_font("ArialTR", bold, size)
        pdf.set_text_color(*BLUE)
        pdf.set_x(pdf.l_margin + indent)
        pdf.cell(0, 6, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ══════════════════════════════════════════════════════════════════════
    # BÖLÜM 1 — GİRİŞ
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.h1("1. Giriş ve Genel Bakış")

    pdf.h2("1.1 EDA Lab Nedir?")
    pdf.body(
        "EDA Lab, kredi riski modelleme sürecinin tamamını tek bir web arayüzünde "
        "birleştiren bir Keşifsel Veri Analizi (Exploratory Data Analysis) platformudur. "
        "Verinin yüklenmesinden model kurulmasına, model kurulduktan sonra performansının "
        "izlenmesine kadar tüm adımları kapsar."
    )
    pdf.body("Uygulama iki ana ekrandan oluşur:")
    pdf.table(
        ["Ekran", "Amaç", "Tipik Kullanıcı"],
        [
            ["Geliştirme", "Veri analizi, değişken seçimi, model kurma", "Risk analisti, veri bilimci"],
            ["İzleme", "Mevcut modelin performansını dönem bazlı takip etme", "Model validasyon ekibi, risk yöneticisi"],
        ],
        [pw * 0.18, pw * 0.48, pw * 0.34],
    )

    pdf.h2("1.2 Temel Kavramlar")
    pdf.body("Manuel boyunca sık kullanılan terimlerin kısa tanımları:")
    pdf.table(
        ["Terim", "Açıklama"],
        [
            ["Target", "Tahmin edilmek istenen ikili (0/1) değişken. Kredi riskinde genellikle temerrüt durumu (1 = kötüye giden, 0 = iyi)."],
            ["WoE", "Weight of Evidence — Bir değişkenin her aralığı (bin) için hesaplanan log-odds oranı. Negatif WoE = düşük risk, pozitif = yüksek risk."],
            ["IV", "Information Value — Bir değişkenin target'ı ne kadar iyi ayırt edebildiğini ölçen 0–1+ aralığında bir değer. IV > 0.1 genellikle \"kullanılabilir\"."],
            ["PSI", "Population Stability Index — İki dönem arasında dağılımdaki kaymayı ölçer. 0.10 altı stabil, 0.25 üstü kritik."],
            ["Gini / AUC", "Modelin iyi ve kötü müşterileri ne kadar iyi ayırt edebildiğini ölçer. AUC = 0.5 rastgele, AUC = 1.0 kusursuz."],
            ["KS", "Kolmogorov-Smirnov — İyi ve kötü müşterilerin kümülatif dağılım farkı. Yüzde olarak ifade edilir."],
            ["OOT", "Out-of-Time — Modelin eğitim döneminden farklı bir zaman dilimindeki verilerle test edilmesi."],
            ["Bin", "Sürekli bir değişkenin bölündüğü aralık. Örneğin: [0, 1000), [1000, 5000), [5000, +∞)."],
            ["HHI", "Herfindahl-Hirschman Endeksi — Rating dağılımının ne kadar dengeli olduğunu ölçer."],
            ["Göç Matrisi", "Müşterilerin bir dönemden diğerine rating değişimlerini gösteren matris."],
        ],
        [pw * 0.16, pw * 0.84],
    )

    pdf.h2("1.3 Uygulamayı Başlatma")
    pdf.body("Komut satırından proje klasörüne gidip aşağıdaki komutu çalıştırın:")
    pdf.bold_body("    python app.py")
    pdf.body("Uygulama http://localhost:8060 adresinde açılacaktır. Tarayıcınızda bu adresi açın.")
    pdf.info_box(
        "Yapılandırma dosyası: Proje klasöründeki config.toml dosyasında varsayılan SQL Server bağlantı "
        "bilgilerini önceden tanımlayabilirsiniz. Bu bilgiler uygulama açıldığında SQL alanlarına otomatik doldurulur."
    )

    # ══════════════════════════════════════════════════════════════════════
    # BÖLÜM 2 — GELİŞTİRME
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.h1("2. Geliştirme Ekranı")
    pdf.body(
        "Üst navigasyon çubuğunda \"Geliştirme\" butonuna tıklayarak bu ekrana ulaşılır (varsayılan ekrandır). "
        "Sol tarafta yapılandırma paneli (sidebar), sağ tarafta analiz sekmeleri yer alır."
    )

    # ── 2.1 Veri Yükleme ──
    pdf.h2("2.1 Veri Yükleme")
    pdf.body("Sidebar'ın üst bölümünde \"Veri Kaynağı\" seçimi bulunur. İki seçenekten biri seçilir:")

    pdf.h3("SQL Server ile Yükleme")
    pdf.body("SQL Server seçeneği işaretlendiğinde aşağıdaki alanlar görünür:")
    pdf.table(
        ["Alan", "Açıklama", "Örnek"],
        [
            ["Server", "SQL Server sunucu adı", "SQLSERVER01"],
            ["Database", "Veritabanı adı", "RiskDB"],
            ["Driver", "ODBC sürücü versiyonu", "ODBC Driver 18"],
            ["Tablo 1", "Ana tablo adı (schema dahil)", "dbo.KREDI_MASTER"],
            ["Tablo 2 (opsiyonel)", "İkinci tablo (join için)", "dbo.MUSTERI_BILGI"],
            ["Tablo 3 (opsiyonel)", "Üçüncü tablo", "dbo.TEMINAT"],
            ["Join Key", "Tabloları birleştiren kolon(lar)", "musteri_no, kredi_no"],
        ],
        [pw * 0.24, pw * 0.46, pw * 0.30],
    )
    pdf.step(1, "Server, Database ve Driver bilgilerini girin.")
    pdf.step(2, "Ana tablo adını yazın.")
    pdf.step(3, "Gerekiyorsa \"+ Tablo Ekle\" ile ek tablolar ekleyin ve join key belirtin.")
    pdf.step(4, "Test için \"İlk 1000 satır\" kutucuğunu işaretleyebilirsiniz (büyük verilerde hızlı kontrol için).")
    pdf.step(5, "\"Veriyi Yükle\" butonuna tıklayın.")
    pdf.tip_box(
        "İpucu: config.toml dosyasında [database] bölümüne server ve database bilgilerini önceden yazarsanız, "
        "her seferinde tekrar girmenize gerek kalmaz."
    )

    pdf.h3("CSV Dosyası ile Yükleme")
    pdf.step(1, "Dosya 1 alanına CSV dosyanızı sürükleyin veya tıklayarak seçin.")
    pdf.step(2, "Birden fazla dosya yüklemeniz gerekiyorsa \"+ Dosya Ekle\" ile ek alanlar ekleyin (en fazla 3).")
    pdf.step(3, "\"Ayırıcı\" açılır listesinden dosyanın ayırıcısını seçin: Virgül (,), Noktalı virgül (;), Tab, Pipe (|).")
    pdf.step(4, "\"Yükle\" butonuna tıklayın.")
    pdf.info_box(
        "Birden fazla CSV yüklendiğinde dosyalar otomatik olarak ortak kolon üzerinden birleştirilir (join). "
        "Her dosyada ortak bir kolon (join key) bulunmalıdır."
    )
    pdf.body(
        "Veri yüklenirken ekranda bir eğitici sunu (tutorial) gösterilir. Bu sunu 8 slayttan oluşur ve "
        "uygulamanın temel özelliklerini kısaca tanıtır. Süreçler arkaplanda devam eder."
    )

    # ── 2.2 Yapılandırma ──
    pdf.h2("2.2 Kolon Yapılandırması")
    pdf.body("Veri yüklendikten sonra sidebar'da \"Kolon Yapılandırması\" bölümü açılır:")
    pdf.table(
        ["Alan", "Zorunlu", "Açıklama"],
        [
            ["Target Kolonu", "Evet", "İkili (0/1) bağımsız değişken. 1 = kötü (temerrüt), 0 = iyi."],
            ["Tarih Kolonu", "Evet", "Zaman boyutu için kullanılan tarih kolonu. OOT ayırma ve trend analizleri için."],
            ["OOT Başlangıç Tarihi", "Hayır", "Bu tarihten önceki veri Train, sonraki OOT olarak ayrılır."],
            ["Train/Test Ayrımı", "Hayır", "İşaretlenirse Train verisinin bir kısmı Test olarak ayrılır (%10-%50)."],
            ["Maks Bin Sayısı", "Hayır", "WoE binleme için maksimum aralık sayısı. Varsayılan: 4, aralık: 2–20."],
            ["Segment Kolonu", "Hayır", "Veriyi alt gruplara ayırmak için kategorik bir kolon (örn: ürün tipi)."],
        ],
        [pw * 0.22, pw * 0.12, pw * 0.66],
    )
    pdf.step(1, "Target ve Tarih kolonlarını seçin.")
    pdf.step(2, "OOT tarihi belirleyin (varsa).")
    pdf.step(3, "Gerekiyorsa segment kolonunu seçin.")
    pdf.step(4, "\"Yapılandırmayı Onayla\" butonuna tıklayın.")
    pdf.warn_box(
        "Dikkat: Yapılandırma onaylandıktan sonra arkaplanda otomatik ön işlem başlar. "
        "Bu işlem veri büyüklüğüne göre 2–15 dakika sürebilir. İşlem sırasında sekmelerde analiz yapabilirsiniz."
    )

    # ── 2.3 Pipeline ──
    pdf.h2("2.3 Ön İşlem Pipeline")
    pdf.body("Yapılandırma onaylandığında otomatik olarak 6 adımlık bir işlem başlar:")
    pdf.table(
        ["Adım", "İşlem", "Açıklama"],
        [
            ["1", "Ön Eleme (Screening)", "%60'tan fazla eksik değer ve sabit (tek değerli) değişkenleri eler."],
            ["2", "Profil Analizi", "Her kolon için temel istatistikleri hesaplar."],
            ["3", "IV Ranking (WoE/IV)", "Her değişken için optimal binleme yapar, WoE ve IV hesaplar."],
            ["4", "Korelasyon Matrisi", "Değişkenler arası Pearson korelasyon matrisini hesaplar."],
            ["5", "Değişken Özeti (WoE)", "WoE dönüşümlü değişken özet tablosunu oluşturur."],
            ["6", "Değişken Özeti (Ham)", "Ham veri üzerinden değişken özet tablosunu oluşturur."],
        ],
        [pw * 0.08, pw * 0.28, pw * 0.64],
    )
    pdf.body(
        "Her adımın yanında durum simgesi görünür: Bekliyor, Calisiyor, Tamamlandi. "
        "\"İptal\" butonu ile işlemi istediğiniz anda durdurabilirsiniz."
    )

    # ── 2.4 Önizleme ──
    pdf.h2("2.4 Önizleme")
    pdf.body("İlk 1000 satırın önizleme tablosu ve uzman eleme paneli içerir:")
    pdf.bullet("Veri Tablosu: Kolon adları, veri tipleri ve değerler görüntülenir.")
    pdf.bullet(
        "Uzman Eleme Paneli: Belirli kolonları analizden elle çıkarmanızı sağlar. "
        "Her kolon için isim, tip, eksik oranı ve tekil değer sayısı görüntülenir."
    )
    pdf.tip_box(
        "İpucu: Uzman elemesi yaptığınızda bu kolonlar tüm sonraki analizlerden hariç tutulur "
        "ancak orijinal veriniz değiştirilmez. İşlemi geri almak için tekrar işaretlemeniz yeterlidir."
    )

    # ── 2.5 Describe ──
    pdf.h2("2.5 Describe (Profilleme)")
    pdf.body("Her kolon için detaylı istatistiksel profili gösteren tablodur.")
    pdf.h4("Sayısal Kolonlar İçin")
    pdf.table(
        ["Metrik", "Açıklama"],
        [
            ["Veri Tipi", "Kolonun veri tipi (int64, float64, vb.)"],
            ["Dolu Sayı / Eksik %", "Boş olmayan satır sayısı ve boş değer yüzdesi"],
            ["Tekil Değer", "Farklı değer sayısı"],
            ["Ortalama / Medyan", "Merkezi eğilim ölçüleri"],
            ["Std Sapma", "Yayılım ölçüsü"],
            ["Min / Max", "En küçük ve en büyük değerler"],
            ["Çarpıklık / Basıklık", "Dağılım şekli göstergeleri"],
        ],
        [pw * 0.28, pw * 0.72],
    )
    pdf.h4("Kategorik Kolonlar İçin")
    pdf.body("Mod (en sık tekrarlanan değer), tekil değer sayısı ve eksik yüzdesi gösterilir.")

    # ── 2.6 Target & IV ──
    pdf.h2("2.6 Target & IV")
    pdf.body("Bu sekme target değişkenin dağılımını ve her değişkenin ayırıcı gücünü (IV) gösterir.")

    pdf.h3("Target İstatistikleri")
    pdf.body("Ekranın üstünde renk kodlu kartlar yer alır:")
    pdf.bullet("Toplam Kayıt: Veri setindeki satır sayısı")
    pdf.bullet("Kötü (1) Sayısı: Target = 1 olan kayıt sayısı")
    pdf.bullet("İyi (0) Sayısı: Target = 0 olan kayıt sayısı")
    pdf.bullet("Temerrüt Oranı (%): Kötü / Toplam × 100")

    pdf.h3("Dönem Seçici")
    pdf.body("Tarih kolonu tanımlanmışsa \"Tüm Veri\", \"Train\" ve \"Test\" seçenekleri görünür.")

    pdf.h3("Temerrüt Oranı Trend Grafiği")
    pdf.body("Zaman içinde temerrüt oranının nasıl değiştiğini gösteren çizgi grafik.")

    pdf.h3("IV Sıralama Tablosu")
    pdf.body(
        "Tüm değişkenlerin IV değerine göre sıralı listesi. Sütunlar: Değişken, IV, Eksik %, Tip. "
        "Tablo herhangi bir kolona göre sıralanabilir. IV değerleri renk koduyla gösterilir "
        "(kırmızı = zayıf, sarı = orta, yeşil = güçlü)."
    )

    # ── 2.7 Outlier ──
    pdf.h2("2.7 Outlier Analizi (Aykırı Değer)")
    pdf.body("Sayısal değişkenlerdeki uç (aşırı) değerleri tespit etmek için kullanılır.")
    pdf.table(
        ["Alan", "Açıklama"],
        [
            ["Müşteri ID Kolonu", "Müşteri bazlı aykırı değer özeti için gerekli (opsiyonel)."],
            ["Yöntem", "IQR: Çeyrekler arası aralık yöntemi. Z-Score: Standart sapma tabanlı yöntem."],
            ["IQR Çarpanı", "1.5 (normal) veya 3.0 (sadece çok aşırı değerler)."],
            ["Z-Score Eşiği", "2.0, 2.5 veya 3.0 standart sapma."],
            ["Değişkenler", "Analiz edilecek değişkenleri seçin (çoklu seçim)."],
        ],
        [pw * 0.24, pw * 0.76],
    )
    pdf.h4("Çıktılar")
    pdf.bullet("Özet Tablo: Her değişken için yöntem, aykırı değer sayısı, yüzdesi, alt/üst sınır.")
    pdf.bullet("Müşteri Aykırı Detay: Hangi müşterilerin kaç değişkende aykırı değere sahip olduğu.")
    pdf.bullet("Dağılım Grafikleri: Kutu grafikleri (box plot) ile aykırı değerler görsel olarak vurgulanır.")

    # ── 2.8 Deep Dive ──
    pdf.h2("2.8 Değişken Analizi (Deep Dive)")
    pdf.body("Tek bir değişkeni derinlemesine incelemenizi sağlar.")
    pdf.bullet("Değişken Seçici: İncelemek istediğiniz değişkeni açılır listeden seçin.")
    pdf.bullet("Tip Zorlama: Otomatik / Sayısal / Kategorik — değişkenin tipini elle değiştirebilirsiniz.")

    pdf.h3("WoE Değerler Sekmesi")
    pdf.table(
        ["Bileşen", "Açıklama"],
        [
            ["Binleme Tablosu", "Her bin için: toplam sayı, kötü oranı (%), WoE değeri, IV katkısı."],
            ["WoE Trend Grafiği", "Her binin WoE değerini gösteren çizgi grafik."],
            ["Kötü Oranı Grafiği", "Her binin temerrüt oranını gösteren bar grafik."],
            ["Dağılım Histogramı", "Verinin binler arasındaki dağılımı."],
            ["Test/OOT Tabloları", "Test ve OOT verilerinde aynı binleme yapısıyla üretilen tablolar."],
            ["PSI Hesabı", "Train ile OOT arasındaki dağılım kayması."],
        ],
        [pw * 0.26, pw * 0.74],
    )
    pdf.info_box(
        "Monotonluk: WoE değerlerinin binler boyunca sürekli artan veya azalan olması istenir. "
        "\"Artan\" veya \"Azalan\" = iyi, \"Monoton Değil\" = dikkatli değerlendirilmeli."
    )

    # ── 2.9 İstatistiksel Testler ──
    pdf.h2("2.9 İstatistiksel Testler")
    pdf.body("Değişkenler arası ilişkileri istatistiksel yöntemlerle test etmenizi sağlar. Dört farklı test mevcuttur.")
    pdf.body("Alt Sekmeler: \"Ham Değerler\" ve \"WoE Değerler\" üzerinde çalıştırabilirsiniz.")

    pdf.h3("a) Korelasyon Analizi (Pearson r)")
    pdf.body("Sayısal değişkenler arasındaki doğrusal ilişkiyi ölçer.")
    pdf.bullet("Korelasyon Matrisi: Isı haritası (heatmap) olarak gösterilir. Mavi = pozitif, kırmızı = negatif.")
    pdf.bullet("Yüksek Korelasyon Çiftleri: |r| ≥ 0.60 olan çiftlerin listesi. Saçılım grafiği oluşturulabilir.")
    pdf.bullet("VIF (Variance Inflation Factor): Çoklu doğrusallık kontrolü. VIF > 5 dikkat, > 10 sorunlu.")

    pdf.h3("b) Chi-Square (Ki-Kare) Bağımsızlık Testi")
    pdf.body("İki kategorik değişken arasında bağımsızlık testi yapar.")
    pdf.bullet("Çıktılar: Chi-square istatistiği, p-değeri, serbestlik derecesi, Cramér's V (etki büyüklüğü).")
    pdf.bullet("Çapraz Tablo Isı Haritası: Gözlenen frekanslar renkli hücrelerde gösterilir.")

    pdf.h3("c) ANOVA (Target vs Sayısal Değişken)")
    pdf.body("Target gruplarının (0 ve 1) bir sayısal değişken açısından farklı olup olmadığını test eder.")
    pdf.bullet("Çıktılar: F-istatistiği, p-değeri, grup ortalamaları tablosu, kutu grafiği.")

    pdf.h3("d) Kolmogorov-Smirnov (KS) Ayırıcılık Testi")
    pdf.body("İyi ve kötü grupların bir değişken üzerindeki dağılım farkını ölçer.")
    pdf.bullet("Çıktılar: KS istatistiği (%), kümülatif dağılım karşılaştırma grafiği.")

    # ── 2.10 Değişken Özeti ──
    pdf.add_page()
    pdf.h2("2.10 Değişken Özeti")
    pdf.body(
        "Tüm değişkenlerin tek bir tabloda özetlendiği, filtreleme ve eleme işlemlerinin yapıldığı "
        "ana karar ekranıdır. Alt Sekmeler: \"WoE Değerler\" ve \"Ham Değerler\"."
    )
    pdf.h3("Filtre Bölümü")
    pdf.table(
        ["Filtre", "Açıklama", "Tipik Kullanım"],
        [
            ["IV", "Information Value filtresi", "IV ≥ 0.02 (zayıfları ele)"],
            ["Korelasyon (Target)", "Target ile korelasyon filtresi", "—"],
            ["Korelasyon (Değişken)", "Değişkenler arası maks. korelasyon eşiği", "≤ 0.70"],
            ["PSI", "Population Stability Index filtresi", "PSI < 0.25"],
            ["Eksik %", "Boş değer yüzdesi filtresi", "Eksik < 50%"],
            ["Test Monotonluk", "Test setinde monoton mu?", "Sadece monoton olanlar"],
            ["OOT Monotonluk", "OOT setinde monoton mu?", "Sadece monoton olanlar"],
        ],
        [pw * 0.24, pw * 0.44, pw * 0.32],
    )
    pdf.body("\"Filtreleri Sıfırla\" butonu tüm filtreleri varsayılana döndürür. Filtrelerin üstünde aktif değişken sayısı gösterilir.")

    pdf.h3("Ana Özet Tablosu")
    pdf.body("Filtreleme sonucunda kalan değişkenler şu sütunlarla gösterilir:")
    pdf.body(
        "Değişken, IV, Eksik %, Bin, PSI Değeri, PSI Durumu (Stabil/Dikkat/Kritik), "
        "Test Monoton, OOT Monoton, Korr Target, Korr Var, VIF."
    )

    pdf.h3("Gelişmiş Eleme Pipeline")
    pdf.body("Tablonun altında açılır bir \"Gelişmiş Eleme\" paneli vardır. Otomatik değişken eleme işlemleri sunar:")
    pdf.table(
        ["Adım", "Açıklama"],
        [
            ["Tekli Gini Kontrolü", "Her değişkenin tek başına oluşturduğu Gini skoru hesaplanır. Eşik altı olanlar elenir."],
            ["Sıfır Kazanç Eleme", "Model içinde sıfır kazanç (zero gain) sağlayan değişkenler elenir."],
            ["RFE", "Özyinelemeli olarak en az önemli değişkenler çıkarılır (Recursive Feature Elimination)."],
            ["Performans Eğrisi", "Değişken sayısı artırılarak model performansı ölçülür. En iyi denge görselleştirilir."],
        ],
        [pw * 0.26, pw * 0.74],
    )
    pdf.body("Her adım için bir onay kutusu vardır — sadece işaretlenenleri çalıştırır. \"Hesapla\" ile pipeline başlatılır.")

    # ── 2.11 Modelleme ──
    pdf.h2("2.11 Modelleme (Playground)")
    pdf.body("Bu sekme iki bölümden oluşur: Grafik Oluşturucu ve Hızlı Model.")

    pdf.h3("Grafik Oluşturucu")
    pdf.body("Veriyi görsel olarak keşfetmek için esnek bir grafik aracı:")
    pdf.table(
        ["Alan", "Açıklama"],
        [
            ["X Ekseni", "Yatay eksen değişkeni"],
            ["Y Ekseni (Sol)", "Dikey eksen değişkeni (birincil)"],
            ["Y2 Ekseni (Sağ)", "İkincil dikey eksen (opsiyonel — çift eksenli grafik için)"],
            ["Grafik Tipi", "Bar / Çizgi / Bar+Çizgi / Saçılım / Kutu / Histogram"],
            ["Agregasyon", "Ortalama / Toplam / Adet"],
            ["Zaman Birimi", "Tarih kolonu seçilmişse: Gün / Ay / Çeyrek / Yıl"],
            ["Grupla (renk)", "Kategorik değişkene göre renklendirme (opsiyonel)"],
        ],
        [pw * 0.24, pw * 0.76],
    )
    pdf.body("\"Çiz\" butonuna tıklandığında interaktif bir Plotly grafik oluşur.")

    pdf.h3("Hızlı Model")
    pdf.body("Seçtiğiniz değişkenlerle hızlıca model kurar, performansını değerlendirir.")
    pdf.step(1, "\"Model Değişkenleri\" açılır listesinden kullanmak istediğiniz değişkenleri seçin.")
    pdf.step(2, "\"Tümünü Ekle\" ile filtrelenmiş tüm değişkenleri ekleyebilir, \"Temizle\" ile sıfırlayabilirsiniz.")
    pdf.step(3, "\"Model Kur\" butonuna tıklayın.")

    pdf.h4("Model Türleri")
    pdf.table(
        ["Model", "Açıklama", "Ne Zaman Kullanılır"],
        [
            ["Logistic Regression", "Doğrusal lojistik regresyon", "Yorumlanabilirlik ön plandaysa"],
            ["LightGBM", "Gradient boosting (hızlı)", "Yüksek performans gerektiğinde"],
            ["XGBoost", "Gradient boosting (hassas)", "Yüksek performans + dayanıklılık"],
            ["Random Forest", "Rastgele orman", "Genel amaçlı, robust"],
        ],
        [pw * 0.22, pw * 0.38, pw * 0.40],
    )
    pdf.body(
        "Her model türü için parametreler \"Model Parametreleri\" panelinden düzenlenebilir. "
        "Ayrıca Optuna hiperparametre optimizasyonu ile en iyi parametreleri otomatik aratabilirsiniz."
    )

    pdf.h4("Çıktılar")
    pdf.body("Model kurulduğunda Ham ve WoE encode karşılaştırması yan yana gösterilir:")
    pdf.bullet("Metrik Kartları: AUC, Gini, KS, F1, Precision, Recall değerleri.")
    pdf.bullet("ROC Eğrisi: Train, Test ve OOT için ayrı renklerle çizilir.")
    pdf.bullet("Confusion Matrix: Doğruluk matrisi (TP, FP, TN, FN).")
    pdf.bullet("Değişken Önemliliği: Her değişkenin modele katkısı.")
    pdf.bullet("SHAP Grafiği: Ağaç tabanlı modellerde her değişkenin tahmine etkisi (beeswarm plot).")

    # ── 2.12 Sonuç ──
    pdf.h2("2.12 Sonuç")
    pdf.body("Kurulan modelin detaylı raporudur. Katlanabilir (accordion) bölümlere ayrılmıştır:")
    pdf.table(
        ["Bölüm", "İçerik"],
        [
            ["Model Özeti", "Serbest metin alanı — model hakkında notlarınızı yazabilirsiniz."],
            ["Metrikler", "Train / Test / OOT metrik kartları: AUC, Gini, KS, F1, Precision, Recall, Accuracy."],
            ["Model Detayları", "LR için katsayı tablosu; ağaç modelleri için değişken önemliliği."],
            ["ROC Eğrisi", "İnteraktif ROC çizgi grafiği."],
            ["Confusion Matrix", "Train / Test / OOT ısı haritaları."],
            ["SHAP", "Beeswarm grafiği — her nokta bir gözlemi temsil eder."],
        ],
        [pw * 0.22, pw * 0.78],
    )
    pdf.h4("Dışa Aktarma Seçenekleri")
    pdf.bullet("Model Kaydetme: Eğitilmiş modeli profil klasörüne pickle olarak kaydeder.")
    pdf.bullet("Tahminleri İndir: Train/Test/OOT tahmin sonuçlarını CSV olarak indirir.")
    pdf.bullet("Değişken Önemliliği: Excel olarak indirir.")

    # ── 2.13 Profil ──
    pdf.h2("2.13 Profil Yönetimi")
    pdf.body("Analizinizi kaydedip daha sonra aynı noktadan devam edebilirsiniz.")

    pdf.h3("Profil Kaydetme")
    pdf.step(1, "Sol panelde \"Profil Kaydet\" butonuna tıklayın.")
    pdf.step(2, "Açılan pencerede profile bir isim verin.")
    pdf.step(3, "\"Kaydet\" butonuna tıklayın. Başarılı kayıt bildirimi görüntülenir.")
    pdf.body(
        "Kaydedilen dosyalar: meta.json (yapılandırma, bağlantı bilgileri), data.pkl (ham veri seti), "
        "cache.pkl (hesaplanmış sonuçlar — IV, WoE, korelasyon vb.)."
    )
    pdf.h3("Profil Yükleme")
    pdf.step(1, "Sol panelin en üstünde \"Kayıtlı Profil\" açılır listesinden profili seçin.")
    pdf.step(2, "\"Yükle\" butonuna tıklayın.")
    pdf.body("Profil yüklendiğinde tüm yapılandırma, filtreler ve sonuçlar geri yüklenir. Yeniden hesaplama gerekmez.")
    pdf.h3("Profil Silme")
    pdf.body("\"Profil Sil\" butonu ile seçili profili kalıcı olarak silebilirsiniz. Silme işlemi onay gerektirir.")

    # ══════════════════════════════════════════════════════════════════════
    # BÖLÜM 3 — İZLEME
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.h1("3. İzleme Ekranı")
    pdf.body(
        "Üst navigasyon çubuğunda \"İzleme\" butonuna tıklayarak bu ekrana geçersiniz. "
        "İzleme ekranı, üretimde olan bir modelin zaman içindeki performansını takip etmek için tasarlanmıştır. "
        "İki farklı veri seti kullanır: Referans (model geliştirme dönemi) ve İzleme (güncel dönem)."
    )

    # ── 3.1 Veri Yükleme ──
    pdf.h2("3.1 Referans ve İzleme Verisi Yükleme")
    pdf.body("Sidebar'da iki ayrı veri yükleme bölümü vardır:")
    pdf.bullet("Referans Verisi: Modelin geliştirildiği dönemdeki veri. Tüm karşılaştırmalar bu veriye göre yapılır.")
    pdf.bullet("İzleme Verisi: Modelin uygulandığı güncel dönem verisi. Dönem bazlı analizler bu veri üzerinde yapılır.")
    pdf.body("Her iki veri için de SQL Server veya CSV yükleme seçenekleri mevcuttur.")
    pdf.warn_box(
        "Önemli: İzleme verisi şu kolonları içermelidir:\n"
        "• Target kolonu: İkili (0/1) temerrüt değişkeni\n"
        "• Tarih kolonu: Dönem bazlı analiz için\n"
        "• PD veya Rating kolonu: Model çıktısı (olasılık 0-1 veya 1-25 rating)\n"
        "• ID kolonu (opsiyonel): Göç Matrisi için gerekli\n"
        "• Model değişkenleri: WoE hesaplaması için girdi kolonları"
    )

    # ── 3.2 Yapılandırma ──
    pdf.h2("3.2 Yapılandırma")
    pdf.table(
        ["Alan", "Zorunlu", "Açıklama"],
        [
            ["Target Kolonu", "Evet", "Temerrüt değişkeni (0/1)"],
            ["Tarih Kolonu", "Evet", "Dönem ayrımı için"],
            ["PD/Rating Kolonu", "Evet", "Model çıktısı. PD ise otomatik olarak 1-25 rating ölçeğine dönüştürülür."],
            ["ID Kolonu", "Hayır", "Müşteri/kredi numarası. Göç Matrisi için gerekli."],
            ["Model Değişkenleri", "Evet", "WoE hesaplaması için girdi kolonları."],
            ["Segment Kolonu", "Hayır", "Alt grup bazlı analiz için."],
            ["Olgunlaşma Süresi (ay)", "Hayır", "Varsayılan 12. Sadece olgun dönemler temerrüt oranı hesabına dahil edilir."],
            ["Dönem Frekansı", "Hayır", "Aylık veya Çeyreklik"],
        ],
        [pw * 0.24, pw * 0.10, pw * 0.66],
    )
    pdf.info_box(
        "Olgunlaşma nedir? Bir kredinin temerrüde düşüp düşmediğini gözlemlemek için gereken süre. "
        "12 ay seçilmişse, sadece 12 ayını tamamlamış dönemler \"olgun\" sayılır ve temerrüt hesabında kullanılır."
    )

    # ── 3.3 Önizleme ──
    pdf.h2("3.3 Önizleme")
    pdf.body("İzleme verisinin ilk 1000 satırını gösterir. Veri kalitesi ve kolon tipleri kontrol edilir.")

    # ── 3.4 PSI ──
    pdf.h2("3.4 PSI (Population Stability Index)")
    pdf.body(
        "Modelin girdi dağılımlarının zamanlı olarak ne kadar değiştiğini ölçer. "
        "Referans döneme kıyasla her izleme döneminde dağılımların ne kadar kaydığı takip edilir."
    )
    pdf.h3("Trend Sekmesi")
    pdf.bullet("Dönem Seçici: İncelemek istediğiniz dönemi seçin.")
    pdf.bullet("Özet Kartları: Referans PSI, seçili dönem PSI değeri ve durum (Stabil / Dikkat / Kritik).")
    pdf.bullet("PSI Trend Grafiği: Tüm dönemlerdeki PSI değerlerini gösteren çizgi grafik. Eşik çizgileri (0.10 ve 0.25) referans olarak gösterilir.")
    pdf.bullet("Değişken PSI Tablosu: Her değişken için bin bazlı dağılım karşılaştırması — Referans %, İzleme %, Fark, PSI katkısı.")
    pdf.bullet("Rating Dağılım Tablosu: 1-25 rating skalasında referans vs izleme dağılımı.")
    pdf.h3("Kümülatif Sekmesi")
    pdf.body("Tüm dönemlerin toplulaştırılmış PSI özeti.")
    pdf.info_box(
        "Nasıl yorumlanır?\n"
        "• PSI < 0.10: Stabil — model güvenilir, aksiyon gerekmez.\n"
        "• 0.10 ≤ PSI < 0.25: Dikkat — izlemeye devam edin, kaymanın nedenini araştırın.\n"
        "• PSI ≥ 0.25: Kritik kayma — model yenilenmeli veya kalibre edilmeli."
    )

    # ── 3.5 Gini/KS ──
    pdf.h2("3.5 Gini / KS (Ayırıcılık Gücü)")
    pdf.body("Modelin iyi ve kötü müşterileri birbirinden ayırma gücünü dönem bazlı ölçer.")
    pdf.h3("Trend Sekmesi")
    pdf.bullet("Metrik Kartları: KS (%), Gini (%) ve AR (Accuracy Ratio, %) değerleri renk kodlu gösterilir.")
    pdf.bullet("Trend Grafiği: Gini ve KS değerlerinin dönem bazlı değişimini gösteren çizgi grafik.")
    pdf.bullet("KS Detay Tablosu: Rating, İyi/Kötü Sayısı, Kümülatif İyi/Kötü %, KS Mesafesi.")
    pdf.h3("Kümülatif Sekmesi")
    pdf.body("Tüm olgun dönemlerin birleştirilerek hesaplanan genel ayırıcılık metrikleri.")
    pdf.info_box(
        "Nasıl yorumlanır?\n"
        "• Gini > 40%: İyi ayırıcılık gücü.\n"
        "• Gini %20-40: Kabul edilebilir, ancak izlenmeli.\n"
        "• Gini < 20%: Zayıf — model yeniden değerlendirilmeli.\n"
        "• KS: Genellikle %20 üstü beklenir."
    )

    # ── 3.6 Bad Rate ──
    pdf.h2("3.6 Temerrüt Oranı (Bad Rate)")
    pdf.body("Dönem bazlı temerrüt oranlarını takip eder.")
    pdf.bullet("Özet Kartları: Toplam gözlem sayısı, temerrüt sayısı, temerrüt oranı (%).")
    pdf.bullet("Trend Grafiği: Dönemlere göre temerrüt oranı değişimi.")
    pdf.bullet("Rating Bazlı Detay Tablosu: Her rating seviyesi için kayıt sayısı, temerrüt sayısı, temerrüt oranı, birikimli dağılımlar.")

    # ── 3.7 HHI ──
    pdf.h2("3.7 HHI (Herfindahl-Hirschman Endeksi)")
    pdf.body("Rating dağılımının yoğunlaşmasını (konsantrasyonunu) ölçer.")
    pdf.bullet("HHI Değer Kartı: Mevcut dönemin HHI değeri.")
    pdf.bullet("Trend Grafiği: HHI zaman serisi, referans çizgileriyle birlikte (0.06 ve 0.10).")
    pdf.bullet("Rating Dağılım Tablosu: Her rating seviyesindeki yoğunlaşma payı.")
    pdf.info_box(
        "Nasıl yorumlanır?\n"
        "• HHI < 0.06: Düşük yoğunlaşma (dengeli dağılım) — ideal.\n"
        "• 0.06 ≤ HHI < 0.10: Orta yoğunlaşma — izlenmeli.\n"
        "• HHI ≥ 0.10: Yüksek yoğunlaşma — dağılım dengelenmeli."
    )

    # ── 3.8 Göç Matrisi ──
    pdf.h2("3.8 Göç Matrisi (Migration Matrix)")
    pdf.body("Müşterilerin bir dönemden diğerine rating değişimlerini izler. ID kolonu tanımlanmış olmalıdır.")
    pdf.bullet("Göç Isı Haritası: Satırlar = önceki rating, sütunlar = yeni rating. Hücrelerde müşteri sayısı ve satır yüzdesi.")
    pdf.bullet("Stabilite Oranı: Köşegen üzerindeki yüzde — aynı ratingde kalan müşterilerin oranı.")
    pdf.bullet("Akış Analizi: Yukarı hareket (upgrade) ve aşağı hareket (downgrade) sayıları.")
    pdf.warn_box(
        "Dikkat: Göç Matrisi sadece ID kolonu tanımlandığında kullanılabilir. "
        "ID kolonu her satırda benzersiz müşteri/kredi numarasını içermelidir."
    )

    # ── 3.9 Backtesting ──
    pdf.h2("3.9 Backtesting")
    pdf.body("Modelin tahminlerinin gerçek sonuçlarla tutarlılığını istatistiksel olarak test eder.")
    pdf.bullet("Binomial Test Sonuçları: Her rating seviyesi için beklenen vs gerçek temerrüt oranı, %95 güven aralığı.")
    pdf.bullet("Muhafazakarlık Kontrolü: Model gerçekten daha yüksek risk mi öngörüyor (istenen durum).")
    pdf.bullet("Monotonluk Kontrolü: Rating arttıkça temerrüt oranının monoton artıp artmadığı.")
    pdf.bullet("İhlal Tablosu: Testleri geçemeyen rating seviyeleri listelenir.")

    # ══════════════════════════════════════════════════════════════════════
    # BÖLÜM 4 — METRİK REFERANS
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.h1("4. Metrik Referans Tablosu")
    pdf.body("Uygulamada kullanılan tüm metriklerin yorumlama kılavuzu:")
    pdf.table(
        ["Metrik", "Aralık", "Yorum"],
        [
            ["IV", "< 0.02", "Çok zayıf — anlamsız, modelde kullanılmamalı"],
            ["IV", "0.02 – 0.10", "Zayıf — sınırlı ayırıcı güç"],
            ["IV", "0.10 – 0.30", "Orta — kullanılabilir"],
            ["IV", "0.30 – 0.50", "Güçlü — iyi ayırıcı güç"],
            ["IV", "> 0.50", "Şüphe — overfitting riski, dikkatle değerlendirilmeli"],
            ["PSI", "< 0.10", "Stabil — model güvenilir"],
            ["PSI", "0.10 – 0.25", "Dikkat — izlemeye devam edin"],
            ["PSI", "≥ 0.25", "Kritik kayma — model yenilenmeli"],
            ["AUC", "< 0.60", "Zayıf ayırıcılık gücü"],
            ["AUC", "0.60 – 0.70", "Kabul edilebilir"],
            ["AUC", "0.70 – 0.80", "İyi"],
            ["AUC", "> 0.80", "Çok iyi"],
            ["p-değeri", "< 0.05", "İstatistiksel olarak anlamlı"],
            ["p-değeri", "0.05 – 0.10", "Sınırda anlamlı"],
            ["p-değeri", "> 0.10", "Anlamlı değil — modelde tutma/çıkar"],
            ["VIF", "< 5", "Sorun yok"],
            ["VIF", "5 – 10", "Dikkat — orta düzey çoklu doğrusallık"],
            ["VIF", "> 10", "Sorunlu — değişkenlerden biri çıkarılmalı"],
            ["HHI", "< 0.06", "Düşük yoğunlaşma (dengeli)"],
            ["HHI", "0.06 – 0.10", "Orta yoğunlaşma"],
            ["HHI", "≥ 0.10", "Yüksek yoğunlaşma — dengelenmeli"],
        ],
        [pw * 0.16, pw * 0.20, pw * 0.64],
    )

    # ══════════════════════════════════════════════════════════════════════
    # BÖLÜM 5 — SSS
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.h1("5. Sık Karşılaşılan Sorunlar ve İpuçları")

    pdf.h3("Veri yüklenmiyor (SQL)")
    pdf.bullet("Sunucu adı yanlış yazılmış olabilir.")
    pdf.bullet("ODBC sürücü versiyonu bilgisayarınızda yüklü değil. Doğru sürücüyü seçin.")
    pdf.bullet("Veritabanına erişim yetkiniz olmayabilir. BT ekibinizle iletişime geçin.")
    pdf.bullet("Firewall SQL portunu (varsayılan 1433) engelliyor olabilir.")

    pdf.h3("Ön işlem çok uzun sürüyor")
    pdf.bullet("Değişken sayısı çok fazlaysa (500+) işlem doğal olarak uzun sürer.")
    pdf.bullet("\"İlk 1000 satır\" seçeneğiyle önce küçük bir örnekle test edin.")
    pdf.bullet("Screening aşamasında düşük kaliteli değişkenler otomatik elenir, bu süreci hızlandırır.")

    pdf.h3("IV değerleri çok düşük çıkıyor")
    pdf.bullet("Target değişkeniniz doğru seçilmemiş olabilir.")
    pdf.bullet("Temerrüt oranı çok düşükse (<%1) IV değerleri doğal olarak düşük olabilir.")
    pdf.bullet("Verinizdeki değişkenler gerçekten zayıf ayırıcı güce sahip olabilir.")

    pdf.h3("Göç Matrisi boş görünüyor")
    pdf.body("Neden: ID kolonu tanımlanmamıştır. Yapılandırmada \"ID Kolonu\" alanını doldurun.")

    pdf.h3("PSI \"Kritik\" gösteriyor ama model iyi çalışıyor")
    pdf.body(
        "PSI dağılım kaymasını ölçer, model performansını değil. Dağılım değişmiş ancak model hâlâ doğru "
        "tahmin yapıyor olabilir. Yine de nedeni araştırın — gelecekte performans düşüşüne yol açabilir."
    )

    pdf.h3("Yardım & Referans")
    pdf.body(
        "Uygulamanın sağ üst köşesindeki \"?\" butonuna tıklayarak kapsamlı yardım ekranına erişebilirsiniz. "
        "Bu ekranda metrik tanımları, eşik değerleri ve iş akışı rehberi yer alır."
    )

    # ── Kaydet ──
    pdf.output(str(OUTPUT))
    return OUTPUT


if __name__ == "__main__":
    print("PDF oluşturuluyor...")
    path = build()
    size_kb = path.stat().st_size / 1024
    print(f"Başarılı: {path}")
    print(f"Boyut: {size_kb:.0f} KB")
    print(f"Sayfa sayısı: Toplam ~25-30 sayfa")
