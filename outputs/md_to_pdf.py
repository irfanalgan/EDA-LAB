"""Convert eda_lab_v2_blueprint.md to PDF using markdown + xhtml2pdf."""

import pathlib
import markdown
from xhtml2pdf import pisa

BASE = pathlib.Path(r"C:\Users\pc\OneDrive\Desktop\Preprocessing")
MD_FILE  = BASE / "eda_lab_v2_blueprint.md"
PDF_FILE = BASE / "eda_lab_v2_blueprint.pdf"

md_text = MD_FILE.read_text(encoding="utf-8")

# Convert MD → HTML
html_body = markdown.markdown(
    md_text,
    extensions=["tables", "fenced_code", "toc", "nl2br"],
)

# Full HTML with CSS
html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  @page {{
    size: A4;
    margin: 1.5cm 1.8cm 1.5cm 1.8cm;
  }}
  body {{
    font-family: "DejaVu Sans", Arial, sans-serif;
    font-size: 9pt;
    line-height: 1.45;
    color: #1a1a1a;
  }}
  h1 {{
    font-size: 18pt;
    color: #1a3c5e;
    border-bottom: 2px solid #1a3c5e;
    padding-bottom: 4px;
    margin-top: 0;
  }}
  h2 {{
    font-size: 13pt;
    color: #1a3c5e;
    border-bottom: 1px solid #bcd0e8;
    padding-bottom: 2px;
    margin-top: 18pt;
  }}
  h3 {{
    font-size: 10.5pt;
    color: #2d5986;
    margin-top: 12pt;
  }}
  h4 {{
    font-size: 9.5pt;
    color: #3a6b9e;
    margin-top: 8pt;
  }}
  p {{
    margin: 4pt 0;
  }}
  code {{
    font-family: "Courier New", monospace;
    font-size: 7.5pt;
    background: #f4f7fb;
    padding: 1px 3px;
    border-radius: 2px;
    color: #c7254e;
  }}
  pre {{
    font-family: "Courier New", monospace;
    font-size: 7pt;
    background: #f4f7fb;
    border: 1px solid #d0dce8;
    border-left: 3px solid #1a3c5e;
    padding: 6pt 8pt;
    margin: 6pt 0;
    white-space: pre-wrap;
    word-break: break-word;
  }}
  pre code {{
    background: none;
    padding: 0;
    color: #1a1a1a;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 6pt 0;
    font-size: 8pt;
  }}
  th {{
    background: #1a3c5e;
    color: #ffffff;
    padding: 4pt 6pt;
    text-align: left;
  }}
  td {{
    padding: 3pt 6pt;
    border: 1px solid #d0dce8;
  }}
  tr:nth-child(even) td {{
    background: #f0f5fb;
  }}
  ul, ol {{
    margin: 4pt 0 4pt 16pt;
    padding: 0;
  }}
  li {{
    margin: 2pt 0;
  }}
  blockquote {{
    margin: 6pt 0 6pt 12pt;
    padding: 4pt 8pt;
    border-left: 3px solid #4a90d9;
    background: #f0f5fb;
    color: #444;
  }}
  hr {{
    border: none;
    border-top: 1px solid #d0dce8;
    margin: 10pt 0;
  }}
  a {{
    color: #1a3c5e;
  }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

print(f"Converting: {MD_FILE}")
print(f"Output:     {PDF_FILE}")

with open(PDF_FILE, "wb") as pdf_out:
    result = pisa.CreatePDF(html.encode("utf-8"), dest=pdf_out, encoding="utf-8")

if result.err:
    print(f"ERROR: {result.err}")
else:
    size_kb = PDF_FILE.stat().st_size // 1024
    print(f"SUCCESS — {size_kb} KB → {PDF_FILE}")
