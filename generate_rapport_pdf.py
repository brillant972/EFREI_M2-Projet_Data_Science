"""
Génère RAPPORT_PROJET.pdf et PRESENTATION.pdf
  - Rapport  : Markdown → HTML stylé (images base64) → PDF via Chrome headless
  - Présentation : PRESENTATION.md → PDF via Marp CLI (outil dédié aux slides)
Usage : python generate_rapport_pdf.py
"""
import subprocess, sys, os, base64, re, pathlib, shutil

BASE = pathlib.Path(__file__).parent

# ── Navigateur Chromium (Chrome ou Edge) ─────────────────────────────────────
BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]
BROWSER = next((b for b in BROWSERS if os.path.exists(b)), None)


# ─────────────────────────────────────────────────────────────────────────────
# 1. RAPPORT PDF
# ─────────────────────────────────────────────────────────────────────────────

def build_rapport_pdf():
    try:
        import markdown
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "markdown", "-q"])
        import markdown

    md_path  = BASE / "RAPPORT_PROJET.md"
    pdf_path = BASE / "RAPPORT_PROJET.pdf"
    html_tmp = BASE / "_rapport_tmp.html"

    text = md_path.read_text(encoding="utf-8")

    # Convertir les images locales en base64
    def img_to_b64(m):
        alt, path = m.group(1), m.group(2)
        full = BASE / path
        if full.exists():
            data = base64.b64encode(full.read_bytes()).decode()
            mime = "image/png" if full.suffix.lower() == ".png" else f"image/{full.suffix.lstrip('.')}"
            return f'![{alt}](data:{mime};base64,{data})'
        return m.group(0)

    text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', img_to_b64, text)

    md_engine = markdown.Markdown(
        extensions=["tables", "fenced_code", "toc", "nl2br"],
        extension_configs={"toc": {"title": "Table des matières"}}
    )
    body = md_engine.convert(text)

    css = """
    @page { size: A4; margin: 2.2cm 2cm 2.2cm 2cm; }
    * { box-sizing: border-box; }
    body {
      font-family: 'Segoe UI', Calibri, Arial, sans-serif;
      font-size: 11pt; line-height: 1.65; color: #222; max-width: 100%;
    }
    h1 {
      font-size: 20pt; color: #1a3a5c;
      border-bottom: 3px solid #e8701a; padding-bottom: 8px;
      margin-top: 0; page-break-before: avoid;
    }
    h2 {
      font-size: 15pt; color: #1a3a5c;
      border-bottom: 2px solid #d0dce8; padding-bottom: 5px;
      margin-top: 28px; page-break-after: avoid;
    }
    h3 { font-size: 12.5pt; color: #2c6e9e; margin-top: 18px; page-break-after: avoid; }
    p { margin: 0.5em 0 0.8em 0; }
    img {
      display: block; max-width: 100%; max-height: 480px; height: auto;
      margin: 16px auto; border-radius: 4px;
      box-shadow: 0 2px 8px rgba(0,0,0,.12); page-break-inside: avoid;
      page-break-before: avoid;
    }
    table {
      width: 100%; border-collapse: collapse;
      font-size: 10pt; margin: 14px 0; page-break-inside: avoid;
    }
    th { background: #1a3a5c; color: white; padding: 7px 10px; text-align: left; }
    td { padding: 6px 10px; border-bottom: 1px solid #dde3ea; vertical-align: top; }
    tr:nth-child(even) td { background: #f4f7fb; }
    code {
      background: #f0f3f7; padding: 2px 5px; border-radius: 3px;
      font-size: 9.5pt; font-family: Consolas,'Courier New',monospace; color: #c0392b;
    }
    pre {
      background: #1e2a3a; color: #cdd9e5; padding: 14px 16px;
      border-radius: 6px; font-size: 9pt; line-height: 1.5;
      page-break-inside: avoid; margin: 12px 0;
    }
    pre code { background: none; color: inherit; padding: 0; font-size: inherit; }
    blockquote {
      border-left: 4px solid #e8701a; margin: 12px 0; padding: 8px 16px;
      background: #fff8f0; border-radius: 0 4px 4px 0; color: #555;
      page-break-inside: avoid;
    }
    ul, ol { padding-left: 1.5em; margin: 6px 0; }
    li { margin-bottom: 4px; }
    hr { border: none; border-top: 1px solid #dde3ea; margin: 20px 0; }
    """

    full_html = f"""<!DOCTYPE html>
<html lang="fr"><head>
<meta charset="UTF-8">
<title>Rapport — Maintenance Prédictive Industrielle</title>
<style>{css}</style>
</head><body>{body}</body></html>"""

    html_tmp.write_text(full_html, encoding="utf-8")

    if not BROWSER:
        print("[RAPPORT] Navigateur introuvable. HTML disponible :", html_tmp)
        return

    subprocess.run([
        BROWSER, "--headless", "--disable-gpu", "--no-sandbox",
        "--disable-web-security",
        f"--print-to-pdf={pdf_path}",
        "--print-to-pdf-no-header",
        "--no-pdf-header-footer",
        str(html_tmp)
    ], capture_output=True, timeout=90)

    html_tmp.unlink(missing_ok=True)

    if pdf_path.exists():
        print(f"[RAPPORT]  {pdf_path.name}  ({pdf_path.stat().st_size // 1024} KB)")
    else:
        print("[RAPPORT] Erreur lors de la génération.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. PRESENTATION PDF
# ─────────────────────────────────────────────────────────────────────────────

def build_presentation_pdf():
    md_path  = BASE / "PRESENTATION.md"
    pdf_path = BASE / "PRESENTATION.pdf"

    result = subprocess.run(
        f'npx @marp-team/marp-cli "{md_path}" --pdf --allow-local-files -o "{pdf_path}"',
        capture_output=True, text=True, timeout=120, shell=True
    )

    if pdf_path.exists():
        print(f"[PRESENTATION] {pdf_path.name}  ({pdf_path.stat().st_size // 1024} KB)")
    else:
        print("[PRESENTATION] Erreur lors de la génération.")
        print(result.stderr[:400])


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Génération des PDFs...")
    build_rapport_pdf()
    build_presentation_pdf()
    print("Terminé.")
