"""
Génère RAPPORT_PROJET.pdf depuis RAPPORT_PROJET.md
Pipeline : Markdown → HTML stylé (images en base64) → PDF via Chrome/Edge headless
"""
import subprocess, sys, os, base64, re, pathlib

# ── 1. Vérifier les dépendances ───────────────────────────────────────────────
try:
    import markdown
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "markdown"])
    import markdown

# ── 2. Lire le markdown ───────────────────────────────────────────────────────
BASE = pathlib.Path(__file__).parent
md_path = BASE / "RAPPORT_PROJET.md"
text = md_path.read_text(encoding="utf-8")

# ── 3. Convertir les chemins d'images en base64 inline ───────────────────────
def img_to_b64(match):
    alt   = match.group(1)
    path  = match.group(2)
    full  = BASE / path
    if full.exists():
        data = base64.b64encode(full.read_bytes()).decode()
        ext  = full.suffix.lstrip(".").lower()
        mime = "image/png" if ext == "png" else f"image/{ext}"
        return f'![{alt}](data:{mime};base64,{data})'
    return match.group(0)

text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', img_to_b64, text)

# ── 4. Markdown → HTML ────────────────────────────────────────────────────────
md = markdown.Markdown(
    extensions=["tables", "fenced_code", "toc", "nl2br"],
    extension_configs={"toc": {"title": "Table des matières"}}
)
body_html = md.convert(text)

# ── 5. Injecter dans un template HTML complet avec CSS ───────────────────────
html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Rapport — Maintenance Prédictive Industrielle</title>
<style>
  @page {{
    size: A4;
    margin: 2.2cm 2cm 2.2cm 2cm;
    @bottom-right {{ content: counter(page); font-size: 10pt; color: #888; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', Calibri, Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.65;
    color: #222;
    max-width: 100%;
  }}
  h1 {{
    font-size: 20pt;
    color: #1a3a5c;
    border-bottom: 3px solid #e8701a;
    padding-bottom: 8px;
    margin-top: 0;
    page-break-before: avoid;
  }}
  h2 {{
    font-size: 15pt;
    color: #1a3a5c;
    border-bottom: 2px solid #d0dce8;
    padding-bottom: 5px;
    margin-top: 28px;
    page-break-after: avoid;
  }}
  h3 {{
    font-size: 12.5pt;
    color: #2c6e9e;
    margin-top: 18px;
    page-break-after: avoid;
  }}
  h4 {{
    font-size: 11.5pt;
    color: #444;
    margin-top: 14px;
    page-break-after: avoid;
  }}
  p {{ margin: 0.5em 0 0.8em 0; }}
  img {{
    display: block;
    max-width: 100%;
    height: auto;
    margin: 16px auto;
    border-radius: 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    page-break-inside: avoid;
  }}
  /* Légende : le texte en italique juste après une image */
  img + em, p:has(img) + p > em {{
    display: block;
    text-align: center;
    font-size: 9.5pt;
    color: #555;
    margin-top: -10px;
    margin-bottom: 16px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 10pt;
    margin: 14px 0;
    page-break-inside: avoid;
  }}
  th {{
    background: #1a3a5c;
    color: white;
    padding: 7px 10px;
    text-align: left;
  }}
  td {{
    padding: 6px 10px;
    border-bottom: 1px solid #dde3ea;
    vertical-align: top;
  }}
  tr:nth-child(even) td {{ background: #f4f7fb; }}
  code {{
    background: #f0f3f7;
    padding: 2px 5px;
    border-radius: 3px;
    font-size: 9.5pt;
    font-family: Consolas, 'Courier New', monospace;
    color: #c0392b;
  }}
  pre {{
    background: #1e2a3a;
    color: #cdd9e5;
    padding: 14px 16px;
    border-radius: 6px;
    font-size: 9pt;
    line-height: 1.5;
    overflow-x: auto;
    page-break-inside: avoid;
    margin: 12px 0;
  }}
  pre code {{
    background: none;
    color: inherit;
    padding: 0;
    font-size: inherit;
  }}
  blockquote {{
    border-left: 4px solid #e8701a;
    margin: 12px 0;
    padding: 8px 16px;
    background: #fff8f0;
    border-radius: 0 4px 4px 0;
    font-style: italic;
    color: #555;
    page-break-inside: avoid;
  }}
  ul, ol {{ padding-left: 1.5em; margin: 6px 0; }}
  li {{ margin-bottom: 4px; }}
  hr {{ border: none; border-top: 1px solid #dde3ea; margin: 20px 0; }}
  a {{ color: #2c6e9e; text-decoration: none; }}
  .toc {{ background: #f4f7fb; border: 1px solid #d0dce8; border-radius: 6px;
         padding: 14px 20px; margin-bottom: 24px; }}
  .toc ul {{ list-style: none; padding-left: 0; }}
  .toc li {{ padding: 2px 0; }}
  .toc a {{ color: #1a3a5c; }}
  /* Page de séparation sur les grandes sections */
  h2.section-break {{ page-break-before: always; }}
</style>
</head>
<body>
{body_html}
</body>
</html>"""

# Sauvegarder le HTML intermédiaire
html_path = BASE / "RAPPORT_PROJET_temp.html"
html_path.write_text(html, encoding="utf-8")
print(f"HTML généré : {html_path}")

# ── 6. Convertir HTML → PDF via Chrome headless ──────────────────────────────
pdf_path = BASE / "RAPPORT_PROJET.pdf"

browsers = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

browser = next((b for b in browsers if os.path.exists(b)), None)
if not browser:
    print("Aucun navigateur Chromium trouvé. Le HTML est disponible pour conversion manuelle.")
    sys.exit(1)

print(f"Navigateur : {browser}")
print("Génération du PDF en cours...")

result = subprocess.run([
    browser,
    "--headless",
    "--disable-gpu",
    "--no-sandbox",
    "--disable-web-security",
    f"--print-to-pdf={pdf_path}",
    "--print-to-pdf-no-header",
    str(html_path)
], capture_output=True, text=True, timeout=60)

if pdf_path.exists():
    size_kb = pdf_path.stat().st_size // 1024
    print(f"PDF généré : {pdf_path}  ({size_kb} KB)")
    # Nettoyer le HTML temporaire
    html_path.unlink()
    print("HTML temporaire supprimé.")
else:
    print("Erreur lors de la génération du PDF.")
    print("STDERR:", result.stderr[:500])
