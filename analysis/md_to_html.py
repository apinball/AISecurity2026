"""
Render WANET_REPORT.md to a self-contained HTML for PDF printing.
Korean fonts via system 'Malgun Gothic'.

Usage:
    python analysis/md_to_html.py WANET_REPORT.md WANET_REPORT.html
"""
import argparse
from pathlib import Path
import markdown


CSS = """
<style>
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
body {
    font-family: "Malgun Gothic", "맑은 고딕", "Apple SD Gothic Neo", "Noto Sans CJK KR", sans-serif;
    font-size: 10.5pt;
    line-height: 1.55;
    color: #1f2937;
    max-width: 100%;
    margin: 0;
    padding: 0;
}
h1 { font-size: 22pt; border-bottom: 2px solid #111827; padding-bottom: 6px; margin-top: 0.6em; }
h2 { font-size: 16pt; border-bottom: 1px solid #d1d5db; padding-bottom: 4px; margin-top: 1.4em; }
h3 { font-size: 13pt; margin-top: 1.2em; color: #111827; }
h4 { font-size: 11pt; margin-top: 1em; color: #374151; }
p { margin: 0.5em 0; }
img { max-width: 100%; height: auto; display: block; margin: 0.6em auto; border: 1px solid #e5e7eb; border-radius: 4px; }
code {
    font-family: "Consolas", "Cascadia Code", "Monaco", monospace;
    background: #f3f4f6; padding: 1px 5px; border-radius: 3px; font-size: 9.5pt;
}
pre {
    background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 5px;
    padding: 8px 12px; overflow-x: auto; font-size: 9pt; line-height: 1.45;
    page-break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: 9pt; }
table {
    border-collapse: collapse; margin: 0.6em 0; font-size: 9.5pt;
    width: auto; max-width: 100%;
}
th, td { border: 1px solid #d1d5db; padding: 4px 8px; text-align: left; }
th { background: #f3f4f6; font-weight: 600; }
tr:nth-child(even) td { background: #fafafa; }
blockquote { border-left: 3px solid #9ca3af; padding-left: 12px; color: #4b5563; margin: 0.5em 0; }
hr { border: none; border-top: 1px solid #d1d5db; margin: 1.4em 0; }
ul, ol { margin: 0.4em 0; padding-left: 1.6em; }
li { margin: 0.18em 0; }
a { color: #2563eb; text-decoration: none; }
a:hover { text-decoration: underline; }
h1, h2, h3, h4 { page-break-after: avoid; }
img, table, pre { page-break-inside: avoid; }
</style>
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("md_path")
    p.add_argument("html_path")
    args = p.parse_args()

    md_text = Path(args.md_path).read_text(encoding="utf-8")
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists"])
    body = md.convert(md_text)

    html = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>WaNet Report</title>
{CSS}
</head>
<body>
{body}
</body>
</html>
"""
    Path(args.html_path).write_text(html, encoding="utf-8")
    print(f"saved: {args.html_path}")


if __name__ == "__main__":
    main()
