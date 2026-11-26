#!/usr/bin/env python3
"""Render the README's table data using LaTeX booktabs and the project palette.

Requires pdflatex and pdftocairo (Poppler). Run from any directory:
    python3 docs/render_tables.py
No model calls or Python dependencies are required.
"""

import html
import json
from pathlib import Path
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "tables"


def tex(value):
    bold = "**" in value or "<strong>" in value
    underline = "<u>" in value
    italic = "<em>" in value
    code = "`" in value
    value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    value = html.unescape(re.sub(r"<[^>]+>", "", value))
    value = value.replace("**", "").replace("`", "")
    chars = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#",
             "$": r"\$", "{": r"\{", "}": r"\}",
             "↑": r"$\uparrow$", "↓": r"$\downarrow$",
             "–": "--", "—": "---"}
    value = "".join(chars.get(c, c) for c in value)
    if code:
        value = r"\texttt{" + value + "}"
    if italic:
        value = r"\textit{" + value + "}"
    if underline:
        value = r"\underline{" + value + "}"
    if bold:
        value = r"\textbf{" + value + "}"
    return value


def source(table):
    name = table["id"]
    count = len(table["headers"])
    width = 250 if name == "tasks" else 210
    available = width * 72.27 / 25.4 - 10 * (count - 1)
    numeric = name in {"main-results", "blocker", "implementation", "dataset"}
    columns = []
    for i, fraction in enumerate(table["widths"]):
        align = "raggedleft" if numeric and i > (1 if name == "dataset" else 0) else "raggedright"
        columns.append(">{" + "\\" + align + r"\arraybackslash}p{" + f"{available * fraction:.2f}pt" + "}")
    lines = [r"\arrayrulecolor{Brick}\toprule[1.2pt]"]
    if name == "main-results":
        lines.extend([
            r"\rowcolor{Cream} \textbf{\color{Brick}Model} & \multicolumn{2}{c}{\textbf{\color{Brick}Task 1: Macro-F1 $\uparrow$}} & \multicolumn{2}{c}{\textbf{\color{Brick}Task 2: Macro DRR $\uparrow$}} & \multicolumn{2}{c}{\textbf{\color{Brick}Task 3: Macro-CAS $\uparrow$}} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
            r"\rowcolor{Cream} & Real & Synth. & Real & Synth. & Real & Synth. \\",
        ])
    else:
        lines.append(r"\rowcolor{Cream} " + " & ".join(r"\textbf{\color{Brick}" + tex(h) + "}" for h in table["headers"]) + r" \\")
    lines.append(r"\arrayrulecolor{Rule}\midrule[0.65pt]")
    for i, row in enumerate(table["rows"]):
        if len(row) == 1:
            if i:
                lines.append(r"\addlinespace[5pt]")
            lines.append(r"\rowcolor{Cream} \multicolumn{" + str(count) + r"}{l}{\color{Brick}" + tex(row[0]) + r"} \\")
            continue
        cells = [tex(c) for c in row]
        if name == "taxonomy":
            cells[1] = cells[1].replace("; ", r"\newline ")
        if name == "main-results" and "GPT-5.6-Sol" in row[0]:
            lines.append(r"\rowcolor{Sage}")
        elif name == "dataset" and "**" in row[0]:
            lines.append(r"\rowcolor{Sage}")
        elif name in {"blocker", "implementation"} and i == 0:
            lines.append(r"\rowcolor{Sage}")
        lines.append(" & ".join(cells) + r" \\")
        if name in {"tasks", "taxonomy", "repository", "construction"} and i < len(table["rows"]) - 1:
            lines.append(r"\addlinespace[6pt]")
    lines.append(r"\arrayrulecolor{Brick}\bottomrule[1.2pt]")
    preamble = r"""\documentclass{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern,booktabs,array}
\usepackage[table]{xcolor}
\definecolor{Brick}{HTML}{AB240E}
\definecolor{Ink}{HTML}{382F2A}
\definecolor{Cream}{HTML}{FFF1DF}
\definecolor{Sage}{HTML}{EEF2DE}
\definecolor{Rule}{HTML}{B8AC94}
\pagestyle{empty}
\begin{document}
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.45}
\setbox0=\hbox{\begin{minipage}{WIDTHmm}
\fontsize{11.5}{14.5}\selectfont\color{Ink}
\begin{tabular}{@{}COLUMNS@{}}
""".replace("WIDTH", str(width)).replace("COLUMNS", "".join(columns))
    ending = r"""
\end{tabular}
\end{minipage}}
\pdfpagewidth=\dimexpr\wd0+20pt\relax
\pdfpageheight=\dimexpr\ht0+\dp0+20pt\relax
\hoffset=-1in
\voffset=-1in
\shipout\hbox{\kern10pt\vbox{\kern10pt\box0\kern10pt}\kern10pt}
\end{document}
"""
    return preamble + "\n".join(lines) + ending


def main():
    tables = json.loads((OUT / "tables.json").read_text())
    with tempfile.TemporaryDirectory(prefix="ideaambig-tables-") as temporary:
        build = Path(temporary)
        for table in tables:
            name = table["id"]
            latex = source(table)
            (OUT / f"{name}.tex").write_text(latex)
            (build / f"{name}.tex").write_text(latex)
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"],
                cwd=build, capture_output=True, text=True,
            )
            if result.returncode:
                raise RuntimeError(result.stdout[-4000:])
            log = (build / f"{name}.log").read_text()
            if "Overfull" in log:
                raise RuntimeError(f"Table {name} overflows: inspect its column widths.")
            subprocess.run(["pdftocairo", "-svg", str(build / f"{name}.pdf"), str(OUT / f"{name}.svg")], check=True)
            svg_path = OUT / f"{name}.svg"
            svg_path.write_text(re.sub(
                r"(<svg\b[^>]*>)", r'\1\n<rect width="100%" height="100%" fill="#ffffff"/>',
                svg_path.read_text(), count=1,
            ))
            print(f"Rendered {name}")


if __name__ == "__main__":
    main()
