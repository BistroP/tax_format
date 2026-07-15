"""Step 5: render a STATIC heatmap of the format tax from results.json — a
self-contained HTML file (no server, no JS libs). This alone is a complete
project: open heatmap.html in any browser.

Run:  python -m eval.make_heatmap   ->   heatmap.html
"""

from __future__ import annotations

import json

from . import config

OUT = config.ROOT / "heatmap.html"


def _tax_color(tax: float) -> str:
    """Green (no tax) -> red (heavy tax). Clamp at a 40-point drop."""
    ratio = max(0.0, min(tax, 0.40)) / 0.40
    hue = 120 * (1 - ratio)  # 120=green, 0=red
    return f"hsl({hue:.0f} 65% 42%)"


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def build_html(results: dict) -> str:
    meta = results["meta"]
    conds = meta["conditions"]
    non_prose = [c for c in conds if c != "prose"]

    rows = []
    for m in meta["models"]:
        mk = m["key"]
        cells = [f'<td class="model">{m["label"]}<br><span class="mid">{m["model"]}</span></td>']
        prose = results["metrics"][mk]["prose"]["correct_rate"]
        cells.append(f'<td class="cell" style="background:hsl(150 30% 30%)">'
                     f'<div class="big">{_pct(prose)}</div>'
                     f'<div class="lab">prose baseline</div></td>')
        for c in non_prose:
            met = results["metrics"][mk][c]
            tax = results["tax"][mk][c]
            extra = ""
            if met["unparseable_rate"]:
                extra += f'<div class="lab">unparseable {_pct(met["unparseable_rate"])}</div>'
            if met.get("pure_json_compliance") is not None:
                extra += f'<div class="lab">pure-JSON {_pct(met["pure_json_compliance"])}</div>'
            if met.get("lexical_violation_rate") is not None:
                extra += f'<div class="lab">banned-word used {_pct(met["lexical_violation_rate"])}</div>'
            cells.append(
                f'<td class="cell" style="background:{_tax_color(tax)}">'
                f'<div class="big">{_pct(met["correct_rate"])}</div>'
                f'<div class="tax">tax &minus;{100 * tax:.0f} pts</div>{extra}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")

    head = ('<th></th><th>prose</th>'
            + "".join(f"<th>{c}</th>" for c in non_prose))
    note = meta.get("note", "")
    gen = meta.get("generated_at", "")
    n = meta.get("n_items", "?")

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Format-Tax Explorer — static heatmap</title>
<style>
  body {{ font: 15px/1.5 -apple-system, system-ui, sans-serif; margin: 2rem auto;
         max-width: 900px; color: #1a1a1a; padding: 0 1rem; }}
  h1 {{ font-size: 1.4rem; margin-bottom: .2rem; }}
  .sub {{ color: #666; margin-bottom: 1.2rem; }}
  table {{ border-collapse: separate; border-spacing: 6px; width: 100%; }}
  th {{ text-align: center; font-size: .85rem; text-transform: uppercase;
        letter-spacing: .04em; color: #555; }}
  td.model {{ text-align: right; font-weight: 600; width: 200px; }}
  td.model .mid {{ font-weight: 400; color: #999; font-size: .8rem; }}
  td.cell {{ color: #fff; text-align: center; border-radius: 8px; padding: 10px 6px;
             min-width: 120px; }}
  .big {{ font-size: 1.5rem; font-weight: 700; }}
  .tax {{ font-size: .9rem; opacity: .95; }}
  .lab {{ font-size: .72rem; opacity: .9; margin-top: 2px; }}
  .foot {{ color: #666; font-size: .85rem; margin-top: 1.5rem;
           border-top: 1px solid #eee; padding-top: 1rem; }}
</style></head><body>
<h1>The format tax, on today's models</h1>
<div class="sub">Cell = accuracy under that condition. Color = the tax
(green: none · red: heavy). {n} piloted items · generated {gen}</div>
<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>
<p class="foot"><b>How to read it:</b> each row is a model; the prose column is the
control (the model gets these right unconstrained — that's the pilot gate). Every
other column is the <i>identical</i> question under a format constraint; the drop
is the tax. <b>unparseable</b> is reported separately from wrong — a high
unparseable rate is itself a finding, not a scoring failure.<br><br>{note}</p>
</body></html>"""


def main():
    if not config.RESULTS_PATH.exists():
        raise SystemExit("results.json not found — run `python -m eval.run_eval` first.")
    results = json.loads(config.RESULTS_PATH.read_text())
    OUT.write_text(build_html(results))
    print(f"Wrote {OUT.relative_to(config.ROOT)} — open it in a browser.")


if __name__ == "__main__":
    main()
