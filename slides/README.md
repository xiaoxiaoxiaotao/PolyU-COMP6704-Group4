# Slides

`6704-Group4.pptx` is the single deck of record for the group presentation: 37 pages,
one divider per part plus four parts.

| Pages | Part |
|---|---|
| 1–7 | Part I — problem, cost model, primal and dual |
| 8–20 | Part II — SMO |
| 21–29 | Part III — ADMM and PA-ADMM |
| 30–37 | Part IV — evaluation, experiments and results (footers `IV · 01` … `IV · 08`) |

No per-part or generated decks are tracked any more; edit this file directly. The
earlier `COMP6704_Group4_PART_IV.pptx` and its builder are in git history if they are
ever needed again.

## Corrections applied to Part IV on 2026-10-31

Checked against the archived runs in `report/data/`:

- Scale ladder redrawn. Every full-data run completes inside the 90-minute budget —
  Pegasos at a median of 726 s and SMO at 1,373 s — so no bar is marked "not run" any
  more. A red cross now marks the bars whose method never reached the 0.1% target.
- `PR-AUC` relabelled `AP`, matching the report, which defines it as
  `average_precision_score` rather than a trapezoidal PR area.
- The grid note now separates the two ideas the deck had merged: reaching the shared
  0.1% target (all 15 ADMM-family runs) versus satisfying a solver's own certificate
  (12 for PA-ADMM, 7 for ADMM).
- The quality footnote scopes its 0.004 gap to the three solvers that reach the target;
  including Pegasos the gap is 0.0105.
- Slide `IV · 08` quotes the actual paired Wilcoxon values, p = 1.00 at 10:1 and
  p = 0.44 at 20:1, instead of "p >= 0.5".
- Tables on `IV · 06` and `IV · 07` lost the explicit black cell borders that the paste
  had baked in, so they now match the rest of the deck.

## Combined submission PDF

Not built yet. The assignment asks for one PDF named `Group4.pdf` — report first, then
slides, under 40 MB — and it should only be assembled once both the report and the deck
have been reviewed by the group. To build it afterwards:

```bash
cd report && latexmk -pdf Group4_Report.tex          # 14 pages
# then export this deck to PDF from PowerPoint, and concatenate:
gs -dBATCH -dNOPAUSE -sDEVICE=pdfwrite \
   -sOutputFile=Group4.pdf report/Group4_Report.pdf <slides>.pdf
```
