# Member 4 slides (PART IV)

`COMP6704_Group4_PART_IV.pptx` is generated, never hand-edited. Rebuild it with:

```bash
.venv/bin/python slides/build_part4_deck.py \
    --template /path/to/6704-Group4.pptx \
    --out slides/COMP6704_Group4_PART_IV.pptx \
    --check
```

`--check` audits every PART IV shape: nothing may leave the 13.33 × 7.5 in canvas
and no two content boxes (text, picture, table) may overlap. It exits non-zero
when a slide needs attention, and the builder separately warns when a text box is
tall enough to run into the takeaway band.

The builder reads `results/raw/<experiment>/**/seed-*.json` and
`results/tables/*.csv`, writes the deck figures to `results/figures/deck/`, and
appends PART IV to the group template using the same master layouts, palette
(`A02337` / `182B3A` / `526574`) and footer conventions as PART III. Every number
on the slides is computed from the run JSONs, so a re-run after new experiments
cannot leave a stale figure or claim behind. Methods that have not run yet are
listed as pending instead of being filled in.

## Run of show (target 8 minutes)

| # | Slide | Seconds |
|---|---|---|
| 1 | Divider: Evaluation, Experiments and Results | 15 |
| 2 | The data behind every number | 55 |
| 3 | How the comparison is made fair | 50 |
| 4 | Primary metric: time to a shared tolerance | 55 |
| 5 | Correctness first: solvers agree on one QP | 45 |
| 6 | The experiment matrix we ran | 45 |
| 7 | Main results: speed and certified convergence | 55 |
| 8 | Full data at a glance | 40 |
| 9 | Performance under the deployment cost model | 50 |
| 10 | What we can and cannot claim | 55 |
| | **Total (slides 1–10)** | **≈ 8:20** |
| 11 | Appendix: fairness controls in detail | Q&A only |
| 12 | Appendix: reproducing these numbers | Q&A only |

Slides 11–12 are backup for the 10-minute Q&A and are not part of the 8 minutes.

Speaker notes with the narration for each slide live in the deck itself
(View → Notes). Keep the ordering: data → protocol → metrics → correctness →
main results → performance → claims.

## Storyline in one line per slide

1. Parts I–III gave the objective and four solvers; Part IV measures them.
2. 283,726 transactions after deduplication, 473 frauds (0.167%), 29 features.
3. Same rows, same fixed C, same 90-minute budget, validation-only thresholds.
4. Ranking metric: seconds to reach within 0.1% of the group-best objective.
5. Correctness gate: independent solvers agree to ~1e-4 on one QP.
6. Scale ladder 7k → 35k → 198k rows, ratios 10/20/50, up to 5 seeds.
7. Speed is similar; certified convergence is where the methods differ.
8. One table the audience can photograph.
9. Detection quality is driven by the data and the cost ratio, not the solver.
10. Claim the protocol and the reliability finding; no speed winner yet.
