# Data samples

Small (<100KB), license-clean **synthetic** CSVs so every Phase 2/3 script
runs offline without downloads. Generated with `random.seed(42)` —
deterministic, no real personal data.

- `titanic_sample.csv` — 61 rows (60 + 1 deliberate duplicate), 8 cols.
  Missing: `age` ×6, `embarked` ×3. Target: `survived`.
  Signal: females + 1st/2nd class survive more often (learnable).
- `housing_sample.csv` — 81 rows (80 + 1 deliberate duplicate), 6 cols.
  Missing: `bathrooms` ×5, `age_years` ×7. Target: `price`.

Rule: never commit raw/scraped datasets. Only curated samples.
See `.gitignore` — `*.csv` is ignored except when force-added from this folder:

```bash
git add -f data/samples/<file>.csv
```
