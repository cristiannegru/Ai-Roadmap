# 02 — Housing Price Regression (Beginner)

Ridge regression on `data/samples/housing_sample.csv` via the shared
`make_regression_pipeline` (median-impute + scale + one-hot).

```bash
cd projects/02_housing_regression
pip install -r requirements.txt
python main.py
python main.py --predict --area 1500 --bedrooms 3 --bathrooms 2 --age 10 --location suburb
```

## Go production

Swap the CSV for a real listings export — same pipeline. Add `location`
target-encoding and `area_sqft` log-transform when R² plateaus.
