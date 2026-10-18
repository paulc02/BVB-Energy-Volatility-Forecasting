# Volatility Modelling & Forecasting of Romanian Energy Stocks (ARMA · GARCH · GARCH-X)

Bachelor's thesis project (*"Analysis and forecasting of stock prices on the stock exchange"*).
It models the daily returns and volatility of three energy companies listed on the Bucharest Stock Exchange (BVB) and of the BET-NG sector index, tests whether **macro drivers** (oil, gas, FX, interest rates) improve the volatility model, and evaluates **out-of-sample forecasts**.

| Asset | Ticker | Source |
|---|---|---|
| OMV Petrom | SNP | Yahoo Finance |
| Nuclearelectrica | SNN | Yahoo Finance |
| Hidroelectrica | H2O | Yahoo Finance |
| BET-NG energy & utilities index | BET-NG | Investing.com (manual export) |

Exogenous variables: Brent crude (USD/bbl), TTF natural gas (EUR/MWh), EUR/RON, USD/RON (Yahoo Finance) and 3-month ROBOR (National Bank of Romania, manual export). Sample: January 2012 – May 2026, daily data.

## What the project does

1. **Data pipeline** – downloads and cleans prices from three sources, aligns them on BET-NG trading days, builds log-returns (and first differences for ROBOR).
2. **Stationarity** – ADF, Phillips–Perron and KPSS tests, ACF/PACF, correlation matrix.
3. **Mean model** – ARMA(p, q) with p, q ≤ 2 selected by AIC among models with significant coefficients; Ljung–Box and ARCH-LM tests on the residuals check for leftover autocorrelation and ARCH effects.
4. **Volatility model** – GARCH, EGARCH, GJR-GARCH and APARCH, each with Normal and Student-t innovations, selected by AIC/BIC; diagnostics on standardized residuals.
5. **GARCH-X / APARCH-X** – custom maximum-likelihood implementation (`src/garchx.py`) where two exogenous macro variables enter the variance equation as lagged, standardized squared shocks. All 10 pairs of exogenous variables are estimated for each asset (40 models) and the best one per asset is selected by AIC.
6. **Out-of-sample forecasting** (80% / 20% chronological split) – rolling one-step-ahead ARMA for returns, recursive GARCH-X / APARCH-X for volatility, price paths with 80 / 90 / 95% intervals, and error metrics.

## Selected models

| Asset | ARMA | Volatility model (no exogenous) | Best GARCH-X / APARCH-X (exogenous variables) |
|---|---|---|---|
| OMV Petrom | (2,1) | APARCH(1,1)-t | APARCH-X(1,1)-t — TTF + EUR/RON |
| Nuclearelectrica | (2,0) | GARCH(1,1)-t | GARCH-X(1,1)-t — Brent + TTF |
| Hidroelectrica | (0,0) | GARCH(1,1)-N | GARCH-X(1,1)-N — EUR/RON + ROBOR |
| BET-NG | (2,0) | APARCH(1,1)-t | APARCH-X(1,1)-t — Brent + TTF |

The selected GARCH-X models leave no significant autocorrelation or remaining ARCH effects in the standardized residuals (Ljung–Box and ARCH-LM, lags up to 20, 5% level).

## Out-of-sample results

Test window: last 20% of each sample (about 446 trading days for SNP, SNN and BET-NG; 145 for H2O, which was listed in 2023).

| Asset | Volatility RMSE (%) | Volatility MAE (%) | Return RMSE | Price MAPE |
|---|---|---|---|---|
| OMV Petrom | 0.99 | 0.87 | 0.0117 | 9.6% |
| Nuclearelectrica | 1.22 | 1.02 | 0.0153 | 25.4% |
| Hidroelectrica | 1.05 | 0.75 | 0.0143 | 7.5% |
| BET-NG | 0.66 | 0.55 | 0.0088 | 10.4% |

Volatility is compared with the absolute out-of-sample ARMA residual (a noisy proxy for realized volatility). Price forecasts are paths built by compounding the forecast returns from the last training price, i.e. a path up to ~446 trading days ahead, which is why price errors are much larger than one-step return errors.

<p align="center"><img src="docs/img/forecast_volatility.png" width="85%" alt="Forecast vs. realized volatility"></p>
<p align="center"><img src="docs/img/forecast_prices_intervals.png" width="70%" alt="Forecast price paths with confidence intervals"></p>
<p align="center"><img src="docs/img/garchx_conditional_volatility.png" width="85%" alt="Conditional volatility of the best GARCH-X models"></p>

## Repository structure

```
├── notebooks/
│   ├── 01_stock_data.ipynb               # download + clean BVB prices
│   ├── 02_macro_indicators.ipynb         # download + clean Brent, TTF, FX, ROBOR
│   ├── 03_dataset_and_stationarity.ipynb # master dataset, EDA, stationarity tests
│   ├── 04_arma_models.ipynb              # ARMA selection and residual diagnostics
│   ├── 05_garch_models.ipynb             # GARCH / EGARCH / GJR / APARCH comparison
│   ├── 06_garchx_models.ipynb            # GARCH-X / APARCH-X, all exogenous pairs
│   └── 07_forecasting.ipynb              # out-of-sample forecasts and metrics
├── src/
│   ├── config.py        # paths, tickers, selected model orders, display names
│   ├── data.py          # download, cleaning, master dataset
│   ├── garchx.py        # custom GARCH-X / APARCH-X estimation, residual tests
│   ├── forecasting.py   # rolling ARMA, recursive volatility, price paths, metrics
│   ├── stationarity.py  # ADF / PP / KPSS table
│   └── plots.py         # EDA panels and correlation heatmap
├── data/                # raw and processed data (not tracked in git)
├── results/             # tables (CSV) and figures generated by the notebooks
└── docs/img/            # figures used in this README
```

Dataset columns: `Pret_<X>` = price/level, `Rand_<X>` = daily log-return, `Diff_ROBOR` = daily change of ROBOR (percentage points).

## How to reproduce

```bash
git clone https://github.com/paulc02/BVB-Energy-Volatility-Forecasting && cd bvb-energy-volatility-forecasting
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

1. Download **BET-NG** history from Investing.com and save it as `data/raw/BETNG.csv` (date, price, open, high, low, volume).
2. Download **ROBOR 3M** from the BNR interactive database and save it as `data/raw/ROBOR.csv` (`;`-separated, 9 header lines).
3. Run the notebooks in order, `01` → `07`. Notebooks 01–02 download the Yahoo Finance data automatically. Notebook 07 re-estimates an ARMA model at every test step and takes a while.

Notebooks and plot labels are in Romanian.

## Limitations

- GARCH-X models are estimated with a custom optimizer that does not produce standard errors, so the significance of the exogenous coefficients is not tested; selection relies on AIC/BIC.
- Realized volatility is proxied by |ARMA residual|, which is noisy and, under normality, about 20% lower than the true standard deviation; the volatility errors above should be read as relative comparisons.
- Returns are close to unpredictable, so mean-forecast MAPE on returns is not informative (values near zero); RMSE/MAE are the meaningful measures.
- Price forecasts are long-horizon compounded paths, not one-step-ahead prices.
