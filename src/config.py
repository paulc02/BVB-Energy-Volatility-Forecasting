"""Configurație centrală: căi, tickere, modele selectate și denumiri pentru grafice."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIG_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"

MASTER_DATASET = DATA_PROCESSED / "master_dataset.csv"
GARCHX_OPTIMAL_MODELS = TABLE_DIR / "garchx_optimal_models.csv"

for _d in (DATA_RAW, DATA_PROCESSED, FIG_DIR, TABLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Surse de date -----------------------------------------------------------
START_DATE = "2000-01-01"
STOCK_TICKERS = {"SNP": "SNP.RO", "H2O": "H2O.RO", "SNN": "SNN.RO"}  # Yahoo Finance
MACRO_TICKERS = {"Brent": "BZ=F", "TTF": "TTF=F", "EURRON": "EURRON=X", "USDRON": "USDRON=X"}
# BET-NG (Investing.com) și ROBOR 3M (BNR) se descarcă manual -> vezi README.
BETNG_FILE = "BETNG.csv"
ROBOR_FILE = "ROBOR.csv"

# --- Seriile modelate --------------------------------------------------------
ASSETS = ["Rand_SNP", "Rand_SNN", "Rand_H2O", "Rand_BET-NG"]
TRAIN_FRACTION = 0.80

# Ordinele ARMA(p, d, q) folosite în etapele GARCH / GARCH-X / prognoză
ARMA_ORDERS = {
    "Rand_SNP": (2, 0, 1),
    "Rand_SNN": (2, 0, 0),
    "Rand_H2O": (0, 0, 0),
    "Rand_BET-NG": (2, 0, 0),
}

# Modelul de volatilitate selectat (în etapa GARCH) pentru fiecare activ
BASE_VOLATILITY_MODEL = {
    "Rand_SNP": "APARCH-X-t",
    "Rand_SNN": "GARCH-X-t",
    "Rand_H2O": "GARCH-X-normal",
    "Rand_BET-NG": "APARCH-X-t",
}

# Variabile exogene candidate (nume afișat -> coloană în master_dataset)
EXOG_CANDIDATES = {
    "Brent": "Rand_Brent",
    "TTF": "Rand_TTF",
    "EUR/RON": "Rand_EUR",
    "USD/RON": "Rand_USD",
    "ROBOR": "Diff_ROBOR",
}

# --- Denumiri pentru grafice și tabele ---------------------------------------
ASSET_NAMES = {
    "Rand_SNP": "OMV Petrom",
    "Rand_SNN": "Nuclearelectrica",
    "Rand_H2O": "Hidroelectrica",
    "Rand_BET-NG": "BET-NG",
}

SERIES_TITLES = {
    "Pret_BET-NG": "Indicele sectorial BET-NG",
    "Pret_SNP": "Prețul acțiunilor OMV Petrom",
    "Pret_SNN": "Prețul acțiunilor Nuclearelectrica",
    "Pret_H2O": "Prețul acțiunilor Hidroelectrica",
    "Pret_Brent": "Prețul petrolului Brent",
    "Pret_TTF": "Prețul gazelor naturale TTF",
    "Pret_EUR": "Cursul valutar EUR/RON",
    "Pret_USD": "Cursul valutar USD/RON",
    "Pret_ROBOR": "Indicele ROBOR la 3 luni",
    "Rand_BET-NG": "Randamentul zilnic BET-NG",
    "Rand_SNP": "Randamentul zilnic OMV Petrom",
    "Rand_SNN": "Randamentul zilnic Nuclearelectrica",
    "Rand_H2O": "Randamentul zilnic Hidroelectrica",
    "Rand_Brent": "Randamentul zilnic petrol Brent",
    "Rand_TTF": "Randamentul zilnic gaze TTF",
    "Rand_EUR": "Randamentul zilnic EUR/RON",
    "Rand_USD": "Randamentul zilnic USD/RON",
    "Diff_ROBOR": "Variația zilnică ROBOR (p.p.)",
}

CORRELATION_LABELS = {
    "Rand_BET-NG": "BET-NG", "Rand_SNP": "OMV Petrom", "Rand_SNN": "Nuclearelectrica",
    "Rand_H2O": "Hidroelectrica", "Rand_Brent": "Petrol Brent", "Rand_TTF": "Gaze TTF",
    "Rand_EUR": "EUR/RON", "Rand_USD": "USD/RON", "Diff_ROBOR": "ROBOR",
}
