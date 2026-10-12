"""Descărcare, curățare și asamblare a bazei de date (prețuri, randamente, indicatori macro)."""
import warnings

import numpy as np
import pandas as pd

from . import config

PRICE_COLUMNS = ["Pret", "Open", "High", "Low", "Volume"]


# --- Descărcare --------------------------------------------------------------
def download_yahoo(tickers: dict) -> None:
    """Descarcă serii zilnice de pe Yahoo Finance și le salvează în data/raw/<nume>.csv."""
    import yfinance as yf

    for name, ticker in tickers.items():
        df = yf.download(ticker, start=config.START_DATE, progress=False)
        df.to_csv(config.DATA_RAW / f"{name}.csv")


# --- Curățare ----------------------------------------------------------------
def _to_numeric(series: pd.Series) -> pd.Series:
    """Elimină separatori de mii/ghilimele și convertește la numeric (None -> NaN)."""
    if not pd.api.types.is_numeric_dtype(series):
        series = series.astype(str).str.replace(",", "", regex=False).str.replace('"', "", regex=False)
    return pd.to_numeric(series, errors="coerce")


def _parse_dates(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
    n_bad = int(df["Data"].isna().sum())
    if n_bad:
        warnings.warn(f"{source}: {n_bad} rânduri cu dată invalidă (metadate sau format neașteptat) au fost eliminate.")
    return df.dropna(subset=["Data"])


def clean_yahoo(path) -> pd.DataFrame:
    """Curăță un CSV descărcat cu yfinance (format cu antet pe 1 sau 3 rânduri).

    Rândurile de metadate (Ticker/Date) devin NaT la parsarea datelor și sunt eliminate.
    Returnează un DataFrame cu coloana 'Data' și coloanele numerice disponibile.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={"Close": "Pret", "Adj Close": "Pret", "Date": "Data", "Dată": "Data"})
    if "Data" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Data"})
    df = _parse_dates(df, str(path))
    cols = [c for c in ["Data"] + PRICE_COLUMNS if c in df.columns]
    df = df[cols]
    for c in cols[1:]:
        df[c] = _to_numeric(df[c])
    return df


# Antete uzuale în exporturile Investing.com (engleză / română)
_INVESTING_HEADERS = {
    "Date": "Data", "Data": "Data",
    "Price": "Pret", "Ultima": "Pret", "Close": "Pret", "Pret": "Pret",
    "Open": "Open", "Deschidere": "Open",
    "High": "High", "Max.": "High", "Max": "High",
    "Low": "Low", "Min.": "Low", "Min": "Low",
    "Vol.": "Volume", "Volume": "Volume",
}


def clean_investing(path) -> pd.DataFrame:
    """Curăță CSV-ul BET-NG exportat manual din Investing.com."""
    df = pd.read_csv(path, encoding="utf-8-sig", thousands=",")
    df = df.rename(columns={c: _INVESTING_HEADERS[c] for c in df.columns if c in _INVESTING_HEADERS})
    missing = {"Data", "Pret"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: lipsesc coloanele {missing}. Antete găsite: {list(df.columns)}")
    df = _parse_dates(df, str(path))
    for c in PRICE_COLUMNS:
        if c in df.columns:
            df[c] = _to_numeric(df[c])
    return df


def clean_robor(path, skiprows: int = 9) -> pd.DataFrame:
    """Curăță CSV-ul ROBOR 3M exportat din baza de date interactivă BNR (separator ';')."""
    df = pd.read_csv(path, sep=";", skiprows=skiprows, names=["Data", "Pret"], encoding="utf-8-sig")
    df = _parse_dates(df, str(path))
    df["Pret"] = pd.to_numeric(df["Pret"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    return df[["Data", "Pret"]]


def _finalize(df: pd.DataFrame, drop_volume: bool = False) -> pd.DataFrame:
    df = df.set_index("Data").sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    if drop_volume and "Volume" in df.columns:
        df = df.drop(columns="Volume")
    return df.ffill().bfill().round(5)


def _clean_and_save(name: str, cleaner, **kwargs) -> None:
    raw = config.DATA_RAW / f"{name}.csv"
    if not raw.exists():
        warnings.warn(f"Lipsește {raw.name} — pas omis.")
        return
    _finalize(cleaner(raw), **kwargs).to_csv(config.DATA_RAW / f"{name}_clean.csv", float_format="%.5f")


def standardize_stock_files() -> None:
    """BET-NG (Investing.com) + SNP/SNN/H2O (Yahoo) -> *_clean.csv în data/raw."""
    _clean_and_save("BETNG", clean_investing)
    for name in config.STOCK_TICKERS:
        _clean_and_save(name, clean_yahoo)


def standardize_macro_files() -> None:
    """Brent, TTF, EUR/RON, USD/RON (Yahoo) + ROBOR (BNR) -> *_clean.csv în data/raw."""
    _clean_and_save("ROBOR", clean_robor)
    _clean_and_save("Brent", clean_yahoo)
    for name in ("TTF", "EURRON", "USDRON"):  # volumul nu e informativ pentru TTF / cursuri valutare
        _clean_and_save(name, clean_yahoo, drop_volume=True)


# --- Baza de date principală -------------------------------------------------
_MASTER_INPUTS = {
    "BET-NG": "BETNG_clean.csv", "SNP": "SNP_clean.csv", "SNN": "SNN_clean.csv", "H2O": "H2O_clean.csv",
    "Brent": "Brent_clean.csv", "TTF": "TTF_clean.csv", "EUR": "EURRON_clean.csv",
    "USD": "USDRON_clean.csv", "ROBOR": "ROBOR_clean.csv",
}


def build_master_dataset(save: bool = True) -> pd.DataFrame:
    """Unește seriile pe date, propagă ultima valoare cunoscută și calculează transformările.

    Pret_<X>  = preț / nivel
    Rand_<X>  = randament logaritmic zilnic (toate seriile, mai puțin ROBOR)
    Diff_ROBOR = variația zilnică în puncte procentuale
    Zilele de tranzacționare sunt cele ale BET-NG.
    """
    master = None
    for name, filename in _MASTER_INPUTS.items():
        s = pd.read_csv(config.DATA_RAW / filename, index_col="Data", parse_dates=True)[["Pret"]]
        s = s.rename(columns={"Pret": f"Pret_{name}"})
        master = s if master is None else master.join(s, how="outer")

    master = master.sort_index().ffill()
    for col in list(master.columns):
        name = col.split("_", 1)[1]
        if name == "ROBOR":
            master["Diff_ROBOR"] = master[col] - master[col].shift(1)
        else:
            master[f"Rand_{name}"] = np.log(master[col] / master[col].shift(1))

    master = master.dropna(subset=["Rand_BET-NG"]).round(6)
    if save:
        master.to_csv(config.MASTER_DATASET)
    return master


def load_master_dataset() -> pd.DataFrame:
    """Încarcă baza de date principală, indexată cronologic după dată."""
    return pd.read_csv(config.MASTER_DATASET, index_col="Data", parse_dates=True).sort_index()
