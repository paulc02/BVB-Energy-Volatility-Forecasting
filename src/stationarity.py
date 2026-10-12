"""Teste de staționaritate: ADF, Phillips-Perron și KPSS."""
import warnings

import pandas as pd
from arch.unitroot import PhillipsPerron
from statsmodels.tsa.stattools import adfuller, kpss

from . import config


def format_pvalue(p) -> str:
    return "< 0.0001" if p < 0.0001 else f"{p:.4f}"


def stationarity_table(df: pd.DataFrame, out_path=None) -> pd.DataFrame:
    """Tabel ADF / PP / KPSS pentru toate randamentele și variația ROBOR.

    O serie e considerată staționară I(0) dacă ADF și PP resping rădăcina unitară (p < 0.05)
    iar KPSS nu respinge staționaritatea (p > 0.05).
    """
    cols = [c for c in df.columns if c.startswith(("Rand_", "Diff_"))]
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # KPSS avertizează când p-value e în afara tabelului
        for col in cols:
            s = df[col].dropna()
            adf_stat, adf_p = adfuller(s)[:2]
            pp = PhillipsPerron(s, trend="c")
            kpss_stat, kpss_p = kpss(s, regression="c", nlags="auto")[:2]
            stationary = adf_p < 0.05 and pp.pvalue < 0.05 and kpss_p > 0.05
            rows.append({
                "Variabilă (Serie Transformată)": config.SERIES_TITLES.get(col, col),
                "Statistică ADF": round(adf_stat, 4), "p-value (ADF)": format_pvalue(adf_p),
                "Statistică PP": round(pp.stat, 4), "p-value (PP)": format_pvalue(pp.pvalue),
                "Statistică KPSS": round(kpss_stat, 4), "p-value (KPSS)": format_pvalue(kpss_p),
                "Decizie Finală": "Staționară I(0)" if stationary else "Non-staționară / Incertă",
            })
    table = pd.DataFrame(rows)
    if out_path is not None:
        table.to_csv(out_path, index=False, encoding="utf-8-sig")
    return table
