"""Prognoze out-of-sample: ARMA rolling (randamente) + GARCH-X / APARCH-X recursiv (volatilitate)."""
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from . import config
from .garchx import fit_volatility_model, split_train, standardized_exog_shocks

Z_SCORES = {"80": 1.28, "90": 1.645, "95": 1.96}


def metrics_basic(y_real, y_pred, include_mape: bool = False) -> dict:
    y_real, y_pred = np.asarray(y_real, float), np.asarray(y_pred, float)
    mask = np.isfinite(y_real) & np.isfinite(y_pred)
    err = y_pred[mask] - y_real[mask]
    out = {"MSE": np.mean(err ** 2), "RMSE": np.sqrt(np.mean(err ** 2)), "MAE": np.mean(np.abs(err))}
    if include_mape:
        nz = np.abs(y_real[mask]) > 1e-12
        out["MAPE (%)"] = np.mean(np.abs(err[nz] / y_real[mask][nz])) * 100 if nz.any() else np.nan
    return out


def rolling_arma_forecast(y: pd.Series, n_train: int, order: tuple, verbose: bool = True):
    """Prognoză one-step-ahead cu reestimare ARMA pe fereastră extinsă. Returnează (prognoze, reziduuri OOS)."""
    forecasts, residuals = [], []
    n_test = len(y) - n_train
    values = y.to_numpy(dtype=float)  # fără index de date: zilele de tranzacționare nu au frecvență fixă
    for i in range(n_test):
        fit = ARIMA(values[:n_train + i], order=order).fit()
        fc = float(np.asarray(fit.forecast(steps=1))[0])
        forecasts.append(fc)
        residuals.append(float(y.iloc[n_train + i]) - fc)
        if verbose and ((i + 1) % 100 == 0 or i == n_test - 1):
            print(f"    {i + 1}/{n_test} pași ARMA")
    idx = y.index[n_train:]
    return pd.Series(forecasts, index=idx), pd.Series(residuals, index=idx)


def recursive_volatility_forecast(kind: str, fit: dict, eps_train: pd.Series, z_test_prev: np.ndarray,
                                  exog_cols: list, resid_oos: pd.Series) -> pd.Series:
    """Volatilitate prognozată (%) pe test: parametrii fixați pe train, șocurile actualizate cu reziduurile OOS.

    z_test_prev[i] = șocurile exogene (standardizate) de la t-1 pentru ziua i din test.
    """
    p = fit["params"]
    theta = np.array([p[f"theta_{c}"] for c in exog_cols])
    omega, alpha, beta = p["omega"], p["alpha"], p["beta"]
    eps_prev = float(eps_train.iloc[-1])
    h_prev = float(fit["h"].iloc[-1])
    sdel_prev = float(fit["sdel"].iloc[-1]) if kind == "APARCH-X" else np.nan

    sigma = []
    for i in range(len(resid_oos)):
        if kind == "GARCH-X":
            h_next = max(float(omega + alpha * eps_prev ** 2 + beta * h_prev + theta @ z_test_prev[i]), 1e-10)
            h_prev = h_next
            s = np.sqrt(h_next)
        else:
            gamma, delta = p["gamma"], p["delta"]
            shock = max(abs(eps_prev) - gamma * eps_prev, 1e-8)
            sdel_next = max(float(omega + alpha * shock ** delta + beta * sdel_prev + theta @ z_test_prev[i]), 1e-10)
            sdel_prev = sdel_next
            s = sdel_next ** (1.0 / delta)
        sigma.append(float(s))
        eps_prev = float(resid_oos.iloc[i] * 100)  # șocul observat actualizează prognoza următoare
    return pd.Series(sigma, index=resid_oos.index, name="Volatilitate_Prognozata")


def reconstruct_price_path(last_train_price: float, returns: pd.Series) -> pd.Series:
    """Traiectorie de preț obținută prin compunerea randamentelor prognozate, pornind de la ultimul preț din train."""
    return pd.Series(last_train_price * np.exp(np.cumsum(returns.values)), index=returns.index, name="Pret_Prognozat")


def forecast_asset(df: pd.DataFrame, var: str, model: dict, verbose: bool = True) -> dict:
    """Prognoză completă pentru un activ.

    model: {'nume', 'exog_cols', 'tip' ('GARCH-X'|'APARCH-X'), 'dist' ('t'|'normal')}.
    """
    exog_cols, kind, dist = model["exog_cols"], model["tip"], model["dist"]
    price_col = var.replace("Rand_", "Pret_")
    order = config.ARMA_ORDERS[var]

    data = df[[var, price_col] + exog_cols].dropna().copy()
    train, test = split_train(data)
    n_train = len(train)

    # 1) ARMA pe train -> reziduuri pentru estimarea GARCH-X
    eps_train = ARIMA(train[var], order=order).fit().resid * 100

    # 2) Șocuri exogene standardizate cu media din train, decalate cu o perioadă
    train_mean = ((train[exog_cols] * 100) ** 2).mean(skipna=True).replace(0, np.nan).fillna(1.0)
    z_train = standardized_exog_shocks(train[exog_cols], train_mean).shift(1).fillna(0.0).loc[eps_train.index]

    # 3) GARCH-X / APARCH-X estimat o singură dată pe train
    fit = fit_volatility_model(kind, eps_train, z_train, dist)

    # 4) ARMA rolling pe test
    if verbose:
        print(f"{var}: {n_train} train | {len(test)} test | {model['nume']}")
    ret_fc, resid_oos = rolling_arma_forecast(data[var], n_train, order, verbose)

    # 5) Volatilitate recursivă (exogenele de la t-1 pentru fiecare zi din test)
    z_all = standardized_exog_shocks(data[exog_cols], train_mean).fillna(0.0)
    z_prev = z_all.iloc[n_train - 1: len(data) - 1].values.astype(float)
    sigma_fc = recursive_volatility_forecast(kind, fit, eps_train, z_prev, exog_cols, resid_oos)

    # 6) Prețuri
    price_fc = reconstruct_price_path(float(train[price_col].iloc[-1]), ret_fc)

    results = pd.DataFrame({
        "Randament_Real": test[var], "Randament_Prognozat": ret_fc,
        "Reziduu_ARIMA_OOS": resid_oos,
        "Volatilitate_Realizata": resid_oos.abs() * 100, "Volatilitate_Prognozata": sigma_fc,
        "Pret_Real": test[price_col], "Pret_Prognozat": price_fc,
    })
    return {
        "serie": var, "info_model": model, "n_train": n_train, "n_test": len(test),
        "fit_garchx_train": fit, "rezultate": results,
        "metrici_vol": metrics_basic(results["Volatilitate_Realizata"], sigma_fc),
        "metrici_rand": metrics_basic(test[var], ret_fc, include_mape=True),
        "metrici_pret": metrics_basic(test[price_col], price_fc, include_mape=True),
    }


def price_intervals(results: pd.DataFrame) -> dict:
    """Intervale de încredere pentru traiectoria de preț, din varianța zilnică prognozată cumulată."""
    sigma_cum = np.sqrt(((results["Volatilitate_Prognozata"] / 100.0) ** 2).cumsum())
    path = results["Pret_Prognozat"]
    return {lvl: (path * np.exp(-z * sigma_cum), path * np.exp(z * sigma_cum)) for lvl, z in Z_SCORES.items()}
