"""Estimare GARCH-X(1,1) și APARCH-X(1,1) prin maximum de verosimilitate (implementare proprie).

Variabilele exogene intră în ecuația varianței ca șocuri pătratice, standardizate și decalate cu o perioadă:
    GARCH-X : h_t      = omega + alpha*e_{t-1}^2 + beta*h_{t-1} + theta' z_{t-1}
    APARCH-X: s_t^d    = omega + alpha*(|e_{t-1}| - gamma*e_{t-1})^d + beta*s_{t-1}^d + theta' z_{t-1}
Inovațiile urmează distribuția Normală sau Student-t (standardizată).

Notă: estimarea nu calculează erori standard, deci semnificația coeficienților nu e testată.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.arima.model import ARIMA

from . import config

STARTS_AB = [(0.04, 0.92), (0.08, 0.88), (0.12, 0.84), (0.16, 0.78)]
STARTS_APARCH = [(0.04, 0.92, 0.05, 2.0), (0.08, 0.88, 0.10, 1.5),
                 (0.12, 0.84, -0.10, 2.0), (0.16, 0.78, 0.15, 1.2)]


# --- Log-verosimilități și criterii de informare -----------------------------
def loglik_normal(eps, h):
    return -0.5 * np.sum(np.log(2 * np.pi) + np.log(h) + eps ** 2 / h)


def loglik_student_t(eps, h, nu):
    c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    return np.sum(c - 0.5 * np.log(h) - ((nu + 1) / 2) * np.log(1 + eps ** 2 / (h * (nu - 2))))


def information_criteria(loglik, k, n):
    return -2 * loglik + 2 * k, -2 * loglik + k * np.log(n)


def _loglik(eps, h, dist, nu=None):
    return loglik_student_t(eps, h, nu) if dist == "t" else loglik_normal(eps, h)


def _minimize_multistart(nll, starts, bounds, maxiter):
    best = None
    for s in starts:
        opt = minimize(nll, s, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "ftol": 1e-8, "disp": False})
        if best is None or opt.fun < best.fun:
            best = opt
    return best


def _pack(best, par_names, eps, h, sigma, extra=None):
    ll = -best.fun
    k = len(best.x)
    aic, bic = information_criteria(ll, k, len(eps))
    out = {
        "params": pd.Series(best.x, index=par_names),
        "h": pd.Series(h, index=eps.index, name="h"),
        "sigma": pd.Series(sigma, index=eps.index, name="sigma"),
        "std_resid": pd.Series(eps.values / sigma, index=eps.index, name="std_resid"),
        "loglik": ll, "aic": aic, "bic": bic,
        "success": bool(best.success), "message": best.message,
        "nobs": len(eps), "k_params": k,
    }
    if extra:
        out.update(extra)
    return out


# --- GARCH-X(1,1) -------------------------------------------------------------
def fit_garch_x(eps: pd.Series, z: pd.DataFrame, dist: str = "normal", maxiter: int = 600) -> dict:
    eps_arr, z_arr = eps.values.astype(float), z.values.astype(float)
    n, m = z_arr.shape
    var0 = max(np.var(eps_arr), 1e-8)

    def rec_h(p):
        omega, alpha, beta, theta = p[0], p[1], p[2], np.array(p[3:3 + m])
        h = np.empty(n)
        h[0] = var0
        for t in range(1, n):
            h[t] = omega + alpha * eps_arr[t - 1] ** 2 + beta * h[t - 1] + np.dot(theta, z_arr[t - 1])
            if not np.isfinite(h[t]) or h[t] <= 1e-10:
                h[t] = 1e-10
        return h

    def nll(p):
        if p[1] + p[2] >= 0.999:
            return 1e10 + (p[1] + p[2] - 0.999) * 1e10
        h = rec_h(p)
        if np.any(~np.isfinite(h)) or np.any(h <= 0):
            return 1e12
        if dist == "t" and p[-1] <= 2.01:
            return 1e12
        ll = _loglik(eps_arr, h, dist, p[-1] if dist == "t" else None)
        return -ll if np.isfinite(ll) else 1e12

    tail_x0, tail_bounds = ([8.0], [(2.05, 100.0)]) if dist == "t" else ([], [])
    x0 = [0.05 * var0, 0.08, 0.88] + [0.01 * var0] * m + tail_x0
    bounds = [(1e-10, 10 * var0), (1e-8, 0.999), (1e-8, 0.999)] + [(0.0, 10 * var0)] * m + tail_bounds
    starts = [x0] + [[0.05 * var0, a, b] + [0.01 * var0] * m + tail_x0 for a, b in STARTS_AB]

    best = _minimize_multistart(nll, starts, bounds, maxiter)
    h = rec_h(best.x)
    names = ["omega", "alpha", "beta"] + [f"theta_{c}" for c in z.columns] + (["nu"] if dist == "t" else [])
    return _pack(best, names, eps, h, np.sqrt(h))


# --- APARCH-X(1,1) ------------------------------------------------------------
def fit_aparch_x(eps: pd.Series, z: pd.DataFrame, dist: str = "t", maxiter: int = 800) -> dict:
    eps_arr, z_arr = eps.values.astype(float), z.values.astype(float)
    n, m = z_arr.shape
    var0 = max(np.var(eps_arr), 1e-8)
    sig0 = np.sqrt(var0)

    def rec(p):
        omega, alpha, gamma, beta, delta = p[:5]
        theta = np.array(p[5:5 + m])
        sdel = np.empty(n)
        sdel[0] = max(sig0 ** delta, 1e-8)
        for t in range(1, n):
            shock = max(abs(eps_arr[t - 1]) - gamma * eps_arr[t - 1], 1e-8)
            sdel[t] = omega + alpha * shock ** delta + beta * sdel[t - 1] + np.dot(theta, z_arr[t - 1])
            if not np.isfinite(sdel[t]) or sdel[t] <= 1e-10:
                sdel[t] = 1e-10
        sigma = sdel ** (1.0 / delta)
        return sdel, sigma, sigma ** 2

    def nll(p):
        if p[1] + p[3] >= 0.999:
            return 1e10 + (p[1] + p[3] - 0.999) * 1e10
        _, _, h = rec(p)
        if np.any(~np.isfinite(h)) or np.any(h <= 0):
            return 1e12
        if dist == "t" and p[-1] <= 2.01:
            return 1e12
        ll = _loglik(eps_arr, h, dist, p[-1] if dist == "t" else None)
        return -ll if np.isfinite(ll) else 1e12

    tail_x0, tail_bounds = ([8.0], [(2.05, 100.0)]) if dist == "t" else ([], [])
    x0 = [0.05 * sig0, 0.08, 0.05, 0.88, 2.0] + [0.01 * sig0] * m + tail_x0
    bounds = ([(1e-10, 10 * sig0), (1e-8, 0.999), (-0.999, 0.999), (1e-8, 0.999), (0.25, 4.0)]
              + [(0.0, 10 * sig0)] * m + tail_bounds)
    starts = [x0] + [[0.05 * sig0, a, g, b, d] + [0.01 * sig0] * m + tail_x0 for a, b, g, d in STARTS_APARCH]

    best = _minimize_multistart(nll, starts, bounds, maxiter)
    sdel, sigma, h = rec(best.x)
    names = (["omega", "alpha", "gamma", "beta", "delta"] + [f"theta_{c}" for c in z.columns]
             + (["nu"] if dist == "t" else []))
    return _pack(best, names, eps, h, sigma, extra={"sdel": pd.Series(sdel, index=eps.index)})


def fit_volatility_model(kind: str, eps, z, dist):
    """kind: 'APARCH-X' sau 'GARCH-X'."""
    return fit_aparch_x(eps, z, dist=dist) if kind == "APARCH-X" else fit_garch_x(eps, z, dist=dist)


# --- Pregătirea datelor -------------------------------------------------------
def split_train(data: pd.DataFrame, fraction: float = config.TRAIN_FRACTION):
    n_train = int(len(data) * fraction)
    return data.iloc[:n_train].copy(), data.iloc[n_train:].copy()


def standardized_exog_shocks(raw: pd.DataFrame, train_mean: pd.Series) -> pd.DataFrame:
    """Șocuri pătratice (randamente în %) standardizate cu media din train."""
    z = ((raw * 100) ** 2) / train_mean
    return z.replace([np.inf, -np.inf], np.nan)


def prepare_inputs(df: pd.DataFrame, serie_y: str, exog_cols: list, fraction: float = config.TRAIN_FRACTION):
    """Reziduuri ARMA (în %) pe train și șocurile exogene decalate cu o perioadă, pe eșantionul comun."""
    missing = [c for c in [serie_y] + exog_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Coloane lipsă în baza de date: {missing}")

    train, _ = split_train(df[[serie_y] + exog_cols].dropna(), fraction)
    arma_fit = ARIMA(train[serie_y], order=config.ARMA_ORDERS[serie_y]).fit()
    eps = arma_fit.resid * 100

    raw = (train[exog_cols] * 100) ** 2
    mean = raw.mean(skipna=True).replace(0, np.nan)
    z = (raw / mean).replace([np.inf, -np.inf], np.nan).shift(1).fillna(0.0).loc[eps.index]
    return eps.astype(float), z.astype(float), arma_fit


# --- Teste de diagnostic pe reziduuri standardizate --------------------------
def residual_diagnostics(std_resid, lags=(1, 5, 10, 20)):
    """Ljung-Box și ARCH-LM pe reziduurile standardizate. Returnează (rânduri_LB, rânduri_ARCH)."""
    s = pd.Series(std_resid).replace([np.inf, -np.inf], np.nan).dropna()
    ljung, arch = [], []
    if len(s) < max(lags) + 5:
        return ljung, arch

    lb = acorr_ljungbox(s, lags=list(lags), return_df=True)
    for lag in lags:
        p = lb.loc[lag, "lb_pvalue"]
        ljung.append({"Lag": lag, "Statistica Ljung-Box": lb.loc[lag, "lb_stat"], "p-value": p,
                      "Autocorelare reziduala?": "DA" if p < 0.05 else "NU"})
    for lag in lags:
        stat, p, _, _ = het_arch(s, nlags=lag)
        arch.append({"Lag": lag, "Statistica ARCH-LM": stat, "p-value": p,
                     "Efecte ARCH reziduale?": "DA" if p < 0.05 else "NU"})
    return ljung, arch
