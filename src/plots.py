"""Funcții de grafică: planșe de serii / ACF / PACF, matrice de corelație."""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from . import config

BLUE, RED, GRAY_BG, GRID = "#1A56A0", "#C0392B", "#F7F9FC", "#DDE3EE"
TITLE_COLOR = "#1A2945"


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "axes.facecolor": GRAY_BG, "figure.facecolor": "white",
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": False,
    })


def plot_series_panels(df, columns, kind: str, out_path, lags: int = 40, n_cols: int = 3):
    """Salvează o planșă cu un panou per serie.

    kind: 'series' (evoluție), 'acf' sau 'pacf'.
    """
    _style()
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()

    for ax, col in zip(axes, columns):
        s = df[col].dropna()
        title = config.SERIES_TITLES.get(col, col)
        if kind == "series":
            ax.plot(s.index, s.values, color=BLUE, linewidth=0.9, alpha=0.85)
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        else:
            color, ci = (BLUE, "#A8C4E8") if kind == "acf" else (RED, "#F0BCBC")
            kwargs = dict(lags=lags, ax=ax, color=color, vlines_kwargs={"colors": color, "linewidth": 1.4},
                          alpha=0.05, zero=False, auto_ylims=True)
            if kind == "acf":
                plot_acf(s, **kwargs)
            else:
                plot_pacf(s, method="ywm", **kwargs)
            for poly in ax.collections:
                poly.set_facecolor(ci)
                poly.set_alpha(0.35)
            title = f"{kind.upper()}: {title}"
            ax.set_xlabel("")
        ax.set_title(title, fontsize=11, fontweight="bold", color=TITLE_COLOR, pad=5)
        ax.tick_params(labelsize=7.5)

    for ax in axes[len(columns):]:
        ax.set_visible(False)
    fig.tight_layout(h_pad=3.0, w_pad=1.5)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(df, out_path):
    """Matricea de corelație Pearson (randamente + variația ROBOR)."""
    cols = [c for c in df.columns if c.startswith(("Rand_", "Diff_"))]
    corr = df[cols].rename(columns=config.CORRELATION_LABELS).corr(method="pearson")
    plt.style.use("default")
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, center=0, linewidths=0.5,
                annot_kws={"size": 10, "weight": "bold"})
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    return corr
