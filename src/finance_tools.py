"""Reusable helpers for the Financial Data Science market-monitoring briefing.

Every function is used by the notebook in ``notebook/``. Nothing in here
hard-codes a personal computer path: pass ``pathlib.Path`` objects in.

Layout of the raw file this module understands
----------------------------------------------
1. Wide format: ``Date`` + one price column per asset/benchmark.
2. yfinance-style format: ``Date`` + columns named ``TICKER|Field``
   (e.g. ``AAPL|Adj Close``), produced by :func:`download_yfinance`.

# TODO(PROMPT): If your dataset has a different layout (long format with a
# 'Symbol' column, one CSV per company, Excel sheets, CSE column names such as
# 'Trade Date' / 'Close (Rs.)'), paste 5 rows of it to Claude with:
# "Adapt load_data() and select_price_field() in finance_tools.py to this layout."
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "load_data",
    "download_yfinance",
    "file_sha256",
    "validate_data",
    "select_price_field",
    "clean_prices",
    "flag_extreme_moves",
    "calculate_returns",
    "normalize_prices",
    "summary_statistics",
    "run_quality_checks",
    "check_files_exist",
    "save_outputs",
    "plot_normalized",
    "plot_histograms",
    "plot_risk_return",
    "plot_heatmap",
]


# --------------------------------------------------------------------------
# Acquisition and file handling
# --------------------------------------------------------------------------
def load_data(path: Path, date_col: str = "Date") -> pd.DataFrame:
    """Load a raw CSV or Excel file and parse its date column.

    Invalid dates become ``NaT`` (they are logged later by ``clean_prices``);
    the raw values of every other column are left untouched.

    Raises:
        FileNotFoundError: the file does not exist.
        ValueError: unsupported extension, missing date column, or no valid date.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {path}. Place the dataset there or set "
            "DATA_SOURCE='yfinance' in the notebook configuration."
        )
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)  # needs openpyxl
    else:
        raise ValueError(f"Unsupported file type: {suffix} (use .csv or .xlsx)")

    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found. Columns: {list(df.columns)}")
    if date_col != "Date":
        df = df.rename(columns={date_col: "Date"})

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().all():
        raise ValueError("No valid dates could be parsed from the date column.")
    return df


def download_yfinance(tickers: Iterable[str], start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV data with yfinance and flatten to ``TICKER|Field`` columns.

    Used when the market-monitoring briefing data come from an API instead of a file.

    Raises:
        ImportError: yfinance is not installed.
        ValueError: invalid dates, no data returned, or some tickers unavailable.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install yfinance first: pip install yfinance") from exc

    tickers = list(tickers)
    try:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid start/end date: {start!r}, {end!r}") from exc
    if start_ts >= end_ts:
        raise ValueError("START_DATE must be earlier than END_DATE")

    data = yf.download(
        tickers, start=start, end=end, auto_adjust=False, progress=False,
        group_by="column",
    )
    if data is None or data.empty:
        raise ValueError(f"No data returned for {tickers}. Check ticker symbols and dates.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [f"{ticker}|{field}" for field, ticker in data.columns]
    else:  # single ticker
        data.columns = [f"{tickers[0]}|{field}" for field in data.columns]

    got = {c.split("|")[0] for c in data.columns}
    missing = [t for t in tickers if t not in got]
    if missing:
        raise ValueError(f"Tickers unavailable from yfinance: {missing}")

    data = data.reset_index().rename(columns={data.index.name or "index": "Date"})
    return data


def file_sha256(path: Path) -> str:
    """SHA-256 fingerprint of a file; proves the raw dataset was never modified."""
    digest = hashlib.sha256()
    with open(Path(path), "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_outputs(df: pd.DataFrame, path: Path, index: bool = True) -> Path:
    """Write a DataFrame to CSV (creating folders) and confirm the file exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    if not path.exists() or path.stat().st_size == 0:
        raise IOError(f"Output file was not created: {path}")
    return path


# --------------------------------------------------------------------------
# Validation and cleaning
# --------------------------------------------------------------------------
def validate_data(df: pd.DataFrame, min_months: int = 24) -> pd.DataFrame:
    """Return a data-audit table (one row per column) and check the date span.

    Raises:
        ValueError: if the valid-date span is shorter than ``min_months``.
    """
    n_bad = int(df["Date"].isna().sum())
    if n_bad:
        print(f"Warning: {n_bad} rows have an invalid date (they will be logged).")
    first, last = df["Date"].min(), df["Date"].max()
    span_days = (last - first).days
    if span_days < min_months * 30:
        raise ValueError(
            f"Only {span_days} days of data ({first:%Y-%m-%d} to {last:%Y-%m-%d}); "
            f"the brief requires at least {min_months} months."
        )

    audit = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "first_valid_date": [df.loc[df[c].notna(), "Date"].min() for c in df.columns],
        "last_valid_date": [df.loc[df[c].notna(), "Date"].max() for c in df.columns],
    })
    audit.index.name = "column"
    return audit


def select_price_field(
    df: pd.DataFrame, field: str = "Adj Close", tickers: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    """Return ``Date`` + one price column per ticker.

    * ``TICKER|Field`` columns -> pick the requested field and strip the suffix.
    * Otherwise the frame is assumed to be wide already and is returned as is.
    """
    if any("|" in str(c) for c in df.columns):
        keep = {c: str(c).split("|")[0] for c in df.columns if str(c).endswith(f"|{field}")}
        if not keep:
            raise KeyError(f"Field '{field}' not found in the raw columns.")
        out = df[["Date", *keep.keys()]].rename(columns=keep)
    else:
        out = df.copy()

    if tickers is not None:
        absent = [t for t in tickers if t not in out.columns]
        if absent:
            raise KeyError(f"Columns missing from the raw data: {absent}")
        out = out[["Date", *tickers]]
    return out


def clean_prices(df: pd.DataFrame, max_ffill: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean a wide price table and log every material decision.

    Steps (each one is written to the returned quality log):
      1. drop rows with an invalid date;
      2. coerce prices to numbers; non-numeric and non-positive values -> NaN;
      3. sort by date;
      4. drop duplicate dates (keep the last record);
      5. forward-fill gaps of at most ``max_ffill`` trading days (holidays);
      6. drop any remaining row that still has a missing analytical value.

    No price is ever edited by hand; nothing is replaced silently.

    Returns:
        (clean_prices indexed by Date, quality_log with columns
        date / column / issue / action)
    """
    log: list[dict] = []

    def add(date, column, issue, action):
        log.append({"date": date, "column": column, "issue": issue, "action": action})

    df = df.copy()
    price_cols = [c for c in df.columns if c != "Date"]

    bad_date = df["Date"].isna()
    for _ in range(int(bad_date.sum())):
        add(pd.NaT, "Date", "invalid or unparsable date", "row removed")
    df = df.loc[~bad_date]

    for col in price_cols:
        converted = pd.to_numeric(df[col], errors="coerce")
        for d in df.loc[converted.isna() & df[col].notna(), "Date"]:
            add(d, col, "non-numeric price", "converted to missing")
        nonpos = converted <= 0
        for d in df.loc[nonpos, "Date"]:
            add(d, col, "zero or negative price", "converted to missing")
        converted[nonpos] = np.nan
        df[col] = converted

    if not df["Date"].is_monotonic_increasing:
        add(pd.NaT, "Date", "dates not in ascending order", "sorted ascending")
    df = df.sort_values("Date", kind="stable")

    dup = df["Date"].duplicated(keep="last")
    for d in df.loc[dup, "Date"]:
        add(d, "Date", "duplicate date", "kept last record, removed earlier one")
    df = df.loc[~dup].set_index("Date")

    filled = df.ffill(limit=max_ffill)
    for col in price_cols:
        was_missing = df[col].isna()
        for d in df.index[was_missing & filled[col].notna()]:
            add(d, col, "missing price (short gap)", f"forward-filled (limit {max_ffill} days)")
    df = filled

    still_missing = df.isna().any(axis=1)
    for d in df.index[still_missing]:
        cols = ", ".join(df.columns[df.loc[d].isna()])
        add(d, cols, "missing price not fillable (leading or long gap)", "row removed")
    df = df.loc[~still_missing]

    quality_log = pd.DataFrame(log, columns=["date", "column", "issue", "action"])
    return df, quality_log


def flag_extreme_moves(returns: pd.DataFrame, threshold: float = 0.10) -> pd.DataFrame:
    """List every daily return whose absolute size exceeds ``threshold`` (e.g. 0.10 = 10%).

    The analyst must then classify each one as a valid market move, a corporate
    action (split, dividend, rights issue) or a data error.
    """
    stacked = returns.stack()
    stacked.index.names = ["date", "asset"]
    flagged = stacked[stacked.abs() > threshold].rename("daily_return").reset_index()
    return flagged.sort_values("daily_return", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Returns and statistics
# --------------------------------------------------------------------------
def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns: Return_t = Price_t / Price_(t-1) - 1."""
    return prices.pct_change(fill_method=None).dropna(how="all")


def normalize_prices(prices: pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
    """Rebase every series so its first valid observation equals ``base``."""
    first = prices.apply(lambda s: s.loc[s.first_valid_index()])
    return prices / first * base


def summary_statistics(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Analyst-ready table from daily returns (one row per asset/benchmark).

    Kurtosis is *excess* kurtosis (normal distribution = 0), as returned by pandas.
    """
    table = pd.DataFrame({
        "Observations": returns.count(),
        "Mean daily return": returns.mean(),
        "Minimum": returns.min(),
        "Maximum": returns.max(),
        "Total period return": prices.iloc[-1] / prices.iloc[0] - 1,
        "Variance": returns.var(),
        "Standard deviation": returns.std(),
        "Skewness": returns.skew(),
        "Kurtosis (excess)": returns.kurt(),
    })
    table.index.name = "Asset"
    return table


# --------------------------------------------------------------------------
# Automated checks
# --------------------------------------------------------------------------
def run_quality_checks(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Run the automated data checks, show a visible PASS/FAIL table, then assert.

    Raises:
        AssertionError: if any check fails (after the table is returned/printed).
    """
    checks = {
        "No duplicate dates": not prices.index.duplicated().any(),
        "Dates in ascending order": prices.index.is_monotonic_increasing,
        "Index is datetime": isinstance(prices.index, pd.DatetimeIndex),
        "All prices numeric": all(pd.api.types.is_numeric_dtype(t) for t in prices.dtypes),
        "All prices positive": bool((prices > 0).all().all()),
        "No missing prices": not prices.isna().any().any(),
        "No missing returns": not returns.isna().any().any(),
        "No infinite returns": bool(np.isfinite(returns.to_numpy()).all()),
        "Prices and returns aligned": bool(returns.index.isin(prices.index).all()),
    }
    result = pd.DataFrame({"check": list(checks), "passed": list(checks.values())})
    result["status"] = np.where(result["passed"], "PASS", "FAIL")
    failed = result.loc[~result["passed"], "check"].tolist()
    if failed:
        print(result.to_string(index=False))
    assert not failed, f"Data quality checks failed: {failed}"
    return result[["check", "status"]]


def check_files_exist(paths: Iterable[Path]) -> pd.DataFrame:
    """Confirm that every expected output file exists and is not empty."""
    rows = []
    for p in map(Path, paths):
        ok = p.exists() and p.stat().st_size > 0
        rows.append({"file": p.name, "status": "PASS" if ok else "FAIL"})
    result = pd.DataFrame(rows)
    assert (result["status"] == "PASS").all(), f"Missing outputs:\n{result}"
    return result


# --------------------------------------------------------------------------
# Decision-oriented figures (all export a PNG)
# --------------------------------------------------------------------------
def _finish(fig, source_note: str, save_path: Path):
    """Add the source/period note, save the PNG and confirm it exists."""
    fig.text(0.01, 0.005, source_note, ha="left", va="bottom", fontsize=8, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    assert save_path.exists(), f"Figure not saved: {save_path}"
    return fig


def plot_normalized(norm: pd.DataFrame, benchmark: str, title: str,
                    source_note: str, save_path: Path):
    """Figure 1: all series rebased to 100 (benchmark drawn as a dashed black line)."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for col in norm.columns:
        if col == benchmark:
            ax.plot(norm.index, norm[col], color="black", ls="--", lw=2, label=f"{col} (benchmark)")
        else:
            ax.plot(norm.index, norm[col], lw=1.5, label=col)
    ax.axhline(100, color="grey", lw=0.8)
    ax.set(title=title, xlabel="Date", ylabel="Index level (start = 100)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    return _finish(fig, source_note, save_path)


def plot_histograms(returns: pd.DataFrame, title: str, source_note: str,
                    save_path: Path, bins: int = 50):
    """Figure 2: one daily-return histogram per series (percent scale)."""
    n = returns.shape[1]
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.6 * rows), squeeze=False)
    for ax, col in zip(axes.ravel(), returns.columns):
        ax.hist(returns[col] * 100, bins=bins, color="#4C72B0", alpha=0.85, label=col)
        ax.axvline(returns[col].mean() * 100, color="red", lw=1.2, label="Mean")
        ax.set(title=col, xlabel="Daily return (%)", ylabel="Frequency")
        ax.legend(fontsize=8)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=13)
    return _finish(fig, source_note, save_path)


def plot_risk_return(returns: pd.DataFrame, title: str, source_note: str, save_path: Path):
    """Figure 3: mean daily return (y) against standard deviation (x), every point labelled."""
    fig, ax = plt.subplots(figsize=(8, 6))
    x, y = returns.std() * 100, returns.mean() * 100
    ax.scatter(x, y, s=90, color="#C44E52", label="Asset / benchmark")
    for name in returns.columns:
        ax.annotate(name, (x[name], y[name]), textcoords="offset points",
                    xytext=(6, 6), fontsize=9)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(title=title, xlabel="Risk: standard deviation of daily return (%)",
           ylabel="Mean daily return (%)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    return _finish(fig, source_note, save_path)


def plot_heatmap(corr: pd.DataFrame, title: str, source_note: str, save_path: Path):
    """Figure 4: annotated correlation heatmap."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr.values, cmap="RdYlBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Correlation of daily returns")
    ax.set(title=title, xlabel="Asset", ylabel="Asset")
    return _finish(fig, source_note, save_path)
