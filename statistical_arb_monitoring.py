# residual_stream.py — prints one DF row per second with residual & live prices
from datetime import datetime, timezone
import time
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

# -------- params you can change --------
SYM1 = "US500"      # dependent (y)
SYM2 = "USTEC"      # independent (x)
TF_STR = "M1"       # M1,M5,M15,M30,H1,H4,D1
BETA_WINDOW = 300   # rolling OLS lookback (bars)
HISTORY_BARS = max(BETA_WINDOW + 50, 600)  # bars fetched each tick
REFRESH_SEC = 1
# --------------------------------------

TF_MAP = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}
TF = TF_MAP[TF_STR]

def init_mt5():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    for s in (SYM1, SYM2):
        mt5.symbol_select(s, True)

def get_last_bars(sym: str, tf_code: int, n: int) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(sym, tf_code, 0, n)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df[["time", "close"]].rename(columns={"close": sym})

def rolling_ols_alpha_beta_past_only(y: pd.Series, x: pd.Series, win: int):
    # vectorized OLS with intercept; shift(1) so α,β at t use data up to t-1
    mx  = x.rolling(win, min_periods=win).mean()
    my  = y.rolling(win, min_periods=win).mean()
    ex2 = (x**2).rolling(win, min_periods=win).mean()
    exy = (x*y).rolling(win, min_periods=win).mean()
    varx  = ex2 - mx**2
    covxy = exy - mx*my
    beta  = covxy / varx.replace(0, np.nan)
    alpha = my - beta * mx
    return alpha.shift(1), beta.shift(1)

def mid_tick(sym: str) -> float:
    t = mt5.symbol_info_tick(sym)
    if t is None:
        return np.nan
    last = getattr(t, "last", 0.0)
    if last not in (None, 0.0):
        return float(last)
    bid = getattr(t, "bid", 0.0)
    ask = getattr(t, "ask", 0.0)
    if bid and ask:
        return (float(bid) + float(ask)) / 2.0
    return np.nan

def compute_row():
    """Return a single-row DataFrame with ts_utc, tf, y_price, x_price, alpha, beta, residual."""
    d1 = get_last_bars(SYM1, TF, HISTORY_BARS)
    d2 = get_last_bars(SYM2, TF, HISTORY_BARS)
    if d1.empty or d2.empty:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return pd.DataFrame([{
            "ts_utc": ts, "tf": TF_STR, SYM1: np.nan, SYM2: np.nan,
            "alpha": np.nan, "beta": np.nan, "residual": np.nan
        }])

    df = d1.merge(d2, on="time", how="inner")
    y = df[SYM1].astype(float)
    x = df[SYM2].astype(float)

    alpha, beta = rolling_ols_alpha_beta_past_only(y, x, BETA_WINDOW)
    a_last = alpha.iloc[-1]
    b_last = beta.iloc[-1]

    # live prices (fallback to last closes if ticks unavailable)
    y_live = mid_tick(SYM1); x_live = mid_tick(SYM2)
    if not np.isfinite(y_live): y_live = float(y.iloc[-1])
    if not np.isfinite(x_live): x_live = float(x.iloc[-1])

    residual = y_live - (a_last + b_last * x_live) if (np.isfinite(a_last) and np.isfinite(b_last)) else np.nan

    return pd.DataFrame([{
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tf": TF_STR,
        SYM1: y_live,
        SYM2: x_live,
        "alpha": a_last,
        "beta": b_last,
        "residual": residual
    }])

def ff(v: float) -> str:
    return "nan" if not np.isfinite(v) else f"{v:.6f}"

if __name__ == "__main__":
    init_mt5()
    try:
        while True:
            row = compute_row()
            # pretty one-line DF print (no index; fixed float format)
            print(row.to_string(index=False, float_format=ff))
            time.sleep(REFRESH_SEC)
    except KeyboardInterrupt:
        print("Stopped.")

        
