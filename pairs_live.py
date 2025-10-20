# This is the script that executes the pairs trading strategy that was developed in previous notebooks. The script will run in the vps and connect to mt5.

# - Rolling OLS (alpha, beta), residual spread, Bollinger bands, z-score
# - Next-bar execution; two-leg orders with beta fixed at entry
# - CSV monitoring (account + spread) every loop; JSON state persistence
# - Configure EVERYTHING via environment variables

import os, time, json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

# =========================
# CONFIG (environment-driven)
# =========================
SYM1 = os.getenv("PAIR_SYM1", "GS.NYSE")
SYM2 = os.getenv("PAIR_SYM2", "MS.NYSE")

TF   = getattr(mt5, os.getenv("TF", "TIMEFRAME_H1"), mt5.TIMEFRAME_H1)

BETA_WINDOW = int(os.getenv("BETA_WINDOW", 300))   # rolling OLS window
BAND_WINDOW = int(os.getenv("BAND_WINDOW", 200))   # bands/z window
Z_IN  = float(os.getenv("Z_IN", 2.0))
Z_OUT = float(os.getenv("Z_OUT", 0.5))
UNITS = float(os.getenv("UNITS", 1.0))             # spread units to trade (maps to lots via broker steps)

# Data + cadence
LOOKBACK = int(os.getenv("LOOKBACK", max(BETA_WINDOW + BAND_WINDOW + 10, 800)))
POLL_SEC = int(os.getenv("POLL_SEC", 15))          # check every 15s for new bar

# Files (make unique per instance/pair/account)
STATE_FILE       = os.getenv("STATE_FILE", "pairs_state.json")
MON_SPREAD_CSV   = os.getenv("MON_SPREAD_CSV", "spread_monitor.csv")
MON_ACCOUNT_CSV  = os.getenv("MON_ACCOUNT_CSV", "account_log.csv")
LOG_PREFIX       = os.getenv("LOG_PREFIX", "")     # optional text prefix in logs (e.g., "ACC1 GS/MS")

# Magic (unique per instance if you like)
MAGIC = int(os.getenv("MAGIC", 20250930))

# Optional MT5 login (if terminal is not already logged in, or you want to switch account)
LOGIN    = os.getenv("MT5_LOGIN", "")
PASSWORD = os.getenv("MT5_PASSWORD", "")
SERVER   = os.getenv("MT5_SERVER", "")

# =========================
# UTILITIES
# =========================
def log(*a):
    prefix = (LOG_PREFIX + " | ") if LOG_PREFIX else ""
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "|", prefix, *a, flush=True)

def ensure_mt5():
    # init
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    # if credentials are provided or terminal has wrong account, log in
    if LOGIN and PASSWORD and SERVER:
        info = mt5.account_info()
        must_login = (info is None) or (str(info.login) != str(LOGIN))
        if must_login:
            if not mt5.login(login=int(LOGIN), password=PASSWORD, server=SERVER):
                raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")

def fetch_bars(symbol, timeframe, bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No data for {symbol}. err={mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df[["time","open","high","low","close","tick_volume"]]

def align_two(s1, s2):
    df = s1[["time","close"]].merge(s2[["time","close"]], on="time", how="inner",
                                    suffixes=(f"_{SYM1}", f"_{SYM2}"))
    df = df.rename(columns={f"close_{SYM1}": SYM1, f"close_{SYM2}": SYM2})
    return df.sort_values("time").reset_index(drop=True)

def rolling_ols_last(y, x, window):
    """alpha,beta using the last `window` points of y,x (past-only)."""
    yW = y[-window:]
    xW = x[-window:]
    mx = xW.mean(); my = yW.mean()
    varx = (xW**2).mean() - mx*mx
    if not np.isfinite(varx) or varx <= 0: return None, None
    covxy = (xW*yW).mean() - mx*my
    beta  = covxy / varx
    alpha = my - beta*mx
    if not np.isfinite(beta): return None, None
    return float(alpha), float(beta)

def calc_z_for_last(residual_series, window):
    rW = residual_series[-window:]
    sma = rW.mean()
    std = rW.std(ddof=1)
    if not np.isfinite(std) or std <= 0: return None, None, None
    z = (residual_series.iloc[-1] - sma) / std
    return float(z), float(sma), float(std)

def get_last_bar_open_time(symbol, timeframe):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 2)
    if rates is None or len(rates) < 2: return None
    return int(rates[0]["time"])  # current bar's open time (unix)

def symbol_info(symbol):
    info = mt5.symbol_info(symbol)
    if info is None: raise RuntimeError(f"symbol_info failed for {symbol}")
    return info

def normalize_volume(symbol, vol_req):
    info = symbol_info(symbol)
    vmin, vmax, vstep = info.volume_min, info.volume_max, info.volume_step
    v = abs(vol_req)
    if v < vmin: v = vmin
    if v > vmax: v = vmax
    if vstep > 0: v = np.floor(v / vstep) * vstep
    return max(v, vmin)

def get_tick(symbol):
    t = mt5.symbol_info_tick(symbol)
    if t is None: raise RuntimeError(f"symbol_info_tick failed for {symbol}")
    return t

def close_positions_for(symbols, magic):
    ok = True
    poss = mt5.positions_get()
    if poss is None: return True
    for pos in poss:
        if pos.magic != magic: continue
        if pos.symbol not in symbols: continue
        vol = pos.volume
        if vol <= 0: continue
        tick = get_tick(pos.symbol)
        if pos.type == mt5.POSITION_TYPE_BUY:
            req = dict(action=mt5.TRADE_ACTION_DEAL, symbol=pos.symbol, volume=vol,
                       type=mt5.ORDER_TYPE_SELL, price=tick.bid, deviation=50, magic=magic)
        else:
            req = dict(action=mt5.TRADE_ACTION_DEAL, symbol=pos.symbol, volume=vol,
                       type=mt5.ORDER_TYPE_BUY, price=tick.ask, deviation=50, magic=magic)
        r = mt5.order_send(req)
        ok = ok and (r and r.retcode == mt5.TRADE_RETCODE_DONE)
        if not ok: log("Close failed:", pos.symbol, r)
    return ok

def enter_spread(units, beta_entry, direction):
    """
    direction: +1 = long spread  (BUY SYM1, SELL beta*SYM2)
               -1 = short spread (SELL SYM1, BUY beta*SYM2)
    """
    v1 = normalize_volume(SYM1, units)
    v2 = normalize_volume(SYM2, units * beta_entry)
    if v1 <= 0 or v2 <= 0: 
        log("Invalid volumes", v1, v2); 
        return False
    t1, t2 = get_tick(SYM1), get_tick(SYM2)

    if direction > 0:
        req1 = dict(action=mt5.TRADE_ACTION_DEAL, symbol=SYM1, volume=v1,
                    type=mt5.ORDER_TYPE_BUY, price=t1.ask, deviation=50, magic=MAGIC)
        req2 = dict(action=mt5.TRADE_ACTION_DEAL, symbol=SYM2, volume=v2,
                    type=mt5.ORDER_TYPE_SELL, price=t2.bid, deviation=50, magic=MAGIC)
    else:
        req1 = dict(action=mt5.TRADE_ACTION_DEAL, symbol=SYM1, volume=v1,
                    type=mt5.ORDER_TYPE_SELL, price=t1.bid, deviation=50, magic=MAGIC)
        req2 = dict(action=mt5.TRADE_ACTION_DEAL, symbol=SYM2, volume=v2,
                    type=mt5.ORDER_TYPE_BUY, price=t2.ask, deviation=50, magic=MAGIC)

    r1 = mt5.order_send(req1)
    r2 = mt5.order_send(req2)
    ok = (r1 and r1.retcode == mt5.TRADE_RETCODE_DONE) and (r2 and r2.retcode == mt5.TRADE_RETCODE_DONE)
    if not ok: log("Enter spread failed", r1, r2)
    return ok

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"pos":0, "beta_entry":None, "entry_spread":None}
    try:
        with open(STATE_FILE,"r") as f:
            return json.load(f)
    except:
        return {"pos":0, "beta_entry":None, "entry_spread":None}

def save_state(state):
    with open(STATE_FILE,"w") as f: json.dump(state,f)

# ------------- Monitoring writers -------------
def write_account_snapshot(state):
    try:
        acct = mt5.account_info()
        now  = datetime.now(timezone.utc).isoformat()
        row = {
            "time_utc": now,
            "login": getattr(acct, "login", None),
            "balance": getattr(acct, "balance", None),
            "equity": getattr(acct, "equity", None),
            "margin": getattr(acct, "margin", None),
            "free_margin": getattr(acct, "margin_free", None),
            "margin_level": getattr(acct, "margin_level", None),
            "pos": state.get("pos", 0),
            "beta_entry": state.get("beta_entry", None),
        }
        hdr = not os.path.exists(MON_ACCOUNT_CSV)
        pd.DataFrame([row]).to_csv(MON_ACCOUNT_CSV, mode="a", header=hdr, index=False)
    except Exception as e:
        log("account snapshot error:", e)

def write_spread_snapshot(ts, spread, sma, upper, lower, z, pos):
    try:
        row = {
            "time_utc": ts.isoformat(),
            "spread": spread,
            "sma": sma,
            "upper": upper,
            "lower": lower,
            "z": z,
            "pos": pos,
        }
        hdr = not os.path.exists(MON_SPREAD_CSV)
        pd.DataFrame([row]).to_csv(MON_SPREAD_CSV, mode="a", header=hdr, index=False)
    except Exception as e:
        log("spread snapshot error:", e)

# =========================
# MAIN LOOP
# =========================
def main():
    ensure_mt5()
    # make sure symbols are visible
    for s in (SYM1, SYM2):
        mt5.symbol_select(s, True)

    state = load_state()
    last_seen_open = None

    acct = mt5.account_info()
    acct_login = acct.login if acct else "N/A"
    log(f"START | {SYM1} vs {SYM2} | TF={TF} | Account={acct_login} | "
        f"WIN(beta/band)={BETA_WINDOW}/{BAND_WINDOW} | Z_IN/Z_OUT={Z_IN}/{Z_OUT} | UNITS={UNITS}")

    while True:
        try:
            # quick account snapshot (every loop)
            write_account_snapshot(state)

            # detect new bar
            bar_open = get_last_bar_open_time(SYM1, TF)
            if bar_open is None:
                time.sleep(POLL_SEC); 
                continue

            if last_seen_open is None:
                last_seen_open = bar_open
                time.sleep(POLL_SEC)
                continue

            if bar_open != last_seen_open:
                # NEW BAR: compute on last closed bar (index -2), act now 
                last_seen_open = bar_open

                s1 = fetch_bars(SYM1, TF, LOOKBACK)
                s2 = fetch_bars(SYM2, TF, LOOKBACK)
                df = align_two(s1, s2)

                if len(df) < (BETA_WINDOW + BAND_WINDOW + 5):
                    log("not enough history:", len(df))
                    time.sleep(POLL_SEC)
                    continue

                y = df[SYM1].astype(float)
                x = df[SYM2].astype(float)

                # OLS on past-only points up to last closed bar: y[:-1], x[:-1]
                alpha, beta = rolling_ols_last(y.iloc[:-1], x.iloc[:-1], BETA_WINDOW)
                if alpha is None or beta is None or beta == 0:
                    log("beta invalid")
                    time.sleep(POLL_SEC)
                    continue

                # residual series over the last BAND_WINDOW bars ending at last closed
                res_series = y.iloc[-BAND_WINDOW-1:-1] - (alpha + beta * x.iloc[-BAND_WINDOW-1:-1])
                z, sma, std = calc_z_for_last(res_series, BAND_WINDOW)
                if z is None:
                    log("std invalid => skip")
                    time.sleep(POLL_SEC)
                    continue

                # monitor snapshot for the last CLOSED bar
                last_closed_ts = df["time"].iloc[-2]
                last_resid = float(y.iloc[-2] - (alpha + beta*x.iloc[-2]))
                upper = float(sma + 2.0*std)
                lower = float(sma - 2.0*std)
                write_spread_snapshot(last_closed_ts, last_resid, float(sma), upper, lower, float(z), int(state.get("pos",0)))

                # decisions (use z from last closed)
                pos = int(state.get("pos", 0))

                # EXIT first
                if pos != 0 and abs(z) < Z_OUT:
                    if close_positions_for([SYM1, SYM2], MAGIC):
                        pnl_spread = (last_resid - (state.get("entry_spread") or 0.0)) * pos if state.get("entry_spread") is not None else None
                        log(f"EXIT pos={pos} | z={z:.2f} | beta_entry={state.get('beta_entry')} | pnl_spread={pnl_spread}")
                        state = {"pos":0, "beta_entry":None, "entry_spread":None}
                        save_state(state)
                    time.sleep(POLL_SEC)
                    continue

                # ENTER
                if pos == 0:
                    if z >= Z_IN:
                        # short spread
                        if enter_spread(UNITS, beta, -1):
                            entry_spread = last_resid
                            state = {"pos":-1, "beta_entry":float(beta), "entry_spread":float(entry_spread)}
                            save_state(state)
                            log(f"ENTER SHORT | z={z:.2f} | beta={beta:.4f} | entry_spread={entry_spread:.4f}")
                    elif z <= -Z_IN:
                        # long spread
                        if enter_spread(UNITS, beta, +1):
                            entry_spread = last_resid
                            state = {"pos":+1, "beta_entry":float(beta), "entry_spread":float(entry_spread)}
                            save_state(state)
                            log(f"ENTER LONG  | z={z:.2f} | beta={beta:.4f} | entry_spread={entry_spread:.4f}")

            time.sleep(POLL_SEC)

        except Exception as e:
            log("ERROR:", e)
            # try to reinitialize MT5 if it dropped
            try:
                mt5.shutdown()
            except: pass
            time.sleep(3)
            try:
                ensure_mt5()
                for s in (SYM1, SYM2): mt5.symbol_select(s, True)
                log("reconnected to MT5")
            except Exception as e2:
                log("reconnect failed:", e2)
                time.sleep(10)

if __name__ == "__main__":
    main() 