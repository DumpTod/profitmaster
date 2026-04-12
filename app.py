from flask import Flask, jsonify, request, redirect, session
from flask_cors import CORS
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
import calendar
import pytz
import os
import json
import time
from urllib.parse import quote

app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get('FLASK_SECRET', 'atr-scanner-secret-key-2024')

# ========================================
# 🔑 UPSTOX CREDENTIALS
# ========================================
API_KEY    = os.environ.get('API_KEY',    'dd06178d-b9a1-4854-b9fc-1bde72620f86')
API_SECRET = os.environ.get('API_SECRET', 'un701txcrg')
REDIRECT_URI = "https://profitmaster-4jdd.onrender.com/callback"

# ========================================
# Scanner Settings (from backtest)
# ========================================
SCANNER_CONFIG = {
    'NIFTY': {
        'instrument_key': 'NSE_INDEX|Nifty 50',
        'timeframe': '1minute',
        'resample_minutes': 15,
        'fast_period': 3,
        'fast_mult': 1.0,
        'slow_period': 25,
        'slow_mult': 2.0,
        'lot_size': 65,
        'strike_step': 50,
        'options_key': 'NSE_INDEX|Nifty 50'
    },
    'BANKNIFTY': {
        'instrument_key': 'NSE_INDEX|Nifty Bank',
        'timeframe': '1minute',
        'resample_minutes': 5,
        'fast_period': 5,
        'fast_mult': 0.7,
        'slow_period': 20,
        'slow_mult': 3.5,
        'lot_size': 30,
        'strike_step': 100,
        'options_key': 'NSE_INDEX|Nifty Bank'
    }
}

IST = pytz.timezone('Asia/Kolkata')

# Token storage
token_data = {
    'access_token': None,
    'token_time': None
}

# Cache
scan_cache = {
    'signals': [],
    'last_scan': None,
    'daily_trades': {}
}

options_cache = {
    'signals': [],
    'last_fetch': None
}

# ========================================
# EXPIRY LOGIC - 2026 NSE HOLIDAYS
# ========================================
TRADING_HOLIDAYS = {
    date(2026,1,26),  # Republic Day
    date(2026,3,3),   # Holi
    date(2026,3,26),  # Shri Ram Navami
    date(2026,3,31),  # Shri Mahavir Jayanti
    date(2026,4,3),   # Good Friday
    date(2026,4,14),  # Dr. Baba Saheb Ambedkar Jayanti
    date(2026,5,1),   # Maharashtra Day
    date(2026,5,28),  # Bakri Id
    date(2026,6,26),  # Muharram
    date(2026,9,14),  # Ganesh Chaturthi
    date(2026,10,2),  # Mahatma Gandhi Jayanti
    date(2026,10,20), # Dussehra
    date(2026,11,10), # Diwali-Balipratipada
    date(2026,11,24), # Prakash Gurpurb Sri Guru Nanak Dev
    date(2026,12,25), # Christmas
}

def is_trading_day(d):
    return d.weekday() < 5 and d not in TRADING_HOLIDAYS

def last_weekday_of_month(year, month, weekday):
    last_day = calendar.monthrange(year, month)[1]
    d = date(year, month, last_day)
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d

def get_monthly_expiry(symbol, year, month):
    """Last Tuesday for Nifty/BankNifty."""
    expiry = last_weekday_of_month(year, month, 1)  # Tuesday
    while not is_trading_day(expiry):
        expiry -= timedelta(days=1)
    return expiry

def get_active_expiry(symbol, signal_date=None):
    """Return active monthly expiry. Roll to next month in last 5 trading days."""
    if signal_date is None:
        signal_date = datetime.now(IST).date()
    if isinstance(signal_date, str):
        signal_date = date.fromisoformat(signal_date[:10])
    y, m = signal_date.year, signal_date.month
    expiry = get_monthly_expiry(symbol, y, m)
    td_left = sum(
        1 for i in range((expiry - signal_date).days + 1)
        if is_trading_day(signal_date + timedelta(days=i))
    )
    if td_left <= 5:
        if m < 12:
            expiry = get_monthly_expiry(symbol, y, m + 1)
        else:
            expiry = get_monthly_expiry(symbol, y + 1, 1)
    return expiry

def round_to_strike(price, step):
    return round(round(price / step) * step, 2)

# ========================================
# TOKEN
# ========================================
def save_token(access_token):
    token_data['access_token'] = access_token
    token_data['token_time'] = datetime.now(IST).isoformat()
    with open('/tmp/token.json', 'w') as f:
        json.dump(token_data, f)
    print(f"✓ Token saved at {datetime.now(IST).strftime('%H:%M:%S')}")

def load_token():
    try:
        with open('/tmp/token.json', 'r') as f:
            data = json.load(f)
            token_data['access_token'] = data.get('access_token')
            token_data['token_time']   = data.get('token_time')
        print(f"✓ Token loaded")
    except:
        print(f"✗ No saved token found")

load_token()

def get_headers():
    if not token_data['access_token']:
        return None
    return {
        'Authorization': f"Bearer {token_data['access_token']}",
        'Accept': 'application/json'
    }

# ========================================
# AUTH ROUTES
# ========================================
@app.route('/refresh')
def refresh_token():
    auth_url = (
        f"https://api.upstox.com/v2/login/authorization/dialog"
        f"?client_id={API_KEY}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&response_type=code"
    )
    return redirect(auth_url)


@app.route('/callback')
def callback():
    code = request.args.get('code')
    if not code:
        return jsonify({'error': 'No authorization code received'}), 400
    try:
        r = requests.post(
            'https://api.upstox.com/v2/login/authorization/token',
            data={
                'grant_type':    'authorization_code',
                'code':          code,
                'client_id':     API_KEY,
                'client_secret': API_SECRET,
                'redirect_uri':  REDIRECT_URI
            },
            headers={
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept':       'application/json'
            }
        )
        if r.status_code == 200:
            data = r.json()
            save_token(data['access_token'])
            return '''
            <html><body style="font-family:sans-serif;text-align:center;padding:50px;background:#1a2a4a;color:white">
            <h1>✅ Token Refreshed!</h1><p>ATR Scanner is ready.</p>
            <a href="/" style="color:#22c55e;font-size:18px">← Go to Scanner</a>
            </body></html>'''
        else:
            return jsonify({'error': 'Token exchange failed', 'details': r.text}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ========================================
# CORE: ATR TRAILING STOP CALCULATOR
# ========================================
def calculate_atr_trailing(df, fast_period, fast_mult, slow_period, slow_mult):
    df   = df.copy()
    high  = df['high'].values
    low   = df['low'].values
    close = df['close'].values
    n     = len(df)

    if n < max(fast_period, slow_period) + 5:
        return df

    # True Range
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))

    # RMA function
    def rma(arr, period):
        alpha = 1.0 / period
        a = np.zeros(n)
        if n < period:
            return a
        a[period-1] = arr[:period].mean()
        for i in range(period, n):
            a[i] = (a[i-1] * (period - 1) + arr[i]) / period
        return a

    fast_atr = rma(tr, fast_period) * fast_mult
    slow_atr = rma(tr, slow_period) * slow_mult

    # Trailing stop function
    def trail(atr_sl):
        t = np.zeros(n)
        for i in range(1, n):
            sc = close[i]
            pt = t[i-1]
            ps = close[i-1]
            if sc > pt and ps > pt:
                t[i] = max(pt, sc - atr_sl[i])
            elif sc < pt and ps < pt:
                t[i] = min(pt, sc + atr_sl[i])
            elif sc > pt:
                t[i] = sc - atr_sl[i]
            else:
                t[i] = sc + atr_sl[i]
        return t

    trail1 = trail(fast_atr)
    trail2 = trail(slow_atr)

    df['trail1']   = trail1
    df['trail2']   = trail2
    df['fast_atr'] = fast_atr / fast_mult
    df['slow_atr'] = slow_atr / slow_mult

    # Generate signals
    buy  = np.zeros(n, bool)
    sell = np.zeros(n, bool)
    for i in range(1, n):
        if trail1[i] > trail2[i] and trail1[i-1] <= trail2[i-1]:
            buy[i] = True
        if trail1[i] < trail2[i] and trail1[i-1] >= trail2[i-1]:
            sell[i] = True

    df['buy_signal']  = buy
    df['sell_signal'] = sell

    # Bar color
    bar_color = []
    for i in range(n):
        if trail1[i] > trail2[i] and close[i] > trail2[i] and low[i] > trail2[i]:
            bar_color.append('green')
        elif trail1[i] > trail2[i] and close[i] > trail2[i] and low[i] < trail2[i]:
            bar_color.append('blue')
        elif trail2[i] > trail1[i] and close[i] < trail2[i] and high[i] < trail2[i]:
            bar_color.append('red')
        elif trail2[i] > trail1[i] and close[i] < trail2[i] and high[i] > trail2[i]:
            bar_color.append('yellow')
        else:
            bar_color.append('neutral')

    df['bar_color'] = bar_color
    df['regime']    = np.where(trail1 > trail2, 'BULL', 'BEAR')
    return df


# ========================================
# DATA FETCHING
# ========================================
def fetch_candles(instrument_key, timeframe='1minute', days=5):
    headers = get_headers()
    if not headers:
        return pd.DataFrame()

    end_date   = datetime.now(IST)
    start_date = end_date - timedelta(days=days)

    url = 'https://api.upstox.com/v2/historical-candle/intraday'
    params = {
        'instrument_key': instrument_key,
        'interval':       timeframe,
        'from_date':      start_date.strftime('%Y-%m-%d'),
        'to_date':        end_date.strftime('%Y-%m-%d')
    }

    try:
        r = requests.get(url, headers=headers, params=params)
        if r.status_code != 200:
            print(f"✗ Fetch failed for {instrument_key}: {r.status_code}")
            return pd.DataFrame()

        data = r.json().get('data', {})
        candles = data.get('candles', [])
        if not candles:
            print(f"✗ No candles for {instrument_key}")
            return pd.DataFrame()

        rows = []
        for c in candles:
            # c = [timestamp_str, open, high, low, close, volume, oi]
            dt = pd.to_datetime(c[0])
            if dt.tzinfo is not None:
                dt = dt.tz_convert(IST).tz_localize(None)
            rows.append({
                'datetime': dt,
                'open':     float(c[1]),
                'high':     float(c[2]),
                'low':      float(c[3]),
                'close':    float(c[4]),
                'volume':   int(c[5])
            })

        df = pd.DataFrame(rows)
        df = df.sort_values('datetime').drop_duplicates('datetime').reset_index(drop=True)
        
        # Filter market hours only
        df['time_val'] = df['datetime'].dt.hour * 100 + df['datetime'].dt.minute
        df = df[(df['time_val'] >= 915) & (df['time_val'] <= 1530)].copy()
        df = df.drop('time_val', axis=1).reset_index(drop=True)
        
        print(f"✓ Fetched {len(df)} candles for {instrument_key}")
        return df

    except Exception as e:
        print(f"✗ Fetch error for {instrument_key}: {e}")
        return pd.DataFrame()


def resample_candles(df_1m, minutes):
    if len(df_1m) == 0:
        return pd.DataFrame()
    
    df = df_1m.copy().set_index('datetime')
    resampled = df.resample(f'{minutes}min').agg({
        'open':   'first',
        'high':   'max',
        'low':    'min',
        'close':  'last',
        'volume': 'sum'
    }).dropna().reset_index()
    
    # Filter market hours
    resampled['time_val'] = resampled['datetime'].dt.hour * 100 + resampled['datetime'].dt.minute
    resampled = resampled[(resampled['time_val'] >= 915) & (resampled['time_val'] <= 1530)].copy()
    resampled = resampled.drop('time_val', axis=1).reset_index(drop=True)
    
    return resampled


# ========================================
# SIGNAL GENERATION - TIMEZONE FIXED
# ========================================
def generate_signals():
    now     = datetime.now(IST)
    signals = []

    print(f"\n🔍 SCAN: {now.strftime('%Y-%m-%d %H:%M:%S')} | Token: {'✓' if token_data['access_token'] else '✗'}")

    for symbol, config in SCANNER_CONFIG.items():
        try:
            df_1m = fetch_candles(config['instrument_key'], '1minute', days=5)
            if len(df_1m) < 50:
                print(f"  {symbol}: Insufficient data ({len(df_1m)} candles)")
                continue

            df = resample_candles(df_1m, config['resample_minutes'])
            if len(df) < max(config['fast_period'], config['slow_period']) + 10:
                print(f"  {symbol}: Insufficient resampled data ({len(df)} candles)")
                continue

            df = calculate_atr_trailing(
                df, config['fast_period'], config['fast_mult'],
                config['slow_period'], config['slow_mult']
            )

            # Remove last incomplete candle
            if len(df) > 0:
                df = df.iloc[:-1].copy()
            
            # Filter today's signals only
            today = now.date()
            df['date'] = pd.to_datetime(df['datetime']).dt.date
            today_df = df[df['date'] == today].copy()

            if len(today_df) == 0:
                print(f"  {symbol}: No candles for today, using tail(20)")
                today_df = df.tail(20).copy()

            signal_count = 0
            for idx, row in today_df.iterrows():
                if not (row.get('buy_signal', False) or row.get('sell_signal', False)):
                    continue

                direction     = 'BUY-LONG' if row['buy_signal'] else 'SELL-SHORT'
                entry         = round(float(row['close']), 2)
                trail2        = round(float(row['trail2']), 2)
                trail1        = round(float(row['trail1']), 2)
                fast_atr_val  = round(float(row['fast_atr']), 2)
                slow_atr_val  = round(float(row['slow_atr']), 2)

                if direction == 'BUY-LONG':
                    sl       = trail2
                    risk     = entry - sl
                    target_1 = round(entry + risk * 1.5, 2)
                    target_2 = round(entry + risk * 2.5, 2)
                else:
                    sl       = trail2
                    risk     = sl - entry
                    target_1 = round(entry - risk * 1.5, 2)
                    target_2 = round(entry - risk * 2.5, 2)

                risk = abs(risk)
                if risk == 0:
                    continue

                reward     = abs(target_2 - entry)
                rr         = round(reward / risk, 2) if risk > 0 else 0
                confidence = 0.5
                bar_c      = row.get('bar_color', 'neutral')

                if direction == 'BUY-LONG':
                    if bar_c == 'green': confidence += 0.2
                    elif bar_c == 'blue': confidence += 0.1
                else:
                    if bar_c == 'red':    confidence += 0.2
                    elif bar_c == 'yellow': confidence += 0.1

                if rr >= 2: confidence += 0.1
                if rr >= 3: confidence += 0.1
                confidence = min(confidence, 0.95)

                if confidence >= 0.8:   grade, grade_score = 'A+', 95
                elif confidence >= 0.7: grade, grade_score = 'A',  85
                elif confidence >= 0.6: grade, grade_score = 'B',  70
                else:                   grade, grade_score = 'C',  55

                # TIMEZONE FIX: Convert to naive datetime, then localize to IST
                signal_time = pd.to_datetime(row['datetime'])
                if isinstance(signal_time, pd.Timestamp):
                    if signal_time.tzinfo is not None:
                        signal_time = signal_time.tz_localize(None)
                    signal_time = IST.localize(signal_time)

                signals.append({
                    '_id':          f"{symbol}_{signal_time.strftime('%Y%m%d_%H%M')}",
                    'symbol':       symbol,
                    'direction':    direction,
                    'model':        'ATR-TS',
                    'entry':        entry,
                    'sl':           sl,
                    'target_1':     target_1,
                    'target_2':     target_2,
                    'target':       target_2,
                    'risk_reward':  f"1:{rr}",
                    'confidence':   round(confidence, 2),
                    'grade':        grade,
                    'grade_score':  grade_score,
                    'scan_date':    signal_time.isoformat(),
                    'scan_time':    signal_time.strftime('%H:%M'),
                    'trail1':       trail1,
                    'trail2':       trail2,
                    'fast_atr':     fast_atr_val,
                    'slow_atr':     slow_atr_val,
                    'bar_color':    bar_c,
                    'regime':       row.get('regime', 'UNKNOWN'),
                    'timeframe':    f"{config['resample_minutes']}m",
                    'lot_size':     config['lot_size'],
                    'scanner_type': 'atr_trailing',
                    'outcome':      'pending'
                })
                signal_count += 1
                print(f"  ✓ {symbol} {direction} @ {signal_time.strftime('%H:%M')} | Entry: {entry} | Grade: {grade}")

            if signal_count == 0:
                print(f"  {symbol}: No signals found")

        except Exception as e:
            print(f"✗ Error scanning {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue

    signals.sort(key=lambda x: x.get('scan_date', ''), reverse=True)

    # Merge with previously cached signals from today
    existing = scan_cache.get('signals', [])
    existing_ids = {s['_id'] for s in signals}
    today_str = datetime.now(IST).strftime('%Y-%m-%d')
    for s in existing:
        if s['_id'] not in existing_ids and s.get('scan_date','')[:10] == today_str:
            signals.append(s)

    signals.sort(key=lambda x: x.get('scan_date', ''), reverse=True)
    print(f"✓ Total signals: {len(signals)}\n")
    return signals


# ========================================
# OPTION CHAIN - CACHE
# ========================================
_option_contracts_cache = {}

def fetch_option_contracts(symbol, expiry_date):
    """Fetch option chain from Upstox and cache."""
    cache_key = f"{symbol}_{expiry_date}"
    if cache_key in _option_contracts_cache:
        return _option_contracts_cache[cache_key]

    headers = get_headers()
    if not headers:
        return {}

    config = SCANNER_CONFIG.get(symbol, {})
    url = 'https://api.upstox.com/v2/option/chain'
    params = {
        'instrument_key': config.get('options_key', ''),
        'expiry_date':    expiry_date.strftime('%Y-%m-%d')
    }

    try:
        r = requests.get(url, headers=headers, params=params)
        if r.status_code != 200:
            return {}

        data = r.json().get('data', [])
        contracts = {}
        for item in data:
            strike = item.get('strike_price')
            ce = item.get('call_options', {})
            pe = item.get('put_options', {})
            if ce:
                contracts[(float(strike), 'CE')] = {
                    'ltp': ce.get('market_data', {}).get('ltp', 0),
                    'key': ce.get('instrument_key', '')
                }
            if pe:
                contracts[(float(strike), 'PE')] = {
                    'ltp': pe.get('market_data', {}).get('ltp', 0),
                    'key': pe.get('instrument_key', '')
                }

        _option_contracts_cache[cache_key] = contracts
        return contracts

    except Exception as e:
        print(f"Option chain fetch error: {e}")
        return {}


def get_option_ltp(symbol, spot_price, option_type, expiry_date, otm=False):
    """Find option strike and LTP. Returns (ltp, strike, key)."""
    contracts = fetch_option_contracts(symbol, expiry_date)
    if not contracts:
        return None, None, None

    config = SCANNER_CONFIG.get(symbol, {})
    step = config.get('strike_step', 50)
    strike = spot_price

    atm_strike = round_to_strike(spot_price, step)
    entry = contracts.get((float(strike), option_type))
    if not entry:
        candidates = [(s, t) for (s, t) in contracts if t == option_type]
        if otm:
            candidates = [(s, t) for (s, t) in candidates if s != float(atm_strike)]
        if not candidates:
            return None, strike, None
        closest = min(candidates, key=lambda x: abs(x[0] - strike))
        strike  = closest[0]
        entry   = contracts[(closest[0], option_type)]

    return entry.get('ltp'), strike, entry.get('key')


def generate_option_signals(futures_signals):
    """For each futures signal, fetch ATM + OTM option LTP."""
    now = datetime.now(IST)
    results = []

    for sig in futures_signals:
        symbol = sig.get('symbol', '')
        config = SCANNER_CONFIG.get(symbol)
        if not config:
            continue

        direction  = sig.get('direction', '')
        opt_type   = 'CE' if direction == 'BUY-LONG' else 'PE'
        spot       = float(sig.get('entry', 0))
        tp1        = float(sig.get('target_1', 0))
        lot        = config['lot_size']
        expiry     = get_active_expiry(symbol, now.date())
        days_left  = (expiry - now.date()).days

        ltp_atm, strike_atm, key_atm = get_option_ltp(symbol, tp1, opt_type, expiry, otm=False)
        ltp_otm, strike_otm, key_otm = get_option_ltp(symbol, tp1, opt_type, expiry, otm=True)

        results.append({
            '_id':           sig['_id'] + '_OPT',
            'futures_id':    sig['_id'],
            'symbol':        symbol,
            'direction':     direction,
            'opt_type':      opt_type,
            'action':        'BUY ' + opt_type,
            'spot':          spot,
            'tp1':           tp1,
            'atm_strike':    strike_atm,
            'atm_ltp':       round(ltp_atm, 2) if ltp_atm else None,
            'atm_key':       key_atm,
            'otm_strike':    strike_otm,
            'otm_ltp':       round(ltp_otm, 2) if ltp_otm else None,
            'otm_key':       key_otm,
            'expiry':        expiry.strftime('%d %b %Y'),
            'days_to_expiry':days_left,
            'lot_size':      lot,
            'max_risk_atm':  round(ltp_atm * lot, 0) if ltp_atm else None,
            'max_risk_otm':  round(ltp_otm * lot, 0) if ltp_otm else None,
            'scan_date':     sig.get('scan_date', ''),
            'scan_time':     sig.get('scan_time', ''),
            'grade':         sig.get('grade', ''),
            'grade_score':   sig.get('grade_score', 0),
            'confidence':    sig.get('confidence', 0),
        })

    return results


def get_scanner_status():
    now      = datetime.now(IST)
    hour     = now.hour
    minute   = now.minute
    day      = now.weekday()
    today    = now.date()

    if not token_data['access_token']:
        return 'NO_TOKEN'
    if day >= 5:
        return 'MARKET_CLOSED'
    if today in TRADING_HOLIDAYS:
        return 'MARKET_CLOSED'

    time_val = hour * 100 + minute
    if 915 <= time_val <= 1530:   return 'ACTIVE'
    elif 900 <= time_val < 915:   return 'PRE_MARKET'
    else:                          return 'MARKET_CLOSED'


# ========================================
# API ROUTES
# ========================================
@app.route('/')
def home():
    return '''
    <html><body style="font-family:sans-serif;text-align:center;padding:50px;background:#1a2a4a;color:white">
    <h1>⚡ ATR Trailing Stop Scanner</h1><p>API is running</p>
    <p><a href="/refresh" style="color:#22c55e">🔑 Refresh Token</a></p>
    <p><a href="/api/status" style="color:#3b82f6">📊 API Status</a></p>
    <p><a href="/api/signals" style="color:#f59e0b">📡 Get Signals</a></p>
    <p><a href="/api/option-signals" style="color:#a78bfa">🎯 Option Signals</a></p>
    </body></html>'''


@app.route('/api/status')
def api_status():
    now = datetime.now(IST)
    return jsonify({
        'status':         'success',
        'scanner_status': get_scanner_status(),
        'server_time_ist':now.isoformat(),
        'token_set':      token_data['access_token'] is not None,
        'token_time':     token_data.get('token_time'),
        'scanner_model':  'ATR Trailing Stop',
        'config': {
            sym: {
                'timeframe': f"{cfg['resample_minutes']}m",
                'fast':      f"({cfg['fast_period']}, {cfg['fast_mult']})",
                'slow':      f"({cfg['slow_period']}, {cfg['slow_mult']})"
            } for sym, cfg in SCANNER_CONFIG.items()
        }
    })


@app.route('/api/signals')
def api_signals():
    now    = datetime.now(IST)
    status = get_scanner_status()

    if status == 'NO_TOKEN':
        return jsonify({'status': 'success', 'scanner_status': 'NO_TOKEN',
                        'signals': [], 'timestamp': now.isoformat()})

    if (scan_cache['last_scan'] and
            (now - scan_cache['last_scan']).total_seconds() < 60):
        return jsonify({
            'status':        'success',
            'scanner_status': status,
            'signals':        scan_cache['signals'],
            'last_scan':      scan_cache['last_scan'].isoformat(),
            'daily_trades':   scan_cache.get('daily_trades', {}),
            'timestamp':      now.isoformat(),
            'cached':         True
        })

    signals = generate_signals() if status in ['ACTIVE', 'SCANNING', 'PRE_MARKET'] else scan_cache.get('signals', [])
    scan_cache['signals']   = signals
    scan_cache['last_scan'] = now

    return jsonify({
        'status':        'success',
        'scanner_status': status,
        'signals':        signals,
        'last_scan':      now.isoformat(),
        'daily_trades':   scan_cache.get('daily_trades', {}),
        'timestamp':      now.isoformat(),
        'cached':         False
    })


@app.route('/api/option-signals')
def api_option_signals():
    """Returns ATM + OTM option data for each active futures signal."""
    now    = datetime.now(IST)
    status = get_scanner_status()

    if status == 'NO_TOKEN':
        return jsonify({'status': 'success', 'scanner_status': 'NO_TOKEN',
                        'option_signals': [], 'timestamp': now.isoformat()})

    if (options_cache['last_fetch'] and
            (now - options_cache['last_fetch']).total_seconds() < 120):
        return jsonify({
            'status':         'success',
            'scanner_status': status,
            'option_signals': options_cache['signals'],
            'last_fetch':     options_cache['last_fetch'].isoformat(),
            'timestamp':      now.isoformat(),
            'cached':         True
        })

    # Always clear contract cache to get fresh LTPs
    _option_contracts_cache.clear()

    futures = generate_signals()
    scan_cache['signals']   = futures
    scan_cache['last_scan'] = now

    opt_signals = generate_option_signals(futures)
    options_cache['signals']    = opt_signals
    options_cache['last_fetch'] = now

    return jsonify({
        'status':         'success',
        'scanner_status': status,
        'option_signals': opt_signals,
        'last_fetch':     now.isoformat(),
        'timestamp':      now.isoformat(),
        'cached':         False
    })


@app.route('/api/track', methods=['POST'])
def api_track():
    try:
        data = request.json
        if not data or 'signals' not in data:
            return jsonify({'status': 'error', 'message': 'No signals provided'})

        headers = get_headers()
        results = []

        for sig in data['signals']:
            symbol    = sig.get('symbol', '')
            config    = SCANNER_CONFIG.get(symbol)
            entry     = float(sig.get('entry', 0))
            sl        = float(sig.get('sl', 0))
            t2        = float(sig.get('target_2', sig.get('target', 0)))
            direction = sig.get('direction', '')
            scan_date = sig.get('scan_date', '')

            if not headers or not config:
                results.append({'_id': sig.get('_id'), 'status': 'pending',
                                 'exit_price': None, 'current_price': None,
                                 'live_pnl_pct': 0, 'track_status': 'no_token'})
                continue

            try:
                signal_time = pd.to_datetime(scan_date).replace(tzinfo=None)
                df_1m       = fetch_candles(config['instrument_key'], '1minute', days=5)

                if len(df_1m) == 0:
                    results.append({'_id': sig.get('_id'), 'status': 'pending',
                                     'exit_price': None, 'current_price': None,
                                     'live_pnl_pct': 0, 'track_status': 'no_candles_fetched'})
                    continue

                df_1m['datetime'] = pd.to_datetime(df_1m['datetime'])
                if df_1m['datetime'].dt.tz is not None:
                    df_1m['datetime'] = df_1m['datetime'].dt.tz_localize(None)
                    
                df_after = df_1m[df_1m['datetime'] > signal_time].reset_index(drop=True)

                if len(df_after) == 0:
                    results.append({'_id': sig.get('_id'), 'status': 'pending',
                                     'exit_price': None, 'current_price': None,
                                     'live_pnl_pct': 0, 'track_status': 'no_candles_after_signal'})
                    continue

                entry_met  = False
                entry_idx  = None
                for idx, row in df_after.iterrows():
                    if direction == 'BUY-LONG' and row['high'] >= entry:
                        entry_met = True; entry_idx = idx; break
                    elif direction != 'BUY-LONG' and row['low'] <= entry:
                        entry_met = True; entry_idx = idx; break

                if not entry_met:
                    current_price = float(df_after.iloc[-1]['close'])
                    results.append({'_id': sig.get('_id'), 'status': 'pending',
                                     'exit_price': None, 'current_price': current_price,
                                     'live_pnl_pct': 0, 'track_status': 'entry_not_met'})
                    continue

                entry_pos     = df_after.index.get_loc(entry_idx)
                df_post_entry = df_after.iloc[entry_pos:].reset_index(drop=True)
                status_val    = 'open'
                exit_price    = None
                current_price = float(df_post_entry.iloc[-1]['close'])

                for idx, row in df_post_entry.iterrows():
                    if direction == 'BUY-LONG':
                        if row['high'] >= t2:  status_val = 'target_hit'; exit_price = t2; break
                        elif row['low'] <= sl: status_val = 'stop_hit';   exit_price = sl; break
                    else:
                        if row['low'] <= t2:   status_val = 'target_hit'; exit_price = t2; break
                        elif row['high'] >= sl:status_val = 'stop_hit';   exit_price = sl; break

                pnl_pct = round((current_price - entry) / entry * 100, 2) if direction == 'BUY-LONG' \
                     else round((entry - current_price) / entry * 100, 2)

                results.append({
                    '_id':          sig.get('_id'),
                    'status':       status_val,
                    'exit_price':   exit_price,
                    'current_price':current_price,
                    'live_pnl_pct': pnl_pct,
                    'track_status': 'tracked'
                })

            except Exception as e:
                print(f"Track error: {e}")
                results.append({'_id': sig.get('_id'), 'status': 'pending',
                                 'exit_price': None, 'current_price': None,
                                 'live_pnl_pct': 0, 'track_status': f'error:{str(e)}'})
                continue

        return jsonify({'status': 'success', 'results': results})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
# Store latest signals in memory
latest_signals = {
    "NIFTY": {"status": "No Signal", "timestamp": None},
    "BANKNIFTY": {"status": "No Signal", "timestamp": None},
    "SENSEX": {"status": "No Signal", "timestamp": None}
}

# Endpoint to receive signals from n8n
@app.route('/api/signals', methods=['POST'])
def receive_signal():
    try:
        data = request.get_json()
        instrument = data.get('instrument')
        
        if instrument in latest_signals:
            latest_signals[instrument] = {
                "status": data.get('status'),
                "timestamp": data.get('timestamp'),
                "entryPrice": data.get('entryPrice'),
                "stopLoss": data.get('stopLoss'),
                "target1": data.get('target1'),
                "target2": data.get('target2')
            }
            return jsonify({"success": True, "message": f"{instrument} signal received"}), 200
        else:
            return jsonify({"success": False, "message": "Invalid instrument"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# Endpoint for webpage to fetch signals
@app.route('/api/signals', methods=['GET'])
def get_signals():
    return jsonify(latest_signals)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n{'='*60}")
    print(f"🚀 PROFITMASTER UPSTOX SCANNER")
    print(f"{'='*60}")
    print(f"Port: {port}")
    print(f"Token: {'✓ Active' if token_data['access_token'] else '✗ Not Set'}")
    print(f"Time: {datetime.now(IST).strftime('%d %b %Y %H:%M:%S IST')}")
    print(f"{'='*60}\n")
    app.run(host='0.0.0.0', port=port, debug=False)
