from flask import Flask, jsonify, request, redirect, session
from flask_cors import CORS
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
import os
import json
import time

app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get('FLASK_SECRET', 'atr-scanner-secret-key-2024')

# ========================================
# 🔑 FILL YOUR UPSTOX CREDENTIALS
# ========================================
API_KEY = os.environ.get('API_KEY', 'dd06178d-b9a1-4854-b9fc-1bde72620f86')
API_SECRET = os.environ.get('API_SECRET', 'un701txcrg')
REDIRECT_URI = "https://profitmaster.onrender.com/callback"

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
        'lot_size': 65
    },
    'BANKNIFTY': {
        'instrument_key': 'NSE_INDEX|Nifty Bank',
        'timeframe': '1minute',
        'resample_minutes': 5,
        'fast_period': 5,
        'fast_mult': 0.7,
        'slow_period': 20,
        'slow_mult': 3.5,
        'lot_size': 30
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

def save_token(access_token):
    token_data['access_token'] = access_token
    token_data['token_time'] = datetime.now(IST).isoformat()
    with open('/tmp/token.json', 'w') as f:
        json.dump(token_data, f)

def load_token():
    try:
        with open('/tmp/token.json', 'r') as f:
            data = json.load(f)
            token_data['access_token'] = data.get('access_token')
            token_data['token_time'] = data.get('token_time')
    except:
        pass

load_token()
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
                'grant_type': 'authorization_code',
                'code': code,
                'client_id': API_KEY,
                'client_secret': API_SECRET,
                'redirect_uri': REDIRECT_URI
            },
            headers={
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept': 'application/json'
            }
        )

        if r.status_code == 200:
            data = r.json()
            save_token(data['access_token'])
            return '''
            <html><body style="font-family:sans-serif;text-align:center;padding:50px;background:#1a2a4a;color:white">
            <h1>✅ Token Refreshed!</h1>
            <p>ATR Scanner is ready.</p>
            <a href="/" style="color:#22c55e;font-size:18px">← Go to Scanner</a>
            </body></html>
            '''
        else:
            return jsonify({'error': 'Token exchange failed', 'details': r.text}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ========================================
# CORE: ATR TRAILING STOP CALCULATOR
# ========================================
def calculate_atr_trailing(df, fast_period, fast_mult, slow_period, slow_mult):
    df = df.copy()
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    n = len(df)

    if n < max(fast_period, slow_period) + 5:
        return df

    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )

    # Fast ATR
    fast_atr = np.zeros(n)
    if n >= fast_period:
        fast_atr[fast_period - 1] = np.mean(tr[:fast_period])
        for i in range(fast_period, n):
            fast_atr[i] = (fast_atr[i - 1] * (fast_period - 1) + tr[i]) / fast_period
    fast_sl = fast_mult * fast_atr

    # Slow ATR
    slow_atr = np.zeros(n)
    if n >= slow_period:
        slow_atr[slow_period - 1] = np.mean(tr[:slow_period])
        for i in range(slow_period, n):
            slow_atr[i] = (slow_atr[i - 1] * (slow_period - 1) + tr[i]) / slow_period
    slow_sl = slow_mult * slow_atr

    # Fast Trail
    trail1 = np.zeros(n)
    for i in range(1, n):
        sc = close[i]
        pt = trail1[i - 1]
        ps = close[i - 1]
        if sc > pt and ps > pt:
            trail1[i] = max(pt, sc - fast_sl[i])
        elif sc < pt and ps < pt:
            trail1[i] = min(pt, sc + fast_sl[i])
        elif sc > pt:
            trail1[i] = sc - fast_sl[i]
        else:
            trail1[i] = sc + fast_sl[i]

    # Slow Trail
    trail2 = np.zeros(n)
    for i in range(1, n):
        sc = close[i]
        pt = trail2[i - 1]
        ps = close[i - 1]
        if sc > pt and ps > pt:
            trail2[i] = max(pt, sc - slow_sl[i])
        elif sc < pt and ps < pt:
            trail2[i] = min(pt, sc + slow_sl[i])
        elif sc > pt:
            trail2[i] = sc - slow_sl[i]
        else:
            trail2[i] = sc + slow_sl[i]

    df['trail1'] = trail1
    df['trail2'] = trail2
    df['fast_atr'] = fast_atr
    df['slow_atr'] = slow_atr

    # Buy/Sell signals
    df['buy_signal'] = False
    df['sell_signal'] = False
    for i in range(1, n):
        if trail1[i] > trail2[i] and trail1[i - 1] <= trail2[i - 1]:
            df.iloc[i, df.columns.get_loc('buy_signal')] = True
        if trail1[i] < trail2[i] and trail1[i - 1] >= trail2[i - 1]:
            df.iloc[i, df.columns.get_loc('sell_signal')] = True

    # Color conditions
    df['bar_color'] = 'neutral'
    for i in range(n):
        if trail1[i] > trail2[i] and close[i] > trail2[i] and low[i] > trail2[i]:
            df.iloc[i, df.columns.get_loc('bar_color')] = 'green'
        elif trail1[i] > trail2[i] and close[i] > trail2[i] and low[i] < trail2[i]:
            df.iloc[i, df.columns.get_loc('bar_color')] = 'blue'
        elif trail2[i] > trail1[i] and close[i] < trail2[i] and high[i] < trail2[i]:
            df.iloc[i, df.columns.get_loc('bar_color')] = 'red'
        elif trail2[i] > trail1[i] and close[i] < trail2[i] and high[i] > trail2[i]:
            df.iloc[i, df.columns.get_loc('bar_color')] = 'yellow'

    df['regime'] = np.where(trail1 > trail2, 'BULL', 'BEAR')

    return df


# ========================================
# DATA FETCHING
# ========================================
def get_headers():
    if not token_data['access_token']:
        return None
    return {
        'Authorization': f"Bearer {token_data['access_token']}",
        'Accept': 'application/json'
    }


def fetch_candles(instrument_key, interval='1minute', days=5):
    headers = get_headers()
    if not headers:
        return pd.DataFrame()

    encoded = requests.utils.quote(instrument_key, safe='')
    all_candles = []
    end_date = datetime.now(IST)
    start_date = end_date - timedelta(days=days)

    current_to = end_date
    while current_to >= start_date:
        current_from = max(start_date, current_to - timedelta(days=30))
        url = (
            f"https://api.upstox.com/v2/historical-candle/{encoded}"
            f"/{interval}"
            f"/{current_to.strftime('%Y-%m-%d')}"
            f"/{current_from.strftime('%Y-%m-%d')}"
        )

        try:
            r = requests.get(url, headers=headers)
            if r.status_code == 200:
                candles = r.json().get('data', {}).get('candles', [])
                for c in candles:
                    all_candles.append({
                        'datetime': c[0],
                        'open': c[1],
                        'high': c[2],
                        'low': c[3],
                        'close': c[4],
                        'volume': c[5]
                    })
        except Exception as e:
            print(f"Fetch error: {e}")

        current_to = current_from - timedelta(days=1)
        time.sleep(0.2)

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').drop_duplicates(subset='datetime').reset_index(drop=True)

    # Market hours only
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['time_val'] = df['hour'] * 100 + df['minute']
    df = df[(df['time_val'] >= 915) & (df['time_val'] <= 1530)]
    df = df.drop(columns=['hour', 'minute', 'time_val'])

    return df.reset_index(drop=True)


def resample_candles(df_1m, minutes):
    if len(df_1m) == 0:
        return pd.DataFrame()

    df = df_1m.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')

    resampled = df.resample(f'{minutes}min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()

    # Market hours
    resampled['hour'] = resampled['datetime'].dt.hour
    resampled['minute'] = resampled['datetime'].dt.minute
    resampled['time_val'] = resampled['hour'] * 100 + resampled['minute']
    resampled = resampled[(resampled['time_val'] >= 915) & (resampled['time_val'] <= 1530)]
    resampled = resampled.drop(columns=['hour', 'minute', 'time_val'])

    return resampled.reset_index(drop=True)


# ========================================
# SIGNAL GENERATION
# ========================================
def generate_signals():
    now = datetime.now(IST)
    signals = []

    for symbol, config in SCANNER_CONFIG.items():
        try:
            # Fetch 1-minute data
            df_1m = fetch_candles(config['instrument_key'], '1minute', days=5)

            if len(df_1m) < 50:
                continue

            # Resample to required timeframe
            df = resample_candles(df_1m, config['resample_minutes'])

            if len(df) < max(config['fast_period'], config['slow_period']) + 10:
                continue

            # Calculate ATR trailing stops
            df = calculate_atr_trailing(
                df,
                config['fast_period'],
                config['fast_mult'],
                config['slow_period'],
                config['slow_mult']
            )

            # Get today's signals only
            today = now.date()
            df['date'] = pd.to_datetime(df['datetime']).dt.date
            today_df = df[df['date'] == today]

            if len(today_df) == 0:
                # Use latest data if market hasn't opened yet
                today_df = df.tail(20)

            # Check for signals
            for idx, row in today_df.iterrows():
                if row.get('buy_signal', False) or row.get('sell_signal', False):
                    direction = 'BUY-LONG' if row['buy_signal'] else 'SELL-SHORT'
                    entry = round(row['close'], 2)
                    trail2 = round(row['trail2'], 2)
                    trail1 = round(row['trail1'], 2)
                    fast_atr_val = round(row['fast_atr'], 2)
                    slow_atr_val = round(row['slow_atr'], 2)

                    # Calculate SL and targets
                    if direction == 'BUY-LONG':
                        sl = trail2
                        risk = entry - sl
                        target_1 = round(entry + risk * 1.5, 2)
                        target_2 = round(entry + risk * 2.5, 2)
                    else:
                        sl = trail2
                        risk = sl - entry
                        target_1 = round(entry - risk * 1.5, 2)
                        target_2 = round(entry - risk * 2.5, 2)

                    risk = abs(risk)
                    if risk == 0:
                        continue

                    reward = abs(target_2 - entry)
                    rr = round(reward / risk, 2) if risk > 0 else 0

                    # Confidence scoring
                    confidence = 0.5
                    bar_c = row.get('bar_color', 'neutral')

                    if direction == 'BUY-LONG':
                        if bar_c == 'green':
                            confidence += 0.2
                        elif bar_c == 'blue':
                            confidence += 0.1
                    else:
                        if bar_c == 'red':
                            confidence += 0.2
                        elif bar_c == 'yellow':
                            confidence += 0.1

                    if rr >= 2:
                        confidence += 0.1
                    if rr >= 3:
                        confidence += 0.1

                    confidence = min(confidence, 0.95)

                    # Grade
                    if confidence >= 0.8:
                        grade = 'A+'
                        grade_score = 95
                    elif confidence >= 0.7:
                        grade = 'A'
                        grade_score = 85
                    elif confidence >= 0.6:
                        grade = 'B'
                        grade_score = 70
                    else:
                        grade = 'C'
                        grade_score = 55

                    signal_time = pd.to_datetime(row['datetime'])

                    signals.append({
                        '_id': f"{symbol}_{signal_time.strftime('%Y%m%d_%H%M')}",
                        'symbol': symbol,
                        'direction': direction,
                        'model': 'ATR-TS',
                        'entry': entry,
                        'sl': sl,
                        'target_1': target_1,
                        'target_2': target_2,
                        'target': target_2,
                        'risk_reward': f"1:{rr}",
                        'confidence': round(confidence, 2),
                        'grade': grade,
                        'grade_score': grade_score,
                        'scan_date': signal_time.isoformat(),
                        'scan_time': signal_time.strftime('%H:%M'),
                        'trail1': trail1,
                        'trail2': trail2,
                        'fast_atr': fast_atr_val,
                        'slow_atr': slow_atr_val,
                        'bar_color': bar_c,
                        'regime': row.get('regime', 'UNKNOWN'),
                        'timeframe': f"{config['resample_minutes']}m",
                        'lot_size': config['lot_size'],
                        'scanner_type': 'atr_trailing',
                        'outcome': 'pending'
                    })

        except Exception as e:
            print(f"Error scanning {symbol}: {e}")
            continue

    # Sort by time descending
    signals.sort(key=lambda x: x.get('scan_date', ''), reverse=True)

    return signals


def get_scanner_status():
    now = datetime.now(IST)
    hour = now.hour
    minute = now.minute
    day = now.weekday()

    if not token_data['access_token']:
        return 'NO_TOKEN'

    if day >= 5:
        return 'MARKET_CLOSED'

    time_val = hour * 100 + minute
    if 915 <= time_val <= 1530:
        return 'ACTIVE'
    elif 900 <= time_val < 915:
        return 'PRE_MARKET'
    else:
        return 'MARKET_CLOSED'


# ========================================
# API ROUTES
# ========================================
@app.route('/')
def home():
    return '''
    <html><body style="font-family:sans-serif;text-align:center;padding:50px;background:#1a2a4a;color:white">
    <h1>⚡ ATR Trailing Stop Scanner</h1>
    <p>API is running</p>
    <p><a href="/refresh" style="color:#22c55e">🔑 Refresh Token</a></p>
    <p><a href="/api/status" style="color:#3b82f6">📊 API Status</a></p>
    <p><a href="/api/signals" style="color:#f59e0b">📡 Get Signals</a></p>
    </body></html>
    '''


@app.route('/api/status')
def api_status():
    now = datetime.now(IST)
    return jsonify({
        'status': 'success',
        'scanner_status': get_scanner_status(),
        'server_time_ist': now.isoformat(),
        'token_set': token_data['access_token'] is not None,
        'token_time': token_data.get('token_time'),
        'scanner_model': 'ATR Trailing Stop',
        'config': {
            sym: {
                'timeframe': f"{cfg['resample_minutes']}m",
                'fast': f"({cfg['fast_period']}, {cfg['fast_mult']})",
                'slow': f"({cfg['slow_period']}, {cfg['slow_mult']})"
            } for sym, cfg in SCANNER_CONFIG.items()
        }
    })


@app.route('/api/signals')
def api_signals():
    now = datetime.now(IST)
    status = get_scanner_status()

    if status == 'NO_TOKEN':
        return jsonify({
            'status': 'success',
            'scanner_status': 'NO_TOKEN',
            'signals': [],
            'timestamp': now.isoformat()
        })

    # Use cache if scanned within last 60 seconds
    if (scan_cache['last_scan'] and
            (now - scan_cache['last_scan']).total_seconds() < 60):
        return jsonify({
            'status': 'success',
            'scanner_status': status,
            'signals': scan_cache['signals'],
            'last_scan': scan_cache['last_scan'].isoformat(),
            'daily_trades': scan_cache.get('daily_trades', {}),
            'timestamp': now.isoformat()
        })

    # Run scan
    if status in ['ACTIVE', 'SCANNING', 'PRE_MARKET']:
        signals = generate_signals()
    else:
        signals = scan_cache.get('signals', [])

    scan_cache['signals'] = signals
    scan_cache['last_scan'] = now

    return jsonify({
        'status': 'success',
        'scanner_status': status,
        'signals': signals,
        'last_scan': now.isoformat(),
        'daily_trades': scan_cache.get('daily_trades', {}),
        'timestamp': now.isoformat()
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
            symbol = sig.get('symbol', '')
            config = SCANNER_CONFIG.get(symbol)
            entry = float(sig.get('entry', 0))
            sl = float(sig.get('sl', 0))
            t1 = float(sig.get('target_1', sig.get('target', 0)))
            t2 = float(sig.get('target_2', sig.get('target', 0)))
            direction = sig.get('direction', '')

            current_price = None

            # Try to get live price if token available
            if headers and config:
                try:
                    df_1m = fetch_candles(config['instrument_key'], '1minute', days=1)
                    if len(df_1m) > 0:
                        current_price = float(df_1m.iloc[-1]['close'])
                except:
                    pass

            # If no live price, use entry as fallback
            if not current_price:
                results.append({
                    '_id': sig.get('_id'),
                    'status': 'pending',
                    'exit_price': None,
                    'current_price': None,
                    'live_pnl_pct': 0,
                    'track_status': 'no_price_available'
                })
                continue

            status = 'open'
            exit_price = None

            if direction == 'BUY-LONG':
                if current_price >= t2:
                    status = 'target_hit'
                    exit_price = t2
                elif current_price <= sl:
                    status = 'stop_hit'
                    exit_price = sl
                pnl_pct = round((current_price - entry) / entry * 100, 2)
            else:
                if current_price <= t2:
                    status = 'target_hit'
                    exit_price = t2
                elif current_price >= sl:
                    status = 'stop_hit'
                    exit_price = sl
                pnl_pct = round((entry - current_price) / entry * 100, 2)

            results.append({
                '_id': sig.get('_id'),
                'status': status,
                'exit_price': exit_price,
                'current_price': current_price,
                'live_pnl_pct': pnl_pct,
                'track_status': 'tracked'
            })

        return jsonify({'status': 'success', 'results': results})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
