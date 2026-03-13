from flask import Flask, jsonify, request, redirect
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
app.secret_key = os.environ.get('FLASK_SECRET', 'striketrail-secret-2025')

# ========================================
# CREDENTIALS
# ========================================
API_KEY      = os.environ.get('API_KEY', 'YOUR_UPSTOX_API_KEY')
API_SECRET   = os.environ.get('API_SECRET', 'YOUR_UPSTOX_SECRET')
REDIRECT_URI = os.environ.get('REDIRECT_URI', 'https://profitmaster.onrender.com/callback')

# ========================================
# VALIDATED SCANNER CONFIG
# Params from walk-forward backtested results:
#   BankNifty 5m  -> Sharpe 2.15 | OOS PF 1.34
#   Nifty50   5m  -> Sharpe 1.00 | OOS PF 1.13
# ========================================
SCANNER_CONFIG = {
    'NIFTY50': {
        'instrument_key'  : 'NSE_INDEX|Nifty 50',
        'timeframe'       : '1minute',
        'resample_minutes': 5,
        'fast_period'     : 5,
        'fast_mult'       : 1.5,
        'slow_period'     : 25,
        'slow_mult'       : 4.0,
        'lot_size'        : 75
    },
    'BANKNIFTY': {
        'instrument_key'  : 'NSE_INDEX|Nifty Bank',
        'timeframe'       : '1minute',
        'resample_minutes': 5,
        'fast_period'     : 5,
        'fast_mult'       : 1.5,
        'slow_period'     : 20,
        'slow_mult'       : 4.0,
        'lot_size'        : 30
    }
}

IST = pytz.timezone('Asia/Kolkata')

token_data = {'access_token': None, 'token_time': None}

scan_cache = {'signals': [], 'last_scan': None}

TOKEN_FILE = '/tmp/token.json'


def save_token(access_token):
    token_data['access_token'] = access_token
    token_data['token_time']   = datetime.now(IST).isoformat()
    try:
        with open(TOKEN_FILE, 'w') as f:
            json.dump(token_data, f)
    except Exception:
        pass


def load_token():
    try:
        with open(TOKEN_FILE, 'r') as f:
            data = json.load(f)
            token_data['access_token'] = data.get('access_token')
            token_data['token_time']   = data.get('token_time')
    except Exception:
        pass


load_token()


# ========================================
# AUTH
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
        return jsonify({'error': 'No code received'}), 400
    try:
        r = requests.post(
            'https://api.upstox.com/v2/login/authorization/token',
            data={
                'grant_type'   : 'authorization_code',
                'code'         : code,
                'client_id'    : API_KEY,
                'client_secret': API_SECRET,
                'redirect_uri' : REDIRECT_URI
            },
            headers={
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept'      : 'application/json'
            }
        )
        if r.status_code == 200:
            save_token(r.json()['access_token'])
            return '''
            <html><body style="font-family:sans-serif;text-align:center;padding:50px;
            background:#0f1f3d;color:white">
            <h1>Token Refreshed!</h1>
            <p>StrikeTrail scanner is ready.</p>
            <a href="/" style="color:#22c55e;font-size:18px">Go to Scanner</a>
            </body></html>
            '''
        return jsonify({'error': 'Token exchange failed', 'details': r.text}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ========================================
# CORE: ATR TRAILING STOP
# ========================================
def calculate_atr_trailing(df, fast_period, fast_mult, slow_period, slow_mult):
    df    = df.copy()
    hi    = df['high'].values
    lo    = df['low'].values
    cl    = df['close'].values
    n     = len(df)

    if n < max(fast_period, slow_period) + 5:
        return df

    # True Range
    tr    = np.empty(n)
    tr[0] = hi[0] - lo[0]
    for i in range(1, n):
        tr[i] = max(hi[i] - lo[i], abs(hi[i] - cl[i-1]), abs(lo[i] - cl[i-1]))

    # Wilder's ATR (RMA)
    def rma(arr, period):
        a = np.zeros(n)
        if n < period:
            return a
        a[period-1] = arr[:period].mean()
        for i in range(period, n):
            a[i] = (a[i-1] * (period - 1) + arr[i]) / period
        return a

    fast_atr = rma(tr, fast_period) * fast_mult
    slow_atr = rma(tr, slow_period) * slow_mult

    # Trailing stop bands
    def trail(atr_sl):
        t = np.zeros(n)
        for i in range(1, n):
            sc, pt, ps = cl[i], t[i-1], cl[i-1]
            if   sc > pt and ps > pt: t[i] = max(pt, sc - atr_sl[i])
            elif sc < pt and ps < pt: t[i] = min(pt, sc + atr_sl[i])
            elif sc > pt:             t[i] = sc - atr_sl[i]
            else:                     t[i] = sc + atr_sl[i]
        return t

    t1 = trail(fast_atr)
    t2 = trail(slow_atr)

    df['trail1']   = t1
    df['trail2']   = t2
    df['fast_atr'] = fast_atr / fast_mult   # raw ATR value
    df['slow_atr'] = slow_atr / slow_mult

    # Crossover signals
    buy  = np.zeros(n, bool)
    sell = np.zeros(n, bool)
    for i in range(1, n):
        if t1[i] > t2[i] and t1[i-1] <= t2[i-1]: buy[i]  = True
        if t1[i] < t2[i] and t1[i-1] >= t2[i-1]: sell[i] = True

    df['buy_signal']  = buy
    df['sell_signal'] = sell

    # Bar colour regime
    bar_color = []
    for i in range(n):
        if   t1[i] > t2[i] and cl[i] > t2[i] and lo[i] > t2[i]: bar_color.append('green')
        elif t1[i] > t2[i] and cl[i] > t2[i] and lo[i] < t2[i]: bar_color.append('blue')
        elif t2[i] > t1[i] and cl[i] < t2[i] and hi[i] < t2[i]: bar_color.append('red')
        elif t2[i] > t1[i] and cl[i] < t2[i] and hi[i] > t2[i]: bar_color.append('yellow')
        else:                                                       bar_color.append('neutral')

    df['bar_color'] = bar_color
    df['regime']    = np.where(t1 > t2, 'BULL', 'BEAR')
    return df


# ========================================
# DATA
# ========================================
def get_headers():
    if not token_data['access_token']:
        return None
    return {
        'Authorization': f"Bearer {token_data['access_token']}",
        'Accept'       : 'application/json'
    }


def fetch_candles(instrument_key, days=10):
    headers = get_headers()
    if not headers:
        return pd.DataFrame()

    encoded     = requests.utils.quote(instrument_key, safe='')
    all_candles = []
    end_date    = datetime.now(IST)
    start_date  = end_date - timedelta(days=days)
    current_to  = end_date

    while current_to >= start_date:
        current_from = max(start_date, current_to - timedelta(days=30))
        url = (
            f"https://api.upstox.com/v2/historical-candle/{encoded}"
            f"/1minute"
            f"/{current_to.strftime('%Y-%m-%d')}"
            f"/{current_from.strftime('%Y-%m-%d')}"
        )
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                candles = r.json().get('data', {}).get('candles', [])
                for c in candles:
                    all_candles.append({
                        'datetime': c[0],
                        'open'    : c[1],
                        'high'    : c[2],
                        'low'     : c[3],
                        'close'   : c[4],
                        'volume'  : c[5]
                    })
        except Exception as e:
            print(f"Fetch error: {e}")

        current_to = current_from - timedelta(days=1)
        time.sleep(0.2)

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').drop_duplicates('datetime').reset_index(drop=True)

    t = df['datetime'].dt.hour * 100 + df['datetime'].dt.minute
    df = df[(t >= 915) & (t <= 1530)].reset_index(drop=True)
    return df


def resample_candles(df_1m, minutes):
    if len(df_1m) == 0:
        return pd.DataFrame()
    df = df_1m.copy().set_index('datetime')
    r  = df.resample(f'{minutes}min').agg(
        open=('open','first'), high=('high','max'),
        low=('low','min'), close=('close','last'), volume=('volume','sum')
    ).dropna().reset_index()
    t  = r['datetime'].dt.hour * 100 + r['datetime'].dt.minute
    return r[(t >= 915) & (t <= 1530)].reset_index(drop=True)


# ========================================
# SIGNAL GENERATION
# ========================================
def generate_signals():
    now     = datetime.now(IST)
    today   = now.date()
    signals = []

    for symbol, config in SCANNER_CONFIG.items():
        try:
            df_1m = fetch_candles(config['instrument_key'], days=10)
            if len(df_1m) < 100:
                print(f"{symbol}: insufficient 1m candles ({len(df_1m)})")
                continue

            df = resample_candles(df_1m, config['resample_minutes'])
            min_bars = max(config['fast_period'], config['slow_period']) + 10
            if len(df) < min_bars:
                print(f"{symbol}: insufficient resampled bars ({len(df)})")
                continue

            df = calculate_atr_trailing(
                df,
                config['fast_period'],  config['fast_mult'],
                config['slow_period'],  config['slow_mult']
            )

            df['date'] = pd.to_datetime(df['datetime']).dt.date
            today_df   = df[df['date'] == today]

            if len(today_df) == 0:
                today_df = df.tail(30)

            for _, row in today_df.iterrows():
                if not (row.get('buy_signal', False) or row.get('sell_signal', False)):
                    continue

                direction     = 'BUY-LONG' if row['buy_signal'] else 'SELL-SHORT'
                entry         = round(float(row['close']),  2)
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

                reward = abs(target_2 - entry)
                rr     = round(reward / risk, 2)

                # Confidence score
                confidence = 0.5
                bar_c      = row.get('bar_color', 'neutral')
                if direction == 'BUY-LONG':
                    if bar_c == 'green':  confidence += 0.2
                    elif bar_c == 'blue': confidence += 0.1
                else:
                    if bar_c == 'red':      confidence += 0.2
                    elif bar_c == 'yellow': confidence += 0.1
                if rr >= 2: confidence += 0.1
                if rr >= 3: confidence += 0.1
                confidence = min(confidence, 0.95)

                if   confidence >= 0.8: grade, grade_score = 'A+', 95
                elif confidence >= 0.7: grade, grade_score = 'A',  85
                elif confidence >= 0.6: grade, grade_score = 'B',  70
                else:                   grade, grade_score = 'C',  55

                # FIX: use row datetime directly (was undefined before)
                signal_dt = pd.to_datetime(row['datetime'])
                if signal_dt.tzinfo is None:
                    signal_dt = IST.localize(signal_dt)
                else:
                    signal_dt = signal_dt.tz_convert(IST)

                signals.append({
                    '_id'        : f"{symbol}_{signal_dt.strftime('%Y%m%d_%H%M')}",
                    'symbol'     : symbol,
                    'direction'  : direction,
                    'model'      : 'ATR-TS',
                    'entry'      : entry,
                    'sl'         : sl,
                    'target_1'   : target_1,
                    'target_2'   : target_2,
                    'target'     : target_2,
                    'risk_reward': f"1:{rr}",
                    'confidence' : round(confidence, 2),
                    'grade'      : grade,
                    'grade_score': grade_score,
                    'scan_date'  : signal_dt.isoformat(),
                    'scan_time'  : signal_dt.strftime('%H:%M'),
                    'trail1'     : trail1,
                    'trail2'     : trail2,
                    'fast_atr'   : fast_atr_val,
                    'slow_atr'   : slow_atr_val,
                    'bar_color'  : bar_c,
                    'regime'     : row.get('regime', 'UNKNOWN'),
                    'timeframe'  : f"{config['resample_minutes']}m",
                    'lot_size'   : config['lot_size'],
                    'scanner_type': 'atr_trailing',
                    'outcome'    : 'pending'
                })

        except Exception as e:
            print(f"Error scanning {symbol}: {e}")
            continue

    signals.sort(key=lambda x: x.get('scan_date', ''), reverse=True)
    return signals


def get_scanner_status():
    now      = datetime.now(IST)
    time_val = now.hour * 100 + now.minute
    day      = now.weekday()

    if not token_data['access_token']:
        return 'NO_TOKEN'
    if day >= 5:
        return 'MARKET_CLOSED'
    if 915 <= time_val <= 1530:
        return 'ACTIVE'
    if 900 <= time_val < 915:
        return 'PRE_MARKET'
    return 'MARKET_CLOSED'


# ========================================
# ROUTES
# ========================================
@app.route('/')
def home():
    return '''
    <html><body style="font-family:sans-serif;text-align:center;padding:50px;
    background:#0f1f3d;color:white">
    <h1>StrikeTrail ATR Scanner</h1>
    <p style="color:#aaa">API running — v2 (Walk-Forward Validated)</p>
    <p><a href="/refresh" style="color:#22c55e">Refresh Token</a></p>
    <p><a href="/api/status" style="color:#3b82f6">API Status</a></p>
    <p><a href="/api/signals" style="color:#f59e0b">Get Signals</a></p>
    </body></html>
    '''


@app.route('/api/status')
def api_status():
    now = datetime.now(IST)
    return jsonify({
        'status'        : 'success',
        'scanner_status': get_scanner_status(),
        'server_time_ist': now.isoformat(),
        'token_set'     : token_data['access_token'] is not None,
        'token_time'    : token_data.get('token_time'),
        'scanner_model' : 'ATR Trailing Stop (Walk-Forward Validated)',
        'config'        : {
            sym: {
                'timeframe' : f"{cfg['resample_minutes']}m",
                'fast'      : f"({cfg['fast_period']}, {cfg['fast_mult']})",
                'slow'      : f"({cfg['slow_period']}, {cfg['slow_mult']})"
            } for sym, cfg in SCANNER_CONFIG.items()
        }
    })


@app.route('/api/signals')
def api_signals():
    now    = datetime.now(IST)
    status = get_scanner_status()

    if status == 'NO_TOKEN':
        return jsonify({
            'status'        : 'success',
            'scanner_status': 'NO_TOKEN',
            'signals'       : [],
            'timestamp'     : now.isoformat()
        })

    # Cache for 60 seconds
    if (scan_cache['last_scan'] and
            (now - scan_cache['last_scan']).total_seconds() < 60):
        return jsonify({
            'status'        : 'success',
            'scanner_status': status,
            'signals'       : scan_cache['signals'],
            'last_scan'     : scan_cache['last_scan'].isoformat(),
            'timestamp'     : now.isoformat()
        })

    if status in ['ACTIVE', 'PRE_MARKET']:
        signals = generate_signals()
    else:
        signals = scan_cache.get('signals', [])

    scan_cache['signals']   = signals
    scan_cache['last_scan'] = now

    return jsonify({
        'status'        : 'success',
        'scanner_status': status,
        'signals'       : signals,
        'last_scan'     : now.isoformat(),
        'timestamp'     : now.isoformat()
    })


@app.route('/api/track', methods=['POST'])
def api_track():
    try:
        data = request.json
        if not data or 'signals' not in data:
            return jsonify({'status': 'error', 'message': 'No signals'})

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
                df_1m       = fetch_candles(config['instrument_key'], days=10)

                if len(df_1m) == 0:
                    results.append({'_id': sig.get('_id'), 'status': 'pending',
                                    'exit_price': None, 'current_price': None,
                                    'live_pnl_pct': 0, 'track_status': 'no_data'})
                    continue

                df_1m['datetime'] = pd.to_datetime(df_1m['datetime']).dt.tz_localize(None)
                df_after = df_1m[df_1m['datetime'] > signal_time].reset_index(drop=True)

                if len(df_after) == 0:
                    results.append({'_id': sig.get('_id'), 'status': 'pending',
                                    'exit_price': None, 'current_price': None,
                                    'live_pnl_pct': 0, 'track_status': 'no_candles_after'})
                    continue

                # Find entry
                entry_met = False
                entry_idx = None
                for idx, row in df_after.iterrows():
                    if direction == 'BUY-LONG'  and row['high'] >= entry:
                        entry_met = True; entry_idx = idx; break
                    elif direction == 'SELL-SHORT' and row['low'] <= entry:
                        entry_met = True; entry_idx = idx; break

                if not entry_met:
                    current_price = float(df_after.iloc[-1]['close'])
                    results.append({'_id': sig.get('_id'), 'status': 'pending',
                                    'exit_price': None, 'current_price': current_price,
                                    'live_pnl_pct': 0, 'track_status': 'entry_not_met'})
                    continue

                entry_pos    = df_after.index.get_loc(entry_idx)
                df_post      = df_after.iloc[entry_pos:].reset_index(drop=True)
                trade_status = 'open'
                exit_price   = None
                current_price = float(df_post.iloc[-1]['close'])

                for _, row in df_post.iterrows():
                    if direction == 'BUY-LONG':
                        if row['high'] >= t2:
                            trade_status = 'target_hit'; exit_price = t2; break
                        if row['low'] <= sl:
                            trade_status = 'stop_hit';   exit_price = sl; break
                    else:
                        if row['low'] <= t2:
                            trade_status = 'target_hit'; exit_price = t2; break
                        if row['high'] >= sl:
                            trade_status = 'stop_hit';   exit_price = sl; break

                if direction == 'BUY-LONG':
                    pnl_pct = round((current_price - entry) / entry * 100, 2)
                else:
                    pnl_pct = round((entry - current_price) / entry * 100, 2)

                results.append({
                    '_id'         : sig.get('_id'),
                    'status'      : trade_status,
                    'exit_price'  : exit_price,
                    'current_price': current_price,
                    'live_pnl_pct': pnl_pct,
                    'track_status': 'tracked'
                })

            except Exception as e:
                results.append({'_id': sig.get('_id'), 'status': 'pending',
                                'exit_price': None, 'current_price': None,
                                'live_pnl_pct': 0, 'track_status': f'error:{str(e)}'})

        return jsonify({'status': 'success', 'results': results})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
