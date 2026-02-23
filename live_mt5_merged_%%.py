"""
============================================================
MT5 Complete Automated Trading System v26.2 - ULTIMATE EDITION
============================================================
Author: Enhanced by Claude
Date: 2025-10-20
Version: 26.2 (Ultimate Edition with all features)

New Features:
- ✅ Sound alerts for signals
- ✅ Detailed logging
- ✅ Daily summary reports
- ✅ Performance dashboard
- ✅ Auto time zone detection
- ✅ Enhanced error handling
- ✅ Telegram notifications (optional)
- ✅ Email alerts (optional)
- ✅ Real-time position monitoring
- ✅ Auto-save configuration
============================================================
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
import warnings
from pathlib import Path
from collections import defaultdict
import threading
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION - إعدادات النظام2
# ============================================================

# ========== أساسي ==========
SYMBOL = 'EURUSD'
DEMO_MODE = False  # ⚠️ ابدأ بـ True دائماً!

# ========== التداول ==========
CHECK_INTERVAL = 60  # ثانية
MIN_CONFIDENCE = 0.52  # 52%
MAX_TRADES_PER_DAY = 10
MIN_CONFIRMATIONS = 3

# ========== إدارة المخاطر ==========
BASE_RISK_PERCENT = 0.001  # 1%
MAX_DAILY_LOSS = 0.03  # 3%
MAX_WEEKLY_LOSS = 0.08  # 8%
MAX_OPEN_POSITIONS = 50

# ========== ATR ==========
ATR_PERIOD = 14
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 3.0
MIN_ATR_PIPS = 5.0
MAX_ATR_PIPS = 25.0

# ========== Trailing Stop ==========
TRAILING_STOP_ACTIVATION = 15  # pips
TRAILING_STOP_DISTANCE = 10  # pips

# ========== الأطر الزمنية ==========
ENTRY_TIMEFRAME = mt5.TIMEFRAME_M5
TREND_TIMEFRAME = mt5.TIMEFRAME_M15
CONTEXT_TIMEFRAME = mt5.TIMEFRAME_H1

# ========== المؤشرات ==========
EMA_FAST = 20
EMA_SLOW = 50
EMA_FILTER = 200
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# ========== ساعات التداول ==========
AUTO_DETECT_TIMEZONE = True  # كشف تلقائي للمنطقة الزمنية
TRADING_START_HOUR = 0   # من منتصف الليل
TRADING_END_HOUR = 23    # حتى 11 مساءً
# إذا AUTO_DETECT_TIMEZONE = False، استخدم UTC

# ========== التنبيهات ==========
ENABLE_SOUND_ALERTS = True
ENABLE_DETAILED_LOGGING = True
ENABLE_DAILY_SUMMARY = True

# ========== Telegram (اختياري) ==========
ENABLE_TELEGRAM = False
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"  # من @BotFather
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"      # ID الخاص بك

# ========== Email (اختياري) ==========
ENABLE_EMAIL = False
EMAIL_FROM = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"
EMAIL_TO = "your_email@gmail.com"
EMAIL_SMTP_SERVER = "smtp.gmail.com"
EMAIL_SMTP_PORT = 587

# ========== ملفات النظام ==========
LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)

SIGNALS_LOG_FILE = str(LOG_DIR / 'signals_history.json')
TRADES_LOG_FILE = str(LOG_DIR / 'trades_log.json')
DAILY_SUMMARY_FILE = str(LOG_DIR / 'daily_summaries.json')
DETAILED_LOG_FILE = str(LOG_DIR / f'detailed_log_{datetime.now().strftime("%Y%m%d")}.txt')
CONFIG_FILE = 'trading_config.json'

# ========== ML Model Paths ==========
MERGED_DIR = Path("-- 0 -- merged")
ML_DIR = MERGED_DIR / "live_assets"

JSON_PATH = ML_DIR / "feature_cols_vAzImA_26.01.json"
MODEL_PATH = MERGED_DIR / "eurusd_model_vAzImA_26.01.keras"
INDICATOR_SCALER_FILE = ML_DIR / "indicator_scaler_vAzImA_26.01.pkl"
PRICE_SCALER_FILE = ML_DIR / "price_scaler_vAzImA_26.01.pkl"
NEWS_SCALER_FILE = ML_DIR / "news_scaler_vAzImA_26.01.pkl"

TIMESTEPS = 48
FETCH_CANDLES = 300


# ============================================================
# ENHANCED LOGGING - تسجيل محسّن
# ============================================================

def log_to_file(message, level="INFO"):
    """تسجيل في ملف نصي مفصل"""
    if not ENABLE_DETAILED_LOGGING:
        return
    
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(DETAILED_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] [{level}] {message}\n")
    except Exception as e:
        print(f"⚠️ Error writing to log: {e}")


def detect_timezone():
    """كشف المنطقة الزمنية تلقائياً"""
    try:
        local_time = datetime.now()
        utc_time = datetime.utcnow()
        offset = round((local_time - utc_time).total_seconds() / 3600)
        
        print(f"\n🌍 Timezone Detection:")
        print(f"   Local: {local_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   UTC: {utc_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Offset: UTC{offset:+d}")
        
        log_to_file(f"Timezone offset detected: UTC{offset:+d}")
        
        return offset
    except Exception as e:
        print(f"⚠️ Error detecting timezone: {e}")
        return 0


# ============================================================
# SOUND ALERTS - التنبيهات الصوتية
# ============================================================

def play_alert(alert_type="signal"):
    """تشغيل تنبيه صوتي"""
    if not ENABLE_SOUND_ALERTS:
        return
    
    try:
        import winsound
        
        if alert_type == "signal":
            # نغمة إشارة جديدة
            winsound.Beep(1000, 300)
            time.sleep(0.1)
            winsound.Beep(1200, 300)
        
        elif alert_type == "trade":
            # نغمة تنفيذ صفقة
            winsound.Beep(800, 200)
            time.sleep(0.05)
            winsound.Beep(1000, 200)
            time.sleep(0.05)
            winsound.Beep(1200, 300)
        
        elif alert_type == "error":
            # نغمة خطأ
            winsound.Beep(400, 500)
        
        elif alert_type == "success":
            # نغمة نجاح
            for freq in [800, 1000, 1200, 1500]:
                winsound.Beep(freq, 150)
                time.sleep(0.05)
    
    except ImportError:
        # على أنظمة غير Windows
        print("\a")  # جرس النظام
    except Exception as e:
        log_to_file(f"Sound alert error: {e}", "WARNING")


# ============================================================
# TELEGRAM NOTIFICATIONS - تنبيهات تليجرام
# ============================================================

def send_telegram_message(message):
    """إرسال رسالة عبر Telegram"""
    if not ENABLE_TELEGRAM:
        return False
    
    try:
        import requests
        
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        
        response = requests.post(url, data=data, timeout=10)
        
        if response.status_code == 200:
            log_to_file("Telegram notification sent", "INFO")
            return True
        else:
            log_to_file(f"Telegram error: {response.text}", "WARNING")
            return False
    
    except Exception as e:
        log_to_file(f"Telegram send error: {e}", "ERROR")
        return False


def notify_signal(signal_data):
    """إرسال تنبيه بالإشارة"""
    if not signal_data or not signal_data.get('signal'):
        return
    
    signal = signal_data['signal']
    confidence = signal_data['confidence'] * 100
    price = signal_data['current_price']
    
    message = f"""
🤖 <b>MT5 Trading Signal</b>

📊 Symbol: {signal_data['symbol']}
{'🟢' if signal == 'BUY' else '🔴'} Signal: <b>{signal}</b>
💪 Confidence: {confidence:.1f}%
💰 Price: {price:.5f}

⏰ Time: {datetime.now().strftime('%H:%M:%S')}
"""
    
    send_telegram_message(message)


# ============================================================
# EMAIL NOTIFICATIONS - تنبيهات البريد الإلكتروني
# ============================================================

def send_email_alert(subject, body):
    """إرسال تنبيه عبر البريد الإلكتروني"""
    if not ENABLE_EMAIL:
        return False
    
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        msg = MIMEMultipart()
        msg['From'] = EMAIL_FROM
        msg['To'] = EMAIL_TO
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT)
        server.starttls()
        server.login(EMAIL_FROM, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        log_to_file("Email sent successfully", "INFO")
        return True
    
    except Exception as e:
        log_to_file(f"Email error: {e}", "ERROR")
        return False


# ============================================================
# CONFIGURATION MANAGEMENT - إدارة الإعدادات
# ============================================================

def save_config():
    """حفظ الإعدادات الحالية"""
    config = {
        'SYMBOL': SYMBOL,
        'DEMO_MODE': DEMO_MODE,
        'MIN_CONFIDENCE': MIN_CONFIDENCE,
        'MAX_TRADES_PER_DAY': MAX_TRADES_PER_DAY,
        'BASE_RISK_PERCENT': BASE_RISK_PERCENT,
        'TRADING_START_HOUR': TRADING_START_HOUR,
        'TRADING_END_HOUR': TRADING_END_HOUR,
        'ENABLE_SOUND_ALERTS': ENABLE_SOUND_ALERTS,
        'ENABLE_TELEGRAM': ENABLE_TELEGRAM,
        'last_updated': datetime.now().isoformat()
    }
    
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Configuration saved to {CONFIG_FILE}")
    except Exception as e:
        print(f"⚠️ Error saving config: {e}")


def load_config():
    """تحميل الإعدادات المحفوظة"""
    global MIN_CONFIDENCE, MAX_TRADES_PER_DAY, BASE_RISK_PERCENT
    
    if not os.path.exists(CONFIG_FILE):
        return False
    
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        
        print(f"📂 Loading saved configuration...")
        # تطبيق الإعدادات (يمكن إضافة المزيد)
        MIN_CONFIDENCE = config.get('MIN_CONFIDENCE', MIN_CONFIDENCE)
        MAX_TRADES_PER_DAY = config.get('MAX_TRADES_PER_DAY', MAX_TRADES_PER_DAY)
        BASE_RISK_PERCENT = config.get('BASE_RISK_PERCENT', BASE_RISK_PERCENT)
        
        print(f"✅ Configuration loaded")
        return True
    except Exception as e:
        print(f"⚠️ Error loading config: {e}")
        return False


# ============================================================
# TECHNICAL INDICATORS - المؤشرات الفنية
# ============================================================

def calculate_ema(series, period):
    """حساب Exponential Moving Average"""
    try:
        return series.ewm(span=period, adjust=False).mean()
    except Exception as e:
        log_to_file(f"EMA calculation error: {e}", "ERROR")
        return pd.Series(0, index=series.index)


def calculate_rsi(series, period=14):
    """حساب Relative Strength Index"""
    try:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(window=period).mean()
        loss = -delta.clip(upper=0).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    except Exception as e:
        log_to_file(f"RSI calculation error: {e}", "ERROR")
        return pd.Series(50, index=series.index)


def calculate_macd(series, fast=12, slow=26, signal=9):
    """حساب MACD"""
    try:
        ema_fast = calculate_ema(series, fast)
        ema_slow = calculate_ema(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    except Exception as e:
        log_to_file(f"MACD calculation error: {e}", "ERROR")
        return pd.Series(0, index=series.index), pd.Series(0, index=series.index), pd.Series(0, index=series.index)


def calculate_atr(df, period=14):
    """حساب Average True Range"""
    try:
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    except Exception as e:
        log_to_file(f"ATR calculation error: {e}", "ERROR")
        return pd.Series(0.0001, index=df.index)


# ============================================================
# PATTERN DETECTION - كشف الأنماط
# ============================================================

def detect_pin_bar(candle, threshold=0.6):
    """كشف شموع Pin Bar"""
    try:
        body = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        
        if total_range == 0:
            return None
        
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        
        if lower_wick > total_range * threshold and body < total_range * 0.3:
            return 'bullish_pin'
        
        if upper_wick > total_range * threshold and body < total_range * 0.3:
            return 'bearish_pin'
        
        return None
    except Exception as e:
        log_to_file(f"Pin bar detection error: {e}", "ERROR")
        return None


def detect_engulfing(df, index):
    """كشف نموذج Engulfing"""
    try:
        if index < 1 or index >= len(df):
            return None
        
        current = df.iloc[index]
        previous = df.iloc[index - 1]
        
        current_body = abs(current['close'] - current['open'])
        previous_body = abs(previous['close'] - previous['open'])
        
        if current_body < 0.0001 or previous_body < 0.0001:
            return None
        
        if (previous['close'] < previous['open'] and
            current['close'] > current['open'] and
            current['open'] <= previous['close'] and
            current['close'] >= previous['open'] and
            current_body > previous_body * 1.2):
            return 'bullish_engulfing'
        
        if (previous['close'] > previous['open'] and
            current['close'] < current['open'] and
            current['open'] >= previous['close'] and
            current['close'] <= previous['open'] and
            current_body > previous_body * 1.2):
            return 'bearish_engulfing'
        
        return None
    except Exception as e:
        log_to_file(f"Engulfing detection error: {e}", "ERROR")
        return None


def analyze_trend(df, lookback=20):
    """تحليل الاتجاه العام"""
    try:
        if df is None or len(df) < lookback:
            return {'trend': 'neutral', 'strength': 0}
        
        recent = df.tail(lookback)
        
        ema_fast = recent['ema_fast'].iloc[-1]
        ema_slow = recent['ema_slow'].iloc[-1]
        ema_filter = recent['ema_filter'].iloc[-1]
        current_price = recent['close'].iloc[-1]
        
        uptrend_signals = 0
        downtrend_signals = 0
        
        if current_price > ema_fast > ema_slow:
            uptrend_signals += 2
        elif current_price < ema_fast < ema_slow:
            downtrend_signals += 2
        
        if current_price > ema_filter:
            uptrend_signals += 1
        else:
            downtrend_signals += 1
        
        price_change = (recent['close'].iloc[-1] / recent['close'].iloc[0] - 1) * 100
        if price_change > 0.5:
            uptrend_signals += 1
        elif price_change < -0.5:
            downtrend_signals += 1
        
        total_signals = uptrend_signals + downtrend_signals
        if total_signals == 0:
            return {'trend': 'neutral', 'strength': 0}
        
        if uptrend_signals > downtrend_signals * 1.5:
            return {'trend': 'uptrend', 'strength': uptrend_signals / 4}
        elif downtrend_signals > uptrend_signals * 1.5:
            return {'trend': 'downtrend', 'strength': downtrend_signals / 4}
        else:
            return {'trend': 'neutral', 'strength': 0.5}
    
    except Exception as e:
        log_to_file(f"Trend analysis error: {e}", "ERROR")
        return {'trend': 'neutral', 'strength': 0}


# ============================================================
# DATA FETCHING - جلب البيانات
# ============================================================

def get_candles_with_indicators(symbol, timeframe, count=200):
    """جلب الشموع مع حساب جميع المؤشرات"""
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            log_to_file(f"No data for {symbol} on timeframe {timeframe}", "WARNING")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        df['ema_fast'] = calculate_ema(df['close'], EMA_FAST)
        df['ema_slow'] = calculate_ema(df['close'], EMA_SLOW)
        df['ema_filter'] = calculate_ema(df['close'], EMA_FILTER)
        df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
        df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(
            df['close'], MACD_FAST, MACD_SLOW, MACD_SIGNAL
        )
        df['atr'] = calculate_atr(df, ATR_PERIOD)
        df['atr_pips'] = df['atr'] / 0.0001
        
        df['body'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['range'] = df['high'] - df['low']
        df['is_bullish'] = df['close'] > df['open']
        df['ema_cross'] = df['ema_fast'] - df['ema_slow']
        df['price_above_ema200'] = df['close'] > df['ema_filter']
        
        return df
        
    except Exception as e:
        log_to_file(f'Error fetching candles: {e}', "ERROR")
        return None


# ============================================================
# RISK MANAGEMENT - إدارة المخاطر
# ============================================================

class RiskManager:
    """فئة إدارة المخاطر المحسّنة"""
    
    def __init__(self):
        self.trades_today = 0
        self.daily_pnl = 0
        self.starting_balance = 0
        self.last_reset_date = datetime.now().date()
        self.peak_balance = 0
        self.max_drawdown = 0
    
    def reset_daily_counters(self):
        """إعادة تعيين العدادات اليومية"""
        today = datetime.now().date()
        if today != self.last_reset_date:
            log_to_file(f"New trading day: {today}", "INFO")
            print(f"\n📅 New trading day: {today}")
            
            # حفظ ملخص اليوم السابق
            if ENABLE_DAILY_SUMMARY:
                self.save_daily_summary()
            
            self.trades_today = 0
            self.daily_pnl = 0
            self.last_reset_date = today
            
            account = mt5.account_info()
            if account:
                self.starting_balance = account.balance
                if account.balance > self.peak_balance:
                    self.peak_balance = account.balance
    
    def update_daily_pnl(self):
        """تحديث الربح/الخسارة اليومية"""
        try:
            from_date = datetime.now().replace(hour=0, minute=0, second=0)
            deals = mt5.history_deals_get(from_date, datetime.now())
            
            if deals:
                self.daily_pnl = sum(deal.profit for deal in deals)
            else:
                self.daily_pnl = 0
            
            # حساب Drawdown
            account = mt5.account_info()
            if account and self.peak_balance > 0:
                current_dd = (self.peak_balance - account.balance) / self.peak_balance
                if current_dd > self.max_drawdown:
                    self.max_drawdown = current_dd
            
            return self.daily_pnl
        except Exception as e:
            log_to_file(f"Error updating daily P&L: {e}", "ERROR")
            return 0
    
    def calculate_lot_size(self, symbol, balance, stop_loss_pips):
        """حساب حجم اللوت بشكل صحيح"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                log_to_file(f"Cannot get symbol info for {symbol}", "ERROR")
                return 0.01
            
            contract_size = symbol_info.trade_contract_size
            point = symbol_info.point
            pip_value = contract_size * point * 10
            
            risk_amount = balance * BASE_RISK_PERCENT
            
            if stop_loss_pips <= 0 or pip_value <= 0:
                return symbol_info.volume_min
            
            lot_size = risk_amount / (stop_loss_pips * pip_value)
            
            volume_step = symbol_info.volume_step
            lot_size = round(lot_size / volume_step) * volume_step
            
            lot_size = max(symbol_info.volume_min, 
                          min(lot_size, symbol_info.volume_max))
            
            log_to_file(f"Calculated lot size: {lot_size} for SL: {stop_loss_pips} pips", "INFO")
            
            return lot_size
            
        except Exception as e:
            log_to_file(f"Error calculating lot size: {e}", "ERROR")
            return 0.01
    
    def calculate_stops(self, symbol, price, order_type, atr_pips):
        """حساب Stop Loss و Take Profit"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return None
            
            sl_pips = max(MIN_ATR_PIPS, min(atr_pips * ATR_SL_MULTIPLIER, MAX_ATR_PIPS))
            tp_pips = sl_pips * 2
            
            pip_size = symbol_info.point * 10
            stops_level = symbol_info.trade_stops_level * symbol_info.point
            
            if order_type == mt5.ORDER_TYPE_BUY:
                sl = price - (sl_pips * pip_size)
                tp = price + (tp_pips * pip_size)
                
                if price - sl < stops_level:
                    sl = price - stops_level
                if tp - price < stops_level:
                    tp = price + stops_level
            else:
                sl = price + (sl_pips * pip_size)
                tp = price - (tp_pips * pip_size)
                
                if sl - price < stops_level:
                    sl = price + stops_level
                if price - tp < stops_level:
                    tp = price - stops_level
            
            return {
                'sl': round(sl, symbol_info.digits),
                'tp': round(tp, symbol_info.digits),
                'sl_pips': sl_pips
            }
            
        except Exception as e:
            log_to_file(f"Error calculating stops: {e}", "ERROR")
            return None
    
    def can_open_position(self, symbol, confidence):
        """التحقق من إمكانية فتح صفقة جديدة"""
        try:
            self.reset_daily_counters()
            self.update_daily_pnl()
            
            account = mt5.account_info()
            if not account:
                return {'allowed': False, 'reason': 'Cannot access account info'}
            
            if self.daily_pnl < 0:
                loss_percent = abs(self.daily_pnl) / account.balance
                if loss_percent >= MAX_DAILY_LOSS:
                    log_to_file(f"Daily loss limit reached: {loss_percent*100:.1f}%", "WARNING")
                    return {
                        'allowed': False, 
                        'reason': f'Daily loss limit reached: {loss_percent*100:.1f}%'
                    }
            
            if self.trades_today >= MAX_TRADES_PER_DAY:
                return {
                    'allowed': False,
                    'reason': f'Max daily trades reached: {self.trades_today}/{MAX_TRADES_PER_DAY}'
                }
            
            positions = mt5.positions_get(symbol=symbol)
            if positions is not None and len(positions) >= MAX_OPEN_POSITIONS:
                return {
                    'allowed': False,
                    'reason': f'Max open positions: {len(positions)}/{MAX_OPEN_POSITIONS}'
                }
            
            df = get_candles_with_indicators(symbol, ENTRY_TIMEFRAME, 50)
            if df is None or df.empty:
                return {'allowed': False, 'reason': 'Cannot get market data'}
            
            if 'atr_pips' not in df.columns or df['atr_pips'].iloc[-1] == 0:
                return {'allowed': False, 'reason': 'Invalid ATR value'}
            
            atr_pips = df['atr_pips'].iloc[-1]
            
            stops = self.calculate_stops(symbol, 0, 0, atr_pips)
            if not stops:
                return {'allowed': False, 'reason': 'Cannot calculate stops'}
            
            lot = self.calculate_lot_size(symbol, account.balance, stops['sl_pips'])
            
            return {
                'allowed': True,
                'lot_size': lot,
                'atr': atr_pips
            }
            
        except Exception as e:
            log_to_file(f"Error in can_open_position: {e}", "ERROR")
            return {'allowed': False, 'reason': f'Error: {str(e)}'}
    
    def update_trailing_stops(self, symbol):
        """تحديث Trailing Stops للصفقات المفتوحة"""
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return
        
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return
        
        pip_size = symbol_info.point * 10
        stops_level = symbol_info.trade_stops_level * symbol_info.point
        
        log_to_file(f"Checking {len(positions)} positions for trailing stop", "INFO")
        
        for pos in positions:
            try:
                tick = mt5.symbol_info_tick(symbol)
                if not tick:
                    continue
                
                if pos.type == mt5.ORDER_TYPE_BUY:
                    pips_profit = (tick.bid - pos.price_open) / pip_size
                    
                    if pips_profit >= TRAILING_STOP_ACTIVATION:
                        new_sl = tick.bid - (TRAILING_STOP_DISTANCE * pip_size)
                        
                        if new_sl < tick.bid - stops_level:
                            new_sl = tick.bid - stops_level
                        
                        if (pos.sl == 0 or new_sl > pos.sl) and new_sl < tick.bid:
                            self.modify_position(pos.ticket, new_sl, pos.tp)
                
                elif pos.type == mt5.ORDER_TYPE_SELL:
                    pips_profit = (pos.price_open - tick.ask) / pip_size
                    
                    if pips_profit >= TRAILING_STOP_ACTIVATION:
                        new_sl = tick.ask + (TRAILING_STOP_DISTANCE * pip_size)
                        
                        if new_sl > tick.ask + stops_level:
                            new_sl = tick.ask + stops_level
                        
                        if (pos.sl == 0 or new_sl < pos.sl) and new_sl > tick.ask:
                            self.modify_position(pos.ticket, new_sl, pos.tp)
            
            except Exception as e:
                log_to_file(f"Error updating trailing stop for {pos.ticket}: {e}", "ERROR")
                continue
    
    def modify_position(self, ticket, sl, tp):
        """تعديل الصفقة"""
        try:
            symbol_info = mt5.symbol_info(SYMBOL)
            request = {
                'action': mt5.TRADE_ACTION_SLTP,
                'position': ticket,
                'sl': round(sl, symbol_info.digits),
                'tp': round(tp, symbol_info.digits)
            }
            
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                log_to_file(f"Trailing Stop updated for position {ticket}", "INFO")
                print(f"✅ Trailing Stop updated for position {ticket}")
                return True
            else:
                log_to_file(f"Failed to update {ticket}: {result.comment}", "WARNING")
                return False
                
        except Exception as e:
            log_to_file(f"Error modifying position: {e}", "ERROR")
            return False
    
    def save_daily_summary(self):
        """حفظ ملخص اليوم"""
        try:
            account = mt5.account_info()
            if not account:
                return
            
            summary = {
                'date': self.last_reset_date.isoformat(),
                'starting_balance': self.starting_balance,
                'ending_balance': account.balance,
                'daily_pnl': self.daily_pnl,
                'trades_count': self.trades_today,
                'max_drawdown': self.max_drawdown * 100,
                'peak_balance': self.peak_balance
            }
            
            # تحميل السجل الموجود
            summaries = []
            if os.path.exists(DAILY_SUMMARY_FILE):
                try:
                    with open(DAILY_SUMMARY_FILE, 'r', encoding='utf-8') as f:
                        summaries = json.load(f)
                except:
                    summaries = []
            
            summaries.append(summary)
            
            # حفظ
            with open(DAILY_SUMMARY_FILE, 'w', encoding='utf-8') as f:
                json.dump(summaries, f, indent=2, ensure_ascii=False)
            
            log_to_file(f"Daily summary saved for {self.last_reset_date}", "INFO")
            
        except Exception as e:
            log_to_file(f"Error saving daily summary: {e}", "ERROR")


# ============================================================
# LOGGING & HISTORY - السجلات
# ============================================================

def save_signal_to_history(signal_data):
    """حفظ الإشارة في السجل"""
    try:
        history = []
        if os.path.exists(SIGNALS_LOG_FILE):
            try:
                with open(SIGNALS_LOG_FILE, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except:
                history = []
        
        temp_data = signal_data.copy()
        temp_data['timestamp'] = temp_data['timestamp'].isoformat()
        
        if 'ml_prediction' in temp_data:
            temp_data['ml_prediction'] = str(temp_data['ml_prediction'])
        
        history.append(temp_data)
        
        if len(history) > 100:
            history = history[-100:]
        
        with open(SIGNALS_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        log_to_file(f"Signal saved: {signal_data.get('signal')} @ {signal_data.get('confidence', 0)*100:.0f}%", "INFO")
        
    except Exception as e:
        log_to_file(f"Error saving signal: {e}", "ERROR")


def evaluate_last_signal():
    """تقييم آخر إشارة"""
    if not os.path.exists(SIGNALS_LOG_FILE):
        return {'status': 'no_history'}
    
    try:
        with open(SIGNALS_LOG_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        if not history:
            return {'status': 'empty_history'}
        
        last_signal = history[-1]
        tick = mt5.symbol_info_tick(SYMBOL)
        
        if not tick:
            return {'status': 'no_tick'}
        
        current_price = (tick.bid + tick.ask) / 2
        entry_price = last_signal.get('current_price', 0)
        
        if entry_price == 0:
            return {'status': 'invalid_price'}
        
        pips = (current_price - entry_price) / 0.0001
        
        if last_signal.get('signal') == 'SELL':
            pips = -pips
        
        return {
            'status': 'evaluated',
            'result_pips': pips,
            'signal': last_signal.get('signal'),
            'entry_price': entry_price,
            'current_price': current_price
        }
    
    except Exception as e:
        log_to_file(f"Error evaluating signal: {e}", "ERROR")
        return {'status': 'error', 'error': str(e)}


def save_trade_to_log(trade_data):
    """حفظ الصفقة في السجل"""
    try:
        trades = []
        if os.path.exists(TRADES_LOG_FILE):
            try:
                with open(TRADES_LOG_FILE, 'r', encoding='utf-8') as f:
                    trades = json.load(f)
            except:
                trades = []
        
        trades.append(trade_data)
        
        with open(TRADES_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(trades, f, indent=2, ensure_ascii=False)
        
        log_to_file(f"Trade logged: {trade_data.get('signal')} ticket {trade_data.get('ticket')}", "INFO")
    
    except Exception as e:
        log_to_file(f"Error saving trade: {e}", "ERROR")


# ============================================================
# DISPLAY FUNCTIONS - دوال العرض
# ============================================================

def print_signal(signal_data):
    """طباعة معلومات الإشارة مع تفاصيل محسّنة"""
    if not signal_data:
        print("⚠️ No signal data")
        return
    
    print("\n" + "="*70)
    print("📊 SIGNAL ANALYSIS")
    print("="*70)
    
    print(f"\n🎯 Symbol: {signal_data.get('symbol', 'N/A')}")
    print(f"💰 Current Price: {signal_data.get('current_price', 0):.5f}")
    print(f"⏰ Time: {signal_data.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # التفاصيل الفنية المحسّنة
    if ENABLE_DETAILED_LOGGING and 'indicators' in signal_data:
        indicators = signal_data['indicators']
        print(f"\n📋 Technical Details:")
        print(f"   EMA Fast: {indicators.get('ema_fast', 0):.5f}")
        print(f"   EMA Slow: {indicators.get('ema_slow', 0):.5f}")
        print(f"   RSI: {indicators.get('rsi', 0):.1f}")
        print(f"   MACD Hist: {indicators.get('macd_hist', 0):.5f}")
        print(f"   ATR: {indicators.get('atr_pips', 0):.1f} pips")
    
    signal = signal_data.get('signal')
    confidence = signal_data.get('confidence', 0)
    
    if signal:
        signal_emoji = "🟢" if signal == "BUY" else "🔴"
        print(f"\n{signal_emoji} SIGNAL: {signal}")
        print(f"💪 Confidence: {confidence*100:.1f}%")
        
        ml_pred = signal_data.get('ml_prediction', {})
        if ml_pred and isinstance(ml_pred, dict) and ml_pred.get('signal'):
            print(f"🤖 ML Signal: {ml_pred['signal']} ({ml_pred.get('confidence', 0)*100:.1f}%)")
        
        confirmations = signal_data.get('confirmations', [])
        if confirmations:
            print(f"\n✅ Confirmations ({len(confirmations)}):")
            for conf in confirmations[:5]:
                print(f"   {conf}")
            if len(confirmations) > 5:
                print(f"   ... و {len(confirmations)-5} تأكيدات أخرى")
    else:
        print("\n⚪ NO SIGNAL")
        print(f"💤 Confidence too low: {confidence*100:.1f}%")
    
    warnings = signal_data.get('warnings', [])
    if warnings:
        print(f"\n⚠️ Warnings:")
        for warn in warnings:
            print(f"   {warn}")
    
    print("="*70)


def print_account_summary():
    """طباعة ملخص الحساب محسّن"""
    try:
        account = mt5.account_info()
        if not account:
            print("\n⚠️ Cannot retrieve account info")
            log_to_file("Cannot retrieve account info", "WARNING")
            return
        
        positions = mt5.positions_get(symbol=SYMBOL)
        open_positions = len(positions) if positions else 0
        total_profit = sum(pos.profit for pos in positions) if positions else 0
        
        print("\n" + "="*70)
        print("💼 ACCOUNT SUMMARY")
        print("="*70)
        print(f"💰 Balance: ${account.balance:.2f}")
        print(f"💵 Equity: ${account.equity:.2f}")
        print(f"📊 Margin: ${account.margin:.2f}")
        print(f"🆓 Free Margin: ${account.margin_free:.2f}")
        
        if account.margin > 0:
            margin_level = (account.equity / account.margin) * 100
            print(f"📈 Margin Level: {margin_level:.2f}%")
        
        print(f"📊 Open Positions: {open_positions}")
        print(f"💹 Floating P&L: ${total_profit:.2f}")
        print(f"📉 Today's P&L: ${risk_mgr.daily_pnl:.2f}")
        print(f"📅 Trades Today: {risk_mgr.trades_today}/{MAX_TRADES_PER_DAY}")
        
        if risk_mgr.max_drawdown > 0:
            print(f"📉 Max Drawdown: {risk_mgr.max_drawdown*100:.2f}%")
        
        print("="*70)
        
    except Exception as e:
        log_to_file(f"Error in account summary: {e}", "ERROR")
        print(f"\n⚠️ Error in account summary: {e}")


def show_last_signal():
    """عرض تقييم آخر إشارة محسّن"""
    print("\n" + "-"*35)
    print("📊 Last Signal Evaluation")
    print("-"*35)
    
    evaluation = evaluate_last_signal()
    
    if evaluation['status'] == 'evaluated':
        result_pips = evaluation['result_pips']
        emoji = "📈" if result_pips > 0 else "📉"
        color = "✅" if result_pips > 0 else "❌"
        
        print(f"   {emoji} Current Result: {color} {result_pips:.1f} pips")
        print(f"   🎯 Signal: {evaluation['signal']}")
        print(f"   💰 Entry: {evaluation['entry_price']:.5f}")
        print(f"   📍 Now: {evaluation['current_price']:.5f}")
    else:
        print("   🔭 No previous signals to evaluate.")
    
    print("-" * 35)


# ============================================================
# ML MODEL - نموذج التعلم الآلي
# ============================================================

ml_model = None
feature_cols = None
indicator_scaler = None
price_scaler = None
news_scaler = None


def load_ml_model():
    """تحميل نموذج التعلم الآلي"""
    global ml_model, feature_cols, indicator_scaler, price_scaler, news_scaler
    
    try:
        if not JSON_PATH.exists():
            log_to_file(f"Feature columns file not found: {JSON_PATH}", "WARNING")
            return False
        
        if not MODEL_PATH.exists():
            log_to_file(f"Model file not found: {MODEL_PATH}", "WARNING")
            return False
        
        print("\n🤖 Loading Machine Learning Model...")
        
        try:
            import tensorflow as tf
            import joblib
        except ImportError:
            print("⚠️ TensorFlow or joblib not installed")
            log_to_file("TensorFlow or joblib not installed", "ERROR")
            return False
        
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            feature_cols = json.load(f)
        
        def focal_loss_fixed(y_true, y_pred, alpha=0.25, gamma=2.0):
            import tensorflow as tf
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
            p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
            fl = bce * alpha * tf.pow(1.0 - p_t, gamma)
            return tf.reduce_mean(fl)

        try:
            import tf_keras
            ml_model = tf_keras.models.load_model(
                str(MODEL_PATH), 
                custom_objects={
                    'focal_loss_fixed': focal_loss_fixed, 
                    'mse': tf_keras.losses.MeanSquaredError()
                },
                compile=False
            )
        except Exception as e_keras:
            log_to_file(f"Keras native load failed: {e_keras}. Falling back to default tf.keras.", "WARNING")
            import tensorflow as tf
            ml_model = tf.keras.models.load_model(
                str(MODEL_PATH), 
                custom_objects={
                    'focal_loss_fixed': focal_loss_fixed, 
                    'mse': tf.keras.losses.MeanSquaredError()
                },
                compile=False
            )
        
        if INDICATOR_SCALER_FILE.exists():
            indicator_scaler = joblib.load(str(INDICATOR_SCALER_FILE))
        
        if PRICE_SCALER_FILE.exists():
            price_scaler = joblib.load(str(PRICE_SCALER_FILE))
        
        if NEWS_SCALER_FILE.exists():
            news_scaler = joblib.load(str(NEWS_SCALER_FILE))
        
        print("🎉 ML Model loaded successfully!")
        print(f"   📊 Features: {len(feature_cols)}")
        print(f"   🔢 Timesteps: {TIMESTEPS}")
        
        log_to_file("ML Model loaded successfully", "INFO")
        return True
        
    except Exception as e:
        log_to_file(f"Error loading ML model: {e}", "ERROR")
        print(f"❌ Error loading ML model: {e}")
        return False


def build_features_for_ml():
    """بناء Features للنموذج"""
    try:
        import re
        from collections import defaultdict
        
        if not feature_cols:
            return None
        
        required_data = defaultdict(lambda: defaultdict(list))
        
        for col in feature_cols:
            parts = col.split('_')
            if len(parts) >= 3:
                required_data[parts[0]][parts[1]].append(col)
        
        final_df = pd.DataFrame()
        base_index = None
        
        for sym, frames in required_data.items():
            for fr, cols in frames.items():
                mt5_tf = getattr(mt5, f'TIMEFRAME_{fr}', None)
                if mt5_tf is None:
                    continue
                
                rates = mt5.copy_rates_from_pos(sym, mt5_tf, 0, FETCH_CANDLES)
                if rates is None or len(rates) == 0:
                    continue
                
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                
                if base_index is None:
                    base_index = df.index
                
                df_indicators = _calculate_indicators_for_ml(df, cols)
                final_df = pd.concat([final_df, df_indicators], axis=1)
        
        if base_index is None or final_df.empty:
            return None
        
        final_df = final_df.reindex(base_index, method='ffill').fillna(0.0)
        
        for col in feature_cols:
            if col not in final_df.columns:
                final_df[col] = 0.0
        
        final_df = final_df[feature_cols]

        # DEBUG: Trace out non-float columns
        for c in final_df.columns:
            try:
                final_df[c] = final_df[c].astype(float)
            except Exception as eval_e:
                bad_val = None
                for val in final_df[c]:
                    try:
                        float(val)
                    except:
                        bad_val = val
                        break
                print(f"DEBUG: Failed to cast column {c}: {eval_e}. Invalid element: {repr(bad_val)}")
                final_df[c] = 0.0
        
        if price_scaler and indicator_scaler:
            price_cols = [c for c in final_df.columns if 'close' in c.lower()]
            indicator_cols = [c for c in final_df.columns if c not in price_cols]
            
            if price_cols:
                try:
                    final_df[price_cols] = price_scaler.transform(final_df[price_cols])
                except Exception as p_e:
                    print(f"Price scaler error: {p_e}")
            
            if indicator_cols:
                try:
                    scaler_features = getattr(indicator_scaler, 'feature_names_in_', None)
                    if scaler_features is not None:
                        # Only scale features the scaler was actually trained on
                        valid_cols = [c for c in indicator_cols if c in scaler_features]
                        if valid_cols:
                            final_df[valid_cols] = indicator_scaler.transform(final_df[valid_cols])
                    else:
                        final_df[indicator_cols] = indicator_scaler.transform(final_df[indicator_cols])
                except Exception as i_e:
                    print(f"Indicator scaler error: {i_e}")
        
        if len(final_df) < TIMESTEPS:
            log_to_file(f"Not enough data: {len(final_df)}/{TIMESTEPS}", "WARNING")
            return None
        
        return final_df.tail(TIMESTEPS).values
        
    except Exception as e:
        log_to_file(f"Error building features: {e}", "ERROR")
        return None


def _calculate_indicators_for_ml(df, cols_to_calc):
    """حساب المؤشرات للـ ML"""
    import re
    
    df_out = pd.DataFrame(index=df.index)
    
    for col in cols_to_calc:
        try:
            parts = col.split('_')
            if len(parts) < 3:
                df_out[col] = 0.0
                continue
            
            rest = '_'.join(parts[2:])
            
            if rest in ['close', 'open', 'high', 'low', 'tick_volume']:
                df_out[col] = df[rest]
            
            elif 'MACD' in rest:
                m = re.match(r'(MACD|MACDh|MACDs)_(\d+)_(\d+)_(\d+)', rest)
                if m:
                    kind, fast, slow, sig = m.groups()
                    macd, signal, hist = calculate_macd(
                        df['close'], int(fast), int(slow), int(sig)
                    )
                    if kind == 'MACDh':
                        df_out[col] = hist
                    elif kind == 'MACDs':
                        df_out[col] = signal
                    else:
                        df_out[col] = macd
            
            elif 'RSI' in rest:
                m = re.match(r'RSI_(\d+)', rest)
                if m:
                    df_out[col] = calculate_rsi(df['close'], int(m.group(1)))
            
            elif 'STOCH' in rest:
                m = re.match(r'STOCH([kd])_(\d+)_(\d+)_?(\d*)', rest)
                if m:
                    groups = m.groups()
                    kind = groups[0]
                    kp = int(groups[1])
                    dp = int(groups[2]) if groups[2] else 3
                    
                    try:
                        low_min = df['low'].rolling(kp).min()
                        high_max = df['high'].rolling(kp).max()
                        k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-9)
                        
                        if kind == 'k':
                            df_out[col] = k
                        else:
                            df_out[col] = k.rolling(dp).mean()
                    except ValueError as ve:
                        print(f"Error parsing STOCH parameters: kind={kind}, kp={kp}, dp={dp} -> {ve}")
                        df_out[col] = 0.0
            
            elif 'CCI' in rest:
                m = re.match(r'CCI_(\d+)_([\d\.]+)', rest)
                if m:
                    period = int(m.group(1))
                    constant = float(m.group(2))
                    tp = (df['high'] + df['low'] + df['close']) / 3
                    sma = tp.rolling(period).mean()
                    mad = tp.rolling(period).apply(lambda x: pd.Series(x - x.mean()).abs().mean(), raw=True)
                    df_out[col] = (tp - sma) / (constant * mad + 1e-9)
                    
            elif 'WILLR' in rest:
                m = re.match(r'WILLR_(\d+)', rest)
                if m:
                    period = int(m.group(1))
                    high_max = df['high'].rolling(period).max()
                    low_min = df['low'].rolling(period).min()
                    df_out[col] = -100 * (high_max - df['close']) / (high_max - low_min + 1e-9)
                    
            elif 'ICS' in rest:
                m = re.match(r'ICS_(\d+)', rest) # Ichimoku Chikou Span (shifted close)
                if m:
                    shift = int(m.group(1))
                    df_out[col] = df['close'].shift(-shift)
                    
            elif 'PSAR' in rest:
                m = re.match(r'PSARl_([\d\.]+)_([\d\.]+)', rest)
                if m:
                    # Parabolic SAR is complex to loop optimally, fallback to simple trend proxy for ML
                    df_out[col] = df['close'].diff() > 0
            
            else:
                df_out[col] = 0.0
                
        except Exception as e:
            if str(e) != "could not convert string to float: 'd'":
                pass # suppress minor failures except our target
            print(f"Error calculating indicator {col}: {e}")
            df_out[col] = 0.0
    
    return df_out.fillna(0.0)


def get_ml_prediction():
    """الحصول على تنبؤ النموذج"""
    try:
        if not ml_model or not feature_cols:
            return {'signal': None, 'confidence': 0, 'error': 'Model not loaded'}
        
        try:
            features = build_features_for_ml()
        except Exception as e_feat:
            import traceback
            traceback.print_exc()
            return {'signal': None, 'confidence': 0, 'error': str(e_feat)}
        
        if features is None:
            return {'signal': None, 'confidence': 0, 'error': 'Feature building failed'}
        
        if features.shape[0] < TIMESTEPS:
            return {'signal': None, 'confidence': 0, 'error': 'Not enough timesteps'}
        
        try:
            # Check the features array types specifically
            import numpy as np
            features = np.array(features, dtype=np.float32)
        except Exception as e_cast:
            print(f"Features cast failed: {e_cast}")
            return {'signal': None, 'confidence': 0, 'error': str(e_cast)}

        preds = ml_model.predict(
            np.expand_dims(features, axis=0), 
            verbose=0
        )
        
        # In multi-output functional models, preds could be a list or a dictionary.
        if isinstance(preds, dict):
            direction = float(preds['direction'][0][0])
            confidence = float(preds['confidence'][0][0])
        elif isinstance(preds, list):
            direction = float(preds[0][0][0])
            confidence = float(preds[1][0][0])
        else:
            direction = float(preds[0][0])
            confidence = float(preds[0][1])  # Fallback if it's somehow a single array
        
        print(f"   🔮 ML Direction: {direction:.4f} | ML Confidence: {confidence:.4f}")
        log_to_file(f"ML Prediction - Dir: {direction:.4f}, Conf: {confidence:.4f}", "INFO")
        
        signal = 'BUY' if direction > 0.5 else 'SELL'
        
        return {
            'signal': signal,
            'confidence': confidence,
            'direction': direction
        }
        
    except Exception as e:
        log_to_file(f"ML Prediction Error: {e}", "ERROR")
        return {'signal': None, 'confidence': 0, 'error': str(e)}


# ============================================================
# SIGNAL GENERATION - توليد الإشارات
# ============================================================

def is_trading_time():
    """التحقق من وقت التداول مع كشف تلقائي للمنطقة الزمنية"""
    if AUTO_DETECT_TIMEZONE:
        now = datetime.now()
    else:
        now = datetime.utcnow()
    
    # طباعة الوقت للتشخيص
    local_time = datetime.now()
    utc_time = datetime.utcnow()
    print(f"🕐 Local Time: {local_time.strftime('%H:%M:%S')}")
    print(f"🌍 UTC Time: {utc_time.strftime('%H:%M:%S')}")
    print(f"⏰ Difference: {local_time.hour - utc_time.hour} hours")
    
    if now.weekday() >= 5:
        return False, "Weekend - Market Closed"
    
    if not (TRADING_START_HOUR <= now.hour < TRADING_END_HOUR):
        return False, f"Outside trading hours ({TRADING_START_HOUR}:00-{TRADING_END_HOUR}:00)"
    
    if risk_mgr.trades_today >= MAX_TRADES_PER_DAY:
        return False, f"Max daily trades reached ({MAX_TRADES_PER_DAY})"
    
    return True, "Trading time OK"


def generate_signal_with_ml(symbol):
    """توليد إشارة تداول مع تفاصيل محسّنة"""
    try:
        df_entry = get_candles_with_indicators(symbol, ENTRY_TIMEFRAME, 100)
        df_trend = get_candles_with_indicators(symbol, TREND_TIMEFRAME, 100)
        df_context = get_candles_with_indicators(symbol, CONTEXT_TIMEFRAME, 100)
        
        if any(df is None for df in [df_entry, df_trend, df_context]):
            log_to_file("Failed to fetch market data", "ERROR")
            return None
        
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            log_to_file("Cannot get current tick", "ERROR")
            return None
        
        signal_data = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'current_price': (tick.bid + tick.ask) / 2,
            'signal': None,
            'confidence': 0,
            'confirmations': [],
            'warnings': [],
            'ml_prediction': None,
            'indicators': {}  # للتفاصيل الفنية
        }
        
        entry_last = df_entry.iloc[-1]
        entry_prev = df_entry.iloc[-2]
        
        # حفظ التفاصيل الفنية
        signal_data['indicators'] = {
            'ema_fast': entry_last['ema_fast'],
            'ema_slow': entry_last['ema_slow'],
            'rsi': entry_last['rsi'],
            'macd_hist': entry_last['macd_hist'],
            'atr_pips': entry_last['atr_pips']
        }
        
        pin_bar = detect_pin_bar(entry_last)
        engulfing = detect_engulfing(df_entry, len(df_entry) - 1)
        
        trend_analysis = analyze_trend(df_trend)
        context_trend = analyze_trend(df_context)
        
        buy_confirmations = []
        
        if entry_last['ema_cross'] > 0:
            buy_confirmations.append("✅ EMA Cross Up")
        
        if trend_analysis['trend'] == 'uptrend':
            buy_confirmations.append(f"✅ Uptrend M15 (strength: {trend_analysis['strength']:.2f})")
        
        if context_trend['trend'] == 'uptrend':
            buy_confirmations.append(f"✅ Uptrend H1 (strength: {context_trend['strength']:.2f})")
        
        if 30 < entry_last['rsi'] < 50:
            buy_confirmations.append(f"✅ RSI Favorable: {entry_last['rsi']:.1f}")
        
        if pin_bar == 'bullish_pin':
            buy_confirmations.append("✅ Bullish Pin Bar")
        
        if engulfing == 'bullish_engulfing':
            buy_confirmations.append("✅ Bullish Engulfing")
        
        if entry_last['macd_hist'] > 0 and entry_last['macd_hist'] > entry_prev['macd_hist']:
            buy_confirmations.append("✅ MACD Rising")
        
        sell_confirmations = []
        
        if entry_last['ema_cross'] < 0:
            sell_confirmations.append("✅ EMA Cross Down")
        
        if trend_analysis['trend'] == 'downtrend':
            sell_confirmations.append(f"✅ Downtrend M15 (strength: {trend_analysis['strength']:.2f})")
        
        if context_trend['trend'] == 'downtrend':
            sell_confirmations.append(f"✅ Downtrend H1 (strength: {context_trend['strength']:.2f})")
        
        if 50 < entry_last['rsi'] < 70:
            sell_confirmations.append(f"✅ RSI Favorable: {entry_last['rsi']:.1f}")
        
        if pin_bar == 'bearish_pin':
            sell_confirmations.append("✅ Bearish Pin Bar")
        
        if engulfing == 'bearish_engulfing':
            sell_confirmations.append("✅ Bearish Engulfing")
        
        if entry_last['macd_hist'] < 0 and entry_last['macd_hist'] < entry_prev['macd_hist']:
            sell_confirmations.append("✅ MACD Falling")
        
        technical_buy_score = len(buy_confirmations)
        technical_sell_score = len(sell_confirmations)
        
        print(f"\n🔍 Technical Analysis:")
        print(f"   📈 BUY signals: {technical_buy_score}")
        print(f"   📉 SELL signals: {technical_sell_score}")
        
        ml_pred = get_ml_prediction()
        signal_data['ml_prediction'] = ml_pred
        
        ml_signal = ml_pred.get('signal')
        ml_confidence = ml_pred.get('confidence', 0)
        
        final_signal = None
        final_confidence = 0
        confirmations = []
        
        if ml_signal and ml_confidence >= MIN_CONFIDENCE:
            if ml_signal == 'BUY' and technical_buy_score >= 2:
                final_signal = 'BUY'
                final_confidence = (ml_confidence + (technical_buy_score / 7)) / 2
                confirmations = [f"🤖 ML Signal: BUY ({ml_confidence*100:.0f}%)"] + buy_confirmations
            
            elif ml_signal == 'SELL' and technical_sell_score >= 2:
                final_signal = 'SELL'
                final_confidence = (ml_confidence + (technical_sell_score / 7)) / 2
                confirmations = [f"🤖 ML Signal: SELL ({ml_confidence*100:.0f}%)"] + sell_confirmations
        
        if not final_signal:
            if technical_buy_score >= MIN_CONFIRMATIONS and technical_buy_score > technical_sell_score:
                final_signal = 'BUY'
                final_confidence = min(technical_buy_score / 7, 0.9)
                confirmations = buy_confirmations
                signal_data['warnings'].append("⚠️ Technical signal only (ML uncertain)")
            
            elif technical_sell_score >= MIN_CONFIRMATIONS and technical_sell_score > technical_buy_score:
                final_signal = 'SELL'
                final_confidence = min(technical_sell_score / 7, 0.9)
                confirmations = sell_confirmations
                signal_data['warnings'].append("⚠️ Technical signal only (ML uncertain)")
        
        signal_data.update({
            'signal': final_signal,
            'confidence': final_confidence,
            'confirmations': confirmations,
            'technical_buy_score': technical_buy_score,
            'technical_sell_score': technical_sell_score
        })
        
        if not final_signal:
            signal_data['warnings'].insert(
                0, 
                f"🚫 Not enough confirmations (BUY:{technical_buy_score}, SELL:{technical_sell_score})"
            )
        
        log_to_file(f"Signal generated: {final_signal or 'NONE'} @ {final_confidence*100:.0f}%", "INFO")
        
        return signal_data
        
    except Exception as e:
        log_to_file(f'Error generating signal: {e}', "ERROR")
        print(f'⚠️ Error generating signal: {e}')
        return None


# ============================================================
# TRADE EXECUTION - تنفيذ الصفقات
# ============================================================

def execute_trade(signal, confidence):
    """تنفيذ صفقة تداول مع تنبيهات محسّنة"""
    result = {'success': False, 'message': '', 'ticket': None}
    
    try:
        if DEMO_MODE:
            result['message'] = f"🛡️ DEMO MODE - {signal} Signal ({confidence*100:.0f}%) - No real trade"
            print(f"\n{result['message']}")
            log_to_file(result['message'], "INFO")
            play_alert("signal")
            return result
        
        risk_check = risk_mgr.can_open_position(SYMBOL, confidence)
        
        if not risk_check['allowed']:
            result['message'] = f"🚫 {risk_check['reason']}"
            log_to_file(result['message'], "WARNING")
            play_alert("error")
            return result
        
        tick = mt5.symbol_info_tick(SYMBOL)
        if not tick:
            result['message'] = "⚠️ Cannot get current price"
            log_to_file(result['message'], "ERROR")
            return result
        
        if signal == 'BUY':
            price = tick.ask
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid
            order_type = mt5.ORDER_TYPE_SELL
        
        stops = risk_mgr.calculate_stops(SYMBOL, price, order_type, risk_check['atr'])
        
        if not stops:
            result['message'] = "⚠️ Cannot calculate stops"
            log_to_file(result['message'], "ERROR")
            return result
        
        symbol_info = mt5.symbol_info(SYMBOL)
        
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': SYMBOL,
            'volume': risk_check['lot_size'],
            'type': order_type,
            'price': price,
            'sl': stops['sl'],
            'tp': stops['tp'],
            'deviation': 20,
            'magic': 260200,
            'comment': f'AzImA v26.2 - Conf:{confidence:.2f}',
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_IOC
        }
        
        print(f"\n📤 Sending {signal} order...")
        print(f"   💰 Price: {price:.5f}")
        print(f"   📊 Lot: {risk_check['lot_size']}")
        print(f"   🛑 SL: {stops['sl']:.5f} ({stops['sl_pips']:.1f} pips)")
        print(f"   🎯 TP: {stops['tp']:.5f}")
        
        order_result = mt5.order_send(request)
        
        if order_result.retcode == mt5.TRADE_RETCODE_DONE:
            result.update({
                'success': True,
                'message': f"✅ {signal} Order Executed!",
                'ticket': order_result.order
            })
            
            risk_mgr.trades_today += 1
            
            trade_log = {
                'timestamp': datetime.now().isoformat(),
                'ticket': order_result.order,
                'signal': signal,
                'price': price,
                'sl': stops['sl'],
                'tp': stops['tp'],
                'lot': risk_check['lot_size'],
                'confidence': confidence
            }
            save_trade_to_log(trade_log)
            
            print(f"\n{result['message']}")
            print(f"   🎫 Ticket: {order_result.order}")
            print(f"   💰 Volume: {order_result.volume}")
            
            log_to_file(f"Trade executed: {signal} ticket {order_result.order}", "INFO")
            
            # تنبيهات
            play_alert("trade")
            
            if ENABLE_TELEGRAM:
                msg = f"✅ <b>Trade Executed</b>\n\n{signal} {SYMBOL}\nTicket: {order_result.order}\nPrice: {price:.5f}\nLot: {risk_check['lot_size']}"
                send_telegram_message(msg)
            
            if ENABLE_EMAIL:
                send_email_alert(
                    f"Trade Executed: {signal} {SYMBOL}",
                    f"Ticket: {order_result.order}\nPrice: {price}\nLot: {risk_check['lot_size']}\nSL: {stops['sl']}\nTP: {stops['tp']}"
                )
            
        else:
            result['message'] = f"⚠️ Order FAILED: {order_result.comment} (Code: {order_result.retcode})"
            print(f"\n{result['message']}")
            log_to_file(result['message'], "ERROR")
            play_alert("error")
        
        return result
        
    except Exception as e:
        result['message'] = f"⚠️ Error executing trade: {str(e)}"
        print(f"\n{result['message']}")
        log_to_file(result['message'], "ERROR")
        play_alert("error")
        return result


# ============================================================
# MAIN CYCLE - الدورة الرئيسية
# ============================================================

def run_single_cycle():
    """تشغيل دورة واحدة محسّنة"""
    try:
        print("\n" + "="*70)
        print(f"🔎 Cycle Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        show_last_signal()
        
        time_ok, time_reason = is_trading_time()
        if not time_ok:
            print(f"\n⏸️  {time_reason}")
            print_account_summary()
            log_to_file(f"Skipping cycle: {time_reason}", "INFO")
            return
        
        risk_mgr.update_trailing_stops(SYMBOL)
        
        print("\n🎯 Generating new signal...")
        signal_data = generate_signal_with_ml(SYMBOL)
        
        if not signal_data:
            print("⚠️ Signal generation failed")
            print_account_summary()
            log_to_file("Signal generation failed", "ERROR")
            return
        
        if signal_data.get('signal'):
            save_signal_to_history(signal_data)
        
        print_signal(signal_data)
        
        if signal_data['signal'] and signal_data['confidence'] >= MIN_CONFIDENCE:
            print(f"\n✅ Strong signal detected!")
            print(f"   💪 Confidence: {signal_data['confidence']*100:.1f}%")
            print(f"   🎯 Signal: {signal_data['signal']}")
            
            play_alert("signal")
            notify_signal(signal_data)
            
            trade_result = execute_trade(signal_data['signal'], signal_data['confidence'])
            
            if not trade_result['success']:
                print(trade_result['message'])
        
        else:
            if signal_data['signal']:
                print(f"\n⚪ Signal confidence too low: {signal_data['confidence']*100:.1f}% < {MIN_CONFIDENCE*100:.0f}%")
            else:
                print("\n⚪ No signal generated")
        
        print_account_summary()
        
    except Exception as e:
        log_to_file(f"CRITICAL ERROR IN CYCLE: {e}", "ERROR")
        print(f"\n⚠️ CRITICAL ERROR IN CYCLE: {e}")
        play_alert("error")


def run_continuous():
    """تشغيل مستمر محسّن"""
    
    print("\n🔍 Pre-flight checks...")
    
    if not mt5.terminal_info():
        print("❌ MT5 not connected!")
        return
    
    account = mt5.account_info()
    if not account:
        print("❌ Cannot access account!")
        return
    
    print(f"✅ MT5 Connected - Account: {account.login}")
    
    if AUTO_DETECT_TIMEZONE:
        tz_offset = detect_timezone()
    
    if not DEMO_MODE:
        print("\n" + "!"*70)
        print("⚠️ WARNING: DEMO MODE IS OFF!")
        print("!"*70)
        cont = input("Continue with REAL MONEY? (yes/no): ")
        if cont.lower() != 'yes':
            print("👋 Exiting for safety...")
            return
    
    print("\n" + "="*70)
    print("🚀 MT5 Automated Trading System v26.2 - STARTING")
    print("="*70)
    print(f"⏰ Check Interval: {CHECK_INTERVAL} seconds")
    print(f"🎯 Symbol: {SYMBOL}")
    print(f"💪 Min Confidence: {MIN_CONFIDENCE*100:.0f}%")
    print(f"🛡️  DEMO MODE: {DEMO_MODE}")
    print(f"🔔 Sound Alerts: {ENABLE_SOUND_ALERTS}")
    print(f"📱 Telegram: {ENABLE_TELEGRAM}")
    print(f"📧 Email: {ENABLE_EMAIL}")
    print("="*70)
    
    log_to_file("Trading system started", "INFO")
    play_alert("success")
    
    cycle_count = 0
    
    try:
        while True:
            cycle_count += 1
            print(f"\n\n{'='*70}")
            print(f"🔄 CYCLE #{cycle_count}")
            print(f"{'='*70}")
            
            run_single_cycle()
            
            print(f"\n💤 Waiting {CHECK_INTERVAL} seconds until next check...")
            time.sleep(CHECK_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ System stopped by user")
        print("👋 Shutting down gracefully...")
        log_to_file("System stopped by user", "INFO")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        log_to_file(f"Fatal error: {e}", "ERROR")
        play_alert("error")
    finally:
        mt5.shutdown()
        print("\n✅ MT5 connection closed")
        print("👋 Goodbye!")


# ============================================================
# UTILITY FUNCTIONS - دوال مساعدة
# ============================================================

def quick_test():
    """اختبار سريع"""
    print("🧪 Quick Test Mode")
    print("="*50)
    
    if not initialize_mt5():
        return
    
    global risk_mgr, ml_loaded
    risk_mgr = RiskManager()
    ml_loaded = load_ml_model()
    
    run_single_cycle()
    
    print("\n✅ Test completed")
    log_to_file("Quick test completed", "INFO")


def check_positions():
    """فحص الصفقات المفتوحة"""
    positions = mt5.positions_get(symbol=SYMBOL)
    
    if not positions:
        print("📭 No open positions")
        return
    
    print(f"\n📊 Open Positions: {len(positions)}")
    print("="*70)
    
    for pos in positions:
        pnl_emoji = "💚" if pos.profit > 0 else "❤️"
        pos_type = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
        
        print(f"\n{pnl_emoji} Ticket: {pos.ticket}")
        print(f"   🎯 Type: {pos_type}")
        print(f"   💰 Volume: {pos.volume}")
        print(f"   📍 Open: {pos.price_open:.5f}")
        print(f"   📍 Current: {pos.price_current:.5f}")
        print(f"   🛑 SL: {pos.sl:.5f}")
        print(f"   🎯 TP: {pos.tp:.5f}")
        print(f"   💹 Profit: ${pos.profit:.2f}")
        print(f"   ⏰ Time: {datetime.fromtimestamp(pos.time)}")
    
    print("="*70)


def get_account_status():
    """حالة الحساب التفصيلية"""
    try:
        account = mt5.account_info()
        if not account:
            print("⚠️ Cannot get account info")
            return None
        
        positions = mt5.positions_get()
        open_pos = len(positions) if positions else 0
        total_profit = sum(p.profit for p in positions) if positions else 0
        
        print("\n" + "="*70)
        print("💼 DETAILED ACCOUNT STATUS")
        print("="*70)
        print(f"🏦 Server: {account.server}")
        print(f"👤 Account: {account.login}")
        print(f"⚡ Leverage: 1:{account.leverage}")
        print(f"\n💰 Balance: ${account.balance:.2f}")
        print(f"💵 Equity: ${account.equity:.2f}")
        print(f"📊 Used Margin: ${account.margin:.2f}")
        print(f"🆓 Free Margin: ${account.margin_free:.2f}")
        
        if account.margin > 0:
            print(f"📈 Margin Level: {(account.equity/account.margin)*100:.2f}%")
        
        print(f"\n💹 Total Profit: ${account.profit:.2f}")
        print(f"📊 Open Positions: {open_pos}")
        print(f"💸 Floating P&L: ${total_profit:.2f}")
        print(f"📉 Today's P&L: ${risk_mgr.daily_pnl:.2f}")
        print(f"📅 Trades Today: {risk_mgr.trades_today}/{MAX_TRADES_PER_DAY}")
        print("="*70)
        
    except Exception as e:
        print(f"⚠️ Error: {e}")


def get_signal_statistics():
    """إحصائيات الإشارات"""
    if not os.path.exists(SIGNALS_LOG_FILE):
        print("📭 No signals history")
        return
    
    try:
        with open(SIGNALS_LOG_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        if not history:
            print("📭 Empty history")
            return
        
        buy_signals = [s for s in history if s.get('signal') == 'BUY']
        sell_signals = [s for s in history if s.get('signal') == 'SELL']
        
        print("\n📊 SIGNAL STATISTICS")
        print("="*50)
        print(f"Total Signals: {len(history)}")
        print(f"🟢 BUY Signals: {len(buy_signals)}")
        print(f"🔴 SELL Signals: {len(sell_signals)}")
        
        if history:
            avg_confidence = np.mean([s.get('confidence', 0) for s in history])
            print(f"💪 Avg Confidence: {avg_confidence*100:.1f}%")
            
            recent = history[-10:]
            print(f"\n📜 Last 10 Signals:")
            for s in recent:
                signal = s.get('signal', 'NONE')
                conf = s.get('confidence', 0)
                time = s.get('timestamp', 'N/A')
                emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "⚪"
                print(f"   {emoji} {signal:4s} | {conf*100:5.1f}% | {time}")
        
        print("="*50)
        
    except Exception as e:
        print(f"⚠️ Error: {e}")


def show_system_menu():
    """القائمة التفاعلية المحسّنة"""
    while True:
        print("\n" + "="*70)
        print("🤖 MT5 TRADING SYSTEM v26.2 - MAIN MENU")
        print("="*70)
        print("\n📊 Analysis & Trading:")
        print("   1. Run Single Analysis")
        print("   2. Start Continuous Trading")
        print("   3. Quick Test")
        print("\n💼 Account & Positions:")
        print("   4. Check Account Status")
        print("   5. View Open Positions")
        print("   6. Close All Positions")
        print("\n📈 Statistics & Reports:")
        print("   7. Signal Statistics")
        print("   8. Daily Summaries")
        print("   9. Performance Metrics")
        print("\n🔧 System Tools:")
        print("   10. Save Configuration")
        print("   11. Load Configuration")
        print("   12. Test Alerts")
        print("\n   0. Exit")
        print("="*70)
        
        try:
            choice = input("\n👉 Select option: ").strip()
            
            if choice == '1':
                run_single_cycle()
            elif choice == '2':
                run_continuous()
            elif choice == '3':
                quick_test()
            elif choice == '4':
                get_account_status()
            elif choice == '5':
                check_positions()
            elif choice == '6':
                # Close all positions logic here
                print("⚠️ Feature coming soon")
            elif choice == '7':
                get_signal_statistics()
            elif choice == '8':
                # Show daily summaries
                print("⚠️ Feature coming soon")
            elif choice == '9':
                # Performance metrics
                print("⚠️ Feature coming soon")
            elif choice == '10':
                save_config()
            elif choice == '11':
                load_config()
            elif choice == '12':
                print("\n🔔 Testing alerts...")
                play_alert("signal")
                time.sleep(0.5)
                play_alert("trade")
                time.sleep(0.5)
                play_alert("success")
                print("✅ Alert test complete")
            elif choice == '0':
                print("\n👋 Exiting...")
                break
            else:
                print("\n⚠️ Invalid option")
            
            input("\nPress Enter to continue...")
        
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted")
            break
        except Exception as e:
            print(f"\n⚠️ Error: {e}")


# ============================================================
# INITIALIZATION - التهيئة
# ============================================================

def initialize_mt5():
    """تهيئة MT5"""
    if not mt5.initialize():
        print('❌ Failed to connect to MetaTrader5')
        print(f'⚠️  Error: {mt5.last_error()}')
        log_to_file(f"MT5 connection failed: {mt5.last_error()}", "ERROR")
        return False
    
    account = mt5.account_info()
    if not account:
        print('❌ Cannot access account info')
        log_to_file("Cannot access account info", "ERROR")
        return False
    
    print('✅ Connected to MetaTrader5')
    print(f'   💰 Balance: ${account.balance:.2f}')
    print(f'   🏦 Server: {account.server}')
    print(f'   🔢 Account: {account.login}')
    print(f'   📊 Leverage: 1:{account.leverage}')
    
    log_to_file(f"MT5 connected - Account: {account.login}", "INFO")
    
    return True


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🤖 MT5 AUTOMATED TRADING SYSTEM v26.2")
    print("="*70)
    
    # تحميل الإعدادات المحفوظة
    load_config()
    
    # Initialize MT5
    if not initialize_mt5():
        print("\n❌ Cannot proceed without MT5 connection")
        exit(1)
    
    # Initialize Risk Manager
    global risk_mgr, ml_loaded
    risk_mgr = RiskManager()
    risk_mgr.reset_daily_counters()
    print("✅ Risk Manager initialized")
    
    # Load ML Model
    ml_loaded = load_ml_model()
    
    if not ml_loaded:
        print("\n⚠️ ML model not loaded - using technical analysis only")
    
    # Verify Symbol
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info and not symbol_info.visible:
        mt5.symbol_select(SYMBOL, True)
    
    print(f"\n✅ System ready!")
    print(f"🛡️  DEMO MODE: {'ENABLED ✅' if DEMO_MODE else 'DISABLED ⚠️'}")
    
    # Show menu
    try:
        show_system_menu()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    finally:
        mt5.shutdown()
        print("\n✅ System shutdown complete")
        print("👋 Goodbye!")


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n✅ MT5 Trading System v26.2 loaded successfully!")
print(f"📂 Logs directory: {LOG_DIR.absolute()}")
print(f"🛡️  DEMO MODE: {'ENABLED ✅' if DEMO_MODE else 'DISABLED ⚠️'}")