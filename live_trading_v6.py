#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 AzImA Live Trading System v6.3 (MT5 Integration)
===================================================
Simulates/Runs live trading by:
1. Connecting to MetaTrader 5 (MT5).
2. Fetching real-time M15/H1 candles.
3. Generating features on-the-fly via AdvancedFeatureEngineerV6.
4. Predicting direction using Base Models + LightGBM Ensemble.
5. Managing Risk (Stops, Targets) and sending Orders to MT5.
6. Sending Alerts to Telegram/Email.

Author: Ahmed (AzImA Team)
Date: Feb 2026
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import joblib

try:
    import MetaTrader5 as mt5
except ImportError:
    print("⚠️ MetaTrader5 module not found. Run: pip install MetaTrader5")
    sys.exit(1)

from config_v6 import ConfigV6
from train_ensemble_v6 import get_base_models
from paper_trading_v6 import StrategyManager

import requests # For Telegram

# ============================================================
# SYSTEM CONFIGURATION
# ============================================================
config = ConfigV6()

# MT5 Trading Settings
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
DEMO_MODE = True  # Set to False to run actual trades
CHECK_INTERVAL_SEC = 60 * 60  # Run once every hour

# Risk Management
MAX_RISK_PER_TRADE = 0.02
MAX_DAILY_DRAWDOWN = 0.05
MAX_TRADES_PER_DAY = 5
ATR_MULTIPLIER_SL = 1.5
ATR_MULTIPLIER_TP = 3.0

# Alert Settings
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"
ENABLE_TELEGRAM = False

# Logs
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "live_trading_mt5.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LiveTradingMT5")

# ============================================================
# UTILITIES AND ALERTS
# ============================================================
def send_telegram_message(message: str):
    if not ENABLE_TELEGRAM or not TELEGRAM_BOT_TOKEN:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")

# ============================================================
# MT5 INTEGRATION & RISK MANAGEMENT
# ============================================================
def initialize_mt5():
    """Initialize MT5 Connection"""
    if not mt5.initialize():
        logger.error("❌ Failed to connect to MetaTrader5")
        logger.error(f"Error: {mt5.last_error()}")
        return False
        
    account = mt5.account_info()
    if not account:
        logger.error("❌ Cannot access account info")
        return False
        
    logger.info(f"✅ Connected to MT5. Account: {account.login}, Balance: ${account.balance:.2f}")
    
    # Enable symbol
    if not mt5.symbol_select(SYMBOL, True):
        logger.error(f"❌ Failed to select {SYMBOL}")
        return False
        
    return True

class LiveRiskManager:
    def __init__(self):
        self.daily_trades = 0
        self.start_balance = 0
        self._set_start_balance()
        
    def _set_start_balance(self):
        acc = mt5.account_info()
        if acc: self.start_balance = acc.balance
            
    def can_trade(self) -> tuple[bool, str]:
        if self.daily_trades >= MAX_TRADES_PER_DAY:
            return False, "Max daily trades reached"
            
        acc = mt5.account_info()
        if not acc: return False, "Cannot read account"
            
        drawdown = (self.start_balance - acc.balance) / self.start_balance
        if drawdown >= MAX_DAILY_DRAWDOWN:
            return False, "Max daily drawdown reached"
            
        return True, "OK"

    def calculate_lot_size(self, sl_pips: float) -> float:
        acc = mt5.account_info()
        if not acc or sl_pips <= 0: return 0.01
        
        risk_amount = acc.balance * MAX_RISK_PER_TRADE
        pip_value = 10.0 # Approx for EURUSD
        lot_size = risk_amount / (sl_pips * pip_value)
        return max(0.01, round(lot_size, 2))

# ============================================================
# DATA PIPELINE & ML INFERENCE
# ============================================================
class PipelineEngine:
    def __init__(self):
        self.base_models = []
        self.ensemble_model = None
        self.feature_engineer = None
        
    def load_pipeline(self):
        logger.info("📡 Loading ML Pipeline...")
        
        # 1. Feature Engineer
        eng_path = Path("models_v6/feature_engineer.joblib")
        if not eng_path.exists():
            logger.error(f"❌ Feature engineer not found: {eng_path}")
            return False
        self.feature_engineer = joblib.load(eng_path)
        
        # 2. Base Models
        self.base_models = get_base_models(config)
        if not self.base_models:
            logger.error("❌ Failed to load Base Models")
            return False
            
        # 3. Ensemble Model
        ens_path = config.ENSEMBLE_DIR / 'lightgbm_ensemble.pkl'
        if not ens_path.exists():
            logger.error(f"❌ Ensemble model not found: {ens_path}")
            return False
        self.ensemble_model = joblib.load(ens_path)
        
        logger.info("✅ Pipeline fully loaded!")
        return True

    def fetch_mt5_data(self) -> pd.DataFrame:
        """Fetch sufficient candles from MT5 and prepare dataframe"""
        # We need enough candles to compute SMA_200 and sequence_length (24)
        num_candles = 300 
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, num_candles)
        if rates is None or len(rates) == 0:
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.rename(columns={'time': 'timestamp', 'tick_volume': 'volume'})
        return df

    def predict(self, df: pd.DataFrame):
        if df is None or len(df) < config.SEQUENCE_LENGTH:
            return None
            
        # 1. Generate Features (using transform to guarantee consistent column order)
        df_features = self.feature_engineer.transform(df)
        
        target_cols = ['timestamp', 'target', 'future_return', 'exit_reason', 'holding_period', 'realized_return']
        feature_cols = [c for c in df_features.columns if c not in target_cols]
        
        latest_sequence = df_features.iloc[-config.SEQUENCE_LENGTH:].copy()
        X_raw_seq = latest_sequence[feature_cols].values
        X_raw_seq = np.nan_to_num(X_raw_seq) # Security fallback
        
        # 2. Base Model Predictions
        meta_features = []
        for mi in self.base_models:
            model = mi['model']
            scaler = mi['scaler']
            m_feats = mi.get('features', None)
            
            X_scaled = scaler.transform(X_raw_seq)
            if m_feats:
                indices = [feature_cols.index(f) for f in m_feats]
                X_input = X_scaled[:, indices]
            else:
                X_input = X_scaled
                
            X_input_reshaped = X_input.reshape(1, config.SEQUENCE_LENGTH, -1)
            preds = model.predict(X_input_reshaped, verbose=0)
            meta_features.append(preds)
            
        X_meta = np.hstack(meta_features)
        
        # 3. Ensemble Prediction
        y_probs = self.ensemble_model.predict(X_meta)[0]
        # Classes: 0: SELL, 1: HOLD, 2: BUY
        predicted_class = np.argmax(y_probs)
        confidence = y_probs[predicted_class]
        
        signal = "HOLD"
        if predicted_class == 0: signal = "SELL"
        if predicted_class == 2: signal = "BUY"
        
        return {
            "signal": signal,
            "confidence": confidence,
            "probs": y_probs,
            "latest_row": df_features.iloc[-1]
        }

# ============================================================
# EXECUTION ENGINE
# ============================================================
def execute_trade(signal: str, confidence: float, latest_row: pd.Series, risk_mgr: LiveRiskManager):
    if signal not in ["BUY", "SELL"]:
        return
        
    can_trade, reason = risk_mgr.can_trade()
    if not can_trade:
        logger.warning(f"🚫 Trade blocked: {reason}")
        return
        
    # Strategy Filters (From paper_trading_v6)
    strat = StrategyManager()
    approved, req_reason = strat.check_filters(signal, latest_row)
    if not approved:
        logger.info(f"⏸️ Strategy Filter: {req_reason}")
        return

    # Execute
    tick = mt5.symbol_info_tick(SYMBOL)
    if not tick:
        logger.error("Cannot fetch tick data")
        return
        
    price = tick.ask if signal == "BUY" else tick.bid
    action = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
    
    # Calculate SL/TP
    # We generated ATR_14 dynamically in feature engineer
    atr = latest_row.get("ATR_14", 0.0010) 
    
    if signal == "BUY":
        sl = price - (atr * ATR_MULTIPLIER_SL)
        tp = price + (atr * ATR_MULTIPLIER_TP)
    else:
        sl = price + (atr * ATR_MULTIPLIER_SL)
        tp = price - (atr * ATR_MULTIPLIER_TP)
        
    # Standardize SL Pips size for lot calc
    sl_pips = abs(price - sl) * 10000 
    lot_size = risk_mgr.calculate_lot_size(sl_pips)

    logger.info(f"📤 Preparing {signal} order... Price: {price:.5f}, SL: {sl:.5f}, TP: {tp:.5f}, Lot: {lot_size}")
    
    if DEMO_MODE:
        logger.info(f"🛡️ DEMO MODE ACTIVE: Trade logically passed but not sent to MT5.")
        send_telegram_message(f"🧪 <b>DEMO MODE SIGNAL</b>\n\n{signal} {SYMBOL}\nConf: {confidence*100:.1f}%\nSL: {sl:.5f}\nTP: {tp:.5f}")
        return
        
    request = {
        'action': mt5.TRADE_ACTION_DEAL,
        'symbol': SYMBOL,
        'volume': lot_size,
        'type': action,
        'price': price,
        'sl': sl,
        'tp': tp,
        'deviation': 20,
        'magic': 60300,
        'comment': f'AzImA v6.3: {confidence:.2f}',
        'type_time': mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"✅ Trade Executed! Ticket: {result.order}")
        risk_mgr.daily_trades += 1
        send_telegram_message(f"✅ <b>TRADE EXECUTED</b>\n\n{signal} {SYMBOL}\nTicket: {result.order}\nPrice: {price:.5f}\nLot: {lot_size}")
    else:
        logger.error(f"❌ Trade Failed: {result.comment} ({result.retcode})")

# ============================================================
# MAIN LOOP
# ============================================================
def main():
    logger.info("=" * 60)
    logger.info("🤖 Starting AzImA MT5 Live Trading System v6.3")
    logger.info("=" * 60)
    
    if not initialize_mt5():
        sys.exit(1)
        
    pipeline = PipelineEngine()
    if not pipeline.load_pipeline():
        mt5.shutdown()
        sys.exit(1)
        
    risk_mgr = LiveRiskManager()
    
    logger.info(f"🔄 Check Interval: {CHECK_INTERVAL_SEC} sec")
    logger.info(f"🛡️ DEMO MODE: {DEMO_MODE}")
    
    try:
        while True:
            logger.info("🔎 Analyzing Market...")
            df = pipeline.fetch_mt5_data()
            
            if df is not None:
                prediction = pipeline.predict(df)
                
                if prediction:
                    signal = prediction['signal']
                    conf = prediction['confidence']
                    logger.info(f"🎯 ML Prediction: {signal} ({conf*100:.2f}%)")
                    
                    if signal in ["BUY", "SELL"] and conf >= config.BUY_THRESHOLD: 
                        execute_trade(signal, conf, prediction["latest_row"], risk_mgr)
            
            time.sleep(CHECK_INTERVAL_SEC)
            
    except KeyboardInterrupt:
        logger.info("🛑 Stopped by user. Shutting down...")
    finally:
        mt5.shutdown()
        logger.info("🔌 MT5 Connection Closed.")

if __name__ == "__main__":
    main()
