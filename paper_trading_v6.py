#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 AzImA Paper Trading System v6.0
==================================
Simulates live trading by:
1. Detecting new data (file-watching or simulated loop)
2. Generating features on the fly
3. Predicting with Ensemble + Filter models
4. Logging signals and simulated trades to CSV

Author: Ahmed (AzImA Team)
Date: Feb 2026
"""

import time
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from datetime import datetime
from config_v6 import ConfigV6

from train_ensemble_v6 import get_base_models, generate_predictions_for_set

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/paper_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PaperTrading")


class StrategyManager:
    """
    🛡️ Strategy Layer (Rule-Based Filters)
    Implements 'Trend & Vigor' strategy to filter AI signals.
    """
    def __init__(self):
        # Strategy Parameters
        self.SMA_PERIOD = 200
        self.RSI_PERIOD = 14
        self.ADX_PERIOD = 14
        
        self.RSI_OVERBOUGHT = 70
        self.RSI_OVERSOLD = 30
        self.MIN_ADX = 20
        
    def check_filters(self, signal, row):
        """
        Apply strategy rules to a signal.
        Returns: (is_approved, reason)
        """
        price = row['close']
        
        # 1. Trend Filter (SMA 200)
        # "The Golden Rule": Trade primarily in direction of major trend
        sma_200 = row.get(f'SMA_{self.SMA_PERIOD}', 0)
        
        # If SMA is 0 (not enough data), skip trend check or be conservative?
        # Let's assume if data exists, check it.
        if sma_200 > 0:
            if signal == "BUY" and price < sma_200:
                return False, f"Price below SMA {self.SMA_PERIOD} (Downtrend)"
            if signal == "SELL" and price > sma_200:
                return False, f"Price above SMA {self.SMA_PERIOD} (Uptrend)"
                
        # 2. Momentum Filter (RSI)
        # "Don't Chase Tops/Bottoms"
        rsi = row.get(f'RSI_{self.RSI_PERIOD}', 50)
        
        if signal == "BUY" and rsi > self.RSI_OVERBOUGHT:
            return False, f"RSI Overbought ({rsi:.1f} > {self.RSI_OVERBOUGHT})"
        
        if signal == "SELL" and rsi < self.RSI_OVERSOLD:
            return False, f"RSI Oversold ({rsi:.1f} < {self.RSI_OVERSOLD})"
            
        # 3. Volatility Filter (ADX)
        # "Avoid Dead Markets"
        adx = row.get(f'ADX_{self.ADX_PERIOD}', 0)
        if adx < self.MIN_ADX:
            return False, f"Low Volatility ADX ({adx:.1f} < {self.MIN_ADX})"
            
        return True, "Strategy Approved"

class PaperTrader:
    def __init__(self):
        self.config = ConfigV6()
        self.base_models = []
        self.ensemble_model = None
        self.filter_model = None
        self.filter_features = None
        self.feature_cols = None
        
        # Initialize Strategy Manager
        self.strategy_manager = StrategyManager()
        
        # Trade Log
        self.trade_log_path = self.config.RESULTS_DIR / "paper_trades.csv"
        # Initial columns if file doesn't exist
        if not self.trade_log_path.exists():
            pd.DataFrame(columns=[
                "timestamp", "price", "signal", "confidence", "filter_prob", 
                "atr", "sl", "tp", "status", "exit_time", "exit_price", "pnl", "strategy_check"
            ]).to_csv(self.trade_log_path, index=False)
            
        self.open_positions = [] # List of dicts
        
        # Load Models
        self._load_models()
        self._load_open_positions() # Simple persistence check?
        
    def _load_models(self):
        logger.info("📡 Loading Models...")
        
        # 1. Base Models
        self.base_models = get_base_models(self.config)
        if not self.base_models:
            logger.error("❌ Failed to load Base Models!")
            # In live script we might want to exit or retry
            return
            
        # 2. Ensemble Model
        ensemble_path = self.config.ENSEMBLE_DIR / 'lightgbm_ensemble.pkl'
        if ensemble_path.exists():
            self.ensemble_model = joblib.load(ensemble_path)
            logger.info("✅ Ensemble Model Loaded")
        else:
            logger.error(f"❌ Ensemble model not found at {ensemble_path}")
            return
            
        # 3. Filter Model
        if self.config.FILTER_MODEL_PATH.exists():
            self.filter_model = joblib.load(self.config.FILTER_MODEL_PATH)
            feat_path = self.config.MODELS_DIR / "filter_features.pkl"
            if feat_path.exists():
                self.filter_features = joblib.load(feat_path)
                logger.info("✅ Filter Model Loaded")
            else:
                logger.warning("⚠️ Filter Features not found.")
        else:
            logger.warning("⚠️ Filter Model not found. Running without filter.")

    def _load_open_positions(self):
        """
        In a real system, we'd read from disk/db to recover state.
        For now, we start fresh or just check the CSV last status?
        Let's keep it simple: in-memory state for this demo session.
        """
        pass

    def fetch_latest_data(self):
        """
        Simulate fetching latest data.
        In a real scenario, this would connect to MT5/Binance or watch a CSV file.
        For now, we read the LAST rows of the test data as a demo.
        """
        # Read the main data file for demo purposes
        try:
            df = pd.read_csv(self.config.LABELED_CSV_FILE)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Return enough history to generate sequence
            # SEQ_LEN=24. We need at least that.
            # Plus mutual info / technical indicators calculation window...
            # The labeled file ALREADY has features calculated.
            # So we just need the last SEQ_LEN rows to feed into the model.
            
            lookback = self.config.SEQUENCE_LENGTH
            if len(df) < lookback:
                logger.warning("Data file too short.")
                return None
                
            return df.iloc[-lookback:].copy()
        except Exception as e:
            logger.error(f"Error reading data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def process_and_predict(self, df):
        """Generate signal for the latest candle"""
        if df is None or len(df) < self.config.SEQUENCE_LENGTH:
            logger.warning("Not enough data to predict.")
            return None
            
        # 1. Feature Extraction
        # In this demo, `df` already has features.
        # We need to extract them matching the training columns.
        
        # Identify feature cols (excluding metadata)
        # Re-use logic from train_ensemble_v6 or config
        exclude_cols = ['timestamp', 'target', 'realized_return', 'exit_reason', 'holding_period']
        feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
        
        # 2. Meta Features Generation
        latest_sequence_df = df.iloc[-self.config.SEQUENCE_LENGTH:].copy()
        X_raw_seq = latest_sequence_df[feature_cols].values
        X_raw_seq = np.nan_to_num(X_raw_seq)
        
        logger.info(f"🔮 Predicting on last {len(X_raw_seq)} candles...")
        
        logger.info(f"🔮 Predicting on last {len(X_raw_seq)} candles...")
        
        try:
            # Generate Meta Features (Level 1) - Manual Inference
            meta_features = []
            
            for model_info in self.base_models:
                model = model_info['model']
                scaler = model_info['scaler']
                model_features = model_info.get('features', None)
                
                # 1. Scale standardly (RobustScaler expects 2D)
                # X_raw_seq is (24, total_features)
                X_scaled_seq = scaler.transform(X_raw_seq)
                
                # 2. Slice features if needed
                if model_features:
                    try:
                        indices = [feature_cols.index(f) for f in model_features]
                        X_input = X_scaled_seq[:, indices]
                    except ValueError:
                        # Fallback
                        X_input = X_scaled_seq
                else:
                    X_input = X_scaled_seq
                    
                # 3. Reshape for LSTM (1, seq_len, n_features)
                # The entire X_input IS one sequence
                X_input_reshaped = X_input.reshape(1, self.config.SEQUENCE_LENGTH, -1)
                
                # 4. Predict
                preds = model.predict(X_input_reshaped, verbose=0) # (1, 3)
                meta_features.append(preds)
            
            # Concatenate all base model predictions
            # Shape: (1, n_models * 3)
            X_meta = np.hstack(meta_features)
            
            # Ensemble Prediction (Level 2)
            y_probs = self.ensemble_model.predict(X_meta) # (1, 3)
            current_prob = y_probs[0] # [p_sell, p_hold, p_buy]
            
            predicted_class_idx = np.argmax(current_prob)
            confidence = current_prob[predicted_class_idx]
            
            class_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
            predicted_signal = class_map.get(predicted_class_idx, "HOLD")
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # 3. Apply Confidence Threshold
        threshold = 0.40 # Default baseline
        final_signal = "HOLD" # Default
        if confidence >= threshold:
            final_signal = predicted_signal
            
        # 4. Filter Check (Level 3)
        filter_prob = -1.0
        if final_signal == "SELL" and self.filter_model and self.filter_features:
            try:
                # Extract filter features for the last row
                last_row = latest_sequence_df.iloc[[-1]].copy()
                
                # Ensure all filter features exist (pad with 0 if missing)
                for c in self.filter_features:
                    if c not in last_row.columns:
                        last_row[c] = 0.0
                
                X_filter = last_row[self.filter_features].values
                X_filter = np.nan_to_num(X_filter)
                
                # Predict
                filter_prob = self.filter_model.predict(X_filter)[0]
                
                if filter_prob < self.config.FILTER_THRESHOLD:
                    logger.info(f"🛡️ Filter BLOCKED Short: Prob {filter_prob:.2f} < {self.config.FILTER_THRESHOLD}")
                    final_signal = "HOLD"
                else:
                    logger.info(f"✅ Filter PASSED Short: Prob {filter_prob:.2f}")
                    
            except Exception as e:
                logger.error(f"Filter check failed: {e}")

        # 5. 🔥 Strategy Layer Check (Level 4)
        strategy_check = "N/A"
        if final_signal in ["BUY", "SELL"]:
            last_row = latest_sequence_df.iloc[-1]
            is_approved, reason = self.strategy_manager.check_filters(final_signal, last_row)
            
            if is_approved:
                logger.info(f"✅ Strategy Approved: {reason}")
                strategy_check = "APPROVED"
            else:
                logger.info(f"🛡️ Strategy BLOCKED {final_signal}: {reason}")
                final_signal = "HOLD"
                strategy_check = f"BLOCKED: {reason}"

        # Get current ATR for sizing/SL/TP
        atr = latest_sequence_df.iloc[-1].get('ATR_14', 0.0010)
        
        result = {
            "timestamp": latest_sequence_df.iloc[-1]['timestamp'],
            "price": latest_sequence_df.iloc[-1]['close'],
            "signal": final_signal,
            "raw_signal": predicted_signal,
            "confidence": confidence,
            "filter_prob": filter_prob,
            "atr": atr,
            "strategy_check": strategy_check
        }
        
        logger.info(f"📊 Analysis: {result['raw_signal']} ({confidence:.2f}) -> Strat: {strategy_check} -> Final: {result['signal']}")
        return result

    def execute_trade(self, signal_data):
        """Simulate trade execution"""
        price = signal_data['price']
        atr = signal_data['atr']
        
        # Calculate SL/TP
        if signal_data['signal'] == "BUY":
            sl = price - (atr * self.config.ATR_MULTIPLIER_SL)
            tp = price + (atr * self.config.ATR_MULTIPLIER_TP)
        else: # SELL
            sl = price + (atr * self.config.ATR_MULTIPLIER_SL)
            tp = price - (atr * self.config.ATR_MULTIPLIER_TP)
            
        trade = {
            "timestamp": signal_data['timestamp'],
            "price": price,
            "signal": signal_data['signal'],
            "confidence": signal_data['confidence'],
            "filter_prob": signal_data['filter_prob'],
            "atr": atr,
            "sl": sl,
            "tp": tp,
            "status": "OPEN",
            "exit_time": None,
            "exit_price": None,
            "pnl": 0.0,
            "strategy_check": signal_data.get('strategy_check', 'N/A')
        }
        
        self.open_positions.append(trade)
        
        # Log to file
        df = pd.DataFrame([trade])
        df.to_csv(self.trade_log_path, mode='a', header=False, index=False)
        
        logger.info(f"🚀 EXECUTE {signal_data['signal']} @ {price:.5f} [TP: {tp:.5f} | SL: {sl:.5f}]")

    def monitor_positions(self, current_price, current_time):
        """Check open positions for exit conditions"""
        active_positions = []
        
        for pos in self.open_positions:
            exit_reason = None
            
            # Check Holding Time
            entry_time = pd.to_datetime(pos['timestamp'])
            try:
                # current_time is from dataframe or system clock? 
                # If dataframe, it's pandas timestamp.
                elapsed_hours = (current_time - entry_time).total_seconds() / 3600
                if elapsed_hours >= self.config.MAX_HOLDING_PERIODS:
                    exit_reason = "TIME_EXIT"
            except:
                pass # Timestamp format issues?
                
            # Check Price Levels
            if pos['signal'] == "BUY":
                if current_price >= pos['tp']:
                    exit_reason = "TAKE_PROFIT"
                elif current_price <= pos['sl']:
                    exit_reason = "STOP_LOSS"
            elif pos['signal'] == "SELL":
                if current_price <= pos['tp']:
                    exit_reason = "TAKE_PROFIT"
                elif current_price >= pos['sl']:
                    exit_reason = "STOP_LOSS"
            
            if exit_reason:
                self.close_trade(pos, current_price, current_time, exit_reason)
            else:
                active_positions.append(pos)
                
        self.open_positions = active_positions

    def run_simulation(self, start_date=None, end_date=None):
        """
        Runs a simulation over a specific date range.
        If dates are None, defaults to last 30 days.
        Format: 'YYYY-MM-DD'
        """
        logger.info(f"� Starting Simulation Mode...")
        if start_date: logger.info(f"� Start Date: {start_date}")
        if end_date: logger.info(f"📅 End Date:   {end_date}")
        
        # 1. Load Data
        df = pd.read_csv(self.config.LABELED_CSV_FILE)
        if 'timestamp' in df.columns:
            # Use dayfirst=True if your data is DD-MM-YYYY, or infer
            # Best practice: coerce errors, infer format
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 2. Filter Data
        mask = pd.Series([True] * len(df))
        
        if start_date:
            # Handle potential User formats like DD-MM-YYYY vs YYYY-MM-DD
            # We try to standarize input first or let pandas guess
            try:
                s_dt = pd.to_datetime(start_date, dayfirst=False) # Assume YYYY-MM-DD or standard
                logger.info(f"Start Date parsed as: {s_dt}")
                mask &= (df['timestamp'] >= s_dt)
            except:
                logger.warning(f"⚠️ Could not parse start date: {start_date}")
        
        if end_date:
            try:
                e_dt = pd.to_datetime(end_date, dayfirst=False) 
                logger.info(f"End Date parsed as:   {e_dt}")
                # Make end date inclusive of the whole day (23:59:59)
                e_dt = e_dt.replace(hour=23, minute=59, second=59)
                mask &= (df['timestamp'] <= e_dt)
            except:
                 logger.warning(f"⚠️ Could not parse end date: {end_date}")
            
        if not start_date and not end_date:
            # Default: Last 30 days
            cutoff = df['timestamp'].max() - pd.Timedelta(days=30)
            mask &= (df['timestamp'] >= cutoff)
            logger.info("⚠️ No dates specified. Defaulting to last 30 days.")

        # checks to ensure we have enough data (SEQUENCE_LENGTH buffer)
        # We need the filtered data PLUS the lookback context for the first row
        filtered_indices = df[mask].index
        
        if len(filtered_indices) == 0:
            logger.error("❌ No data found for the specified date range.")
            return

        start_idx = filtered_indices[0]
        end_idx = filtered_indices[-1]
        
        # Ensure we have context for the first prediction
        if start_idx < self.config.SEQUENCE_LENGTH:
            logger.warning("⚠️ Start date is too early for lookback. Starting from first possible index.")
            start_idx = self.config.SEQUENCE_LENGTH

        # Reset Stats
        self.stats = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'fees': 0.0 # Placeholder
        }
        
        logger.info(f"📊 Simulating from {df.iloc[start_idx]['timestamp']} to {df.iloc[end_idx]['timestamp']} ({end_idx - start_idx} bars)...")
        
        # 3. Iterate
        # We iterate from start_idx to end_idx
        for curr_abs_idx in range(start_idx, end_idx + 1):
            
            # Slice window for feature generation
            window_df = df.iloc[curr_abs_idx - self.config.SEQUENCE_LENGTH : curr_abs_idx + 1].copy()
            
            # Run Logic
            current_time = window_df.iloc[-1]['timestamp']
            current_price = window_df.iloc[-1]['close']
            
            # Monitor Open Positions
            self.monitor_positions(current_price, current_time)
            
            # Check for New Signals
            if len(self.open_positions) < self.config.MAX_POSITIONS:
                result = self.process_and_predict(window_df)
                if result and result['signal'] in ["BUY", "SELL"]:
                    self.execute_trade(result)
            
            if (curr_abs_idx - start_idx) % 100 == 0:
                print(f"Propagating... {window_df.iloc[-1]['timestamp']}", end='\r')
        
        # Force close remaining positions at end price
        final_price = df.iloc[end_idx]['close']
        final_time = df.iloc[end_idx]['timestamp']
        for pos in self.open_positions:
            self.close_trade(pos, final_price, final_time, "END_OF_SIM")

        # 4. Report
        print("\n" + "="*50)
        print(f"🏁 SIMULATION RESULTS ({start_date} to {end_date})")
        print("="*50)
        print(f"Total Trades: {self.stats['total_trades']}")
        print(f"Wins:         {self.stats['wins']}")
        print(f"Losses:       {self.stats['losses']}")
        win_rate = (self.stats['wins'] / self.stats['total_trades'] * 100) if self.stats['total_trades'] > 0 else 0
        print(f"Win Rate:     {win_rate:.2f}%")
        print(f"Total PnL:    ${self.stats['total_pnl']:.2f}")
        print("="*50)
        logger.info("✅ Simulation Complete.")

    def close_trade(self, trade, price, time, reason):
        """Close trade and calculate PnL"""
        pnl = 0.0
        if trade['signal'] == "BUY":
            pnl = price - trade['price']
        else:
            pnl = trade['price'] - price
            
        trade['status'] = f"CLOSED_{reason}"
        trade['exit_time'] = time
        trade['exit_price'] = price
        trade['pnl'] = pnl
        
        # Update Stats (if simulation mode)
        if hasattr(self, 'stats'):
            self.stats['total_trades'] += 1
            self.stats['total_pnl'] += pnl
            if pnl > 0: self.stats['wins'] += 1
            else: self.stats['losses'] += 1
        
        logger.info(f"💰 CLOSED {trade['signal']} ({reason}): PnL {pnl:.5f}")
        
        # Log Closure (Append as new row or update? Simplified append)
        log_entry = trade.copy()
        df = pd.DataFrame([log_entry])
        df.to_csv(self.trade_log_path, mode='a', header=False, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='AzImA Paper Trading')
    parser.add_argument('--mode', type=str, choices=['live', 'sim'], default=None, help='Mode: live or sim')
    parser.add_argument('--start', type=str, default=None, help='Start Date YYYY-MM-DD')
    parser.add_argument('--end', type=str, default=None, help='End Date YYYY-MM-DD')
    
    args = parser.parse_args()
    
    bot = PaperTrader()
    
    if args.mode == 'sim':
        bot.run_simulation(start_date=args.start, end_date=args.end)
    elif args.mode == 'live':
        bot.run_loop()
    else:
        # Interactive Mode
        print("\n🚀 AzImA Paper Trading System v6.0")
        print("----------------------------------")
        print("1. Live/Paper Trading (Real-time Looping)")
        print("2. Simulation (Backtest specific period)")
        choice = input("Select Mode [1/2]: ").strip()
        
        if choice == "2":
            s_date = input("Start Date (YYYY-MM-DD) [Enter for 30 days ago]: ").strip()
            e_date = input("End Date   (YYYY-MM-DD) [Enter for Today]:       ").strip()
            
            s_date = s_date if s_date else None
            e_date = e_date if e_date else None
            
            bot.run_simulation(start_date=s_date, end_date=e_date)
        else:
            bot.run_loop()

