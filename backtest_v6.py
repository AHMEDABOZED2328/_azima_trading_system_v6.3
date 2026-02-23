
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import json
import logging
from pathlib import Path
from config_v6 import ConfigV6
from triple_barrier_backtest import TripleBarrierBacktest

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_ensemble_model(config):
    """Load the trained LightGBM ensemble model"""
    model_path = config.ENSEMBLE_DIR / 'lightgbm_ensemble.pkl'
    if not model_path.exists():
        raise FileNotFoundError(f"Ensemble model not found at {model_path}")
    
    logger.info(f"Loading ensemble model from {model_path}...")
    return joblib.load(model_path)

def load_data(config):
    """Load labeled data for backtesting"""
    data_path = config.LABELED_CSV_FILE
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        logger.warning("⚠️ 'timestamp' column missing. Generating synthetic timestamps.")
        # Create dummy hourly timestamps starting from 2020-01-01
        dates = pd.date_range(start='2020-01-01', periods=len(df), freq='h')
        df['timestamp'] = dates
    
    return df

def get_base_models(config):
    """
    Load base models (reused logic from train_ensemble_v6, 
    but we only need to know WHERE they are to generate predictions if not already saved)
    
    其实 we should have saved the meta-features or predictions? 
    No, for a fresh backtest on new data we need the full pipeline.
    But here we are backtesting on the TEST set (labeled data).
    
    To avoid reloading all heavy TF models again, we can re-use the Meta-Features generation logic
    OR just run the backtest on the Test Split where we already have ground truth labels?
    
    Wait, backtest needs SIGNALs. Signals come from the Ensemble Model.
    Ensemble Model needs Meta-Features.
    Meta-Features come from Base Models.
    
    So yes, we strictly need to load Base Models to generate predictions for the Test Set 
    (or use the ones generated during training if we saved them... which we didn't explicitly save to disk).
    
    For efficiency, let's just Import the generation logic from train_ensemble_v6.
    """
    from train_ensemble_v6 import get_base_models as get_models_func
    from train_ensemble_v6 import generate_predictions_for_set
    
    return get_models_func(config), generate_predictions_for_set

def run_backtest():
    config = ConfigV6()
    
    # 1. Load Data
    df = load_data(config)
    
    # Feature columns
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'target', 'realized_return', 'exit_reason', 'holding_period']]
    
    # Time-based split to get Test Set
    n = len(df)
    train_end = int(n * config.TRAIN_SPLIT)
    val_end = int(n * (config.TRAIN_SPLIT + config.VALIDATION_SPLIT))
    
    test_df = df.iloc[val_end:].copy()
    
    logger.info(f"Backtesting on Test Set: {len(test_df)} samples")
    
    # 2. Generate Ensemble Predictions
    # We need to run the full stack: Base Models -> Meta Features -> Ensemble -> Class Prediction
    
    # Load Base Models
    from train_ensemble_v6 import get_base_models, generate_predictions_for_set
    
    base_models = get_base_models(config)
    if not base_models:
        logger.error("Failed to load base models.")
        return
        
    # Generate Meta Features for Test Set
    # Note: Test set needs to be processed. 
    # generate_predictions_for_set expects raw 2D array
    X_test_raw = test_df[feature_cols].values
    X_test_raw = np.nan_to_num(X_test_raw, nan=0.0)
    
    logger.info("Generating meta-features for Test Set...")
    X_meta = generate_predictions_for_set(base_models, X_test_raw, config, feature_cols)
    
    if X_meta is None:
        logger.error("Failed to generate meta-features.")
        return
        
    # Load Ensemble Model
    ensemble_model = load_ensemble_model(config)
    
    # Predict
    logger.info("Predicting with Ensemble Model...")
    y_pred_probs = ensemble_model.predict(X_meta)
    
    # --- DEBUG: Inspect Probabilities ---
    logger.info("🔍 Probability Inspection (First 5 samples):")
    logger.info(f"\n{y_pred_probs[:5]}")
    
    mean_probs = np.mean(y_pred_probs, axis=0)
    logger.info(f"📊 Average Probabilities: {mean_probs}")
    
    # --- DYNAMIC THRESHOLDING LOGIC ---
    # Smart Thresholding: Adjust confidence based on Volatility (ATR)
    # High Volatility -> Lower Threshold (Catch Trends)
    # Low Volatility -> Higher Threshold (Avoid Chop)
    
    # 1. Get ATR for volatility context
    seq_len = config.SEQUENCE_LENGTH
    
    atr_col = next((col for col in df.columns if 'ATR_14' in col), 'ATR_14')
    if atr_col in test_df.columns:
        atr = test_df[atr_col].values[seq_len:]
        avg_atr = np.mean(atr)
        std_atr = np.std(atr)
        logger.info(f"📊 Market Volatility (ATR): Mean={avg_atr:.5f}, Std={std_atr:.5f}")
    else:
        logger.warning("⚠️ ATR column not found! Using static threshold.")
        atr = np.zeros(len(y_pred_probs))
        avg_atr = 1.0

    # --- LOAD FILTER MODEL ---
    filter_model = None
    filter_probs = None
    
    if hasattr(config, 'FILTER_MODEL_PATH') and config.FILTER_MODEL_PATH.exists():
        try:
            logger.info(f"Loading Filter Model from {config.FILTER_MODEL_PATH}")
            filter_model = joblib.load(config.FILTER_MODEL_PATH)
            
            # Load feature columns
            feat_path = config.MODELS_DIR / "filter_features.pkl"
            if feat_path.exists():
                filter_cols = joblib.load(feat_path)
                
                # Align features
                # predictions start from seq_len (loss of initial window)
                # So we need features from the same corresponding rows
                offset = len(test_df) - len(y_pred_probs)
                
                # Ensure columns exist
                missing_cols = [c for c in filter_cols if c not in test_df.columns]
                if missing_cols:
                    logger.warning(f"Filter missing {len(missing_cols)} columns, filling 0")
                    for c in missing_cols:
                        test_df[c] = 0
                
                # Extract features for legitimate rows
                X_filter = test_df[filter_cols].iloc[offset:].values
                X_filter = np.nan_to_num(X_filter)
                
                logger.info(f"Generating Filter Probabilities for {len(X_filter)} samples...")
                # Booster.predict returns probabilities directly for binary class
                filter_probs = filter_model.predict(X_filter)
            else:
                logger.error("Filter features file not found!")
        except Exception as e:
            logger.error(f"Failed to load/predict filter model: {e}")
            import traceback
            traceback.print_exc()

    y_pred = []
    forced_holds = 0
    filtered_shorts = 0
    dynamic_thresholds = []
    
    BASE_THRESHOLD = 0.38  # 🔥 Relaxed from 0.40 to let BUYs pass
    SENSITIVITY = 0.5      # How much ATR affects threshold
    MIN_THRESH = 0.30      # Never go below this (safety)
    MAX_THRESH = 0.55      # Never go above this (too strict)

    for i, probs in enumerate(y_pred_probs):
        max_prob = np.max(probs)
        cls = np.argmax(probs)
        
        # Calculate Dynamic Threshold
        if avg_atr > 0:
            # Normalized ATR deviation: (Current - Avg) / Avg
            atr_dev = (atr[i] - avg_atr) / avg_atr
            
            # Logic: Higher ATR -> Lower Threshold
            # threshold = Base - (Deviation * Sensitivity)
            # Example: ATR is 50% higher -> 0.40 - (0.5 * 0.5) = 0.15 (Too low!) -> Clamp it
            current_threshold = BASE_THRESHOLD - (atr_dev * SENSITIVITY)
            current_threshold = np.clip(current_threshold, MIN_THRESH, MAX_THRESH)
        else:
            current_threshold = BASE_THRESHOLD

        dynamic_thresholds.append(current_threshold)

        # 1. Base Confidence Check
        if max_prob < current_threshold:
            y_pred.append(1)  # Force HOLD
            forced_holds += 1
            continue
            
        # 2. Filter Logic (Only for Short Signals)
        if cls == 0 and filter_probs is not None:
            # Check Filter Probability
            f_prob = filter_probs[i]
            
            # If Filter says "Low Quality Short" (< Threshold), reject it
            if f_prob < config.FILTER_THRESHOLD:
                y_pred.append(1) # Force HOLD
                filtered_shorts += 1
                continue
                
        y_pred.append(cls)
            
    y_pred = np.array(y_pred)
    avg_thresh = np.mean(dynamic_thresholds)
    logger.info(f"🛡️ Dynamic Thresholds: Avg={avg_thresh:.3f} (Min={min(dynamic_thresholds):.3f}, Max={max(dynamic_thresholds):.3f})")
    logger.info(f"🛡️ forced HOLDs (Confidence): {forced_holds} ({forced_holds/len(y_pred)*100:.1f}%)")
    logger.info(f"🛡️ Filtered Shorts (Quality): {filtered_shorts} ({filtered_shorts/len(y_pred)*100:.1f}%)")
    
    # 3. Run Triple Barrier Backtest
    # Need ATR values for the test set. 
    # Assuming 'atr' column exists in dataframe?
    # Let's check feature columns or raw data.
    # config.LABELED_CSV_FILE usually has engineered features.
    
    # If 'ATR_14' or similar is in features, use it. 
    # Otherwise recalculate?
    # Let's check columns for 'atr'
    atr_col = next((col for col in df.columns if 'atr' in col.lower()), None)
    
    if atr_col:
        logger.info(f"Using ATR column: {atr_col}")
        atr_values = test_df[atr_col].values
    else:
        logger.warning("No ATR column found! Defaulting to 0.0010 (10 pips)")
        atr_values = np.full(len(test_df), 0.0010)
    
    # Align arrays
    # X_meta generation consumes SEQ_LEN samples.
    # So y_pred corresponds to test_df[SEQ_LEN:]
    seq_len = config.SEQUENCE_LENGTH
    
    test_df_trimmed = test_df.iloc[seq_len:].reset_index(drop=True)
    atr_values_trimmed = atr_values[seq_len:]
    
    if len(y_pred) != len(test_df_trimmed):
        logger.error(f"Shape mismatch! Preds: {len(y_pred)}, Data: {len(test_df_trimmed)}")
        # Truncate to match header/tail
        min_len = min(len(y_pred), len(test_df_trimmed))
        y_pred = y_pred[:min_len]
        test_df_trimmed = test_df_trimmed.iloc[:min_len]
        atr_values_trimmed = atr_values_trimmed[:min_len]
    
    # Initialize Backtest
    tb_backtest = TripleBarrierBacktest(
        atr_multiplier_tp=config.ATR_MULTIPLIER_TP,
        atr_multiplier_sl=config.ATR_MULTIPLIER_SL,
        max_holding_periods=config.MAX_HOLDING_PERIODS,
        initial_capital=config.INITIAL_CAPITAL,
        position_size_pct=config.POSITION_SIZE_PCT,
        verbose=True
    )
    
    # Run
    results = tb_backtest.run_backtest(
        predictions=y_pred,
        prices_df=test_df_trimmed,
        atr_values=atr_values_trimmed
    )
    
    # Save Results
    results_dir = config.RESULTS_DIR / 'backtest_v6_final'
    tb_backtest.save_results(results_dir)
    
    logger.info(f"🎉 Backtest complete! Results saved to {results_dir}")

if __name__ == "__main__":
    try:
        run_backtest()
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
