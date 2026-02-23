#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════
AzImA Trading System v6.3 - Filter Model Trainer
═══════════════════════════════════════════════════════════════════

🎯 Goal: Train a secondary "Filter Model" (Binary Classifier) to 
   clean up Short signals.
   
Strategy:
- Inputs: Technical Features (volatility, trend, momentum)
- Target: 1 if Short Trade Profitable, 0 otherwise
- Model: LightGBM Binary Classifier

Author: Ahmed (AzImA Team)
Date: February 2026
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import logging
from config_v6 import ConfigV6
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainFilterModel")

def create_filter_target(df, horizon=24, min_profit=0.002):
    """
    Create binary target for Short signals.
    1 = Good Short (Price drops by min_profit within horizon)
    0 = Bad Short (Price rises or doesn't drop enough)
    
    We want to catch moves that go DOWN by at least min_profit.
    """
    try:
        # Calculate future minimum Low to see maximum potential profit
        # If min(Low[t+1:t+h]) <= Close[t] * (1 - min_profit) -> Success
        future_low = df['low'].rolling(horizon).min().shift(-horizon)
        target_price = df['close'] * (1 - min_profit)
        
        # If future low is below target price, it's a win for Short
        binary_target = (future_low <= target_price).astype(int)
        
        # Determine valid index (where we have future data)
        valid_idx = binary_target.notna()
        return binary_target[valid_idx]
    except Exception as e:
        logger.error(f"Error creating filter target: {e}")
        return None

def train_filter_model():
    config = ConfigV6()
    config.print_config_summary()
    
    logger.info("loading data for filter model...")
    
    # Load labeled data (contains features)
    if not config.LABELED_CSV_FILE.exists():
        logger.error(f"Data file not found: {config.LABELED_CSV_FILE}")
        # Fallback to features file if labeled not found
        if config.FEATURES_CSV_FILE.exists():
             logger.warning(f"Falling back to features file: {config.FEATURES_CSV_FILE}")
             df = pd.read_csv(config.FEATURES_CSV_FILE)
        else:
             return
    else:
        df = pd.read_csv(config.LABELED_CSV_FILE)
    
    logger.info(f"Loaded {len(df)} rows.")

    # 1. Create Target
    # We want to filter SHORT signals. So we train on "What makes a Short profitable?"
    # We use all data to learn this distinction.
    logger.info("Creating binary target for Short Filtering...")
    
    # Target: 1 if Short hits TP (0.2%), 0 otherwise
    target_series = create_filter_target(df, horizon=config.MAX_HOLDING_PERIODS, min_profit=0.002)
    
    if target_series is None:
        logger.error("Failed to create target.")
        return

    # Align align dataframe
    df = df.loc[target_series.index].copy()
    df['filter_target'] = target_series
    
    logger.info(f"Aligned Data: {len(df)} rows")
    logger.info(f"Target Balance (1=Profitable Short): {df['filter_target'].value_counts(normalize=True).to_dict()}")
    
    # 2. Select Features
    # Use features except targets/metadata/timestamps
    exclude_cols = ['timestamp', 'target', 'realized_return', 'exit_reason', 'holding_period', 
                    'filter_target', 'future_return', 'date', 'Unnamed: 0']
    
    # Identify numeric columns only
    feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    X = df[feature_cols]
    y = df['filter_target']
    
    logger.info(f"Training Features: {X.shape[1]}")
    
    # 3. Handling NaN/Inf
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0) # Simple imputation for now
    
    # 4. Split Data (Time-based, last 20% for validation)
    split_idx = int(len(df) * 0.8)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")
    
    # 5. Train LightGBM
    logger.info("Training LightGBM Binary Classifier...")
    
    # Dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Parameters optimized for binary classification
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'n_estimators': 1000,
        'learning_rate': 0.03,
        'num_leaves': 31,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1,
        'seed': 42,
        'is_unbalance': False, # We prefer balanced learning or specific weights
       # 'scale_pos_weight': 1.0 # Adjust if needed
    }
    
    # Callbacks
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50)
    ]
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, test_data],
        callbacks=callbacks
    )
    
    # 6. Evaluate
    y_pred_prob = model.predict(X_test)
    y_pred_class = (y_pred_prob > 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred_class)
    auc = roc_auc_score(y_test, y_pred_prob)
    
    logger.info("\n" + "="*40)
    logger.info("📊 Filter Model Results")
    logger.info("="*40)
    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info(f"Test ROC AUC:  {auc:.4f}")
    logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred_class))
    
    # 7. Save Model & Features
    if not config.MODELS_DIR.exists():
        config.MODELS_DIR.mkdir(parents=True)
        
    logger.info(f"Saving model to {config.FILTER_MODEL_PATH}...")
    joblib.dump(model, config.FILTER_MODEL_PATH)
    
    # Save feature names to ensure alignment during inference
    feature_path = config.MODELS_DIR / "filter_features.pkl"
    joblib.dump(feature_cols, feature_path)
    logger.info(f"Features list saved to {feature_path}")
    
    logger.info("✅ Filter Model Training Phase Complete!")

if __name__ == "__main__":
    train_filter_model()
