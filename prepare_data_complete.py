#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════
Complete Data Preparation Pipeline for AzImA v6.3
═══════════════════════════════════════════════════════════════════

Includes all fixes:
- v6.1 Feature Engineering (balanced per-class MI selection)
- v6.2 Triple Barrier Labeling (symmetric barriers, longer holding)

Author: Ahmed (AzImA Team)
Date: February 2026
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from feature_engineering_v6 import AdvancedFeatureEngineerV6
from triple_barrier_labeling import TripleBarrierLabeler
from config_v6 import ConfigV6
import joblib

# Setup Logging
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "data_prep_complete.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DataPrepComplete")

# Configuration (Matching config_v6.py)
RAW_DATA_PATH = Path('data/raw/eurusd_hourly.csv')
PROCESSED_DATA_PATH = Path('data/processed/eurusd_features_labeled_v6.csv')

def load_raw_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.error(f"Raw data file not found: {path}")
        raise FileNotFoundError(f"Raw data file not found: {path}")
    
    df = pd.read_csv(path)
    # Ensure standard names
    df.columns = [c.lower() for c in df.columns]
    
    # Identify timestamp column
    time_cols = [c for c in df.columns if 'time' in c or 'date' in c]
    if not time_cols:
        raise ValueError("No timestamp column found")
        
    df = df.rename(columns={time_cols[0]: 'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    logger.info(f"Loaded raw data: {df.shape} rows")
    return df

def run_pipeline():
    logger.info("🚀 Starting Complete Data Preparation Pipeline")
    config = ConfigV6()
    
    # 1. Load Data
    try:
        df = load_raw_data(RAW_DATA_PATH)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # 2. Feature Engineering
    logger.info("🛠️ Starting Feature Engineering...")
    # Initialize Engineer with v6 settings
    engineer = AdvancedFeatureEngineerV6(config)
    
    # Fit and transform
    # Note: fit_transform does NOT add 'target', we do that next with Labeler
    # 🔥 FIXED: Disable feature selection here (prevent leakage)
    df_features = engineer.fit_transform(df, select_features=False)
    logger.info(f"Features generated: {df_features.shape}")
    
    # Save feature columns for later
    feature_cols = [c for c in df_features.columns if c not in ['timestamp', 'target']]
    
    # 3. Labeling
    logger.info("🏷️ Starting Triple Barrier Labeling...")
    labeler = TripleBarrierLabeler(
        atr_multiplier_tp=3.0,
        atr_multiplier_sl=1.5,
        max_holding_periods=16,
        min_return_threshold=0.0015,
        use_dynamic_barriers=False,
        enforce_balance=True 
    )
    
    df_labeled = labeler.fit_transform(df_features)
    
    # Check Class Balance
    balance = df_labeled['target'].value_counts(normalize=True)
    logger.info(f"📊 Class Balance:\n{balance}")
    
    # 4. Save
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_labeled.to_csv(PROCESSED_DATA_PATH, index=False)
    logger.info(f"✅ Saved processed data to: {PROCESSED_DATA_PATH}")
    
    # Save Feature Engineer state
    engineer_path = Path('models_v6/feature_engineer.joblib')
    engineer_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(engineer, engineer_path)
    logger.info(f"✅ Saved feature engineer to: {engineer_path}") 

if __name__ == "__main__":
    run_pipeline()
