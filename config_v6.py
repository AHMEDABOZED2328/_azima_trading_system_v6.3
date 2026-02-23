#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════
AzImA Trading System v6.0 - REVOLUTIONARY Configuration
═══════════════════════════════════════════════════════════════════

🎯 CRITICAL CHANGES FROM v5:
✅ Triple Barrier Labeling (replaced Q20/Q80 quantiles)
✅ Realistic ATR-based thresholds
✅ Advanced architectures (BiLSTM + Attention, Transformer)
✅ Proper ensemble with out-of-fold predictions
✅ LightGBM meta-learner (instead of XGBoost)
✅ Reduced class weight extremes
✅ Better regularization strategy
✅ SWA (Stochastic Weight Averaging)

Author: Ahmed (AzImA Team)
Date: January 2026
"""

from pathlib import Path
from datetime import datetime
import json

class ConfigV6:
    """
    🚀 v6.0 - Production-Ready Configuration
    
    Key Philosophy:
    - Realistic FOREX thresholds (Triple Barrier)
    - Strong regularization (prevent overfitting)
    - Diverse ensemble (out-of-fold stacking)
    - Conservative risk management
    """
    
    def __init__(self):
        # ══════════════════════════════════════════════════════════
        # Project Structure
        # ══════════════════════════════════════════════════════════
        self.PROJECT_ROOT = Path(__file__).parent
        
        # Directories - ✅ تعريف جميع المجلدات أولاً
        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.RAW_DATA_DIR = self.DATA_DIR / "raw"
        self.PROCESSED_DATA_DIR = self.DATA_DIR / "processed"
        
        # ✅ CRITICAL FIX: Define MODELS_DIR and ENSEMBLE_DIR BEFORE using them
        self.MODELS_DIR = self.PROJECT_ROOT / "models_v6"
        self.ENSEMBLE_DIR = self.MODELS_DIR / "ensemble"
        self.CHECKPOINTS_DIR = self.MODELS_DIR / "checkpoints_v6"
        self.RESULTS_DIR = self.PROJECT_ROOT / "results_v6"
        self.LOGS_DIR = self.PROJECT_ROOT / "logs"
        
        # Model paths - ✅ الآن يمكن استخدام MODELS_DIR بأمان
        self.MODEL_PATH = self.MODELS_DIR / "production_model_dl_v7.keras"
        self.MODEL_TYPE = "BiLSTM_Attention"  # نوع النموذج
        
        # Data Files
        self.DATA_FILE = self.RAW_DATA_DIR / "eurusd_hourly.csv"
        self.PROCESSED_FEATURES_FILE = self.PROCESSED_DATA_DIR / "processed_features_v6.pkl"
        
        # External Data (Week 2 - DXY + Gold)
        self.EXTERNAL_DATA_DIR = self.DATA_DIR / "external"
        self.MERGED_DATA_FILE = self.PROCESSED_DATA_DIR / "eurusd_external_merged.csv"
        self.USE_EXTERNAL_DATA = True  # 🆕 Enable DXY + Gold features (Week 2)!
        # ============================================================
        # ✅ Pipeline Outputs (Week 3 - Features + Labeling Ready)
        # ============================================================

        # After feature engineering
        self.FEATURES_CSV_FILE = self.PROCESSED_DATA_DIR / "eurusd_features_v6.csv"

        # After triple barrier labeling (FINAL training dataset)
        self.LABELED_CSV_FILE = self.PROCESSED_DATA_DIR / "eurusd_features_labeled_v6.csv"

        # Default training input (use this in notebooks/scripts)
        self.TRAINING_DATA_FILE = self.LABELED_CSV_FILE

        
        # ══════════════════════════════════════════════════════════
        # 🔥 TRIPLE BARRIER LABELING (NEW!)
        # ══════════════════════════════════════════════════════════
        self.NUM_CLASSES = 3  # SELL, HOLD, BUY
        self.CLASS_NAMES = ["SELL", "HOLD", "BUY"]
        
        # ✅ Triple Barrier Settings (replaces quantile-based)
        self.LABELING_METHOD = "triple_barrier"  # NEW!
        
        # Triple Barrier Labeling Parameters - ✅ SYNCED WITH add_labels.py
        # These values MUST match the labeling script for consistent results!
        self.ATR_MULTIPLIER_TP = 3.0     # ✅ FIXED: Match add_labels.py (was 1.5)
        self.ATR_MULTIPLIER_SL = 1.5     # ✅ FIXED: Match add_labels.py (was 1.0)
        self.MAX_HOLDING_PERIODS = 24    # ✅ INCREASED: 24h (was 16) to let profitable trades run
        self.MIN_RETURN_THRESHOLD = 0.002  # ✅ FIXED: Match add_labels.py (was 0.0005)
        
        # Expected distribution: ~40% SELL, ~16% HOLD, ~43% BUY (balanced!)
        
        # ✅ Old quantile settings (DEPRECATED - kept for compatibility)
        self.TARGET_QUANTILE_LOW = 0.20   # Not used in v6
        self.TARGET_QUANTILE_HIGH = 0.80  # Not used in v6
        self.USE_ATR_THRESHOLDS = False   # Not used in v6
        
        # ══════════════════════════════════════════════════════════
        # Model Architecture (ADVANCED)
        # ══════════════════════════════════════════════════════════
        self.SEQUENCE_LENGTH = 24 # ✅ IMPROVED: 24 hours (من 48 - أقصر = أسرع تدريب)

        # ✅ Default architecture (BiLSTM + Attention)
        self.DEFAULT_LSTM_UNITS = [128, 64]  # ✅ INCREASED (was [69, 48]) for better learning

        # 🔥 IMPROVED: Balanced Regularization
        self.DROPOUT_RATE = 0.25  # ✅ Slightly increased (was 0.2) for generalization
        self.RECURRENT_DROPOUT = 0.15  # ✅ Slightly increased (was 0.1)
        self.L2_REGULARIZATION = 0.001  # ✅ Increased (was 0.0005) for regularization
        self.LABEL_SMOOTHING = 0.05  # ✅ Reduced (was 0.1) for clearer labels

        # ✅ Architecture options
        self.ARCHITECTURE_TYPES = [
            'attention_bilstm',  # Default: BiLSTM + Multi-Head Attention
            'transformer',       # Alternative: Pure Transformer
            'hybrid'             # Hybrid: CNN + BiLSTM + Attention
        ]

        
        # ══════════════════════════════════════════════════════════
        # Training Parameters
        # ══════════════════════════════════════════════════════════
        self.LEARNING_RATE = 0.001 # ✅ FURTHER INCREASED from 0.0005 (faster learning!)
        self.EPOCHS = 100 # ✅ INCREASED from 50 (مزيد من التدريب)
        self.BATCH_SIZE = 128 # ✅ INCREASED from 64 (أسرع + أفضل generalization)

        
        # Data splits
        self.TRAIN_SPLIT = 0.70           # 70% training
        self.VALIDATION_SPLIT = 0.15      # 15% validation
        self.TEST_SPLIT = 0.15            # 15% test
        
        # Early stopping (more aggressive)
        self.EARLY_STOPPING_PATIENCE = 7  # Reduced from 10
        self.EARLY_STOPPING_MIN_DELTA = 0.0005  # Stricter
        
        # Learning rate scheduler
        self.REDUCE_LR_PATIENCE = 4       # Reduced from 5
        self.REDUCE_LR_FACTOR = 0.5
        self.MIN_LEARNING_RATE = 1e-7
        
        # Random seed
        self.RANDOM_SEED = 42
        
        # ✅ NEW: Stochastic Weight Averaging (SWA)
        self.USE_SWA = True
        self.SWA_START_EPOCH = 25         # Start SWA at epoch 25 (50% of training)
        self.SWA_FREQ = 2                 # Average every 2 epochs
        
        # ══════════════════════════════════════════════════════════
        # 🔥 FIXED: Balanced Class Weights (not extreme!)
        # ══════════════════════════════════════════════════════════
        self.USE_CLASS_WEIGHTS = True
        self.CLASS_WEIGHT_MODE = "balanced"  # Auto-calculate from data
        
        # Custom Class Weights - ✅ BALANCED FOR SELL/BUY
        # ✅ CLASS WEIGHTS (v6.1 Golden Config)
        # 4.0 for SELL/BUY was the sweet spot for PF 0.96
        # ✅ Optimization for Stagnation Fix (v6.4)
        # Inverse Frequency Weights:
        # Class 0 (45%): 0.75
        # Class 1 (21%): 1.55
        # Class 2 (34%): 0.98
        self.CUSTOM_CLASS_WEIGHTS = {
            0: 0.75,  # SELL (Majority - Downweighted)
            1: 1.55,  # HOLD (Minority - Boosted)
            2: 1.00   # BUY (Moderate)
        }
        
        # ✅ SMOTE (class balancing) - DISABLED (causes overfitting!)
        # Better approach: Use class_weights instead of SMOTE
        self.APPLY_SMOTE = False  # ✅ FIXED: Was True, but SMOTE causes data leakage
        self.SMOTE_SAMPLING_STRATEGY = 'auto'
        self.SMOTE_K_NEIGHBORS = 5
        
        # Focal Loss Parameters - ✅ TUNED for Hard Examples
        self.USE_FOCAL_LOSS = True
        self.FOCAL_ALPHA = [0.25, 0.50, 0.25]  # ✅ Emphasize HOLD (Class 1)
        self.FOCAL_GAMMA = 3.0  # ✅ Increased from 2.0 to focus on hard examples
        
        # ══════════════════════════════════════════════════════════
        # Feature Engineering
        # ══════════════════════════════════════════════════════════
        self.PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]
        
        # Feature selection (optimized)
        self.MAX_FEATURES = 65            # Increased from 60 (for new cyclical features)
        self.FEATURE_SELECTION_METHOD = "mutual_info"
        self.CORRELATION_THRESHOLD = 0.90  # Relaxed from 0.85 (allow more correlated features)
        self.FEATURE_IMPORTANCE_THRESHOLD = 0.01
        
        # ══════════════════════════════════════════════════════════
        # 🔥 ENSEMBLE v6.0 (OUT-OF-FOLD STACKING)
        # ══════════════════════════════════════════════════════════
        self.ENSEMBLE_SIZE = 5            # Reduced from 7 (quality over quantity)
        self.ENSEMBLE_METHOD = "stacking_oof"  # ✅ REVERTED: Back to Stacking (Best PF 0.96)
        self.ENSEMBLE_META_LEARNER = "lightgbm"  # Back to LightGBM
        self.ENSEMBLE_TOP_N = 5           # Use all 5 models
        
        # ✅ Diversity requirements
        self.MIN_MODEL_DIVERSITY = 0.05   # Min 5% disagreement rate
        self.MIN_MODEL_ACCURACY = 0.45    # Min 45% test accuracy
        
        # ✅ Out-of-fold CV
        self.OOF_N_FOLDS = 5              # 5-fold CV for meta-features
        
        # ✅ LightGBM meta-learner params (TUNED)
        self.LGBM_PARAMS = {
            'num_leaves': 63,           # Increased from 31 (more complexity)
            'max_depth': 7,             # Increased from 5
            'learning_rate': 0.03,      # Reduced from 0.05 (slower, more robust)
            'n_estimators': 500,        # Increased from 200
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_weight': 5,
            'random_state': 42
        }
        
        # ══════════════════════════════════════════════════════════
        # Confidence & Risk Management
        # ══════════════════════════════════════════════════════════
        self.CONFIDENCE_THRESHOLD = 0.65  # Reduced from 0.70 (more trades)
        self.AGREEMENT_THRESHOLD = 0.60   # 60% models must agree
        
        # Position sizing
        self.INITIAL_CAPITAL = 10000.0
        self.POSITION_SIZE_PCT = 0.05     # 5% per trade (was 2% - aggressive due to low DD)
        self.MAX_POSITIONS = 1
        self.MAX_DAILY_TRADES = 3
        
        # ✅ ATR-based stops (dynamic)
        self.USE_ATR_STOPS = True
        self.ATR_STOP_MULTIPLIER = 2.0    # 2x ATR for SL
        self.ATR_TAKE_PROFIT_MULTIPLIER = 3.0  # 3x ATR for TP (R:R = 1:1.5)
        
        # Fixed stops (fallback)
        self.STOP_LOSS_PCT = 0.012        # 1.2%
        self.TAKE_PROFIT_PCT = 0.036      # 3.6%
        
        # Trailing stop
        self.USE_TRAILING_STOP = True
        self.TRAILING_STOP_ACTIVATION = 0.018  # 1.8%
        self.TRAILING_STOP_DISTANCE = 0.010    # 1.0%
        
        # ✅ Volatility filter
        self.USE_VOLATILITY_FILTER = True
        self.MIN_VOLATILITY_PERCENTILE = 20
        self.MAX_VOLATILITY_PERCENTILE = 95
        
        # ✅ Filter Model (v6.3)
        self.FILTER_MODEL_PATH = self.MODELS_DIR / "filter_model_v1.pkl"
        self.FILTER_THRESHOLD = 0.60      # Minimum confidence for filter to accept trade
        
        # Commission & Slippage
        self.COMMISSION = 0.0002          # 2 pips
        self.SLIPPAGE = 0.0001            # 1 pip
        
        # ══════════════════════════════════════════════════════════
        # 🔥 ENSEMBLE MODEL CONFIGURATIONS v6.0
        # ══════════════════════════════════════════════════════════
        self.ENSEMBLE_CONFIGS = self._generate_ensemble_configs_v6()
        
        # ══════════════════════════════════════════════════════════
        # Create directories
        # ══════════════════════════════════════════════════════════
        self._create_directories()
    
    def _generate_ensemble_configs_v6(self):
        """
        🔥 v6.0: Diverse ensemble with IMPROVED parameters
        Strategy:
        - Mix architectures (BiLSTM, Transformer, Hybrid)
        - Balanced regularization (avoid underfitting)
        - Higher learning rates
        - Mix batch sizes
        """
        configs = [
            # Model 1: Balanced BiLSTM + Attention (baseline) - IMPROVED
            {
                'id': 1,
                'name': 'balanced_bilstm_v6',
                'arch_type': 'attention_bilstm',
                'lstm_units': [96, 48],        # ✅ من [128, 64]
                'dropout_rate': 0.2,           # ✅ REDUCED from 0.3
                'recurrent_dropout': 0.1,      # ✅ REDUCED from 0.2
                'learning_rate': 0.001,        # ✅ INCREASED from 0.0005
                'epochs': 80,
                'batch_size': 128,
                'l2_reg': 0.0005,             # ✅ REDUCED from 0.001
                'seed': 42
            },

            
            # Model 2: Conservative (moderate regularization) - IMPROVED
            {
                'id': 2,
                'name': 'conservative_bilstm_v6',
                'arch_type': 'attention_bilstm',
                'lstm_units': [96, 48],
                'dropout_rate': 0.25,       # ✅ FURTHER REDUCED from 0.4
                'recurrent_dropout': 0.15,  # ✅ FURTHER REDUCED from 0.25
                'learning_rate': 0.0012,    # ✅ INCREASED from 0.0008
                'epochs': 90,               # ✅ من 60
                'batch_size': 128,          # ✅ من 64
                'l2_reg': 0.001,           # ✅ REDUCED from 0.002
                'seed': 777
            },
            
            # Model 3: Transformer Encoder - IMPROVED
            {
                'id': 3,
                'name': 'transformer_v6',
                'arch_type': 'transformer',
                'lstm_units': [128, 64],
                'dropout_rate': 0.2,        # ✅ FURTHER REDUCED from 0.25
                'recurrent_dropout': 0.0,
                'learning_rate': 0.0015,    # ✅ INCREASED from 0.001
                'epochs': 80,               # ✅ من 50
                'batch_size': 64,           # ✅ نفس القيمة (transformer يحب batch أصغر)
                'l2_reg': 0.0003,          # ✅ REDUCED from 0.0005
                'seed': 314
            },
            
            # Model 4: Deep BiLSTM - IMPROVED
            {
                'id': 4,
                'name': 'deep_bilstm_v6',
                'arch_type': 'attention_bilstm',
                'lstm_units': [128, 64, 32],
                'dropout_rate': 0.25,       # ✅ FURTHER REDUCED from 0.35
                'recurrent_dropout': 0.15,  # ✅ REDUCED from 0.2
                'learning_rate': 0.0012,    # ✅ INCREASED from 0.0009
                'epochs': 80,               # ✅ من 50
                'batch_size': 128,          # ✅ من 64
                'l2_reg': 0.0008,          # ✅ REDUCED from 0.0015
                'seed': 1337
            },
            
            # Model 5: Fast Learner - IMPROVED
            {
                'id': 5,
                'name': 'fast_bilstm_v6',
                'arch_type': 'attention_bilstm',
                'lstm_units': [128, 64],
                'dropout_rate': 0.2,        # ✅ FURTHER REDUCED from 0.25
                'recurrent_dropout': 0.1,   # ✅ REDUCED from 0.15
                'learning_rate': 0.002,     # ✅ INCREASED from 0.0015
                'epochs': 70,               # ✅ من 40
                'batch_size': 128,          # ✅ من 64
                'l2_reg': 0.0005,          # ✅ REDUCED from 0.0008
                'seed': 2026
            },
        ]
        
        return configs

    
    def _create_directories(self):
        
        """Create necessary directories"""
        directories = [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.MODELS_DIR,
            self.ENSEMBLE_DIR,
            self.CHECKPOINTS_DIR,
            self.RESULTS_DIR,
            self.LOGS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def print_config_summary(self):
        """Print configuration summary"""
        print("=" * 80)
        print("🚀 AzImA Trading System v6.0 - REVOLUTIONARY Configuration")
        print("=" * 80)
        
        print(f"\n🎯 KEY IMPROVEMENTS FROM v5:")
        print(f" ✅ Triple Barrier Labeling (replaced Q20/Q80)")
        print(f"    • TP: {self.ATR_MULTIPLIER_TP}x ATR")
        print(f"    • SL: {self.ATR_MULTIPLIER_SL}x ATR")
        print(f"    • Max Hold: {self.MAX_HOLDING_PERIODS}h")
        print(f" ✅ Advanced Architectures (BiLSTM + Attention, Transformer)")
        print(f" ✅ Out-of-Fold Stacking ({self.OOF_N_FOLDS}-fold CV)")
        print(f" ✅ LightGBM Meta-Learner (better than XGBoost)")
        print(f" ✅ Stochastic Weight Averaging (SWA)")
        print(f" ✅ Balanced Class Weights (not extreme)")
        
        print(f"\n🏗️ MODEL ARCHITECTURE:")
        print(f" • Classes: {self.NUM_CLASSES} ({', '.join(self.CLASS_NAMES)})")
        print(f" • Sequence Length: {self.SEQUENCE_LENGTH}h")
        print(f" • LSTM Units: {self.DEFAULT_LSTM_UNITS}")
        print(f" • Dropout: {self.DROPOUT_RATE}")
        print(f" • Recurrent Dropout: {self.RECURRENT_DROPOUT}")
        print(f" • L2 Reg: {self.L2_REGULARIZATION}")
        print(f" • Label Smoothing: {self.LABEL_SMOOTHING}")
        
        print(f"\n🎯 TRAINING:")
        print(f" • Epochs: {self.EPOCHS}")
        print(f" • Batch Size: {self.BATCH_SIZE}")
        print(f" • Learning Rate: {self.LEARNING_RATE}")
        print(f" • Early Stopping: {self.EARLY_STOPPING_PATIENCE} epochs")
        print(f" • SWA: {'Enabled' if self.USE_SWA else 'Disabled'}")
        
        print(f"\n🤖 ENSEMBLE:")
        print(f" • Models: {self.ENSEMBLE_SIZE}")
        print(f" • Method: {self.ENSEMBLE_METHOD}")
        print(f" • Meta-Learner: {self.ENSEMBLE_META_LEARNER}")
        print(f" • Out-of-Fold CV: {self.OOF_N_FOLDS} folds")
        print(f" • Min Diversity: {self.MIN_MODEL_DIVERSITY}")
        print(f" • Min Accuracy: {self.MIN_MODEL_ACCURACY}")
        
        print(f"\n💰 RISK MANAGEMENT:")
        print(f" • Capital: ${self.INITIAL_CAPITAL:,.0f}")
        print(f" • Position Size: {self.POSITION_SIZE_PCT*100:.1f}%")
        print(f" • ATR Stops: {'Enabled' if self.USE_ATR_STOPS else 'Disabled'}")
        print(f" • Confidence Threshold: {self.CONFIDENCE_THRESHOLD*100:.0f}%")
        print(f" • Max Daily Trades: {self.MAX_DAILY_TRADES}")
        
        print("=" * 80)
    
    def load_from_json(self, filepath):
        """Load config from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Update attributes from JSON
        if 'atr_multiplier_tp' in config_dict:
            self.ATR_MULTIPLIER_TP = config_dict['atr_multiplier_tp']
        if 'atr_multiplier_sl' in config_dict:
            self.ATR_MULTIPLIER_SL = config_dict['atr_multiplier_sl']
        if 'max_holding_periods' in config_dict:
            self.MAX_HOLDING_PERIODS = config_dict['max_holding_periods']
        
        if 'sequence_length' in config_dict:
            self.SEQUENCE_LENGTH = config_dict['sequence_length']
        if 'dropout_rate' in config_dict:
            self.DROPOUT_RATE = config_dict['dropout_rate']
        if 'l2_regularization' in config_dict:
            self.L2_REGULARIZATION = config_dict['l2_regularization']
        
        if 'learning_rate' in config_dict:
            self.LEARNING_RATE = config_dict['learning_rate']
        if 'batch_size' in config_dict:
            self.BATCH_SIZE = config_dict['batch_size']
        if 'epochs' in config_dict:
            self.EPOCHS = config_dict['epochs']
        
        if 'ensemble_method' in config_dict:
            self.ENSEMBLE_METHOD = config_dict['ensemble_method']
        
        # Regenerate ensemble configs with new parameters
        self.ENSEMBLE_CONFIGS = self._generate_ensemble_configs_v6()
        
        print(f"✅ Config loaded from: {filepath}")
        return self
    
    def save_to_json(self, filepath=None):
        """Save config to JSON"""
        if filepath is None:
            filepath = self.RESULTS_DIR / "config_v6.json"
        
        config_dict = {
            'version': '6.0',
            'creation_date': datetime.now().isoformat(),
            
            # Labeling
            'labeling_method': self.LABELING_METHOD,
            'atr_multiplier_tp': self.ATR_MULTIPLIER_TP,
            'atr_multiplier_sl': self.ATR_MULTIPLIER_SL,
            'max_holding_periods': self.MAX_HOLDING_PERIODS,
            
            # Architecture
            'sequence_length': self.SEQUENCE_LENGTH,
            'default_lstm_units': self.DEFAULT_LSTM_UNITS,
            'dropout_rate': self.DROPOUT_RATE,
            'l2_regularization': self.L2_REGULARIZATION,
            
            # Training
            'learning_rate': self.LEARNING_RATE,
            'batch_size': self.BATCH_SIZE,
            'epochs': self.EPOCHS,
            'use_swa': self.USE_SWA,
            
            # Ensemble
            'ensemble_method': self.ENSEMBLE_METHOD,
            'ensemble_size': self.ENSEMBLE_SIZE,
            'oof_n_folds': self.OOF_N_FOLDS,
            'meta_learner': self.ENSEMBLE_META_LEARNER,
            
            # Risk
            'confidence_threshold': self.CONFIDENCE_THRESHOLD,
            'position_size_pct': self.POSITION_SIZE_PCT,
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"✅ Config saved to: {filepath}")

# Create global config instance
Config = ConfigV6

# Test
if __name__ == "__main__":
    config = ConfigV6()
    config.print_config_summary()
    print("\n✅ Config v6.0 loaded successfully!")
