
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from sklearn.metrics import accuracy_score, classification_report, f1_score
from pathlib import Path
import joblib
import logging

# Import Custom Modules
import json
from datetime import datetime
from config_v6 import ConfigV6
# Import AdaptiveFocalLoss so Keras can load the model
from advanced_architecture_v6 import AdaptiveFocalLoss, SoftVotingEnsemble

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(config):
    """Load and prepare data for ensemble training"""
    data_path = config.LABELED_CSV_FILE
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Time-based split
    n = len(df)
    train_end = int(n * config.TRAIN_SPLIT)
    val_end = int(n * (config.TRAIN_SPLIT + config.VALIDATION_SPLIT))
    
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'target', 'realized_return', 'exit_reason', 'holding_period']]
    
    # Extract features and targets
    X = df[feature_cols].values
    y = df['target'].values
    
    # Clean NaN
    X = np.nan_to_num(X, nan=0.0)
    
    # Split
    X_train = X[:train_end]
    y_train = y[:train_end]
    
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    
    X_test = X[val_end:]
    y_test = y[val_end:]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols

def create_sequences(X, sequence_length):
    """Create sequences for LSTM input"""
    X_seq = []
    # If len(X) < sequence_length, return empty
    if len(X) < sequence_length:
        return np.array([])
        
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
    return np.array(X_seq)

from advanced_architecture_v6 import AdvancedLSTMBuilder

def get_base_models(config):
    """Load all trained base models by rebuilding and loading weights"""
    models = []
    
    # Iterate through defined configs to ensure we match correct architecture
    for model_config in config.ENSEMBLE_CONFIGS:
        name = model_config['name']
        
        # Find directory starting with name
        # We look for a directory that starts with the model name
        potential_dirs = sorted([d for d in config.MODELS_DIR.glob(f"{name}_*") if d.is_dir()])
        
        if not potential_dirs:
            logger.warning(f"⚠️ No directory found for model config: {name}")
            continue
            
        # Use the latest one if multiple
        model_dir = potential_dirs[-1]
        weight_path = model_dir / 'final_model.h5'
        scaler_path = model_dir / 'scaler.pkl'
        
        if not weight_path.exists():
            logger.warning(f"❌ No weights found at {weight_path}")
            continue
            
        if not scaler_path.exists():
             logger.warning(f"❌ No scaler found at {scaler_path}")
             continue
             
        try:
            logger.info(f"Rebuilding & Loading {name} from {model_dir.name}...")
            
            # 1. Build Model
            builder = AdvancedLSTMBuilder(config, model_config)
            # Input shape: (SEQ_LEN, n_features)
            # We need n_features. Let's get it from the scaler or data?
            # We can't know n_features without data. 
            # But we can assume it's same as config.MAX_FEATURES? No.
            # We can load feature_columns.json if it exists.
            
            # Load selected features (for slicing later)
            selected_features = None
            feat_cols_path = model_dir / 'selected_features.pkl'
            
            if feat_cols_path.exists():
                selected_features = joblib.load(feat_cols_path)
            else:
                # Fallback to JSON if PKL not found
                json_path = model_dir / 'feature_columns.json'
                if json_path.exists():
                     import json
                     with open(json_path, 'r') as f:
                         selected_features = json.load(f)
            
            if selected_features is None:
                logger.warning(f"⚠️ No selected features found for {name}. Assuming ALL features used.")
                n_features = scaler.center_.shape[0] if hasattr(scaler, 'center_') else scaler.n_features_in_
            else:
                n_features = len(selected_features)

            input_shape = (config.SEQUENCE_LENGTH, n_features)
            
            if model_config.get('arch_type') == 'transformer':
                model = builder.build_transformer_encoder(input_shape)
            else:
                model = builder.build_attention_bilstm(input_shape)
            
            # 2. Load Weights
            model.load_weights(str(weight_path))
            
            # 3. Load Scaler
            if 'scaler' not in locals():
                 scaler = joblib.load(scaler_path)
            
            models.append({
                'name': name,
                'model': model,
                'scaler': scaler,
                'features': selected_features # Store for slicing
            })
            logger.info(f"✅ Successfully loaded {name} (Features: {n_features})")
            
        except Exception as e:
            logger.error(f"❌ Failed to load {name}: {e}")
            import traceback
            traceback.print_exc()
            
    return models

def generate_predictions_for_set(models, X_set, config, global_feature_cols):
    """
    Generate predictions for a specific dataset (Train, Val, or Test).
    X_set: Raw 2D features (samples, features)
    global_feature_cols: List of all feature names corresponding to X_set columns
    Returns: (n_sequences, n_models * 3) array of probabilities
    """
    seq_len = config.SEQUENCE_LENGTH
    meta_features_list = []
    
    sequences_count = len(X_set) - seq_len
    if sequences_count <= 0:
        logger.warning(f"Dataset too small for sequence length {seq_len}")
        return None
        
    for i, model_info in enumerate(models):
        model = model_info['model']
        scaler = model_info['scaler']
        name = model_info['name']
        model_features = model_info.get('features', None)
        
        # 1. Scale RAW data first (RobustScaler is 2D, fitted on ALL features)
        X_scaled_raw = scaler.transform(X_set)
        
        # 2. Slice features if model uses a subset
        if model_features is not None:
            # Map feature names to indices in the global set
            # Optimized lookup
            try:
                indices = [global_feature_cols.index(f) for f in model_features]
                X_scaled_subset = X_scaled_raw[:, indices]
            except ValueError as e:
                logger.error(f"Feature mismatch for {name}: {e}")
                # Fallback to all if fail (dangerous but prevents crash)
                X_scaled_subset = X_scaled_raw
        else:
            X_scaled_subset = X_scaled_raw
            
        # 3. Create sequences from SCALED SUBSET
        X_seq_scaled = create_sequences(X_scaled_subset, seq_len)
        
        # 4. Predict
        # Use batch prediction
        if i == 0: logger.info(f"Predicting with {name} on {len(X_seq_scaled)} sequences...")
        preds = model.predict(X_seq_scaled, batch_size=config.BATCH_SIZE, verbose=0)
        
        meta_features_list.append(preds)
        
    # Concatenate columns: Model1_Cls0, Model1_Cls1, Model1_Cls2, Model2_Cls0...
    return np.hstack(meta_features_list)

def train_ensemble():
    logger.info("🚀 Starting Ensemble Training Pipeline")
    config = ConfigV6()
    
    # 1. Load Data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols = load_data(config)
    
    # --- Calculate Class Weights ---
    # from sklearn.utils.class_weight import compute_class_weight
    # class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    # class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    # 🔥 FIX: Use Custom Weights from Config to ensure SYMMETRY
    class_weight_dict = config.CUSTOM_CLASS_WEIGHTS
    logger.info(f"⚖️ Class Weights (Forced): {class_weight_dict}")
    
    # Create sample weights
    sample_weights_full = np.array([class_weight_dict[cls] for cls in y_train])
    
    # 2. Load Base Models
    models = get_base_models(config)
    if not models:
        logger.error("❌ No base models loaded. Cannot proceed.")
        return
    
    # 3. Generate Meta-Features
    # Targets must be aligned (sliced by sequence length)
    seq_len = config.SEQUENCE_LENGTH
    
    y_train_seq = y_train[seq_len:]
    y_val_seq = y_val[seq_len:]
    y_test_seq = y_test[seq_len:]
    
    # Align sample weights
    sample_weights_seq = sample_weights_full[seq_len:]
    
    logger.info("Generating Training Meta-Features...")
    X_meta_train = generate_predictions_for_set(models, X_train, config, feature_cols)
    
    logger.info("Generating Validation Meta-Features...")
    X_meta_val = generate_predictions_for_set(models, X_val, config, feature_cols)
    
    logger.info("Generating Test Meta-Features...")
    X_meta_test = generate_predictions_for_set(models, X_test, config, feature_cols)
    
    # Check shapes
    if X_meta_train is None or X_meta_val is None:
        logger.error("Failed to generate meta-features.")
        return

    logger.info(f"Meta-Features Shape (Train): {X_meta_train.shape}")
    
    # --- Evaluate Base Models Individually ---
    # Meta features are [Model1_C0, M1_C1, M1_C2, Model2_C0, ...]
    # We can check accuracy of each model    # 4. Train LightGBM Meta-Learner (Hold-Out Stacking on Validation Set)
    if config.ENSEMBLE_METHOD == "stacking_oof":
        logger.info("🧠 Training LightGBM Meta-Learner on VALIDATION Set (Hold-Out Stacking)...")
        logger.warning("⚠️ Ignoring Training Set for Meta-Learner to avoid Overfitting Leakage")
        
        # Split Validation Set for Ensemble Training/Validation
        # 80% of Val for Ensembling, 20% for Ensemble Early Stopping
        from sklearn.model_selection import train_test_split
        
        # Compute weights for VALIDATION set (since we are stacking on Val)
        sample_weights_val_full = np.array([class_weight_dict[cls] for cls in y_val])
        sample_weights_val_seq = sample_weights_val_full[seq_len:]

        # X_meta_val and y_val_seq are the clean hold-out data from Base Models
        X_ens_train, X_ens_val, y_ens_train, y_ens_val, w_ens_train, w_ens_val = train_test_split(
            X_meta_val, y_val_seq, sample_weights_val_seq, test_size=config.TEST_SPLIT, random_state=42
        )
        
        logger.info(f"Ensemble Data: Train={X_ens_train.shape}, Val={X_ens_val.shape}")
        
        # LightGBM Dataset
        lgb_train = lgb.Dataset(X_ens_train, label=y_ens_train, weight=w_ens_train)
        lgb_val = lgb.Dataset(X_ens_val, label=y_ens_val, reference=lgb_train, weight=w_ens_val)
        
        params = config.LGBM_PARAMS.copy()
        params.update({
            'objective': 'multiclass',
            'num_class': config.NUM_CLASSES,
            'metric': 'multi_logloss',
            'verbose': -1,
            'class_weight': None
        })
        
        evals_result = {}
        
        gbm = lgb.train(
            params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_train, lgb_val],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50),
                lgb.record_evaluation(evals_result)
            ]
        )
        
        ensemble_model = gbm
        
    else:
        # Fallback or Voting
        logger.warning(f"Unknown Ensemble Method: {config.ENSEMBLE_METHOD}")
        return

    # 5. Save Ensemble
    config.ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.ENSEMBLE_DIR / 'lightgbm_ensemble.pkl'
    joblib.dump(ensemble_model, model_path)
    logger.info(f"✅ Ensemble model saved to {model_path}")
    # 6. Evaluate on Test Set
    logger.info("Evaluating on Test Set...")
    y_pred_probs = ensemble_model.predict(X_meta_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    acc = accuracy_score(y_test_seq, y_pred)
    f1 = f1_score(y_test_seq, y_pred, average='weighted')
    
    logger.info("="*60)
    logger.info(f"🏆 ENSEMBLE TEST RESULTS")
    logger.info("="*60)
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info("\n" + classification_report(y_test_seq, y_pred, target_names=config.CLASS_NAMES))
    
    # 7. Final Save (Redundant but ensures file exists)
    logger.info(f"✅ Ensemble Pipeline Complete. Model at {model_path}")

    # ---------------------------------------------------------
    # 📝 R2 Enhancements: Save Ensemble Metrics & LightGBM Internals
    # ---------------------------------------------------------
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save Metrics
    ensemble_eval = {
        'accuracy': float(acc),
        'f1': float(f1),
        'classification_report': classification_report(y_test_seq, y_pred, target_names=config.CLASS_NAMES, output_dict=True),
        'timestamp': ts
    }
    
    with open(config.ENSEMBLE_DIR / f'ensemble_eval_{ts}.json', 'w') as f:
        json.dump(ensemble_eval, f)
        logger.info(f"✅ Saved ensemble evaluation to ensemble_eval_{ts}.json")

    # Save LightGBM evals_result if available
    if 'evals_result' in locals():
         with open(config.ENSEMBLE_DIR / f'lgb_evals_{ts}.json', 'w') as f:
             json.dump(evals_result, f)
             logger.info(f"✅ Saved LightGBM training history to lgb_evals_{ts}.json")

if __name__ == "__main__":
    try:
        train_ensemble()
    except Exception as e:
        logger.error(f"❌ Critical Error in Ensemble Training: {e}")
        import traceback
        traceback.print_exc()
