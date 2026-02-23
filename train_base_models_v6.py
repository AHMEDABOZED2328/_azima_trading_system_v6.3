
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import joblib
import logging
import json
import time
from config_v6 import ConfigV6
from advanced_architecture_v6 import AdvancedLSTMBuilder, AdaptiveFocalLoss

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainBaseModels")

def load_and_prepare_data(config):
    """Load, scale, and prepare sequences"""
    data_path = config.TRAINING_DATA_FILE
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
        
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Drop non-feature columns
    drop_cols = ['timestamp', 'target', 'realized_return', 'exit_reason', 'holding_period']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    # Validate features
    logger.info(f"Features: {len(feature_cols)}")
    
    # ✅ FIX: Drop NaNs (Critical for preventing NaN Loss)
    initial_shape = df.shape
    df = df.dropna()
    dropped_count = initial_shape[0] - df.shape[0]
    if dropped_count > 0:
        logger.warning(f"⚠️ Dropped {dropped_count} rows containing NaNs (likely due to rolling windows)")
    
    X = df[feature_cols].values
    # Encode target
    y = pd.get_dummies(df['target']).values.astype('float32')
    
    # Split
    n = len(df)
    train_end = int(n * config.TRAIN_SPLIT)
    val_end = int(n * (config.TRAIN_SPLIT + config.VALIDATION_SPLIT))
    
    X_train_raw = X[:train_end]
    y_train = y[:train_end]
    X_val_raw = X[train_end:val_end]
    y_val = y[train_end:val_end]
    
    # Create Scaler (RobustScaler)
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    
    # Save Scaler (Generic one, though each model might want its own if doing feature selection)
    # For now, we use one scaler for simplicity, or we can follow the notebook pattern
    # The notebook saved scaler per model. Let's do that in the training loop.
    
    return (X_train_raw, y_train), (X_val_raw, y_val), feature_cols

def create_dataset(X, y, time_steps, batch_size):
    """Create TF Dataset"""
    # Fix Import: It is in utils, not preprocessing
    from tensorflow.keras.utils import timeseries_dataset_from_array
    
    # We need to align y. The yield of timeseries_dataset is (batch, time_steps, features)
    # The target for a sequence ending at t is y[t].
    # So if sequence is [t-L+1 ... t], target is y[t].
    
    # Note: X[:-time_steps] is not correct for alignment if using standard Keras logic.
    # Keras: defaults to target at the END of the sequence.
    
    ds = timeseries_dataset_from_array(
        data=X,
        targets=y[time_steps-1:],
        sequence_length=time_steps,
        sequence_stride=1,
        shuffle=True,
        batch_size=batch_size
    )
    
    # 🔥 ALREADY ONE-HOT ENCODED via load_and_prepare_data (pd.get_dummies)
    # ds = ds.map(lambda x, y: (x, tf.one_hot(tf.cast(y, tf.int32), depth=3)))
    
    return ds

def train_base_models():
    config = ConfigV6()
    config.print_config_summary()
    
    # 1. Load Raw Data
    (X_train_raw, y_train), (X_val_raw, y_val), feature_cols = load_and_prepare_data(config)
    
    # 2. Iterate Models
    for model_cfg in config.ENSEMBLE_CONFIGS:
        name = model_cfg['name']
        logger.info(f"\n{'='*40}\n🚀 Training {name}\n{'='*40}")
        
        # Setup directories
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        model_dir = config.MODELS_DIR / f"{name}_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)

        # 0. Setup Per-Model Logging
        file_handler = logging.FileHandler(model_dir / 'train.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        start_time = time.time()
        
        # Scale Data (Specific to this run to ensure no leakage if we added feature selection later)
        from sklearn.preprocessing import RobustScaler
        from feature_engineering_v6 import AdvancedFeatureEngineerV6
        
        # 1. Scale first
        scaler = RobustScaler()
        X_train_scaled_full = scaler.fit_transform(X_train_raw)
        X_val_scaled_full = scaler.transform(X_val_raw)
        
        # Save scaler
        joblib.dump(scaler, model_dir / 'scaler.pkl')
        
        # 2. 🔥 IN-LOOP FEATURE SELECTION (Leakage Free)
        logger.info("Performing in-loop feature selection...")
        
        # Create temporary dataframe for feature selection (needs column names)
        df_train_fs = pd.DataFrame(X_train_scaled_full, columns=feature_cols)
        # Add target for mutual info calculation (reverse 1-hot for selection)
        y_train_indices = np.argmax(y_train, axis=1)
        
        # Initialize Engineer
        # We use a FRESH engineer for each model to ensure no state leakage
        fs_engineer = AdvancedFeatureEngineerV6(config)
        
        # Fit SELECTION on Training Data
        # We use the internal method directly or fit_transform with select_features=True
        # But fit_transform expects raw dataframe. Our scaler is already applied.
        # Mutual Info works on scaled data too.
        
        # Let's manually trigger the selection logic on the scaled data
        # We can reuse the _fit_balanced_feature_selection method if we pass it the dataframe
        df_train_selected = fs_engineer._fit_balanced_feature_selection(df_train_fs, y_train_indices)
        selected_features = fs_engineer.selected_features_
        
        logger.info(f"✅ Selected {len(selected_features)} features for this model")
        
        # Get indices of selected features
        selected_indices = [feature_cols.index(f) for f in selected_features]
        
        # Subset the arrays
        X_train_selected = X_train_scaled_full[:, selected_indices]
        X_val_selected = X_val_scaled_full[:, selected_indices]
        
        # Save Selected Features
        joblib.dump(selected_features, model_dir / 'selected_features.pkl')
        joblib.dump(selected_indices, model_dir / 'selected_indices.pkl')
        
        # Create Datasets
        train_ds = create_dataset(X_train_selected, y_train, config.SEQUENCE_LENGTH, model_cfg['batch_size'])
        val_ds = create_dataset(X_val_selected, y_val, config.SEQUENCE_LENGTH, model_cfg['batch_size'])
        
        # Build Model
        input_shape = (config.SEQUENCE_LENGTH, len(selected_features))
        builder = AdvancedLSTMBuilder(config, model_cfg)
        
        if model_cfg['arch_type'] == 'transformer':
            model = builder.build_transformer_encoder(input_shape)
        else:
            model = builder.build_attention_bilstm(input_shape)
            
        # Compile
        optimizer = tf.keras.optimizers.Adam(learning_rate=model_cfg['learning_rate'])
        
        # Loss: Use Adaptive Focal Loss
        loss_fn = AdaptiveFocalLoss(gamma=config.FOCAL_GAMMA, alpha=config.FOCAL_ALPHA)
        
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=config.EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
            ModelCheckpoint(str(model_dir / 'final_model.h5'), save_best_only=True, monitor='val_loss', verbose=1)
        ]
        
        # Train
        logger.info(f"Starting training for {model_cfg['epochs']} epochs...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=model_cfg['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Save Feature Columns (Legacy support & Inference consistency)
        with open(model_dir / 'feature_columns.json', 'w') as f:
            json.dump(selected_features, f)

        # ---------------------------------------------------------
        # 📝 R2 Enhancements: History, Metrics, Metadata
        # ---------------------------------------------------------
        
        # 1. Training History
        history_clean = {k: [float(x) for x in v] for k, v in history.history.items()}
        elapsed_seconds = time.time() - start_time
        
        best_epoch = None
        if 'val_loss' in history.history:
            best_epoch = int(np.argmin(history.history['val_loss']) + 1)

        with open(model_dir / 'training_history.json', 'w') as f:
            json.dump({'history': history_clean, 'best_epoch': best_epoch, 'elapsed_seconds': elapsed_seconds}, f)

        # 2. Train Metrics
        train_metrics = {
            'final_train_loss': history_clean.get('loss', [None])[-1],
            'final_val_loss': history_clean.get('val_loss', [None])[-1],
            'final_train_acc': history_clean.get('accuracy', [None])[-1],
            'final_val_acc': history_clean.get('val_accuracy', [None])[-1],
            'best_epoch': best_epoch,
            'elapsed_seconds': elapsed_seconds
        }
        with open(model_dir / 'train_metrics.json', 'w') as f:
            json.dump(train_metrics, f)

        # 3. Metadata
        try:
            import subprocess
            git_rev = subprocess.check_output(['git','rev-parse','HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            git_rev = None
            
        metadata = {
            'timestamp': timestamp, 
            'git_rev': git_rev, 
            'config': str(vars(config)),
            'model_name': name,
            'features_count': len(selected_features)
        }
        with open(model_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f)
            
        logger.info(f"✅ {name} Training Complete. Artifacts saved to {model_dir}")
        
        # Remove handler
        logger.removeHandler(file_handler)
        file_handler.close()

if __name__ == "__main__":
    try:
        train_base_models()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
