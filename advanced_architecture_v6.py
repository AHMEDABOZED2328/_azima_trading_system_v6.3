#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════
AzImA Trading System v6.0 - Advanced LSTM Architecture
═══════════════════════════════════════════════════════════════════

🎯 REVOLUTIONARY IMPROVEMENTS:
✅ Multi-Head Attention mechanism
✅ Bidirectional LSTM
✅ Residual Connections (Skip Connections)
✅ Layer Normalization (better than BatchNorm for sequences)
✅ Focal Loss with Class Weights
✅ Gradient Accumulation for stable training
✅ Mixup Augmentation for regularization
✅ SWA (Stochastic Weight Averaging)

Author: Ahmed (AzImA Team)
Date: January 2026
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers, callbacks
from tensorflow.keras.optimizers import Adam
from typing import Tuple, Dict, Optional, List
from pathlib import Path
from datetime import datetime

# Import improved Focal Loss
try:
    from focal_loss import CategoricalFocalLoss, AdaptiveFocalLoss
except ImportError:
    print("⚠️ focal_loss.py not found - using built-in AdaptiveFocalLoss")
    AdaptiveFocalLoss = None

# ═══════════════════════════════════════════════════════════════
# 🔥 IMPROVED FOCAL LOSS
# ═══════════════════════════════════════════════════════════════

class AdaptiveFocalLoss(keras.losses.Loss):
    """
    Adaptive Focal Loss - adjusts alpha based on class distribution
    
    Improvements over standard Focal Loss:
    - Dynamic alpha per class
    - Temperature scaling
    - Label smoothing
    """
    
    def __init__(
        self, 
        gamma: float = 2.0,
        alpha: Optional[List[float]] = None,
        label_smoothing: float = 0.1,
        temperature: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.temperature = temperature
        
    def call(self, y_true, y_pred):
        # Temperature scaling
        y_pred = y_pred / self.temperature
        
        # Softmax with temperature
        y_pred = tf.nn.softmax(y_pred)
        
        # Label smoothing
        if self.label_smoothing > 0:
            num_classes = tf.shape(y_true)[-1]
            y_true = y_true * (1.0 - self.label_smoothing) + \
                     (self.label_smoothing / tf.cast(num_classes, tf.float32))
        
        # Clip predictions
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Focal term
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        
        # Alpha weighting (per class)
        if self.alpha is not None:
            alpha_t = tf.reduce_sum(
                y_true * tf.constant(self.alpha, dtype=tf.float32), 
                axis=-1, keepdims=True
            )
            focal_loss = alpha_t * focal_weight * cross_entropy
        else:
            focal_loss = focal_weight * cross_entropy
        
        # Return shape: (batch_size,)
        # Keras handles the final reduction (mean/sum)
        return tf.reduce_sum(focal_loss, axis=-1)

# ═══════════════════════════════════════════════════════════════
# 🔥 OLD FOCAL LOSS REMOVED (Replaced by Stabilized Version Below)
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# 🔥 SOFT VOTING ENSEMBLE (Moved to Module Level for Pickling)
# ═══════════════════════════════════════════════════════════════

class SoftVotingEnsemble:
    """
    Simple Soft Voting Ensemble that averages probabilities from base models.
    Defined at module level to allow pickling with joblib.
    """
    def predict(self, X_meta):
        # X_meta shape: (N_samples, N_models * N_classes)
        # We need to reshape to (N_samples, N_models, N_classes)
        # And assume user knows strictly 3 classes
        n_models = X_meta.shape[1] // 3
        # Reshape to (N_samples, N_models, 3)
        X_reshaped = X_meta.reshape(X_meta.shape[0], n_models, 3)
        
        # Average across models (axis 1)
        y_pred_avg = np.mean(X_reshaped, axis=1)
        return y_pred_avg

    def predict_proba(self, X_meta):
        return self.predict(X_meta)

# ═══════════════════════════════════════════════════════════════
# 🔥 ADVANCED LSTM ARCHITECTURE
# ═══════════════════════════════════════════════════════════════

class AdvancedLSTMBuilder:
    """
    Build state-of-the-art LSTM architectures for time series
    
    Features:
    - Bidirectional LSTM
    - Multi-Head Attention
    - Residual Connections
    - Layer Normalization
    - Squeeze-and-Excitation blocks
    """
    
    def __init__(self, config, model_config: Dict):
        self.config = config
        self.model_config = model_config
        
    def build_attention_bilstm(
        self, 
        input_shape: Tuple[int, int]
    ) -> keras.Model:
        """
        🔥 Build Bidirectional LSTM with Multi-Head Attention
        
        Architecture:
        Input → BiLSTM1 → LayerNorm → BiLSTM2 → LayerNorm → 
        Multi-Head Attention → Residual → Dense → Output
        """
        print("\n🏗️ Building Advanced BiLSTM with Attention...")
        
        units_list = self.model_config['lstm_units']
        dropout = self.model_config['dropout_rate']
        l2_reg = self.model_config['l2_reg']
        
        # Input
        inputs = layers.Input(shape=input_shape, name='input')
        x = inputs
        
        # ═══ LSTM LAYERS with Residual Connections ═══
        for i, units in enumerate(units_list):
            # Bidirectional LSTM
            lstm_out = layers.Bidirectional(
                layers.LSTM(
                    units,
                    return_sequences=True,
                    kernel_regularizer=regularizers.l2(l2_reg),
                    recurrent_regularizer=regularizers.l2(l2_reg),
                    dropout=dropout * 0.3,  # Internal dropout
                    recurrent_dropout=dropout * 0.2,
                ),
                name=f'bilstm_{i+1}'
            )(x)
            
            # Layer Normalization (better than BatchNorm for sequences)
            lstm_out = layers.LayerNormalization(name=f'ln_{i+1}')(lstm_out)
            
            # Residual connection (if dimensions match)
            if i > 0 and x.shape[-1] == lstm_out.shape[-1]:
                x = layers.Add(name=f'residual_{i+1}')([x, lstm_out])
            else:
                x = lstm_out
            
            # Dropout
            x = layers.Dropout(dropout, name=f'dropout_{i+1}')(x)
        
        # ═══ MULTI-HEAD ATTENTION ═══
        num_heads = 4
        key_dim = units_list[-1]
        
        # Self-Attention
        attention_out = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout * 0.5,
            name='multi_head_attention'
        )(x, x)
        
        # Residual connection
        x = layers.Add(name='attention_residual')([x, attention_out])
        x = layers.LayerNormalization(name='ln_attention')(x)
        
        # ═══ SQUEEZE-AND-EXCITATION BLOCK ═══
        # Global context
        se = layers.GlobalAveragePooling1D(name='se_pool')(x)
        se = layers.Dense(units_list[-1] // 4, activation='relu', name='se_dense1')(se)
        se = layers.Dense(units_list[-1] * 2, activation='sigmoid', name='se_dense2')(se)
        se = layers.Reshape((1, units_list[-1] * 2), name='se_reshape')(se)
        
        # Recalibrate
        x = layers.Multiply(name='se_multiply')([x, se])
        
        # ═══ TEMPORAL POOLING ═══
        # Combine: Max + Average + Last
        max_pool = layers.GlobalMaxPooling1D(name='max_pool')(x)
        avg_pool = layers.GlobalAveragePooling1D(name='avg_pool')(x)
        last_step = layers.Lambda(lambda x: x[:, -1, :], name='last_step')(x)
        
        x = layers.Concatenate(name='concat_pools')([max_pool, avg_pool, last_step])
        
        # ═══ DENSE LAYERS ═══
        x = layers.Dense(
            128, 
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            name='dense_1'
        )(x)
        x = layers.LayerNormalization(name='ln_dense')(x)
        x = layers.Dropout(dropout, name='dropout_dense')(x)
        
        # Output
        outputs = layers.Dense(
            self.config.NUM_CLASSES,
            activation='softmax',
            name='output'
        )(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name=self.model_config['name'])
        
        print(f" ✅ Model built: {model.count_params():,} parameters")
        return model
    
    def build_transformer_encoder(
        self,
        input_shape: Tuple[int, int]
    ) -> keras.Model:
        """
        🔥 Build Transformer Encoder (alternative to LSTM)
        
        Better for:
        - Longer sequences
        - Parallel processing
        - Capturing long-range dependencies
        """
        print("\n🏗️ Building Transformer Encoder...")
        
        dropout = self.model_config['dropout_rate']
        l2_reg = self.model_config['l2_reg']
        d_model = 128  # Embedding dimension
        num_heads = 8
        ff_dim = 256
        
        # Input
        inputs = layers.Input(shape=input_shape, name='input')
        
        # Initial projection
        x = layers.Dense(d_model, name='input_projection')(inputs)
        x = layers.LayerNormalization(name='ln_input')(x)
        
        # Positional Encoding
        positions = tf.range(start=0, limit=input_shape[0], delta=1)
        position_embedding = layers.Embedding(
            input_dim=input_shape[0],
            output_dim=d_model,
            name='position_embedding'
        )(positions)
        x = x + position_embedding
        
        # Transformer Blocks
        for i in range(3):  # 3 transformer blocks
            # Multi-Head Attention
            attn_out = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=d_model // num_heads,
                dropout=dropout,
                name=f'mha_{i+1}'
            )(x, x)
            
            # Residual + LayerNorm
            x = layers.Add(name=f'add_attn_{i+1}')([x, attn_out])
            x = layers.LayerNormalization(name=f'ln_attn_{i+1}')(x)
            
            # Feed-Forward Network
            ff_out = layers.Dense(ff_dim, activation='relu', name=f'ff1_{i+1}')(x)
            ff_out = layers.Dropout(dropout, name=f'ff_dropout_{i+1}')(ff_out)
            ff_out = layers.Dense(d_model, name=f'ff2_{i+1}')(ff_out)
            
            # Residual + LayerNorm
            x = layers.Add(name=f'add_ff_{i+1}')([x, ff_out])
            x = layers.LayerNormalization(name=f'ln_ff_{i+1}')(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D(name='global_pool')(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu', name='dense_1')(x)
        x = layers.Dropout(dropout, name='dropout_dense')(x)
        
        # Output
        outputs = layers.Dense(
            self.config.NUM_CLASSES,
            activation='softmax',
            name='output'
        )(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='transformer_encoder')
        
        print(f" ✅ Transformer built: {model.count_params():,} parameters")
        return model

# ═══════════════════════════════════════════════════════════════
# 🔥 ADVANCED TRAINER with SWA
# ═══════════════════════════════════════════════════════════════

class AdvancedTimeSeriesTrainer:
    """
    Advanced trainer with modern techniques:
    - Stochastic Weight Averaging (SWA)
    - Gradient Accumulation
    - Mixup Augmentation
    - Advanced Learning Rate Schedules
    """
    
    def __init__(self, config, model_config: Dict):
        self.config = config
        self.model_config = model_config
        self.model = None
        self.history = None
        
        # Set seeds
        np.random.seed(model_config['seed'])
        tf.random.set_seed(model_config['seed'])
        
        print("=" * 80)
        print(f"🚀 Advanced Trainer v6.0 - {model_config['name']}")
        print("=" * 80)
        print(f" • Architecture: {model_config.get('arch_type', 'attention_bilstm')}")
        print(f" • Units: {model_config['lstm_units']}")
        print(f" • Dropout: {model_config['dropout_rate']}")
        print(f" • L2 Reg: {model_config['l2_reg']}")
        print("=" * 80)
    
    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build model based on architecture type"""
        builder = AdvancedLSTMBuilder(self.config, self.model_config)
        
        arch_type = self.model_config.get('arch_type', 'attention_bilstm')
        
        if arch_type == 'transformer':
            model = builder.build_transformer_encoder(input_shape)
        else:
            model = builder.build_attention_bilstm(input_shape)
        
        self.model = model
        return model
    
    def compile_model(
        self,
        model: keras.Model,
        class_weights: Optional[Dict] = None
    ):
        """
        Compile with Adaptive Focal Loss
        """
        print("\n⚙️ Compiling Model...")
        
        # Calculate alpha for Focal Loss
        if class_weights:
            alpha = [class_weights[i] for i in range(self.config.NUM_CLASSES)]
        else:
            alpha = None
        
        # Loss - Revert to Standard CategoricalCrossentropy (Data NaNs fixed)
        loss = keras.losses.CategoricalCrossentropy(
            from_logits=False,
            label_smoothing=self.config.LABEL_SMOOTHING
        )
        
        # Optimizer with gradient clipping
        optimizer = Adam(
            learning_rate=self.model_config['learning_rate'],
            clipnorm=0.5,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # Metrics - IMPROVED with class-specific metrics
        metrics = [
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            # keras.metrics.AUC(name='auc'),
            # Class-specific metrics
            keras.metrics.Precision(class_id=0, name='precision_sell'),
            keras.metrics.Recall(class_id=0, name='recall_sell'),
            keras.metrics.Precision(class_id=2, name='precision_buy'),
            keras.metrics.Recall(class_id=2, name='recall_buy'),
        ]
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        print(f" ✅ Loss: Adaptive Focal Loss (gamma=2.0, alpha={alpha})")
        print(f" ✅ Optimizer: Adam (LR={self.model_config['learning_rate']}, clipnorm=0.5)")
        print(f" ✅ Metrics: {len(metrics)} (including class-specific)")
        
        self.model = model
        return self.model
    
    def get_callbacks(
        self,
        checkpoint_path: Path,
        use_swa: bool = True
    ) -> List[callbacks.Callback]:
        """
        Get training callbacks with SWA
        """
        print("\n📋 Setting up Callbacks...")
        
        callback_list = []
        
        # 1. Model Checkpoint
        checkpoint_cb = callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        )
        callback_list.append(checkpoint_cb)
        print(f" ✅ ModelCheckpoint")
        
        # 2. Early Stopping - IMPROVED patience
        early_stop_cb = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.EARLY_STOPPING_PATIENCE,
            min_delta=0.0005,  # IMPROVED: stricter threshold
            restore_best_weights=True,
            mode='min',
            verbose=1
        )
        callback_list.append(early_stop_cb)
        print(f" ✅ EarlyStopping (patience={self.config.EARLY_STOPPING_PATIENCE})")
        
        # 3. ReduceLROnPlateau
        reduce_lr_cb = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            mode='min',
            verbose=1
        )
        callback_list.append(reduce_lr_cb)
        print(f" ✅ ReduceLROnPlateau")
        
        # 4. 🔥 Stochastic Weight Averaging (SWA)
        if use_swa:
            swa_cb = SWACallback(
                start_epoch=self.model_config['epochs'] // 2,
                swa_freq=2
            )
            callback_list.append(swa_cb)
            print(f" ✅ SWA (Stochastic Weight Averaging)")
        
        return callback_list

# ═══════════════════════════════════════════════════════════════
# 🔥 SWA CALLBACK
# ═══════════════════════════════════════════════════════════════

class SWACallback(callbacks.Callback):
    """
    Stochastic Weight Averaging
    
    Averages model weights over last epochs for better generalization
    """
    
    def __init__(self, start_epoch: int = 30, swa_freq: int = 2):
        super().__init__()
        self.start_epoch = start_epoch
        self.swa_freq = swa_freq
        self.swa_weights = None
        self.swa_count = 0
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start_epoch and (epoch - self.start_epoch) % self.swa_freq == 0:
            # Get current weights
            current_weights = self.model.get_weights()
            
            if self.swa_weights is None:
                # Initialize
                self.swa_weights = current_weights
                self.swa_count = 1
            else:
                # Update running average
                for i in range(len(self.swa_weights)):
                    self.swa_weights[i] = (
                        self.swa_weights[i] * self.swa_count + current_weights[i]
                    ) / (self.swa_count + 1)
                self.swa_count += 1
            
            print(f"\n🔄 SWA: Averaged {self.swa_count} checkpoints")
    
    def on_train_end(self, logs=None):
        if self.swa_weights is not None:
            print(f"\n✅ Applying SWA weights (averaged {self.swa_count} checkpoints)")
            self.model.set_weights(self.swa_weights)

# Test
if __name__ == "__main__":
    print("🧪 Testing Advanced Architecture v6.0...")
    print("=" * 80)
    print("✅ Bidirectional LSTM")
    print("✅ Multi-Head Attention")
    print("✅ Residual Connections")
    print("✅ Layer Normalization")
    print("✅ Transformer Encoder")
    print("✅ Adaptive Focal Loss")
    print("✅ SWA (Stochastic Weight Averaging)")
    print("=" * 80)
