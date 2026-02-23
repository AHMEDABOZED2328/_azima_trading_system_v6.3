"""
🔄 Data Conversion Script for AzImA Trading System v6.3
Converts new multi-currency data to training format with smart feature selection.

Author: AzImA System
Date: 2026-02-08
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime


class DataConverter:
    """Converts multi-currency data to training format."""
    
    def __init__(self, target_features: int = 100, verbose: bool = True):
        self.target_features = target_features
        self.verbose = verbose
        
        # Primary currency
        self.primary_pair = 'EURUSD'
        
        # Correlated pairs to include (based on correlation analysis)
        self.correlated_pairs = ['USDCHF', 'GBPUSD', 'USDCAD']
        
        # Priority timeframes
        self.timeframes = ['H1', 'H4', 'D1']
        
        # Core features to always include (per timeframe)
        self.core_features = [
            'open', 'high', 'low', 'close', 'tick_volume',
            'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
            'RSI_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
            'EMA_20', 'EMA_50',
            'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0', 'BBB_5_2.0', 'BBP_5_2.0',
            'ATRr_14', 'ADX_14', 'WILLR_14', 'CCI_14_0.015'
        ]
        
        # Extra features for primary pair only (H1)
        self.extra_features = [
            'EMA_5', 'EMA_10', 'EMA_100',
            'DMP_14', 'DMN_14',
            # Ichimoku
            'ISA_9', 'ISB_26', 'ITS_9', 'IKS_26', 'ICS_26',
            # Fibonacci
            'fib_fib_0.382', 'fib_fib_0.5', 'fib_fib_0.618',
            # PSAR
            'PSARl_0.02_0.2', 'PSARs_0.02_0.2',
            # Standard deviation
            'STDEV_30',
            # Lag features
            'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_4',
            'close_lag_12', 'close_lag_24',
            # Rolling statistics
            'close_rolling_mean_24', 'close_rolling_std_24',
            'close_rolling_mean_48', 'close_rolling_std_48',
            'close_rolling_mean_72', 'close_rolling_std_72',
            'RSI_14_rolling_mean_24', 'RSI_14_rolling_std_24',
            'RSI_14_rolling_mean_48', 'RSI_14_rolling_std_48'
        ]
        
        # Features for correlated pairs (extended set)
        self.corr_pair_features = [
            'close', 'RSI_14', 'MACD_12_26_9', 'ATRr_14', 'ADX_14',
            'STOCHk_14_3_3', 'WILLR_14', 'EMA_50'
        ]
    
    def log(self, msg):
        """Print log message if verbose."""
        if self.verbose:
            print(msg)
    
    def load_data(self, input_path: str) -> pd.DataFrame:
        """Load input data."""
        self.log(f"📂 Loading data from: {input_path}")
        df = pd.read_csv(input_path)
        self.log(f"   ✅ Loaded {len(df):,} rows × {len(df.columns):,} columns")
        return df
    
    def get_column_if_exists(self, df: pd.DataFrame, col_name: str) -> str | None:
        """Get column name if it exists."""
        if col_name in df.columns:
            return col_name
        return None
    
    def select_features(self, df: pd.DataFrame) -> list:
        """Select features based on strategy."""
        selected = ['time']  # Always include time
        
        self.log("\n🎯 Feature Selection:")
        
        # 1. EURUSD H1 core features (primary timeframe)
        self.log(f"   📊 Adding {self.primary_pair} H1 features...")
        for feat in self.core_features + self.extra_features:
            col = f"{self.primary_pair}_H1_{feat}"
            if col in df.columns:
                selected.append(col)
        
        # Also add ATR_14 (for labeling)
        atr_col = f"{self.primary_pair}_H1_ATR_14"
        if atr_col in df.columns and atr_col not in selected:
            selected.append(atr_col)
        
        self.log(f"      → Added {len(selected) - 1} H1 features")
        
        # 2. EURUSD H4 and D1 (multi-timeframe) - more features
        for tf in ['H4', 'D1']:
            count = 0
            for feat in self.core_features[:15]:  # Top 15 features
                col = f"{self.primary_pair}_{tf}_{feat}"
                if col in df.columns:
                    selected.append(col)
                    count += 1
            self.log(f"      → Added {count} {tf} features")
        
        # 3. Correlated pairs (minimal features)
        for pair in self.correlated_pairs:
            count = 0
            for feat in self.corr_pair_features:
                col = f"{pair}_H1_{feat}"
                if col in df.columns:
                    selected.append(col)
                    count += 1
            self.log(f"   💱 {pair}: Added {count} features")
        
        # Remove duplicates while preserving order
        selected = list(dict.fromkeys(selected))
        
        self.log(f"\n   ✅ Total features selected: {len(selected)}")
        
        return selected
    
    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns to match expected training format."""
        self.log("\n🔄 Renaming columns...")
        
        # Create rename mapping
        rename_map = {
            'time': 'timestamp',
            f'{self.primary_pair}_H1_open': 'open',
            f'{self.primary_pair}_H1_high': 'high',
            f'{self.primary_pair}_H1_low': 'low',
            f'{self.primary_pair}_H1_close': 'close',
            f'{self.primary_pair}_H1_tick_volume': 'volume',
        }
        
        df = df.rename(columns=rename_map)
        
        # Simplify other column names (remove EURUSD_H1_ prefix for primary features)
        new_cols = {}
        for col in df.columns:
            if col.startswith(f'{self.primary_pair}_H1_'):
                new_name = col.replace(f'{self.primary_pair}_H1_', '')
                new_cols[col] = new_name
            elif col.startswith(f'{self.primary_pair}_'):
                # Keep H4 and D1 prefix for clarity
                new_name = col.replace(f'{self.primary_pair}_', '')
                new_cols[col] = new_name
        
        df = df.rename(columns=new_cols)
        
        self.log(f"   ✅ Columns renamed")
        return df
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate output data."""
        self.log("\n🔍 Validating data...")
        
        errors = []
        
        # Check required columns
        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        
        # Check for NaN values
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            self.log(f"   ⚠️ Found {nan_count:,} NaN values - filling with forward fill")
            df = df.ffill().bfill()
        
        # Check data types
        if df['open'].dtype not in [np.float64, np.float32]:
            errors.append(f"Invalid open dtype: {df['open'].dtype}")
        
        if errors:
            for err in errors:
                self.log(f"   ❌ {err}")
            return False
        
        self.log("   ✅ Validation passed")
        return True
    
    def convert(self, input_path: str, output_path: str) -> pd.DataFrame:
        """Main conversion method."""
        self.log("=" * 60)
        self.log("🚀 AzImA Data Conversion Pipeline")
        self.log("=" * 60)
        
        # Load data
        df = self.load_data(input_path)
        
        # Select features
        selected_cols = self.select_features(df)
        df_selected = df[selected_cols].copy()
        
        # Rename columns
        df_final = self.rename_columns(df_selected)
        
        # Validate
        if not self.validate_data(df_final):
            raise ValueError("Data validation failed!")
        
        # Save output
        self.log(f"\n💾 Saving to: {output_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(output_path, index=False)
        
        # Summary
        self.log("\n" + "=" * 60)
        self.log("📊 CONVERSION SUMMARY")
        self.log("=" * 60)
        self.log(f"   Input:  {len(df):,} rows × {len(df.columns):,} columns")
        self.log(f"   Output: {len(df_final):,} rows × {len(df_final.columns):,} columns")
        self.log(f"   Time:   {df_final['timestamp'].iloc[0]} → {df_final['timestamp'].iloc[-1]}")
        self.log(f"   File:   {output_path}")
        self.log("=" * 60)
        
        return df_final


def main():
    parser = argparse.ArgumentParser(description='Convert multi-currency data to training format')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file path')
    parser.add_argument('--output', '-o', default='data/raw/eurusd_hourly.csv', help='Output CSV file path')
    parser.add_argument('--features', '-f', type=int, default=100, help='Target number of features')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode')
    
    args = parser.parse_args()
    
    converter = DataConverter(
        target_features=args.features,
        verbose=not args.quiet
    )
    
    converter.convert(args.input, args.output)
    print("\n✅ Conversion complete!")


if __name__ == "__main__":
    main()
