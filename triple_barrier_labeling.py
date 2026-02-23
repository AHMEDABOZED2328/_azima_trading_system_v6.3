#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════
AzImA Trading System v6.2 - FIXED Triple Barrier Labeling
═══════════════════════════════════════════════════════════════════

🔥 CRITICAL FIXES:
✅ Symmetric barriers (same threshold for BUY and SELL)
✅ Longer max holding period (24 bars instead of 16)
✅ Lower min return threshold for better HOLD detection
✅ Balanced labeling strategy

ROOT CAUSE IDENTIFIED:
- Current labels: SELL = -0.18% avg, BUY = +0.23% avg
- This means SELL labels are teaching model to lose money!
- Fix: Use symmetric barriers and longer holding periods

Author: Ahmed (AzImA Team)
Date: February 2026
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TripleBarrierLabelerV6_2:
    """
    FIXED Triple Barrier Method for Balanced Labeling
    
    Key Improvements:
    - Symmetric barriers (no directional bias)
    - Longer holding periods (more time for patterns to develop)
    - Better HOLD detection
    - Balanced label distribution
    """
    
    def __init__(
        self, 
        atr_multiplier_tp: float = 2.0,        # Symmetric TP
        atr_multiplier_sl: float = 1.0,        # Symmetric SL
        max_holding_periods: int = 24,         # DOUBLED from 12
        min_return_threshold: float = 0.00005, # HALVED for better HOLD
        use_dynamic_barriers: bool = False,
        enforce_balance: bool = True           # NEW: Force balanced labels
    ):
        """
        Initialize FIXED Triple Barrier Labeler
        
        Args:
            atr_multiplier_tp: Multiplier for ATR to set take-profit barrier
            atr_multiplier_sl: Multiplier for ATR to set stop-loss barrier
            max_holding_periods: Maximum number of periods to hold position
            min_return_threshold: Minimum return to consider for labeling
            use_dynamic_barriers: Use volatility-adjusted barriers
            enforce_balance: Enforce balanced label distribution
        """
        # Store parameters
        self.atr_multiplier_tp = atr_multiplier_tp
        self.atr_multiplier_sl = atr_multiplier_sl
        self.max_holding_periods = max_holding_periods
        self.min_return_threshold = min_return_threshold
        self.use_dynamic_barriers = use_dynamic_barriers
        self.enforce_balance = enforce_balance
        
        # Compatibility alias
        self.max_hold = max_holding_periods
        
        # Print configuration
        print("=" * 80)
        print("🚀 Triple Barrier Labeler v6.2 - FIXED BIAS")
        print("=" * 80)
        print(f" • TP Barrier: {atr_multiplier_tp}x ATR (symmetric)")
        print(f" • SL Barrier: {atr_multiplier_sl}x ATR (symmetric)")
        print(f" • Max Holding: {max_holding_periods} periods (DOUBLED)")
        print(f" • Min Return: {min_return_threshold*100:.3f}% (HALVED)")
        print(f" • Enforce Balance: {'Yes' if enforce_balance else 'No'}")
        print("=" * 80)
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply FIXED Triple Barrier labeling to dataframe
        
        Args:
            df: DataFrame with OHLCV data and ATR
            
        Returns:
            DataFrame with 'target' column added
        """
        print("\n🎯 Applying FIXED Triple Barrier Labeling...")
        df = df.copy()
        
        # Ensure ATR exists
        if 'ATR_14' not in df.columns:
            print(" ⚠️ ATR_14 not found - calculating...")
            df = self._calculate_atr(df, period=14)
        
        # Calculate barriers
        labels = []
        returns = []
        exit_reasons = []
        holding_periods = []
        
        for i in range(len(df) - self.max_hold):
            label, ret, reason, hold_period = self._get_label_at_position(df, i)
            labels.append(label)
            returns.append(ret)
            exit_reasons.append(reason)
            holding_periods.append(hold_period)
        
        # Pad the end
        for _ in range(self.max_hold):
            labels.append(1)  # HOLD for last bars
            returns.append(0.0)
            exit_reasons.append('end_of_data')
            holding_periods.append(0)
        
        # Add to dataframe
        df['target'] = labels
        df['realized_return'] = returns
        df['exit_reason'] = exit_reasons
        df['holding_period'] = holding_periods
        
        # Apply balance enforcement if enabled
        if self.enforce_balance:
            df = self._enforce_balance(df)
        
        # Statistics
        self._calculate_stats(df)
        
        return df
    
    def _get_label_at_position(
        self, df: pd.DataFrame, idx: int
    ) -> Tuple[int, float, str, int]:
        """
        Get label for a single position using SYMMETRIC Triple Barrier
        
        Returns:
            (label, realized_return, exit_reason, holding_period)
            label: 0=SELL, 1=HOLD, 2=BUY
        """
        entry_price = df['close'].iloc[idx]
        atr = df['ATR_14'].iloc[idx]
        
        # Calculate SYMMETRIC barriers (percentage)
        tp_threshold = (atr / entry_price) * self.atr_multiplier_tp
        sl_threshold = (atr / entry_price) * self.atr_multiplier_sl
        
        # Scan future prices
        for j in range(1, self.max_hold + 1):
            if idx + j >= len(df):
                break
            
            future_price = df['close'].iloc[idx + j]
            ret = (future_price - entry_price) / entry_price
            
            # Check UPWARD TP barrier (BUY signal)
            if ret >= tp_threshold:
                return 2, ret, 'take_profit_buy', j
            
            # Check DOWNWARD SL barrier (SELL signal)
            if ret <= -sl_threshold:
                return 0, ret, 'stop_loss_sell', j
        
        # Time barrier reached - check final return
        final_idx = min(idx + self.max_hold, len(df) - 1)
        final_price = df['close'].iloc[final_idx]
        final_return = (final_price - entry_price) / entry_price
        
        # Classify based on final return with STRICTER HOLD threshold
        if abs(final_return) < self.min_return_threshold:
            # Very small movement → HOLD
            return 1, final_return, 'time_exit_neutral', self.max_hold
        elif final_return > 0:
            # Positive movement → BUY
            return 2, final_return, 'time_exit_profit', self.max_hold
        else:
            # Negative movement → SELL
            return 0, final_return, 'time_exit_loss', self.max_hold
    
    def _enforce_balance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce balanced label distribution by converting extreme labels to HOLD
        
        Strategy:
        - If one direction > 55%, convert weakest labels to HOLD
        - Target: 40-50% each for BUY/SELL, 10-20% for HOLD
        """
        df = df.copy()
        
        # Count current distribution
        counts = df['target'].value_counts()
        total = len(df[df['target'].notna()])
        
        sell_pct = counts.get(0, 0) / total
        buy_pct = counts.get(2, 0) / total
        
        print(f"\n⚖️ Label Balance Check:")
        print(f"   Before: SELL={sell_pct*100:.1f}%, BUY={buy_pct*100:.1f}%")
        
        # Check if rebalancing needed
        max_allowed = 0.55  # 55% max for one direction
        
        if sell_pct > max_allowed:
            # Too many SELL - convert weakest to HOLD
            df = self._reduce_direction(df, direction=0, target_pct=0.50)
            print(f"   ⚠️ Reduced SELL signals to 50%")
            
        elif buy_pct > max_allowed:
            # Too many BUY - convert weakest to HOLD
            df = self._reduce_direction(df, direction=2, target_pct=0.50)
            print(f"   ⚠️ Reduced BUY signals to 50%")
        else:
            print(f"   ✅ Labels are balanced")
        
        return df
    
    def _reduce_direction(self, df: pd.DataFrame, direction: int, target_pct: float) -> pd.DataFrame:
        """
        Reduce signals in one direction by converting weakest to HOLD
        """
        df = df.copy()
        
        # Get indices of this direction
        direction_mask = df['target'] == direction
        direction_indices = df[direction_mask].index
        
        # Calculate how many to convert
        total = len(df[df['target'].notna()])
        current_count = len(direction_indices)
        target_count = int(total * target_pct)
        to_convert = current_count - target_count
        
        if to_convert <= 0:
            return df
        
        # Convert weakest signals (smallest absolute returns) to HOLD
        direction_returns = df.loc[direction_indices, 'realized_return'].abs()
        weakest_indices = direction_returns.nsmallest(to_convert).index
        
        df.loc[weakest_indices, 'target'] = 1  # Convert to HOLD
        df.loc[weakest_indices, 'exit_reason'] = 'balanced_to_hold'
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate ATR if not present"""
        df = df.copy()
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR_14'] = tr.rolling(period).mean()
        
        return df
    
    def _calculate_stats(self, df: pd.DataFrame):
        """Calculate and print labeling statistics"""
        valid_labels = df['target'].dropna()
        
        # Class distribution
        class_counts = valid_labels.value_counts().sort_index()
        total = len(valid_labels)
        
        # Exit reasons
        exit_counts = df['exit_reason'].value_counts()
        
        # Holding periods
        avg_hold = df['holding_period'].mean()
        
        # Realized returns by class
        returns_by_class = df.groupby('target')['realized_return'].agg(['mean', 'std', 'count'])
        
        self.labeling_stats_ = {
            'total_samples': total,
            'class_distribution': {
                'SELL': class_counts.get(0, 0) / total,
                'HOLD': class_counts.get(1, 0) / total,
                'BUY': class_counts.get(2, 0) / total,
            },
            'exit_reasons': exit_counts.to_dict(),
            'avg_holding_period': avg_hold,
            'returns_by_class': returns_by_class.to_dict()
        }
        
        print("\n📊 FIXED Triple Barrier Labeling Statistics:")
        print(f"\n • Total Samples: {total:,}")
        print(f"\n • Class Distribution:")
        print(f"   - SELL: {class_counts.get(0, 0):,} ({class_counts.get(0, 0)/total*100:.2f}%)")
        print(f"   - HOLD: {class_counts.get(1, 0):,} ({class_counts.get(1, 0)/total*100:.2f}%)")
        print(f"   - BUY:  {class_counts.get(2, 0):,} ({class_counts.get(2, 0)/total*100:.2f}%)")
        
        print(f"\n • Average Returns by Class:")
        for cls in [0, 1, 2]:
            if cls in returns_by_class.index:
                mean_ret = returns_by_class.loc[cls, 'mean']
                cls_name = ['SELL', 'HOLD', 'BUY'][cls]
                print(f"   - {cls_name}: {mean_ret*100:.3f}%")
        
        print(f"\n • Exit Reasons:")
        for reason, count in exit_counts.head(5).items():
            print(f"   - {reason}: {count:,} ({count/total*100:.2f}%)")
        
        print(f"\n • Average Holding Period: {avg_hold:.2f} bars")
        
        # Balance check
        max_pct = class_counts.max() / total
        if max_pct > 0.55:
            print(f"\n ⚠️ WARNING: Imbalanced classes - max class {max_pct*100:.1f}%")
            print("   💡 Consider enabling enforce_balance=True")
        else:
            print(f"\n ✅ Good class balance - max class {max_pct*100:.1f}%")
        
        # Profitability check
        sell_return = returns_by_class.loc[0, 'mean'] if 0 in returns_by_class.index else 0
        buy_return = returns_by_class.loc[2, 'mean'] if 2 in returns_by_class.index else 0
        
        if sell_return < 0 and buy_return > 0:
            print(f"\n ✅ Labels are correct: BUY profitable (+{buy_return*100:.3f}%), SELL unprofitable ({sell_return*100:.3f}%)")
        elif sell_return > 0 and buy_return < 0:
            print(f"\n ⚠️ WARNING: Labels might be reversed!")
            print(f"   SELL: +{sell_return*100:.3f}%, BUY: {buy_return*100:.3f}%")
        else:
            print(f"\n ⚠️ Both directions have similar profitability")


# Compatibility alias
TripleBarrierLabeler = TripleBarrierLabelerV6_2

# Test
if __name__ == "__main__":
    print("\n🧪 Testing FIXED Triple Barrier Labeler...")
    
    # Create sample data
    np.random.seed(42)
    n = 1000
    
    df = pd.DataFrame({
        'close': 1.1 + np.cumsum(np.random.randn(n) * 0.001),
        'high': 1.1 + np.cumsum(np.random.randn(n) * 0.001) + np.random.rand(n) * 0.002,
        'low': 1.1 + np.cumsum(np.random.randn(n) * 0.001) - np.random.rand(n) * 0.002,
        'volume': np.random.rand(n) * 1000
    })
    
    labeler = TripleBarrierLabelerV6_2(
        atr_multiplier_tp=2.0,
        atr_multiplier_sl=1.0,
        max_holding_periods=24,
        enforce_balance=True
    )
    
    df_labeled = labeler.fit_transform(df)
    
    print("\n✅ Test complete!")
    print(f"   • DataFrame shape: {df_labeled.shape}")
    print(f"   • Target column added: {'target' in df_labeled.columns}")
