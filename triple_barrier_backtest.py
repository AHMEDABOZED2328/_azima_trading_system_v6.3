"""
Triple Barrier Backtest Engine for AzImA Trading System v6.3

This module implements a realistic backtest that simulates Triple Barrier execution,
matching the labeling methodology used during training.

Key Features:
- Simulates TP/SL/Max Holding exits
- Tracks exact entry/exit prices
- Calculates realistic costs (spread + commission + slippage)
- Provides detailed trade-by-trade analytics
- Aligns with Triple Barrier labeling logic

Author: AzImA Trading System
Version: 6.3
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TradeResult:
    """Container for individual trade results"""
    entry_idx: int
    entry_time: pd.Timestamp
    entry_price: float
    exit_idx: int
    exit_time: pd.Timestamp
    exit_price: float
    direction: int  # 1 for BUY, -1 for SELL, 0 for HOLD
    signal_class: int  # 0=SELL, 1=HOLD, 2=BUY
    pnl: float
    pnl_pct: float
    exit_reason: str  # 'tp', 'sl', 'time_exit'
    holding_period: int
    atr_at_entry: float
    tp_price: float
    sl_price: float
    max_holding: int
    

class TripleBarrierBacktest:
    """
    Triple Barrier Backtest Engine
    
    Simulates realistic trading with TP/SL/Max Holding exits,
    matching the Triple Barrier labeling methodology.
    """
    
    def __init__(
        self,
        atr_multiplier_tp: float = 3.0,   # ✅ FIXED: Match labeling (was 1.5)
        atr_multiplier_sl: float = 1.5,   # ✅ FIXED: Match labeling (was 1.0)
        max_holding_periods: int = 16,    # ✅ FIXED: Match labeling (was 12)
        spread_pips: float = 1.0,
        commission_pips: float = 0.2,
        slippage_pips: float = 0.5,
        pip_value: float = 0.0001,
        initial_capital: float = 10000.0,
        position_size_pct: float = 0.02,
        verbose: bool = True
    ):
        """
        Initialize Triple Barrier Backtest Engine
        
        Parameters:
        -----------
        atr_multiplier_tp : float
            ATR multiplier for Take Profit (default: 1.5)
        atr_multiplier_sl : float
            ATR multiplier for Stop Loss (default: 1.0)
        max_holding_periods : int
            Maximum holding period in bars (default: 12)
        spread_pips : float
            Bid-ask spread in pips (default: 1.0)
        commission_pips : float
            Commission per trade in pips (default: 0.2)
        slippage_pips : float
            Slippage per trade in pips (default: 0.5)
        pip_value : float
            Value of one pip (default: 0.0001 for EURUSD)
        initial_capital : float
            Starting capital (default: 10000)
        position_size_pct : float
            Position size as % of capital (default: 0.02)
        verbose : bool
            Print progress messages (default: True)
        """
        self.atr_multiplier_tp = atr_multiplier_tp
        self.atr_multiplier_sl = atr_multiplier_sl
        self.max_holding_periods = max_holding_periods
        self.spread_pips = spread_pips
        self.commission_pips = commission_pips
        self.slippage_pips = slippage_pips
        self.pip_value = pip_value
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.verbose = verbose
        
        # Calculate total cost per trade in price units
        self.total_cost_pips = spread_pips + commission_pips + slippage_pips
        self.total_cost_price = self.total_cost_pips * pip_value
        
        # Storage for results
        self.trades: List[TradeResult] = []
        self.equity_curve: np.ndarray = None
        self.metrics: Dict = {}
        
    def run_backtest(
        self,
        predictions: np.ndarray,
        prices_df: pd.DataFrame,
        atr_values: np.ndarray,
        start_idx: int = 0
    ) -> Dict:
        """
        Run Triple Barrier backtest on predictions
        
        Parameters:
        -----------
        predictions : np.ndarray
            Model predictions (0=SELL, 1=HOLD, 2=BUY)
        prices_df : pd.DataFrame
            DataFrame with columns: ['timestamp', 'open', 'high', 'low', 'close']
        atr_values : np.ndarray
            ATR values aligned with predictions
        start_idx : int
            Starting index in prices_df (to account for sequence length)
            
        Returns:
        --------
        Dict : Backtest results and metrics
        """
        if self.verbose:
            print("🚀 Starting Triple Barrier Backtest...")
            print(f"   • Predictions: {len(predictions):,}")
            print(f"   • TP: {self.atr_multiplier_tp}×ATR")
            print(f"   • SL: {self.atr_multiplier_sl}×ATR")
            print(f"   • Max Hold: {self.max_holding_periods}h")
            print(f"   • Total Cost: {self.total_cost_pips:.1f} pips/trade")
        
        self.trades = []
        equity = self.initial_capital
        equity_history = [equity]
        
        # Ensure we have enough data
        max_idx = len(prices_df) - self.max_holding_periods - 1
        
        i = 0
        while i < len(predictions) and (start_idx + i) < max_idx:
            pred = predictions[i]
            
            # Skip HOLD signals
            if pred == 1:
                equity_history.append(equity)
                i += 1
                continue
            
            # Get entry details
            entry_idx = start_idx + i
            entry_time = prices_df.loc[prices_df.index[entry_idx], 'timestamp']
            entry_price = prices_df.loc[prices_df.index[entry_idx], 'close']
            atr = atr_values[i]
            
            # Determine direction
            direction = 1 if pred == 2 else -1  # BUY=1, SELL=-1
            
            # Calculate TP/SL levels
            tp_price = entry_price + (direction * self.atr_multiplier_tp * atr)
            sl_price = entry_price - (direction * self.atr_multiplier_sl * atr)
            
            # Simulate trade execution
            trade_result = self._simulate_trade(
                entry_idx=entry_idx,
                entry_time=entry_time,
                entry_price=entry_price,
                direction=direction,
                signal_class=pred,
                tp_price=tp_price,
                sl_price=sl_price,
                atr=atr,
                prices_df=prices_df
            )
            
            # Update equity
            equity += trade_result.pnl
            equity_history.append(equity)
            
            # Store trade
            self.trades.append(trade_result)
            
            # Skip to after trade exit to avoid overlapping trades
            i = trade_result.exit_idx - start_idx + 1
        
        # Convert equity history to numpy array
        self.equity_curve = np.array(equity_history)
        
        # Calculate metrics
        self.metrics = self._calculate_metrics()
        
        if self.verbose:
            self._print_summary()
        
        return {
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'metrics': self.metrics
        }
    
    def _simulate_trade(
        self,
        entry_idx: int,
        entry_time: pd.Timestamp,
        entry_price: float,
        direction: int,
        signal_class: int,
        tp_price: float,
        sl_price: float,
        atr: float,
        prices_df: pd.DataFrame
    ) -> TradeResult:
        """
        Simulate a single trade with Triple Barrier logic
        
        Returns:
        --------
        TradeResult : Complete trade details
        """
        # Apply entry costs
        effective_entry = entry_price + (direction * self.total_cost_price / 2)
        
        # Search for exit within max holding period
        exit_idx = entry_idx
        exit_reason = 'time_exit'
        exit_price = None
        
        for h in range(1, self.max_holding_periods + 1):
            check_idx = entry_idx + h
            
            if check_idx >= len(prices_df):
                # Reached end of data
                exit_idx = len(prices_df) - 1
                exit_price = prices_df.loc[prices_df.index[exit_idx], 'close']
                break
            
            bar_high = prices_df.loc[prices_df.index[check_idx], 'high']
            bar_low = prices_df.loc[prices_df.index[check_idx], 'low']
            bar_close = prices_df.loc[prices_df.index[check_idx], 'close']
            
            # Check TP/SL hits
            if direction == 1:  # BUY
                if bar_high >= tp_price:
                    exit_idx = check_idx
                    exit_price = tp_price
                    exit_reason = 'tp'
                    break
                elif bar_low <= sl_price:
                    exit_idx = check_idx
                    exit_price = sl_price
                    exit_reason = 'sl'
                    break
            else:  # SELL
                if bar_low <= tp_price:
                    exit_idx = check_idx
                    exit_price = tp_price
                    exit_reason = 'tp'
                    break
                elif bar_high >= sl_price:
                    exit_idx = check_idx
                    exit_price = sl_price
                    exit_reason = 'sl'
                    break
        
        # If no TP/SL hit, exit at max holding
        if exit_price is None:
            exit_idx = min(entry_idx + self.max_holding_periods, len(prices_df) - 1)
            exit_price = prices_df.loc[prices_df.index[exit_idx], 'close']
            exit_reason = 'time_exit'
        
        # Apply exit costs
        effective_exit = exit_price - (direction * self.total_cost_price / 2)
        
        # Calculate P&L
        price_change = (effective_exit - effective_entry) * direction
        pnl_pct = price_change / effective_entry
        pnl = self.initial_capital * self.position_size_pct * pnl_pct
        
        # Get exit time
        exit_time = prices_df.loc[prices_df.index[exit_idx], 'timestamp']
        holding_period = exit_idx - entry_idx
        
        return TradeResult(
            entry_idx=entry_idx,
            entry_time=entry_time,
            entry_price=entry_price,
            exit_idx=exit_idx,
            exit_time=exit_time,
            exit_price=exit_price,
            direction=direction,
            signal_class=signal_class,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=exit_reason,
            holding_period=holding_period,
            atr_at_entry=atr,
            tp_price=tp_price,
            sl_price=sl_price,
            max_holding=self.max_holding_periods
        )
    
    def _calculate_metrics(self) -> Dict:
        """Calculate comprehensive backtest metrics"""
        if len(self.trades) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_trade': 0.0,
                'max_drawdown': 0.0,
                'final_equity': self.initial_capital,
                'total_return': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'total_profit': 0.0,
                'total_loss': 0.0,
                'sell_signals': 0,
                'hold_signals': 0,
                'buy_signals': 0,
                'tp_exits': 0,
                'sl_exits': 0,
                'time_exits': 0,
                'tp_rate': 0.0,
                'sl_rate': 0.0,
                'avg_holding_period': 0.0
            }
        
        # Extract trade data
        pnls = np.array([t.pnl for t in self.trades])
        winning_trades = pnls > 0
        losing_trades = pnls < 0
        
        # Basic metrics
        total_trades = len(self.trades)
        win_count = winning_trades.sum()
        loss_count = losing_trades.sum()
        win_rate = win_count / total_trades if total_trades > 0 else 0.0
        
        # P&L metrics
        total_profit = pnls[winning_trades].sum() if win_count > 0 else 0.0
        total_loss = abs(pnls[losing_trades].sum()) if loss_count > 0 else 0.0
        profit_factor = total_profit / total_loss if total_loss > 0 else np.inf
        avg_trade = pnls.mean()
        avg_win = pnls[winning_trades].mean() if win_count > 0 else 0.0
        avg_loss = pnls[losing_trades].mean() if loss_count > 0 else 0.0
        
        # Drawdown
        running_max = np.maximum.accumulate(self.equity_curve)
        drawdown = (self.equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Returns
        final_equity = self.equity_curve[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        # Exit reason distribution
        exit_reasons = [t.exit_reason for t in self.trades]
        tp_count = exit_reasons.count('tp')
        sl_count = exit_reasons.count('sl')
        time_count = exit_reasons.count('time_exit')
        
        # Signal distribution
        signal_counts = {0: 0, 1: 0, 2: 0}
        for t in self.trades:
            signal_counts[t.signal_class] += 1
        
        # Holding period stats
        holding_periods = [t.holding_period for t in self.trades]
        avg_holding = np.mean(holding_periods)
        
        return {
            'total_trades': total_trades,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade': avg_trade,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'max_drawdown': max_drawdown,
            'final_equity': final_equity,
            'total_return': total_return,
            'tp_exits': tp_count,
            'sl_exits': sl_count,
            'time_exits': time_count,
            'tp_rate': tp_count / total_trades if total_trades > 0 else 0.0,
            'sl_rate': sl_count / total_trades if total_trades > 0 else 0.0,
            'sell_signals': signal_counts[0],
            'hold_signals': signal_counts[1],
            'buy_signals': signal_counts[2],
            'avg_holding_period': avg_holding
        }
    
    def _print_summary(self):
        """Print backtest summary"""
        m = self.metrics
        
        print("\n" + "="*80)
        print("📊 TRIPLE BARRIER BACKTEST RESULTS")
        print("="*80)
        
        print(f"\n💼 Trading Activity:")
        print(f"   • Total Trades:     {m['total_trades']:,}")
        print(f"   • SELL Signals:     {m['sell_signals']:,}")
        print(f"   • BUY Signals:      {m['buy_signals']:,}")
        print(f"   • Avg Holding:      {m['avg_holding_period']:.1f} bars")
        
        print(f"\n🎯 Performance:")
        print(f"   • Win Rate:         {m['win_rate']*100:.2f}%")
        print(f"   • Profit Factor:    {m['profit_factor']:.3f}")
        print(f"   • Avg Trade:        ${m['avg_trade']:.2f}")
        print(f"   • Avg Win:          ${m['avg_win']:.2f}")
        print(f"   • Avg Loss:         ${m['avg_loss']:.2f}")
        
        print(f"\n📈 Returns:")
        print(f"   • Initial Capital:  ${self.initial_capital:,.2f}")
        print(f"   • Final Equity:     ${m['final_equity']:,.2f}")
        print(f"   • Total Return:     {m['total_return']*100:.2f}%")
        print(f"   • Max Drawdown:     {m['max_drawdown']*100:.2f}%")
        
        print(f"\n🚪 Exit Analysis:")
        print(f"   • TP Exits:         {m['tp_exits']:,} ({m['tp_rate']*100:.1f}%)")
        print(f"   • SL Exits:         {m['sl_exits']:,} ({m['sl_rate']*100:.1f}%)")
        print(f"   • Time Exits:       {m['time_exits']:,}")
        
        print("="*80)
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame for analysis"""
        if len(self.trades) == 0:
            return pd.DataFrame()
        
        trades_data = []
        for t in self.trades:
            trades_data.append({
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'direction': 'BUY' if t.direction == 1 else 'SELL',
                'signal_class': t.signal_class,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct * 100,
                'exit_reason': t.exit_reason,
                'holding_period': t.holding_period,
                'tp_price': t.tp_price,
                'sl_price': t.sl_price,
                'atr': t.atr_at_entry
            })
        
        return pd.DataFrame(trades_data)
    
    def save_results(self, output_dir: Path):
        """Save backtest results to files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save trades
        trades_df = self.get_trades_dataframe()
        trades_path = output_dir / 'backtest_trades.csv'
        trades_df.to_csv(trades_path, index=False)
        print(f"✅ Trades saved to: {trades_path}")
        
        # Save equity curve
        equity_df = pd.DataFrame({
            'step': np.arange(len(self.equity_curve)),
            'equity': self.equity_curve
        })
        equity_path = output_dir / 'backtest_equity.csv'
        equity_df.to_csv(equity_path, index=False)
        print(f"✅ Equity curve saved to: {equity_path}")
        
        # Save metrics
        import json
        metrics_path = output_dir / 'backtest_metrics.json'
        with open(metrics_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            metrics_json = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                          for k, v in self.metrics.items()}
            json.dump(metrics_json, f, indent=2)
        print(f"✅ Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    print("Triple Barrier Backtest Engine v6.3")
    print("This module should be imported, not run directly.")
