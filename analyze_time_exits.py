import pandas as pd
import json

trades_path = 'results_v6/backtest_v6_final/backtest_trades.csv'
metrics_path = 'results_v6/backtest_v6_final/backtest_metrics.json'

try:
    df = pd.read_csv(trades_path)
    time_exits = df[df['exit_reason'] == 'time_exit']
    
    print(f"Total Time Exits: {len(time_exits)}")
    print(f"Time Exits Win Rate: {len(time_exits[time_exits['pnl'] > 0]) / len(time_exits) * 100:.2f}%")
    print(f"Average PnL (Time Exits): {time_exits['pnl'].mean():.4f}")
    print(f"Max PnL (Time Exits): {time_exits['pnl'].max():.4f}")
    print(f"Min PnL (Time Exits): {time_exits['pnl'].min():.4f}")
    
    # Check what the holding period actually is
    print(f"Avg Holding Period (Time Exits): {time_exits['holding_period'].mean():.2f}")
    
except Exception as e:
    print(f"Error: {e}")
