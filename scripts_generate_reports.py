import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
from datetime import datetime

# Setup Paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = ROOT_DIR / "results_v6"
BACKTEST_DIR = RESULTS_DIR / "backtest_v6_final"
PLOTS_DIR = RESULTS_DIR / "plots"


def calculate_metrics(equity_df):
    """Calculate comprehensive performance metrics"""
    # Calculate returns
    equity_df["returns"] = equity_df["equity"].pct_change().fillna(0)

    # annualized periods (Hourly data) — use full calendar days for hourly series (24 * 365)
    PERIODS_PER_YEAR = 24 * 365

    # 1. CAGR (Compound Annual Growth Rate)
    total_return = (equity_df["equity"].iloc[-1] / equity_df["equity"].iloc[0]) - 1
    n_periods = len(equity_df)
    n_years = n_periods / PERIODS_PER_YEAR
    if n_years > 0:
        cagr = (equity_df["equity"].iloc[-1] / equity_df["equity"].iloc[0]) ** (
            1 / n_years
        ) - 1
    else:
        cagr = 0

    # 2. Sharpe Ratio
    mean_return = equity_df["returns"].mean()
    std_return = equity_df["returns"].std()
    sharpe = np.sqrt(PERIODS_PER_YEAR) * (mean_return / (std_return + 1e-10))

    # 3. Sortino Ratio (Downside deviation only)
    downside_returns = equity_df["returns"][equity_df["returns"] < 0]
    std_downside = downside_returns.std()
    if std_downside is None or np.isnan(std_downside) or std_downside == 0:
        sortino = float("nan")
    else:
        sortino = np.sqrt(PERIODS_PER_YEAR) * (mean_return / std_downside)

    # 4. Max Drawdown
    equity_df["peak"] = equity_df["equity"].cummax()
    equity_df["drawdown"] = (equity_df["equity"] - equity_df["peak"]) / equity_df[
        "peak"
    ]
    max_drawdown = equity_df["drawdown"].min()

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(max_drawdown),
        "final_equity": float(equity_df["equity"].iloc[-1]),
    }


def generate_plots(equity_df, timestamp, output_dir):
    """Generate and save plots"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set Style
    plt.style.use("ggplot")  # standard matplotlib style

    # 1. Equity Curve
    plt.figure(figsize=(12, 6))
    plt.plot(
        equity_df["step"],
        equity_df["equity"],
        label="Equity",
        color="blue",
        linewidth=1.5,
    )
    plt.title("AzImA v6.3 - Equity Curve")
    plt.xlabel("Hours")
    plt.ylabel("Account Balance ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f"equity_curve_{timestamp}.png")
    plt.close()

    # 2. Drawdown Chart
    plt.figure(figsize=(12, 4))
    plt.fill_between(
        equity_df["step"], equity_df["drawdown"] * 100, color="red", alpha=0.3
    )
    plt.plot(equity_df["step"], equity_df["drawdown"] * 100, color="red", linewidth=1)
    plt.title("Drawdown (%)")
    plt.xlabel("Hours")
    plt.ylabel("Drawdown %")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f"drawdown_{timestamp}.png")
    plt.close()


def main():
    print("🚀 Generating AzImA Reports...")

    equity_file = BACKTEST_DIR / "backtest_equity.csv"
    if not equity_file.exists():
        print(f"❌ Error: {equity_file} not found.")
        return

    # Load Data
    df = pd.read_csv(equity_file)
    print(f"Loaded {len(df)} records from equity curve.")

    # Calculate Metrics
    metrics = calculate_metrics(df)

    # Timestamp for this report run
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = PLOTS_DIR / ts

    # Generate Plots
    generate_plots(df, ts, report_dir)

    # Save Metrics
    metrics_file = RESULTS_DIR / f"metrics_report_{ts}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"✅ Report Generated!")
    print(f"   - Metrics: {metrics_file}")
    print(f"   - Plots: {report_dir}")
    print("-" * 30)
    for k, v in metrics.items():
        print(f"{k:<20}: {v:.4f}")


if __name__ == "__main__":
    main()
