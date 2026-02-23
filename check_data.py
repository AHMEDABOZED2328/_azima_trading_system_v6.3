
import pandas as pd
import numpy as np

FILE = 'c:/Users/ahmed/--/azima_trading_system_v6.3/data/processed/eurusd_features_labeled_v6.csv'
print(f"Loading {FILE}...")

df = pd.read_csv(FILE)

# Check target
print("\nTarget NaN Count:", df['target'].isna().sum())

# Check features
numeric_cols = df.select_dtypes(include=[np.number]).columns
features = [c for c in numeric_cols if c not in ['timestamp', 'target']]
print(f"\nChecking {len(features)} numeric features...")

nans = df[features].isna().sum()
infs = np.isinf(df[features]).sum()

has_nan = nans[nans > 0]
has_inf = infs[infs > 0]

if len(has_nan) > 0:
    print("\n⚠️ Features with NaNs:")
    print(has_nan)
else:
    print("\n✅ No NaNs in features")

if len(has_inf) > 0:
    print("\n⚠️ Features with Infs:")
    print(has_inf)
else:
    print("\n✅ No Infs in features")

# Check if target is valid
print("\nTarget Values:", df['target'].unique())
