import pandas as pd

print("Starting feature engineering process...")

# --- 1. Load the cleaned dataset ---
# We use the index_col to keep 'utc_time' as our index
df = pd.read_csv('cleaned_satellite_data.csv', index_col='utc_time', parse_dates=True)

# --- 2. Define our main target variable ---
# We'll focus on predicting the satellite clock error first.
TARGET = 'satclockerror_m'

# --- 3. Create Lag Features ---
# We'll create 4 lags (representing the state from the previous hour, since data is at 15min intervals)
for i in range(1, 5):
    df[f'lag_{TARGET}_{i}'] = df[TARGET].shift(i)

print("Created lag features.")

# --- 4. Create Rolling Window Features ---
# We'll use a window of 4, which represents 1 hour.
df[f'rolling_mean_{TARGET}'] = df[TARGET].shift(1).rolling(window=4).mean()
df[f'rolling_std_{TARGET}'] = df[TARGET].shift(1).rolling(window=4).std()

print("Created rolling window features.")

# --- 5. Create Time-Based Features ---
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek # Monday=0, Sunday=6

print("Created time-based features.")

# --- 6. Handle Missing Values ---
# The lag and rolling features will create NaNs at the beginning of the dataset.
# The simplest and cleanest way to handle this is to remove those rows.
df = df.dropna()
print(f"\nDropped rows with NaN values. Shape of final data: {df.shape}")

# --- 7. Save the final dataset ---
# This file now contains our features (X) and our target (y)
df.to_csv('final_model_data.csv')

print("\n✅ Feature engineering complete!")
print("Your final dataset is saved to 'final_model_data.csv' and is ready for model training.")