import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load the new combined data ---
df = pd.read_csv('combined_satellite_data.csv', parse_dates=['utc_time'])
print("Original column names:")
print(df.columns.tolist())

# --- 2. Clean all column names ---
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
print("\nCleaned column names:")
print(df.columns.tolist())

# --- 3. Merge the duplicate 'y_error' columns ---
# FIX #1: Use the correct double-underscore name found in your output
df['y_error_m'] = df['y_error_m'].fillna(df['y_error__m'])

# FIX #2: Drop the correct double-underscore column name
df = df.drop(columns=['y_error__m'])
print("\nMerged 'y_error' columns successfully.")

# --- 4. Handle remaining missing values (if any) ---
print("\nMissing values before interpolation:")
print(df.isnull().sum())

df = df.interpolate(method='linear', limit_direction='forward')
print("\nMissing values after interpolation:")
print(df.isnull().sum())


# --- 5. Visualize the Cleaned Data ---
print("\nGenerating plots...")
df = df.set_index('utc_time')

error_columns = ['x_error_m', 'y_error_m', 'z_error_m', 'satclockerror_m']
df[error_columns].plot(figsize=(15, 8), subplots=True, layout=(2, 2), title='Satellite Errors Over Time')
plt.tight_layout()
# Note: In a script, plt.show() will pause execution.
# You can save the figures instead if you prefer.
plt.savefig('error_plots_over_time.png')
print("Saved time-series plot to 'error_plots_over_time.png'")
# plt.show()


plt.figure(figsize=(12, 7))
sns.boxplot(data=df[error_columns])
plt.title('Distribution of Satellite Errors')
plt.ylabel('Error (meters)')
plt.grid(True)
plt.savefig('error_distribution_boxplot.png')
print("Saved boxplot to 'error_distribution_boxplot.png'")
# plt.show()

# --- 6. Save the final, clean dataset ---
df.to_csv('cleaned_satellite_data.csv')
print("\n✅ EDA and cleaning complete. Your dataset is now ready for feature engineering.")