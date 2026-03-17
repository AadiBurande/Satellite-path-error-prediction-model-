import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import joblib # For saving the model

print("--- Training Final XGBoost Model ---")

# 1. Load the feature-engineered dataset
df = pd.read_csv('final_model_data.csv', index_col='utc_time', parse_dates=True)

# 2. Define Features (X) and Target (y)
TARGET = 'satclockerror_m'
features = [col for col in df.columns if col not in [TARGET, 'satellite_type']]
X = df[features]
y = df[TARGET]

# 3. Chronological Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

# 4. Initialize and Train the XGBoost Model
# Using the default parameters which gave the best results
xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
print("Training the model...")
xgb_model.fit(X_train, y_train)

# 5. Make Predictions on the Test Set
y_pred = xgb_model.predict(X_test)

# 6. Evaluate the Model's Performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print("\n--- Model Evaluation ---")
print(f'Root Mean Squared Error (RMSE): {rmse:.4f} meters')
print(f'Mean Absolute Error (MAE): {mae:.4f} meters')

# 7. Visualize the Results
plt.figure(figsize=(15, 7))
plt.plot(y_test.index, y_test.values, label='Actual Values', color='blue', marker='.')
plt.plot(y_test.index, y_pred, label='Predicted Values', color='red', linestyle='--')
plt.title('XGBoost Model: Actual vs. Predicted Satellite Clock Error')
plt.xlabel('Time')
plt.ylabel('Clock Error (meters)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('final_xgboost_predictions.png')
print("\nPlot saved to 'final_xgboost_predictions.png'")

# 8. Save the Trained Model for Future Use
joblib.dump(xgb_model, 'xgboost_satellite_model.joblib')
print("✅ Model has been saved to 'xgboost_satellite_model.joblib'")

import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- Load your saved model and test data ---
model = joblib.load('xgboost_satellite_model.joblib')
df = pd.read_csv('final_model_data.csv', index_col='utc_time', parse_dates=True)

# --- Recreate the test set to get predictions ---
TARGET = 'satclockerror_m'
features = [col for col in df.columns if col not in [TARGET, 'satellite_type']]
X = df[features]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
y_pred = model.predict(X_test)

# --- Calculate the errors ---
errors = y_test.values - y_pred

# --- Plot the Error Distribution ---
plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True, bins=15)
plt.title('Distribution of Prediction Errors')
plt.xlabel('Prediction Error (meters)')
plt.ylabel('Frequency')
plt.grid(True)
plt.axvline(x=0, color='red', linestyle='--') # Add a line at zero error
plt.savefig('error_distribution.png')

print("✅ Error distribution plot saved to 'error_distribution.png'")

import pandas as pd
import joblib
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- Load your saved model and test data ---
model = joblib.load('xgboost_satellite_model.joblib')
df = pd.read_csv('final_model_data.csv', index_col='utc_time', parse_dates=True)

# --- Recreate the test set to get predictions ---
TARGET = 'satclockerror_m'
features = [col for col in df.columns if col not in [TARGET, 'satellite_type']]
X = df[features]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
y_pred = model.predict(X_test)

# --- Calculate the errors (residuals) ---
errors = y_test.values - y_pred

# --- 1. Create a Q-Q Plot ---
plt.figure(figsize=(8, 6))
stats.probplot(errors, dist="norm", plot=plt)
plt.title("Q-Q Plot of Prediction Errors")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.grid(True)
plt.savefig('qq_plot.png')
print("✅ Q-Q plot saved to 'qq_plot.png'")
# The points should lie closely on the red line for a normal distribution.

# --- 2. Perform the Shapiro-Wilk Test ---
stat, p_value = stats.shapiro(errors)
print("\n--- Shapiro-Wilk Test for Normality ---")
print(f"Statistic: {stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value > alpha:
    print("Conclusion: The errors look normally distributed (fail to reject H0).")
else:
    print("Conclusion: The errors do not look normally distributed (reject H0).")