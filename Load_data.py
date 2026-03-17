import pandas as pd
import os

print("Starting the data loading process...")

# --- Corrected Line ---
# The placeholder has been replaced with the actual column name.
timestamp_col = 'utc_time'

# --- No other changes needed below ---

path_geo = os.path.join('data', 'DATA_GEO_Train.csv')
path_meo1 = os.path.join('data', 'DATA_MEO_Train.csv')
path_meo2 = os.path.join('data', 'DATA_MEO_Train2.csv')

try:
    df_geo = pd.read_csv(path_geo, parse_dates=[timestamp_col])
    df_meo1 = pd.read_csv(path_meo1, parse_dates=[timestamp_col])
    df_meo2 = pd.read_csv(path_meo2, parse_dates=[timestamp_col])
    print("Successfully loaded all three CSV files.")

    df_geo['satellite_type'] = 'GEO'
    df_meo1['satellite_type'] = 'MEO'
    df_meo2['satellite_type'] = 'MEO'
    print("Added 'satellite_type' column for tracking.")

    df_combined = pd.concat([df_geo, df_meo1, df_meo2], ignore_index=True)
    print("Concatenated all DataFrames into a single one.")

    print("\n" + "="*50)
    print("         Combined DataFrame Information")
    print("="*50)
    df_combined.info()

    output_path = 'combined_satellite_data.csv'
    df_combined.to_csv(output_path, index=False)
    print(f"\n✅ Success! Combined data saved to '{output_path}'")

except KeyError:
    print(f"\n❌ Error: Still couldn't find the column named '{timestamp_col}'.")
    print("Please double-check the spelling and case.")
except Exception as e:
    print(f"An error occurred: {e}")