import pandas as pd
import numpy as np

# ------------------------------
# Step 1: Load ERA5 hourly historical data
# ------------------------------
# Assume the ERA5 CSV file ("era5_hourly.csv") has columns: "datetime" and "temperature"
era5 = pd.read_csv("era5_hourly.csv", parse_dates=["datetime"])
era5.set_index("datetime", inplace=True)

# ------------------------------
# Step 2: Compute the diurnal cycle from ERA5 data
# ------------------------------
# Calculate the daily mean temperature
era5['daily_mean'] = era5.resample('D')['temperature'].transform('mean')

# Compute hourly deviations from the daily mean
era5['hourly_deviation'] = era5['temperature'] - era5['daily_mean']

# Extract the hour of the day (0-23)
era5['hour'] = era5.index.hour

# Calculate the average hourly deviation (diurnal pattern) for each hour of the day
diurnal_pattern = era5.groupby('hour')['hourly_deviation'].mean()
print("Diurnal pattern (average hourly deviation):")
print(diurnal_pattern)

# ------------------------------
# Step 3: Load CMIP6 daily downscaled projections
# ------------------------------
# Assume the CMIP6 CSV file ("cmip6_daily.csv") has columns: "date" and "temperature"
cmip6 = pd.read_csv("cmip6_daily.csv", parse_dates=["date"])
cmip6.set_index("date", inplace=True)

# ------------------------------
# Step 4: Disaggregate daily CMIP6 data to hourly using the diurnal pattern
# ------------------------------
# Create an empty list to store hourly records
hourly_records = []

# Loop over each day in the CMIP6 daily dataset
for current_date, row in cmip6.iterrows():
    daily_temp = row['temperature']
    # Loop over each hour of the day (0 to 23)
    for hour in range(24):
        # Add the diurnal deviation for this hour to the daily mean
        hourly_temp = daily_temp + diurnal_pattern.loc[hour]
        # Create a timestamp for the current day and hour
        timestamp = pd.Timestamp(current_date) + pd.Timedelta(hours=hour)
        hourly_records.append({'datetime': timestamp, 'temperature': hourly_temp})

# Create a DataFrame for the hourly CMIP6 data
hourly_cmip6 = pd.DataFrame(hourly_records)
hourly_cmip6.set_index('datetime', inplace=True)
hourly_cmip6.sort_index(inplace=True)

# ------------------------------
# Step 5: Save or inspect the resulting hourly CMIP6 dataset
# ------------------------------
print("Sample of disaggregated hourly CMIP6 data:")
print(hourly_cmip6.head(24))

# Optionally, save to CSV
hourly_cmip6.to_csv("cmip6_hourly_projection.csv")
