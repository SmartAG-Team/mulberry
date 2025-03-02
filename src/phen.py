import pandas as pd
import numpy as np
from datetime import timedelta

# -------------------------------
# Helper function: Calculate Photoperiod
# -------------------------------
def calculate_photoperiod(date, latitude):
    """
    Calculate day length (in hours) for a given date and latitude.
    Uses a simple astronomical formula.
    
    Parameters:
        date (pd.Timestamp): The date.
        latitude (float): Latitude in degrees.
        
    Returns:
        float: Photoperiod (day length) in hours.
    """
    # Convert latitude to radians
    lat_rad = np.radians(latitude)
    
    # Day of year (1-366)
    day_of_year = date.timetuple().tm_yday
    
    # Approximate solar declination (in radians)
    decl = np.radians(23.44) * np.sin(2 * np.pi * (day_of_year - 81) / 365)
    
    # Calculate the hour angle at sunrise/sunset
    cos_omega = -np.tan(lat_rad) * np.tan(decl)
    # Ensure cos_omega is within [-1, 1]
    cos_omega = np.clip(cos_omega, -1, 1)
    omega = np.arccos(cos_omega)
    
    # Day length: (24/π) * omega
    day_length = (24 / np.pi) * omega
    return day_length

# -------------------------------
# Chilling unit function
# -------------------------------
def chilling_unit(T):
    """
    Compute chilling unit for a given temperature.
    Here, we assume a unit of 1 is accumulated when temperature is between 0 and 7°C.
    
    Parameters:
        T (float): Daily average temperature.
    
    Returns:
        float: Chilling unit (1 if T in [0,7], else 0).
    """
    if 0 <= T <= 7:
        return 1
    else:
        return 0

# -------------------------------
# Forcing unit function
# -------------------------------
def forcing_unit(T, photoperiod, T_base=5, alpha=0.1):
    """
    Compute forcing unit based on temperature above a base temperature,
    modulated by a photoperiod factor.
    
    Parameters:
        T (float): Daily average temperature.
        photoperiod (float): Day length in hours.
        T_base (float): Base temperature for forcing (default 5°C).
        alpha (float): Coefficient for photoperiod influence.
    
    Returns:
        float: Forcing unit (0 if T <= T_base).
    """
    if T > T_base:
        # Forcing unit: (T - T_base) multiplied by a photoperiod factor.
        # The photoperiod factor increases with longer days.
        factor = 1 + alpha * (photoperiod - 12)  # assuming 12h is a neutral day length
        return (T - T_base) * factor
    else:
        return 0

# -------------------------------
# Phenology model function
# -------------------------------
def predict_budburst(df, latitude, chilling_threshold, forcing_threshold, T_base=5, alpha=0.1):
    """
    Predict the budburst date based on a two-phase model:
    first accumulate chilling units until a threshold is met, then accumulate forcing units.
    
    Parameters:
        df (pd.DataFrame): DataFrame with columns "date" and "temperature".
                           It should cover the period from autumn into spring.
        latitude (float): Latitude (in degrees) for photoperiod calculation.
        chilling_threshold (float): Total chilling units required to break dormancy.
        forcing_threshold (float): Total forcing units required for budburst.
        T_base (float): Base temperature for forcing (default 5°C).
        alpha (float): Coefficient for photoperiod influence in forcing accumulation.
    
    Returns:
        pd.Timestamp: Predicted budburst date (or None if not reached).
    """
    # Ensure DataFrame is sorted by date
    df = df.sort_values('date').reset_index(drop=True)
    
    chilling_sum = 0.0
    forcing_sum = 0.0
    budburst_date = None
    chilling_phase = True  # Start in the chilling phase
    
    for idx, row in df.iterrows():
        current_date = row['date']
        T = row['temperature']
        # Calculate photoperiod for current date at given latitude
        photoperiod = calculate_photoperiod(current_date, latitude)
        
        if chilling_phase:
            # Accumulate chilling units
            chilling_sum += chilling_unit(T)
            # Check if chilling requirement is met
            if chilling_sum >= chilling_threshold:
                # Start the forcing phase on the next day
                chilling_phase = False
                forcing_sum = 0.0  # Reset forcing accumulation
        else:
            # Accumulate forcing units
            forcing_sum += forcing_unit(T, photoperiod, T_base, alpha)
            if forcing_sum >= forcing_threshold:
                budburst_date = current_date
                break
                
    return budburst_date

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    # For this example, we create a synthetic dataset spanning from October 1 of one year to May 31 of the next.
    start_date = pd.Timestamp("2023-10-01")
    end_date = pd.Timestamp("2024-05-31")
    dates = pd.date_range(start_date, end_date, freq="D")
    
    # Generate synthetic daily average temperatures.
    # Here we simulate a typical seasonal cycle: cool in winter, warm in summer.
    # For simplicity, we use a sinusoidal function plus some noise.
    day_of_year = dates.dayofyear
    # Amplitude and mean chosen for demonstration (values in °C)
    mean_temp = 10 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    noise = np.random.normal(0, 2, len(dates))
    temperatures = mean_temp + noise
    
    # Create DataFrame
    data = pd.DataFrame({
        "date": dates,
        "temperature": temperatures
    })
    
    # Define model parameters (example values)
    latitude = 25.0           # For example, Hechi is around 25° N
    chilling_threshold = 60   # Total chilling units required (e.g., 60 chilling units)
    forcing_threshold = 100   # Total forcing units required for budburst (e.g., 100 units)
    T_base = 5                # Base temperature for forcing accumulation (°C)
    alpha = 0.1               # Photoperiod influence coefficient
    
    budburst = predict_budburst(data, latitude, chilling_threshold, forcing_threshold, T_base, alpha)
    
    if budburst is not None:
        print("Predicted budburst date:", budburst.strftime("%Y-%m-%d"))
    else:
        print("Budburst date could not be determined from the data provided.")
