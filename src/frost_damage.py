import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def compute_forcing(hourly_temps, T_base=5.0):
    """
    Compute cumulative forcing from hourly temperatures.
    
    Parameters:
    -----------
    hourly_temps : pd.Series
        Hourly temperature data (indexed by datetime).
    T_base : float
        Base temperature for forcing accumulation (°C).
    
    Returns:
    --------
    pd.Series
        Cumulative forcing (degree-hours) calculated from T_base.
    """
    # Forcing unit: positive difference from T_base (if any)
    forcing_units = np.maximum(0, hourly_temps - T_base)
    cumulative_forcing = forcing_units.cumsum()
    return cumulative_forcing

def compute_phenological_stage(cum_forcing, F_mature):
    """
    Compute the phenological stage s(t) as a normalized fraction of cumulative forcing.
    
    Parameters:
    -----------
    cum_forcing : pd.Series
        Cumulative forcing values.
    F_mature : float
        Forcing required for full maturation.
    
    Returns:
    --------
    pd.Series
        Phenological stage (0 to 1).
    """
    s = cum_forcing / F_mature
    # Cap the stage at 1 (i.e., fully mature)
    s[s > 1] = 1
    return s

def sensitivity_function(s):
    """
    Sensitivity function S(s) indicating frost vulnerability.
    Assume maximum sensitivity at budburst (s = 0) and zero sensitivity at full maturation (s = 1).
    
    Parameters:
    -----------
    s : float or pd.Series
        Phenological stage.
    
    Returns:
    --------
    float or pd.Series
        Sensitivity value.
    """
    return 1 - s

def hourly_frost_damage(T_hour, s, T_crit=0.0):
    """
    Calculate the frost damage for one hour based on the current temperature and phenological stage.
    
    Parameters:
    -----------
    T_hour : float
        Hourly temperature (°C).
    s : float
        Phenological stage (0 to 1) at that hour.
    T_crit : float
        Critical temperature threshold (°C) below which frost damage begins.
    
    Returns:
    --------
    float
        Damage value for the hour (thermal deficit weighted by sensitivity).
    """
    # Damage is zero if temperature is above the critical threshold.
    deficit = max(0, T_crit - T_hour)
    damage = sensitivity_function(s) * deficit
    return damage

def compute_cumulative_frost_damage(df, budburst_time, end_time, T_crit=0.0, T_base=5.0, F_mature=1000.0):
    """
    Compute cumulative frost damage over a vulnerable period using hourly data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime index and a column 'temperature' (hourly data).
    budburst_time : datetime
        The time of budburst.
    end_time : datetime
        End of the vulnerable period.
    T_crit : float
        Critical temperature threshold for frost damage (°C).
    T_base : float
        Base temperature for forcing accumulation (°C).
    F_mature : float
        Total forcing required for full leaf maturation.
    
    Returns:
    --------
    float
        Total cumulative frost damage.
    pd.DataFrame
        DataFrame containing hourly computations (forcing, phenological stage, hourly damage).
    """
    # Filter the data to the vulnerable period (from budburst to end_time)
    df_period = df.loc[budburst_time:end_time].copy()
    
    # Calculate forcing units for the period
    df_period['forcing'] = np.maximum(0, df_period['temperature'] - T_base)
    df_period['cum_forcing'] = df_period['forcing'].cumsum()
    
    # Compute phenological stage based on cumulative forcing
    df_period['s'] = compute_phenological_stage(df_period['cum_forcing'], F_mature)
    
    # Calculate hourly damage using the temperature and current phenological stage
    df_period['hourly_damage'] = df_period.apply(
        lambda row: hourly_frost_damage(row['temperature'], row['s'], T_crit), axis=1
    )
    
    # Total cumulative damage is the sum of hourly damages
    total_damage = df_period['hourly_damage'].sum()
    
    return total_damage, df_period

# Example usage
if __name__ == "__main__":
    # Create sample hourly data from budburst to 10 days later
    start_time = datetime(2023, 4, 1, 0, 0)
    end_time = datetime(2023, 4, 10, 23, 0)
    times = pd.date_range(start=start_time, end=end_time, freq='H')
    
    # Generate synthetic hourly temperature data with a diurnal cycle
    # (for demonstration purposes, using a sine function plus random noise)
    temperatures = 5 + 10 * np.sin(2 * np.pi * (times.hour) / 24) + np.random.normal(0, 1, len(times))
    df = pd.DataFrame({'temperature': temperatures}, index=times)
    
    # Define model parameters
    budburst_time = start_time   # Assume budburst occurs at start_time
    vulnerable_end_time = end_time
    T_crit = 0.0                 # Critical temperature for frost damage (°C)
    T_base = 5.0                 # Base temperature for forcing (°C)
    F_mature = 1000.0            # Forcing required for full maturation (calibrated value)
    
    # Compute cumulative frost damage over the vulnerable period
    total_damage, df_damage = compute_cumulative_frost_damage(
        df, budburst_time, vulnerable_end_time, T_crit, T_base, F_mature
    )
    
    print("Total cumulative frost damage:", total_damage)
    print(df_damage.head(10))
