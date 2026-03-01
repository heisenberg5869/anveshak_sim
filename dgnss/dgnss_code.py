import pandas as pd

# 1. Setup Known Constants
BASE_ACTUAL_LAT = 12.990123
BASE_ACTUAL_LON = 80.223452

# 2. Reading Data
data = pd.read_csv(r"C:\Users\vtsar\projects\anveshak\dgps_rover_data.csv")

# 3. Calculate DGNSS
def calculate_dgnss(df):
    # Calculate the error at the base station
    # Error = Actual - Observed
    df['lat_error'] = BASE_ACTUAL_LAT - df['Base Latitude (Measured)']
    df['lon_error'] = BASE_ACTUAL_LON - df['Base Longitude (Measured)']
    
    # Apply the correction to the rover
    # Corrected = Observed + Error
    df['Rover Latitude (Corrected)'] = df['Rover Latitude (Measured)'] + df['lat_error']
    df['Rover Longitude (Corrected)'] = df['Rover Longitude (Measured)'] + df['lon_error']
    
    return df[['Row', 'Base Latitude (Measured)', 'Base Longitude (Measured)', 'Rover Latitude (Measured)', 'Rover Longitude (Measured)', 'Rover Latitude (Corrected)', 'Rover Longitude (Corrected)']]

# 4. Save
corrected_df = calculate_dgnss(data)
corrected_df.to_csv('dgnss_rover_coords.csv', index=False)

print("Correction Complete. Sample of corrected coordinates:")
print(corrected_df.head(2))