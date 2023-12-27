import re
import pandas as pd
import os
from .path import *

# Step 2: Read CSV file
df = pd.read_csv(fmm_path, delimiter=';')

# Step 3: Define function to extract coordinates
def extract_coordinates(line_string):
    coordinates = re.findall(r'(\d+\.\d+)\s(\d+\.\d+)', line_string)
    return [(float(lon), float(lat)) for lon, lat in coordinates]

# Step 4: Apply function and create a new column
df['coordinates'] = df['mgeom'].apply(extract_coordinates)

df = df[df['coordinates'].apply(len) > 0]

# Step 5: Display the result
print(df[['id', 'coordinates']])

df.drop(columns=['mgeom'], inplace=True)
df.drop(columns=['cpath'], inplace=True)

df.to_csv(fmm_out_path, index=False)
