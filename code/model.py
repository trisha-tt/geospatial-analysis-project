""" GENERIC MODEL FOR CLIENTS LIVING IN PORTLAND/VANCOUVER AREA WHO WORK FROM HOME """
""" IDS 64, 182, 272, 273  """

import pandas as pd # type: ignore

#----------------------------------------------------------------------------------------------------------------------------------------------

# LOAD & CLEAN DATASET:

# importing dataset into dataframes
locations64_raw = pd.read_csv('../locations/locations_64.csv', parse_dates=['datetime'])
# clean data - drop rows with missing values
locations64_clean = locations64_raw.dropna()
# filter data - drop rows with low accuracy
locations64 = locations64_clean[locations64_clean['accuracy'] < 1500].copy()

