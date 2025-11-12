import pandas as pd # type: ignore
# import numpy as np
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import confusion_matrix, accuracy_score
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import seaborn as sns
# from ISLP import load_data


# importing dataset into dataframes
locations64_raw = pd.read_csv('locations/locations_64.csv', parse_dates=['datetime'])
# locations67 = pd.read_csv('locations/locations_67.csv')
# locations68 = pd.read_csv('locations/locations_68.csv')
# locations69 = pd.read_csv('locations/locations_69.csv')
# locations70 = pd.read_csv('locations/locations_70.csv')
# locations175 = pd.read_csv('locations/locations_175.csv')
# locations177 = pd.read_csv('locations/locations_177.csv')
# locations179 = pd.read_csv('locations/locations_179.csv')
# locations181 = pd.read_csv('locations/locations_181.csv')
# locations182 = pd.read_csv('locations/locations_182.csv')
# locations258 = pd.read_csv('locations/locations_258.csv')
# locations269 = pd.read_csv('locations/locations_269.csv')
# locations272 = pd.read_csv('locations/locations_272.csv')
# locations273 = pd.read_csv('locations/locations_273.csv')
# locations276 = pd.read_csv('locations/locations_276.csv')
# locations328 = pd.read_csv('locations/locations_328.csv')
# locations336 = pd.read_csv('locations/locations_336.csv')
# locations338 = pd.read_csv('locations/locations_338.csv')
# locations343 = pd.read_csv('locations/locations_343.csv')
# locations344 = pd.read_csv('locations/locations_344.csv')

# clean data - drop rows with missing values
locations64_clean = locations64_raw.dropna()

# filter data - drop rows with low accuracy
locations64 = locations64_clean[locations64_clean['accuracy'] < 1500].copy()

# convert date and time to month and year for filtering
locations64['month_year'] = locations64['datetime'].dt.to_period('M')

# group dataset by month for further analysis

# for simplicity, convert longitudes and latitudes into integers
locations64['int_latitude'] = locations64['latitude'].astype(int)
locations64['int_longitude'] = locations64['longitude'].astype(int)

# first, we can find out how many locations a person visited in a month
monthly_count = locations64.groupby('month_year').size().reset_index(name='count')
print("\n==Number of Locations visited in a Month==\n")
print(monthly_count)

# find out how many times a person visits a location in a month
location_counts = locations64.groupby(['month_year', 'int_latitude', 'int_longitude']).size().reset_index(name='count')

# first, find out top 5 locations in each month
# group by month then find 5 largest count
top_5_per_month = location_counts.groupby('month_year').apply(lambda x: x.nlargest(5, 'count'), include_groups=False).reset_index(drop=True)

# then find top 5 locations per month
# use the count to find latitude & longitudes
overall_top_5 = top_5_per_month.groupby(['int_latitude', 'int_longitude']).size().reset_index(name='monthly_top5_appearances')

# sort to find the most visited locations across dataset
overall_most_visited = overall_top_5.sort_values(by='monthly_top5_appearances',ascending=False)

print("\n==Most Visited Locations Each Month==\n")
print(top_5_per_month.to_string())

print("\n==Most Visited Locations Overall==\n")
print(overall_most_visited.to_string())