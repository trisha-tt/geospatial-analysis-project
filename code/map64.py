""" CLIENT ID 64 ANALYSIS """
import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.cluster import DBSCAN # type: ignore
import pyproj # type: ignore
import folium # type: ignore
from folium.plugins import HeatMap # type: ignore

#----------------------------------------------------------------------------------------------------------------------------------------------

# LOAD & CLEAN DATASET:

# importing dataset into dataframes
locations64_raw = pd.read_csv('../locations/locations_64.csv', parse_dates=['datetime'])

# clean data - drop rows with missing values
locations64_clean = locations64_raw.dropna()

# filter data - drop rows with low accuracy
locations64 = locations64_clean[locations64_clean['accuracy'] < 1500].copy()

# convert date and time to month and year for filtering
locations64['month_year'] = locations64['datetime'].dt.to_period('M')

#----------------------------------------------------------------------------------------------------------------------------------------------

# EXPLORATORY DATA ANALYSIS:

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
top_5_per_month = (
    location_counts
        .sort_values(['month_year','count'], ascending=[True, False])
        .groupby('month_year')
        .head(5)
        .reset_index(drop=True)
)

print("\n==Most Visited Locations Each Month==\n")
print(top_5_per_month)

#----------------------------------------------------------------------------------------------------------------------------------------------

# DBSCAN CLUSTERING:

# now find overall top 5 locations using clustering algortihms

# downsize dataset (to prevent memory kill 9 issue)
locations64["time_bucket"] = locations64["datetime"].dt.floor("30min")
locations64_buckets = locations64.groupby("time_bucket").first().reset_index()

# Keep client_id column if exists
if "client_id" not in locations64_buckets.columns:
    locations64_buckets["client_id"] = locations64["client_id"].iloc[0]

# calculate time spent at each bucket location

locations64_buckets = locations64_buckets.sort_values("datetime")
locations64_buckets["next_time"] = locations64_buckets["datetime"].shift(-1)
locations64_buckets["time_spent"] = (locations64_buckets["next_time"] - locations64_buckets["datetime"]).dt.total_seconds()
locations64_buckets["time_spent"] = locations64_buckets["time_spent"].clip(0, 8*3600)

# project coordinates to meters ( also due to memory kill 9 error)
proj = pyproj.Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
locations64_buckets["x"], locations64_buckets["y"] = proj.transform(locations64_buckets["longitude"].values, locations64_buckets["latitude"].values)

# clustering per month
locations64_buckets["month"] = locations64_buckets["datetime"].dt.to_period("M")
df_list = []

for month, df_month in locations64_buckets.groupby("month"):
    coords = df_month[["x", "y"]].to_numpy()
    db = DBSCAN(eps=50, min_samples=3, metric="euclidean")  # eps in meters
    df_month["cluster_id"] = db.fit_predict(coords)
    df_list.append(df_month)

# combine
locations64_buckets = pd.concat(df_list)

# find centroids of the clusters
centroids = (
    locations64_buckets[locations64_buckets["cluster_id"] != -1]
    .groupby("cluster_id")[["latitude", "longitude"]]
    .mean()
    .reset_index()
)

# now to find top 5 locations per month
monthly = (
    locations64_buckets[locations64_buckets["cluster_id"] != -1]
    .groupby(["month", "cluster_id"])["time_spent"]
    .sum()
    .reset_index()
)

top5 = (
    monthly.sort_values(["month", "time_spent"], ascending=[True, False])
    .groupby("month")
    .head(5)
    .reset_index(drop=True)
)

print("\n=== TOP 5 LOCATIONS PER MONTH ===\n")
print(top5)

#----------------------------------------------------------------------------------------------------------------------------------------------

# DATA VISUALIZATION:

# creating a heatmap using folium

df_heat = locations64_buckets[locations64_buckets["cluster_id"] != -1].copy()

center_lat = df_heat["latitude"].mean()
center_lon = df_heat["longitude"].mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

#weighted by time spent
df_heat_w = df_heat.copy()
df_heat_w["weight"] = df_heat_w["time_spent"].fillna(0)

heat_data_weighted = df_heat_w[["latitude", "longitude", "weight"]].values.tolist()

m2 = folium.Map(location=[center_lat, center_lon], zoom_start=12)
HeatMap(heat_data_weighted, radius=18, blur=22, max_zoom=13).add_to(m2)

m2.save("./maps/heatmap64.html")
print("weighted heatmap saved → heatmap64.html")

#----------------------------------------------------------------------------------------------------------------------------------------------

# FREQUENT LOCATION ANALYSIS:

# find home and work locations

