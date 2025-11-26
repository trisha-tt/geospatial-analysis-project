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

# for month, df_month in locations64_buckets.groupby("month"):
#     coords = df_month[["x", "y"]].to_numpy()
#     db = DBSCAN(eps=50, min_samples=3, metric="euclidean")  # eps in meters
#     df_month["cluster_id"] = db.fit_predict(coords)
#     df_list.append(df_month)

for month, df_month in locations64_buckets.groupby("month"):
    coords = df_month[["x", "y"]].to_numpy()
    db = DBSCAN(eps=50, min_samples=3).fit(coords)

    df_month["cluster_id"] = db.labels_
    df_month["month_str"] = df_month["month"].astype(str)

    # Create UNIQUE CLUSTER IDENTIFIER
    df_month["cluster_uid"] = df_month["month_str"] + "_" + df_month["cluster_id"].astype(str)

    df_list.append(df_month)

# combine
locations64_buckets = pd.concat(df_list)

# find centroids of the clusters
# centroids = (
#     locations64_buckets[locations64_buckets["cluster_id"] != -1]
#     .groupby("cluster_id")[["latitude", "longitude"]]
#     .mean()
#     .reset_index()
# )

centroids = (
    locations64_buckets[locations64_buckets["cluster_uid"].str.contains("_-1") == False]
    .groupby("cluster_uid")[["latitude", "longitude"]]
    .mean()
    .reset_index()
)

# now to find top 5 locations per month
monthly = (
    locations64_buckets[locations64_buckets["cluster_id"] != -1]
    .groupby(["month", "cluster_uid"])["time_spent"]
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

# FREQUENT LOCATION ANALYSIS:

# find home and work locations

# categorize records with hour of day, day of week
locations64_buckets["hour"] = locations64_buckets["datetime"].dt.hour
locations64_buckets["weekday"] = locations64_buckets["datetime"].dt.weekday  # 0=Mon ... 6=Sun

# add category labels to each row
def categorize(row):
    hr = row["hour"]
    wd = row["weekday"]

    if 21 <= hr or hr < 6:
        return "night"
    if 9 <= hr < 17 and wd < 5:
        return "workday"
    return "other"

locations64_buckets["time_category"] = locations64_buckets.apply(categorize, axis=1)

valid = locations64_buckets[locations64_buckets["cluster_id"] != -1]

home_cluster = (
    valid[valid["time_category"] == "night"]
    .groupby("cluster_uid")["time_spent"]
    .sum()
    .sort_values(ascending=False)
    .index[0]
)

work_cluster = (
    valid[valid["time_category"] == "workday"]
    .groupby("cluster_uid")["time_spent"]
    .sum()
    .sort_values(ascending=False)
    .index[0]
)

home_coords = centroids[centroids["cluster_uid"] == home_cluster][["latitude", "longitude"]].iloc[0]
work_coords = centroids[centroids["cluster_uid"] == work_cluster][["latitude", "longitude"]].iloc[0]

print("HOME cluster:", home_cluster)
print("HOME coords:", home_coords.values)

print("WORK cluster:", work_cluster)
print("WORK coords:", work_coords.values)

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

# add home and work markers on heat map

df_heat = valid.copy()

center_lat = df_heat["latitude"].mean()
center_lon = df_heat["longitude"].mean()

m2 = folium.Map(location=[center_lat, center_lon], zoom_start=12)

df_heat["weight"] = df_heat["time_spent"].fillna(0)
heat_data_weighted = df_heat[["latitude", "longitude", "weight"]].values.tolist()

HeatMap(heat_data_weighted, radius=18, blur=22, max_zoom=13).add_to(m2)

# HOME marker
folium.Marker(
    [home_coords["latitude"], home_coords["longitude"]],
    popup="HOME",
    tooltip="Home",
    icon=folium.Icon(color="blue", icon="home")
).add_to(m2)

folium.Circle(
    [home_coords["latitude"], home_coords["longitude"]],
    radius=100,
    color="blue",
    fill=True,
    fill_opacity=0.15,
).add_to(m2)

# WORK marker
folium.Marker(
    [work_coords["latitude"], work_coords["longitude"]],
    popup="WORK",
    tooltip="Work",
    icon=folium.Icon(color="red", icon="briefcase")
).add_to(m2)

folium.Circle(
    [work_coords["latitude"], work_coords["longitude"]],
    radius=100,
    color="red",
    fill=True,
    fill_opacity=0.15,
).add_to(m2)

m2.save("./maps/heatmap64.html")
print("heatmap saved → heatmap64.html")