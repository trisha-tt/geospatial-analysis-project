""" CLIENT ID 182 ANALYSIS """
import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.cluster import DBSCAN # type: ignore
import pyproj # type: ignore
import folium # type: ignore
from folium.plugins import HeatMap # type: ignore
from folium.elements import MacroElement # type: ignore
from jinja2 import Template # type: ignore

#----------------------------------------------------------------------------------------------------------------------------------------------

# LOAD & CLEAN DATASET:

# importing dataset into dataframes
locations182_raw = pd.read_csv('../locations/locations_182.csv', parse_dates=['datetime'])

print("=== Client ID 182 Dataset ===")

# clean data - drop rows with missing values
locations182_clean = locations182_raw.dropna()

# filter data - drop rows with low accuracy
locations182 = locations182_clean[locations182_clean['accuracy'] < 1500].copy()

# convert date and time to month and year for filtering
locations182['month_year'] = locations182['datetime'].dt.to_period('M')

#----------------------------------------------------------------------------------------------------------------------------------------------

# EXPLORATORY DATA ANALYSIS:

# group dataset by month for further analysis
# for simplicity, convert longitudes and latitudes into integers
locations182['int_latitude'] = locations182['latitude'].astype(int)
locations182['int_longitude'] = locations182['longitude'].astype(int)

# first, we can find out how many locations a person visited in a month
monthly_count = locations182.groupby('month_year').size().reset_index(name='count')
# print("\n==Number of Locations visited in a Month==\n")
# print(monthly_count)

# find out how many times a person visits a location in a month
location_counts = locations182.groupby(['month_year', 'int_latitude', 'int_longitude']).size().reset_index(name='count')

# first, find out top 5 locations in each month
# group by month then find 5 largest count
top_5_per_month = (
    location_counts
        .sort_values(['month_year','count'], ascending=[True, False])
        .groupby('month_year')
        .head(5)
        .reset_index(drop=True)
)

# print("\n==Most Visited Locations Each Month==\n")
# print(top_5_per_month)

#----------------------------------------------------------------------------------------------------------------------------------------------

# DBSCAN CLUSTERING:

# now find overall top 5 locations using clustering algortihms

# downsize dataset (to prevent memory kill 9 issue)
locations182["time_bucket"] = locations182["datetime"].dt.floor("30min")
locations182_buckets = locations182.groupby("time_bucket").first().reset_index()

# Keep client_id column if exists
if "client_id" not in locations182_buckets.columns:
    locations182_buckets["client_id"] = locations182["client_id"].iloc[0]

# calculate time spent at each bucket location
locations182_buckets = locations182_buckets.sort_values("datetime")
locations182_buckets["next_time"] = locations182_buckets["datetime"].shift(-1)
locations182_buckets["time_spent"] = (locations182_buckets["next_time"] - locations182_buckets["datetime"]).dt.total_seconds()
locations182_buckets["time_spent"] = locations182_buckets["time_spent"].clip(0, 8*3600)

# project coordinates to meters ( also due to memory kill 9 error)
proj = pyproj.Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
locations182_buckets["x"], locations182_buckets["y"] = proj.transform(locations182_buckets["longitude"].values, locations182_buckets["latitude"].values)

# clustering per month
locations182_buckets["month"] = locations182_buckets["datetime"].dt.to_period("M")
df_list = []

for month, df_month in locations182_buckets.groupby("month"):
    coords = df_month[["x", "y"]].to_numpy()
    db = DBSCAN(eps=30, min_samples=3).fit(coords)

    df_month["cluster_id"] = db.labels_
    df_month["month_str"] = df_month["month"].astype(str)

    # Create UNIQUE CLUSTER IDENTIFIER
    df_month["cluster_uid"] = df_month["month_str"] + "_" + df_month["cluster_id"].astype(str)

    df_list.append(df_month)

# combine
locations182_buckets = pd.concat(df_list)

# find centroids of the clusters

centroids = (
    locations182_buckets[locations182_buckets["cluster_uid"].str.contains("_-1") == False]
    .groupby("cluster_uid")[["latitude", "longitude"]]
    .mean()
    .reset_index()
)

# now to find top 5 locations per month
monthly = (
    locations182_buckets[locations182_buckets["cluster_id"] != -1]
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

# print("\n=== TOP 5 LOCATIONS PER MONTH ===\n")
# print(top5)

#----------------------------------------------------------------------------------------------------------------------------------------------

# FREQUENT LOCATION ANALYSIS:

# find home and work locations

# categorize records with hour of day, day of week
locations182_buckets["hour"] = locations182_buckets["datetime"].dt.hour
locations182_buckets["weekday"] = locations182_buckets["datetime"].dt.weekday  # 0=Mon ... 6=Sun

# add category labels to each row
def categorize(row):
    hr = row["hour"]
    wd = row["weekday"]

    if 21 <= hr or hr < 6:
        return "night"
    if 9 <= hr < 17 and wd < 5:
        return "workday"
    return "other"

locations182_buckets["time_category"] = locations182_buckets.apply(categorize, axis=1)

valid = locations182_buckets[locations182_buckets["cluster_id"] != -1]

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

# print("HOME cluster:", home_cluster)
print("HOME coords:", home_coords.values)

# print("WORK cluster:", work_cluster)
print("WORK coords:", work_coords.values)

#----------------------------------------------------------------------------------------------------------------------------------------------

# DATA VISUALIZATION:

# creating a heatmap using folium

df_heat = locations182_buckets[locations182_buckets["cluster_id"] != -1].copy()

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

# add movement sequences 

# prepare coordinates for movement path
df_seq = locations182_buckets.sort_values("datetime")
df_seq = df_seq[df_seq["cluster_id"] != -1]

coords = df_seq[["latitude", "longitude"]].values.tolist()

# convert Python list to JavaScript array string
js_coords = str([[lat, lon] for lat, lon in coords])

# define the PolylineDecorator class
arrow_js = Template("""
{% macro script(this, kwargs) %}
var latlngs = {{ this.coords }};
var line = L.polyline(latlngs, {color: 'grey', weight:3, opacity:0.5}).addTo({{this._parent.get_name()}});
var decorator = L.polylineDecorator(line, {
    patterns: [
        {offset: '5%', repeat: '10%', symbol: L.Symbol.arrowHead({pixelSize: 8, polygon: true, pathOptions: {color: 'blue', fillOpacity: 1}})}
    ]
}).addTo({{this._parent.get_name()}});
{% endmacro %}
""")

# create PolylineDecorator element
class PolylineDecorator(MacroElement):
    def __init__(self, coords):
        super().__init__()
        self._name = 'PolylineDecorator'
        self.coords = coords
        self._template = arrow_js

# add movement path to map
decorator = PolylineDecorator(js_coords)
m2.add_child(decorator)

# save heatmap to html file
m2.save("./maps/heatmap182.html")
print("heatmap saved → heatmap182.html")