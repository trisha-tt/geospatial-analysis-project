""" GENERIC MODEL FOR CLIENTS LIVING IN PORTLAND/VANCOUVER AREA WHO WORK FROM HOME """
""" IDS 64, 182, 272, 273  """

import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.cluster import KMeans # type: ignore
import datetime # type: ignore

#----------------------------------------------------------------------------------------------------------------------------------------------

# PROCESSING DATASET:

# function to load, clean, and process each user's dataset
def process_user(path, k=6):
    """
    load, clean, downsample, cluster, and label a single user's dataset.
    returns:
        df_buckets: per-user dataframe with state_label
        kmeans: fitted KMeans object
    """
    # load & clean
    df = pd.read_csv(path, parse_dates=['datetime'])
    df = df.dropna()
    df = df[df['accuracy'] < 1500].copy()

    # downsize dataset (to prevent memory kill 9 issue)
    df['time_bucket'] = df['datetime'].dt.floor('30min')
    df_buckets = df.groupby('time_bucket').first().reset_index()

    # filter to remove noise from outside the Portland/Vancouver area
    df_buckets = df_buckets[
        (df_buckets.latitude > 45.0) & (df_buckets.latitude < 49.0) &
        (df_buckets.longitude > -123.0) & (df_buckets.longitude < -122.0)
    ]

    # KMeans clustering
    coords = df_buckets[['latitude', 'longitude']].to_numpy()
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
    df_buckets['state'] = kmeans.fit_predict(coords)

    # add time features
    df_buckets['hour'] = df_buckets['time_bucket'].dt.hour
    df_buckets['weekday'] = df_buckets['time_bucket'].dt.weekday
    df_buckets['time_spent'] = 30

    # identify home cluster (night + early morning)
    night_data = df_buckets[(df_buckets['hour'] >= 21) | (df_buckets['hour'] < 8)]
    home_cluster = night_data.groupby('state')['time_spent'].sum().idxmax()

    # identify work cluster (weekdays 9–17)
    work_data = df_buckets[(df_buckets['hour'] >= 9) & (df_buckets['hour'] < 17) & (df_buckets['weekday'] < 5)]
    work_cluster = work_data.groupby('state')['time_spent'].sum().idxmax()
    
    # if work_cluster == home_cluster, force a different work cluster
    # to improve model accuracy
    if work_cluster == home_cluster:
    # find another cluster that appears during work hours
        if not work_data.empty:
            work_counts = work_data.groupby('state')['time_spent'].sum().sort_values(ascending=False)
        
            for cid in work_counts.index:
                if cid != home_cluster:
                    work_cluster = cid
                    break
                else:
                    work_cluster = None
        else:
            work_cluster = None

    # Label all clusters
    labels = {s: 'other' for s in range(k)}
    labels[work_cluster] = 'work'
    labels[home_cluster] = 'home'
    
    labels = {s: 'other' for s in range(k)}
    labels[home_cluster] = 'home'

    if work_cluster is not None:
        labels[work_cluster] = 'work'
    else:
        # pure time-based work override later
        pass

    df_buckets['state_label'] = df_buckets['state'].map(labels)
    
    print("Cluster labels:")
    for cluster_id, label in labels.items():
        centroid = kmeans.cluster_centers_[cluster_id]
        print(f"Cluster {cluster_id}: {label}, centroid = {centroid}")

    return df_buckets, kmeans

# now process each user's dataset
print("Client 64 Clustering: ")  
locations64_buckets, _ = process_user('../locations/locations_64.csv')

print("Client 181 Clustering: ")  
locations181_buckets, _ = process_user('../locations/locations_181.csv')

print("Client 272 Clustering: ")  
locations272_buckets, _ = process_user('../locations/locations_272.csv')

print("Client 273 Clustering: ")
locations273_buckets, _ = process_user('../locations/locations_273.csv')

#----------------------------------------------------------------------------------------------------------------------------------------------

# COMBINE DATASETS:

locations_all = pd.concat([
    locations64_buckets,
    locations181_buckets,
    locations272_buckets,
    locations273_buckets
], ignore_index=True)

#----------------------------------------------------------------------------------------------------------------------------------------------

# NAIVE BAYES MODEL: 
# P(state_label | hour, weekday)

# tabulate counts
count_table = (
    locations_all
    .groupby(['hour', 'weekday', 'state_label'])
    .size()
    .unstack(fill_value=0)
)

# laplace smoothing
count_table += 1

# convert counts to probabilities
prob_table = count_table.div(count_table.sum(axis=1), axis=0)

#----------------------------------------------------------------------------------------------------------------------------------------------

# PREDICTION FUNCTION:

def predict_location_category(dt):
    """
    predict general location category given a datetime for all users.
    returns:
        state_label: human-friendly label ('home', 'work', 'other')
        probabilities: P(state_label | hour, weekday)
    """
    hr = dt.hour
    wd = dt.weekday()
    
    # # Force home for early morning
    # if hr < 9:
    #     return 'home', {'home': 1.0, 'work': 0.0, 'other': 0.0}

    # Select row for hour & weekday
    row = prob_table.loc[(hr, wd)]
    state_label = row.idxmax()
    probs = row.to_dict()

    return state_label, probs

#----------------------------------------------------------------------------------------------------------------------------------------------

# DEMO:
print("=====================================================") 
print("Model Demonstration: ")  

dt = datetime.datetime(2024, 11, 18, 4, 30)
label, probs = predict_location_category(dt)

print("Datetime:", dt)
print("Predicted category:", label)
print("Probabilities:", probs)
print()

dt = datetime.datetime(2024, 11, 18, 10, 30)
label, probs = predict_location_category(dt)

print("Datetime:", dt)
print("Predicted category:", label)
print("Probabilities:", probs)
print()

dt = datetime.datetime(2024, 11, 18, 18, 30)
label, probs = predict_location_category(dt)

print("Datetime:", dt)
print("Predicted category:", label)
print("Probabilities:", probs)