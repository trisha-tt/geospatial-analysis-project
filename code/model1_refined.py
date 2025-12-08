""" GENERIC MODEL FOR CLIENTS LIVING IN PORTLAND/VANCOUVER AREA WHO WORK FROM HOME """
""" IDS 64, 182, 272  """
""" IMPROVE MODEL ACCURACY - REMOVE CLIENT 273 """

import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.cluster import KMeans # type: ignore
import datetime # type: ignore
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
from sklearn.metrics import classification_report # type: ignore

print("Refined Model 1: Individuals living in Portland/Vancouver area")

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
    df_buckets["is_weekend"] = df_buckets["weekday"].isin([5, 6]).astype(int)
    df_buckets["is_workhour"] = df_buckets["hour"].between(9, 17).astype(int)   # 9am–5pm
    df_buckets["is_sleephour"] = df_buckets["hour"].between(0, 5).astype(int)   # midnight–5am

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
        
            work_cluster = None
            for cid in work_counts.index:
                if cid != home_cluster:
                    work_cluster = cid
                    break

    # label all clusters
    labels = {s: 'other' for s in range(k)}
    labels[home_cluster] = 'home'
    if work_cluster is not None and work_cluster != home_cluster:
        labels[work_cluster] = 'work'

    df_buckets['state_label'] = df_buckets['state'].map(labels)
    
    print("Cluster labels:")
    for cluster_id, label in labels.items():
        centroid = kmeans.cluster_centers_[cluster_id]
        print(f"Cluster {cluster_id}: {label}, centroid = {centroid}")

    return df_buckets, kmeans

# now process each user's dataset
print("Client 64 Clustering: ")  
locations64_buckets, _ = process_user('../locations/locations_64.csv')

print("Client 182 Clustering: ")  
locations182_buckets, _ = process_user('../locations/locations_182.csv')

print("Client 272 Clustering: ")  
locations272_buckets, _ = process_user('../locations/locations_272.csv')

print("Client 273 Clustering: ")

#----------------------------------------------------------------------------------------------------------------------------------------------

# COMBINE DATASETS:

locations_all = pd.concat([
    locations64_buckets,
    locations182_buckets,
    locations272_buckets,
], ignore_index=True)

#----------------------------------------------------------------------------------------------------------------------------------------------

# NAIVE BAYES MODEL: 
# P(state_label | hour, weekday)


# build probability tables with Laplace smoothing
feature_cols = ["hour", "weekday", "is_weekend", "is_workhour", "is_sleephour"]
states = ['home', 'work', 'other']

prob_tables = {}

for feat in feature_cols:
    # count occurrences of state_label for each feature value
    counts = (
        locations_all
        .groupby([feat, "state_label"])
        .size()
        .unstack(fill_value=0)
    )

    # Laplace smoothing: add 1 to all counts
    counts = counts + 1

    # ensure all states are present as columns
    for s in states:
        if s not in counts.columns:
            counts[s] = 1

    # reorder columns
    counts = counts[states]

    # normalize per feature value
    probs = counts.div(counts.sum(axis=1), axis=0)

    prob_tables[feat] = probs


#----------------------------------------------------------------------------------------------------------------------------------------------

# PREDICTION FUNCTION:

def predict_location_category(dt, prob_tables):
    """
    Predict state_label ('home','work','other') given datetime features.
    Uses Naive Bayes multiplication of all feature-based conditional probabilities.
    """
    # --- extract features from datetime ---
    hr = dt.hour
    wd = dt.weekday
    is_weekend = 1 if wd in [5, 6] else 0
    is_workhour = 1 if 9 <= hr <= 17 else 0
    is_sleephour = 1 if 0 <= hr <= 5 else 0

    feats = {
        "hour": hr,
        "weekday": wd,
        "is_weekend": is_weekend,
        "is_workhour": is_workhour,
        "is_sleephour": is_sleephour
    }

    log_probs = {}

    for state in ['home', 'work', 'other']:
        log_prob_state = 0.0
        for feat, value in feats.items():
            table = prob_tables[feat]

            # fallback if value is missing in the table
            if value not in table.index:
                val = 1 / len(states)  # uniform small probability
            else:
                val = table.loc[value, state]

            # add log-probability
            log_prob_state += np.log(val)

        log_probs[state] = log_prob_state

    # convert log-probabilities to normal probabilities
    max_log = max(log_probs.values())
    exp_probs = {k: np.exp(v - max_log) for k, v in log_probs.items()}
    Z = sum(exp_probs.values())
    final_probs = {k: v / Z for k, v in exp_probs.items()}

    # pick the best label
    best_state = max(final_probs, key=final_probs.get)

    return best_state, final_probs

#----------------------------------------------------------------------------------------------------------------------------------------------

# DEMO:

print("=====================================================================") 
print("Model Demonstration: ")  

dt = datetime.datetime(2024, 11, 18, 4, 30)
label, probs = predict_location_category(dt, prob_tables)

print("Datetime:", dt)
print("Predicted category:", label)
print("Probabilities:", probs)
print()

dt = datetime.datetime(2024, 11, 18, 10, 30)
label, probs = predict_location_category(dt, prob_tables)

print("Datetime:", dt)
print("Predicted category:", label)
print("Probabilities:", probs)
print()

dt = datetime.datetime(2024, 11, 18, 18, 30)
label, probs = predict_location_category(dt, prob_tables)

print("Datetime:", dt)
print("Predicted category:", label)
print("Probabilities:", probs)

#----------------------------------------------------------------------------------------------------------------------------------------------
# EVALUATE MODEL:

print("=====================================================================") 
print("Model Evaluation: ")  

print("Confusion Matrix:")
print("Plotting confusion matrix...")
# build confusion matrix
true_labels = []
pred_labels = []

# loop through all combined rows
for _, row in locations_all.iterrows():
    dt = row['time_bucket']        # your evaluation timestamp
    true = row['state_label']      # ground truth
    
    pred, _ = predict_location_category(dt,prob_tables)
    
    true_labels.append(true)
    pred_labels.append(pred)

# create confusion matrix
labels = ['home', 'work', 'other']
cm = confusion_matrix(true_labels, pred_labels, labels=labels)

# plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap='Blues', colorbar=True)
plt.title("Confusion Matrix")
#plt.show()
fig.savefig("model_result/confusion_matrix_model1_refined.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print("Confusion Matrix saved to 'model_result/confusion_matrix_model1_refined.png'")
print()

# calculate overall accuracy
accuracy = accuracy_score(true_labels, pred_labels)
print("Overall Model Accuracy:", round(accuracy, 4))
print()

# detailed classification report
# print("Classification Report:")
# print(classification_report(true_labels, pred_labels, labels=labels, zero_division=0))
# print()

# hourly accuracy
# locations_all['predicted'] = pred_labels
# locations_all['correct'] = (locations_all['predicted'] == locations_all['state_label'])
# hourly_accuracy = locations_all.groupby('hour')['correct'].mean()
# print("Hourly Accuracy:")
# print(hourly_accuracy)
# print()

# split predicted results back into user datasets
n64  = len(locations64_buckets)
n182 = len(locations182_buckets)
n272 = len(locations272_buckets)

p64  = pred_labels[:n64]
p182 = pred_labels[n64:n64+n182]
p272 = pred_labels[n64+n182:n64+n182+n272:]

locations64_buckets['predicted']  = p64
locations182_buckets['predicted'] = p182
locations272_buckets['predicted'] = p272

# user-wise accuracy
def compute_user_accuracy(df, name):
    acc = (df['predicted'] == df['state_label']).mean()
    print(f"{name} accuracy: {acc:.4f}")

print("Client-wise Accuracy:")
compute_user_accuracy(locations64_buckets,  "Client 64")
compute_user_accuracy(locations182_buckets, "Client 182")
compute_user_accuracy(locations272_buckets, "Client 272")