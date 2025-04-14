# src/model.py
import joblib
from sklearn.cluster import KMeans
import os
os.environ["OMP_NUM_THREADS"] = "1"

def train_kmeans(data, k):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(data)

    return model, model.labels_, model.inertia_

def save_model(model, path="models/kmeans_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path="models/kmeans_model.pkl"):
    model = joblib.load(path)
    return model
