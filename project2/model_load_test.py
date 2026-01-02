import numpy as np
import joblib
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load the KMeans model
try:
    kmeans_model = joblib.load("kmeans_model.pkl")
    print("KMeans model loaded successfully.")
except Exception as e:
    print(f"Failed to load KMeans model: {e}")
    exit()

# Test model attributes
try:
    print(f"Number of clusters: {kmeans_model.n_clusters}")
    print(f"Cluster centers:\n{kmeans_model.cluster_centers_}")
    print(f"Cluster centers dtype: {kmeans_model.cluster_centers_.dtype}")
except AttributeError as e:
    print(f"Missing attribute in KMeans model: {e}")

# Generate diverse test data
test_data = np.random.uniform(
    low=np.min(kmeans_model.cluster_centers_, axis=0),
    high=np.max(kmeans_model.cluster_centers_, axis=0),
    size=(100, kmeans_model.cluster_centers_.shape[1])
).astype(kmeans_model.cluster_centers_.dtype)

print(f"Test data dtype: {test_data.dtype}")

# Visualize test data and cluster centers
plt.scatter(test_data[:, 0], test_data[:, 1], c="blue", label="Test Data")
plt.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:, 1], c="red", label="Cluster Centers")
plt.legend()
plt.show()

# Predict clusters for the test data
try:
    test_labels = kmeans_model.predict(test_data)
    print(f"Predicted cluster labels for test data:\n{test_labels}")

    # Evaluate clustering with silhouette score
    silhouette_avg = silhouette_score(test_data, test_labels, metric="euclidean")
    print(f"Silhouette Score: {silhouette_avg}")
except Exception as e:
    print(f"Error during prediction or evaluation: {e}")
