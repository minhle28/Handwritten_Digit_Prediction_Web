# from remote_plot import plt
# from pyspark.sql import SparkSession
# from pyspark.ml.clustering import KMeans
# from pyspark.sql import SparkSession
# from pyspark.ml.linalg import SparseVector
# from pyspark.ml.linalg import Vectors

# spark = SparkSession.builder.appName("KMeansClustering").getOrCreate()          #Open spark seesion
# data = spark.read.format("libsvm").load("/user/hadoop/data/kmeans_input.txt")   # Load the list file
# # Format data as array [x,y] 
# data_array = data.rdd.map(lambda x: SparseVector(2, x.features.indices, x.features.values).toArray()).collect()
# data_df = spark.createDataFrame([(Vectors.dense(row),) for row in data_array], ["features"])    # cover to data frame

# kmeans = KMeans(k=2, seed=42)           # Create a KMeans with clusters is 2
# model = kmeans.fit(data_df)             # Fit the KMeans model to the data frame
# predictions = model.transform(data_df)  # Get the clusters point [x,y,prediction]

# cluster_labels = predictions.select("prediction").toPandas()    # Get colum prediction as list [1,0,....]
# data_points = data_df.select("features").toPandas()             # Extract point X,Y as DataFrame
# data_points['cluster_label'] = cluster_labels['prediction']     # Add new column to the data_points
# marker_shapes = {0: 'o', 1: 's'}                                # Each group cluster have difference symbol
# # Each forloop select cluter members and sign Marker by prediction label
# for cluster_label in marker_shapes.keys():
#     cluster_data = data_points[data_points['cluster_label'] == cluster_label]
#     plt.scatter(cluster_data['features'].apply(lambda x: x[0]),
#                 cluster_data['features'].apply(lambda x: x[1]),
#                 marker=marker_shapes[cluster_label],
#                 label=f'Cluster {cluster_label + 1}')
# # Set plot title and labels
# plt.title('K-means Clustering Results')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()        # Add a legend
# print("Showing PLot in localhost:8000 in HOST browers")
# plt.show()          # Show the plot

from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Open Spark session
spark = SparkSession.builder.appName("KMeansClustering").getOrCreate()

# Load the data
# data = spark.read.format("libsvm").load("/user/hadoop/data/kmeans_input.txt")
data = spark.read.format("libsvm").load("file:///home/ble16/cloud/kmeans_input.txt")

# Fit K-means clustering model
kmeans = KMeans(k=2, seed=42)
model = kmeans.fit(data)

# Get predictions for the original data
predictions = model.transform(data)

# Convert predictions to Pandas DataFrame
cluster_labels = predictions.select("prediction").toPandas()
data_points = data.select("features").toPandas()
data_points['cluster_label'] = cluster_labels['prediction']

# Plot clusters
marker_shapes = {0: 'o', 1: 's'}  # Marker shapes for different clusters

for cluster_label in marker_shapes.keys():
    cluster_data = data_points[data_points['cluster_label'] == cluster_label]
    plt.scatter(cluster_data['features'].apply(lambda x: x[0]),
                cluster_data['features'].apply(lambda x: x[1]),
                marker=marker_shapes[cluster_label],
                label=f'Cluster {cluster_label + 1}')

# Load and preprocess your handwritten image
image_path = '/home/ble16/cloud/img/custom_image.png'
handwritten_img = Image.open(image_path)
handwritten_array = np.array(handwritten_img)
# Preprocess your image here to match the format expected by the model

# Make predictions on your handwritten image
handwritten_data = spark.createDataFrame([(Vectors.dense(handwritten_array.flatten()),)], ["features"])
handwritten_predictions = model.transform(handwritten_data)

# Plot the predicted handwritten image
predicted_cluster = handwritten_predictions.select("prediction").first()[0]
plt.scatter(handwritten_array[:, 0], handwritten_array[:, 1], marker='*', label=f'Handwritten Prediction: Cluster {predicted_cluster + 1}', color='red')

# Set plot title and labels
plt.title('K-means Clustering Results')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Show the plot
plt.show()
