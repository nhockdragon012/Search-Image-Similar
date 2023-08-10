import faiss
import numpy as np
import store_vectors 

dimension = 128  # Change this based on your feature dimension
index = faiss.IndexFlatL2(dimension)

data = np.array([store_vectors.feature_vectors])  # Replace the ellipsis with your preprocessed feature vectors
index.add(data)

query_vector = np.array([...])  # Replace the ellipsis with the feature vector of your query image
k = 16  # Replace with the desired number of closest images to retrieve

# Perform similarity search
D, I = index.search(np.array([query_vector]), k)
