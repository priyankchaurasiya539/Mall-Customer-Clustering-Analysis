import pandas as pd
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score
from kneed import  KneeLocator
import joblib

#Load the processed csv file 
df = pd.read_csv("data/processed_data.csv")
print(df.head())

#Elbow Method + Knee Locator to find the value of k 

wcss = []

for k in range(1 , 21) :
    kmeans = KMeans(n_clusters=k , random_state=42 , init="k-means++")
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)

#Find the optimal value of k using kneelocator
knee = KneeLocator( range(1 , 21) , wcss , curve= "convex" , direction="decreasing")
best_value_k = knee.knee
print("Optimal value of k is " , best_value_k)


#Train the model

kmeans = KMeans(n_clusters=best_value_k , random_state=42)
clusters = kmeans.fit_predict(df)
df["Cluster"] = clusters

#Evaluate the silhoutte scoring 
score = silhouette_score(df.drop("Cluster" , axis = 1 ) , df["Cluster"])
print("Silhouette Scoring : " , score)

#Save trained model + clustered dataset
joblib.dump(kmeans, "models/kmeans_model.pkl")
df.to_csv("data/customers_with_clusters.csv", index=False)

print("Model training complete. Model and clustered dataset saved.")