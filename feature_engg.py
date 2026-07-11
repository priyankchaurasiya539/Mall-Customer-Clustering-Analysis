import pandas as pd 
from sklearn.preprocessing import StandardScaler , LabelEncoder
import joblib

#Load the dataset
df = pd.read_csv("data/Mall_Customers (1).csv")
print(df.head(10))

print("Shape : " , df.shape)

#Check missing values 
print(df.isnull().sum())

#Droping unncessary column 
df_new = df.drop('CustomerID' , axis = 1)
print(df_new.head(10))

#Info
print(df_new.info())


#Label Encoding for gender column 
encoder = LabelEncoder()
df_new["Gender"] = encoder.fit_transform(df_new["Gender"])

#Standardization 
scaler = StandardScaler()
scaler.fit(df_new)
scaled_data = scaler.transform(df_new)

#Convert data back to dataframe
df_scaled = pd.DataFrame(scaled_data , columns=df_new.columns)

#Save the data
print(df_scaled.to_csv("data/processed_data.csv" , index=False) )
joblib.dump(scaler , "models/scaler.pkl")
joblib.dump(encoder , "models/encoder.pkl")
print("Feature engineering done and all files saved successfully.")


