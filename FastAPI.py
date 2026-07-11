from fastapi import FastAPI , HTTPException
from pydantic import BaseModel , Field
from typing import List , Literal , Annotated
import pandas as pd 
import joblib 

#Load the saved models 
kmeans = joblib.load("models/kmeans_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/encoder.pkl")

app = FastAPI(title="Mall Customer Clustering System")

class Customer(BaseModel):
    age : Annotated[float , Field(... , description="Age of the customer" , gt=0 , lt=100 ,examples=[45])]
    gender : Annotated[Literal["Male" , "Female"] , Field(... , description="Customer's Gender" , examples=["Male"])]
    annual_income_k : Annotated[float , Field(... , description="Annual income of the customer (1-100)(k)" , examples=[45] , gt=0 , le=150)]
    spending_score : Annotated[float , Field(... , description="Spending score of the customer (1-100)" , examples=[80] , gt=0 , lt=100)]

@app.get("/")
def root():
    return{
        "message" : "Hello ,  User !!"
    }

@app.get("/about")
def about():
    return{
        "message" : "This is a clustering system which helps us to cluster the customers of mall according to their annual income and spending score"
    }

@app.get("/health")
def health():
    return{
        "Status" : "running",
        "Error" : "No",
        "All task performing ? (Yes/No)" : "Yes"
    }

@app.post("/predict")
def preprocess (data : Customer):
    try :
        gender_encoded = encoder.transform([data.gender])[0]

    except ValueError:
        raise HTTPException(status_code=400 , detail=f"Unknown Gender value : {data.gender}")
    
    input_data = pd.DataFrame(
        [[gender_encoded , data.age , data.annual_income_k , data.spending_score]] , 
        columns=["Gender" , "Age" , "Annual Income (k$)" , "Spending Score (1-100)"]
    )
    
    try :
        scaled = scaler.transform(input_data)
        cluster = kmeans.predict(scaled)[0]
    
    except Exception as e :
        raise HTTPException(status_code=500 , detail="Model prediction failed.")

    return {
        "Cluster" : int(cluster), 
        "customer" : data.model_dump()
    }