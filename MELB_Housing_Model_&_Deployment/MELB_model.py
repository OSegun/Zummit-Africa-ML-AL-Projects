import uvicorn
import pickle
from fastapi import FastAPI
from main import Housedata
import pandas as pd
import numpy as np

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


app = FastAPI()
pickle_model = open("HFR.pkl", "rb")
model = pickle.load(pickle_model)


@app.get("/")
def root():
    return {"Message": "Building an API"}

@app.get("/Welcome")
def get_name(name: str):
    return {"""Hi {} Welcome To House Price Dataset 
    Prediction With Randam Forest Regressor, Make Your Predictions""".format(name)}

@app.post("/predict") 
def predict(data:Housedata):
    data = data.dict()
    Suburb        = data['Suburb']
    Address       = data['Address']
    Rooms         = data['Rooms']
    Type          = data['Type']
    Method        = data['Method'] 
    SellerG       = data['SellerG']
    Date          = data['Date']
    Distance      = data['Distance']   
    Postcode      = data['Postcode']   
    Bedroom2      = data['Bedroom2']   
    Bathroom      = data['Bathroom']   
    Car           = data['Car']   
    Landsize      = data['Landsize']
    BuildingArea  = data['BuildingArea']
    YearBuilt     = data['YearBuilt']
    CouncilArea   = data['CouncilArea']
    Lattitude     = data['Lattitude']  
    Longtitude    = data['Longtitude']  
    Regionname    = data['Regionname']   
    Propertycount = data['Propertycount'] 

    feature = pd.Series([Suburb, Address, Rooms, Type, Method, SellerG, 
                Date, Distance, Postcode, Bedroom2, Bathroom,
                Car, Landsize, BuildingArea, YearBuilt, CouncilArea, 
                Lattitude, Longtitude, Regionname, Propertycount], 
                index = ['Suburb', 'Address', 'Rooms', 'Type', 'Method', 'SellerG', 
                'Date', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom',
                'Car', 'Landsize', 'BuildingArea', 'YearBuilt', 'CouncilArea', 
                'Lattitude', 'Longtitude', 'Regionname', 'Propertycount'])

    House_Prediction = model.predict(np.asarray(feature.drop(labels=['Suburb', 
    'Address', 'SellerG', 'Date', 'BuildingArea', 'YearBuilt', 'CouncilArea', 
    'Type', 'Method', 'Regionname'])).reshape(1, -1))
    
    return {"This House Cost ${}".format(np.round(House_Prediction, 2))}

@app.exception_handler(ValueError)
def value_error_exception_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code= 500,
        content={"message": str(exc)},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)