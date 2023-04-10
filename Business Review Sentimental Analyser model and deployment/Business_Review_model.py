import uvicorn
import pickle
from fastapi import FastAPI
import pandas as pd
import numpy as np
#Importing all dependences
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import tensorflow as tf
import requests
from bs4 import BeautifulSoup
import re


from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
app = FastAPI()
pickle_model = open("BRM.pkl", "rb")
model = pickle.load(pickle_model)


@app.get("/")
def root():
    return {"Message": "Building a Business Review Sentimental Analyser API"}

@app.get("/Welcome")
def get_name(name: str):
    return {"""Hi Welcome {} Business,  
    This is a Business Review api, proceed ahead to input your comments""".format(name)}

@app.post("/predict") 
def predict(
    text              :str
):
    
    

    result = model(tokenizer.encode(text[:512], return_tensors='pt'))
    review = int(torch.argmax(result.logits))+1
    return {"'{}' comment gives your business a {} rating ".format(text, review)}

@app.exception_handler(ValueError)
def value_error_exception_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code= 500,
        content={"message": str(exc)},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)