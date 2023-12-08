import uvicorn
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


model_filename = ' '
model = joblib.load(model_filename)

app = FastAPI()

class Item(BaseModel):



@app.post('/predict')
async def predict(item:Item):



model.predict()


responses = {}

return responses