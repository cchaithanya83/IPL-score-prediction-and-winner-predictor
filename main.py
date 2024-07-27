from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from passlib.context import CryptContext
import joblib
import numpy as np
from typing import Optional
import database as _database
import auth as _auth
from pydantic import BaseModel
from typing import Dict, Any
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# Initialize the database
_database.init_db()

# Load models
lr = joblib.load('linear_regression_model.pkl')
rf_reg = joblib.load('random_forest_regressor_model.pkl')
# svr = joblib.load('svr_model.pkl')
knn_reg = joblib.load('knn_regressor_model.pkl')
rf_clf = joblib.load('random_forest_classifier_model.pkl')
# svc = joblib.load('svc_model.pkl')
knn_clf = joblib.load('knn_classifier_model.pkl')
features = joblib.load('features.pkl')
# Set up static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class OAuth2PasswordBearer:
    def __init__(self, tokenUrl: str):
        self.tokenUrl = tokenUrl

    def __call__(self, request: Request):
        token = request.cookies.get("access_token")
        if not token:
            raise HTTPException(status_code=401, detail="Not authenticated")
        return token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.get("/", response_class=HTMLResponse)
async def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/register", response_class=HTMLResponse)
async def register(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(_auth.get_db)
):
    if _auth.get_user(db, username):
        raise HTTPException(status_code=400, detail="Username already registered")
    user = _auth.create_user(db, username, password)
    return templates.TemplateResponse("login.html", {"request": request, "message": "User registered successfully"})

@app.post("/token")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(_auth.get_db)
):
    user = _auth.authenticate_user(db, username, password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    response = RedirectResponse(url="/predict", status_code=302)
    response.set_cookie(key="access_token", value=username)
    return response


@app.get("/predict", response_class=HTMLResponse)
async def predict_form(request: Request):
    return templates.TemplateResponse("predict_form.html", {"request": request})


def predict_win(new_data, model):
    # One-hot encode the new data
    new_data = pd.get_dummies(new_data, columns=['city', 'venue', 'team1', 'team2'])
    
    # Ensure all columns from training are in new data
    missing_cols = set(features.columns) - set(new_data.columns)
    for col in missing_cols:
        new_data[col] = 0
    
    # Reorder columns to match training data
    new_data = new_data[features.columns]
    
    return model.predict(new_data)

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

def predict_score(
    current_runs: int = Form(...),
    current_over: float = Form(...),
    total_overs: int = Form(...),
):
    # Calculate the current run rate
    run_rate = current_runs / current_over if current_over > 0 else 0

    # Calculate remaining overs
    remaining_overs = total_overs - current_over

    # Predict the final score
    predicted_score = current_runs + (run_rate * remaining_overs)

    return predicted_score


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,

    venue: str = Form(...),
    team1: str = Form(...),
    team2: str = Form(...),
    over: float = Form(...),
    cumulative_runs: float = Form(...),
    current_wickets: int = Form(...),
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(_auth.get_db)
):
    user = _auth.get_user(db, token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    

    new_data =pd.DataFrame({
    'venue': [venue],
    'team1': [team1],
    'team2': [team2],
    'over': [over],
    'cumulative_runs': [cumulative_runs],
    'cumulative_wickets': [current_wickets]
    })

    new_data = pd.get_dummies(new_data, columns=[ 'venue', 'team1', 'team2'])

    missing_cols = set(features.columns) - set(new_data.columns)
    for col in missing_cols:
        new_data[col] = 0
    
    # Reorder columns to match training data
    new_data = new_data[features.columns]

    rrprediction= predict_score(cumulative_runs,over,20)
    
    
    # Predictions
    predictions = {
        "Linear_Regression": int(((max(lr.predict(features)[0], 0))+rrprediction)/2),
        "Random_Forest_Regressor": int(((max(rf_reg.predict(features)[0], 0))+rrprediction)/2),
        # "SVR": max(svr.predict(features)[0], 0),
        "kNN_Regressor": int((max(knn_reg.predict(features)[0], 0)+rrprediction)/2),
        "Random_Forest_Classifier": rf_clf.predict(features)[0],
        # "SVC": svc.predict(features)[0],
        # "kNN_Classifier": knn_clf.predict(features)[0],        
    }
    predictions['avg_run']=int((predictions["Linear_Regression"]+predictions["Random_Forest_Regressor"]+predictions["kNN_Regressor"])/3)
    
    return templates.TemplateResponse("result.html", {"request": request, "predictions": predictions})
