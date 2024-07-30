from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from passlib.context import CryptContext
import joblib
import numpy as np
from typing import Optional
import pandas as pd
import database as _database
import auth as _auth

app = FastAPI()
_database.init_db()
lr = joblib.load('first-innings-score-lr-model.pkl')
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

@app.get("/register", response_class=HTMLResponse)
async def login(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

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
COLUMN_ORDER = ['runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5', 'total',
                'bat_team_Chennai Super Kings', 'bat_team_Delhi Capitals', 'bat_team_Delhi Daredevils',
                'bat_team_Gujarat Lions', 'bat_team_Gujarat Titans', 'bat_team_Kings XI Punjab',
                'bat_team_Kolkata Knight Riders', 'bat_team_Lucknow Super Giants', 'bat_team_Mumbai Indians',
                'bat_team_Punjab Kings', 'bat_team_Rajasthan Royals', 'bat_team_Royal Challengers Bangalore',
                'bat_team_Royal Challengers Bengaluru', 'bat_team_Sunrisers Hyderabad', 'bowl_team_Chennai Super Kings',
                'bowl_team_Delhi Capitals', 'bowl_team_Delhi Daredevils', 'bowl_team_Gujarat Lions',
                'bowl_team_Gujarat Titans', 'bowl_team_Kings XI Punjab', 'bowl_team_Kolkata Knight Riders',
                'bowl_team_Lucknow Super Giants', 'bowl_team_Mumbai Indians', 'bowl_team_Punjab Kings',
                'bowl_team_Rajasthan Royals', 'bowl_team_Royal Challengers Bangalore', 'bowl_team_Royal Challengers Bengaluru',
                'bowl_team_Sunrisers Hyderabad']

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    batting_team: str = Form(...),
    bowling_team: str = Form(...),
    overs: float = Form(...),
    runs: int = Form(...),
    wickets: int = Form(...),
    runs_in_prev_5: int = Form(...),
    wickets_in_prev_5: int = Form(...),
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(_auth.get_db)
):
    temp_array = []

    temp_array.extend([runs, wickets, overs, runs_in_prev_5, wickets_in_prev_5])  # 'total' set to 0

    teams = ['Chennai Super Kings', 'Delhi Capitals', 'Delhi Daredevils', 'Gujarat Lions', 'Gujarat Titans',
             'Kings XI Punjab', 'Kolkata Knight Riders', 'Lucknow Super Giants', 'Mumbai Indians', 'Punjab Kings',
             'Rajasthan Royals', 'Royal Challengers Bangalore', 'Royal Challengers Bengaluru', 'Sunrisers Hyderabad']
    for team in teams:
        temp_array.append(1 if batting_team == team else 0)
    
    for team in teams:
        temp_array.append(1 if bowling_team == team else 0)
    
    data = np.array([temp_array])
    print(data)
    prediction = int(lr.predict(data)[0])
    
    return templates.TemplateResponse('result.html', {
        'request': request,
        'lower_limit': prediction - 10,
        'upper_limit': prediction + 5
    })

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})
