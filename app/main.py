# app/main.py
import json
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from statistics import mode

app = FastAPI(title="Disease Predictor API")

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from fastapi.responses import FileResponse

# Serve the index.html file at the root URL
@app.get("/", response_class=FileResponse)
async def read_index():
    return "app/static/index.html"

# Serve static files from app/static
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Load models and metadata
MODEL_DIR = "app/models"
svm = joblib.load(f"{MODEL_DIR}/svm.pkl")
nb = joblib.load(f"{MODEL_DIR}/nb.pkl")
rf = joblib.load(f"{MODEL_DIR}/rf.pkl")

with open(f"{MODEL_DIR}/data_dict.json", "r") as f:
    data_dict = json.load(f)

SYMPTOM_INDEX = data_dict["symptom_index"]      # mapping "Skin Rash" -> index
PRED_CLASSES = data_dict["predictions_classes"]  # list of disease names

class PredictRequest(BaseModel):
    # Accepts either a list of symptoms or a comma-separated string
    symptoms: list | str

@app.get("/")
def read_index():
    # Send the frontend page (index.html)
    return FileResponse("app/static/index.html")

@app.get("/symptoms")
def get_symptoms():
    # Useful for frontend: list of supported symptoms
    return {"symptoms": list(SYMPTOM_INDEX.keys())}

@app.post("/predict")
def predict(req: PredictRequest):
    # normalize input to a list of strings
    if isinstance(req.symptoms, str):
        symptoms = [s.strip() for s in req.symptoms.split(",") if s.strip()]
    else:
        symptoms = [s.strip() for s in req.symptoms]

    # build input vector (binary flags)
    input_vec = [0] * len(SYMPTOM_INDEX)
    for s in symptoms:
        match = None
        for key in SYMPTOM_INDEX.keys():
            if key.lower() == s.lower():
                match = key
                break
        if match is None:
            s_alt = s.replace("_", " ").strip().lower()
            for key in SYMPTOM_INDEX.keys():
                if key.lower() == s_alt:
                    match = key
                    break
        if match is None:
            return JSONResponse(status_code=400, content={
                "error": f"Symptom '{s}' not recognized. Use GET /symptoms to list supported symptoms."
            })
        idx = SYMPTOM_INDEX[match]
        input_vec[int(idx)] = 1

    input_arr = np.array(input_vec).reshape(1, -1)

    # predictions
    svm_pred = PRED_CLASSES[int(svm.predict(input_arr)[0])]
    nb_pred = PRED_CLASSES[int(nb.predict(input_arr)[0])]
    rf_pred = PRED_CLASSES[int(rf.predict(input_arr)[0])]

    # majority vote (fallback to rf if no unique mode)
    try:
        final_pred = mode([svm_pred, nb_pred, rf_pred])
    except Exception:
        final_pred = rf_pred

    return {
        "svm_model_prediction": svm_pred,
        "naive_bayes_prediction": nb_pred,
        "rf_model_prediction": rf_pred,
        "final_prediction": final_pred
    }
