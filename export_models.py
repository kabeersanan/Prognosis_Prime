# export_models.py
import pandas as pd
import joblib
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# 1) Load data (adjust path if needed)
data = pd.read_csv("Training.csv").dropna(axis=1)

# 2) Encode labels
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

# 3) Split X/y (we train on full dataset for final models)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 4) Initialize & train final models
svm = SVC()                # you had this in notebook
nb = GaussianNB()
rf = RandomForestClassifier(random_state=18)

svm.fit(X, y)
nb.fit(X, y)
rf.fit(X, y)

# 5) Build symptom index (human readable keys)
symptoms = X.columns.values
symptom_index = {}
for index, value in enumerate(symptoms):
    # convert "skin_rash" -> "Skin Rash"
    symptom = " ".join([w.capitalize() for w in value.split("_")])
    symptom_index[symptom] = int(index)

# 6) Save models + metadata
import os
os.makedirs("app/models", exist_ok=True)
joblib.dump(svm, "app/models/svm.pkl")
joblib.dump(nb, "app/models/nb.pkl")
joblib.dump(rf, "app/models/rf.pkl")

data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": list(encoder.classes_)
}
with open("app/models/data_dict.json", "w") as f:
    json.dump(data_dict, f)

print("Saved models and app/models/data_dict.json")
