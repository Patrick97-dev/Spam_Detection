"""REST-API fuer den Spam-Filter (Deployment-Prototyp der Fallstudie).

Laedt das serialisierte Modell (spam_model.joblib) und stellt einen /predict-Endpoint
bereit. So koennte der Mail-Server jede eingehende E-Mail in Echtzeit klassifizieren.

Start (lokal):
    pip install fastapi uvicorn scikit-learn joblib
    uvicorn app:app --reload

Test (Browser):  http://127.0.0.1:8000/docs   -> interaktive API-Dokumentation
Test (curl):
    curl -X POST http://127.0.0.1:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"features": [0,0,0, ... 57 Werte ...]}'
"""

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# Modell-Artefakt laden (Modell + Scaler + Merkmalsreihenfolge)
bundle = joblib.load("spam_model.joblib")
model = bundle["model"]
scaler = bundle["scaler"]
columns = bundle["columns"]

app = FastAPI(title="Spam-Filter API", version="1.0",
              description="Klassifiziert E-Mails anhand von 57 Merkmalen in Spam / legitim.")


class Email(BaseModel):
    # 57 numerische Merkmale in der Reihenfolge von 'columns'
    features: list[float]


@app.get("/")
def root():
    return {"status": "ok", "n_features": len(columns)}


@app.post("/predict")
def predict(email: Email):
    if len(email.features) != len(columns):
        return {"error": f"Erwarte {len(columns)} Merkmale, erhalten {len(email.features)}."}
    x = scaler.transform([email.features])
    proba = float(model.predict_proba(x)[0, 1])
    return {"spam": bool(proba >= 0.5), "spam_probability": round(proba, 3)}
