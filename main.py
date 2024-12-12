from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Initialiser l'application FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Remplacez "*" par une liste des origines autorisées, ex: ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # Autoriser toutes les méthodes (GET, POST, etc.)
    allow_headers=["*"],  # Autoriser tous les en-têtes
)

# Charger les modèles et encodeurs sauvegardés
state_encoder = joblib.load('state_encoder.pkl')
intl_plan_encoder = joblib.load('intl_plan_encoder.pkl')
vm_plan_encoder = joblib.load('vm_plan_encoder.pkl')
scaler = joblib.load('scaler.pkl')
rf_model_balanced = joblib.load('rf_model_balanced.pkl')

# Définir les données d'entrée avec Pydantic
class InputData(BaseModel):
    State: str
    Account_length: int
    Area_code: int
    International_plan: str
    Voice_mail_plan: str
    Number_vmail_messages: int
    Total_day_calls: int
    Total_day_charge: float
    Total_eve_calls: int
    Total_eve_charge: float
    Total_night_calls: int
    Total_night_charge: float
    Total_intl_calls: int
    Total_intl_charge: float
    Customer_service_calls: int

# Endpoint pour faire une prédiction
@app.post("/predict")
def predict(data: InputData):
    # Convertir les données en dictionnaire et renommer les colonnes
    input_dict = data.dict()
    renamed_input_dict = {
        'State': input_dict['State'],
        'Account length': input_dict['Account_length'],
        'Area code': input_dict['Area_code'],
        'International plan': input_dict['International_plan'],
        'Voice mail plan': input_dict['Voice_mail_plan'],
        'Number vmail messages': input_dict['Number_vmail_messages'],
        'Total day calls': input_dict['Total_day_calls'],
        'Total day charge': input_dict['Total_day_charge'],
        'Total eve calls': input_dict['Total_eve_calls'],
        'Total eve charge': input_dict['Total_eve_charge'],
        'Total night calls': input_dict['Total_night_calls'],
        'Total night charge': input_dict['Total_night_charge'],
        'Total intl calls': input_dict['Total_intl_calls'],
        'Total intl charge': input_dict['Total_intl_charge'],
        'Customer service calls': input_dict['Customer_service_calls'],
    }

    # Convertir en DataFrame
    input_data = pd.DataFrame([renamed_input_dict])

    # Appliquer les transformations avec les encodeurs
    input_data['State'] = state_encoder.transform(input_data['State'])
    input_data['International plan'] = intl_plan_encoder.transform(input_data['International plan'])
    input_data['Voice mail plan'] = vm_plan_encoder.transform(input_data['Voice mail plan'])

    # Colonnes numériques à standardiser
    numeric_features = [
        'Account length', 'Area code', 'Number vmail messages',
        'Total day calls', 'Total day charge', 'Total eve calls',
        'Total eve charge', 'Total night calls', 'Total night charge',
        'Total intl calls', 'Total intl charge', 'Customer service calls'
    ]
    input_data[numeric_features] = scaler.transform(input_data[numeric_features])

    # Faire une prédiction avec le modèle
    prediction = rf_model_balanced.predict(input_data)
    proba = rf_model_balanced.predict_proba(input_data)

    # Retourner les résultats
    return {
        "predicted_class": "Churn" if prediction[0] else "No Churn",
        "probabilities": {
            "No Churn": proba[0][0],
            "Churn": proba[0][1]
        }
    }
