from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os
import tarfile
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()

app = Flask(__name__, template_folder="templates", static_folder="static")

# ✅ Ensure secrets are loaded
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("CONTAINER_NAME")
BLOB_NAME_MODELS = os.getenv("BLOB_NAME_MODELS")  # models.tar.gz
BLOB_NAME_DATASET = os.getenv("BLOB_NAME_DATASET")  # updated_cloud_cost_dataset.csv

if not AZURE_CONNECTION_STRING or not CONTAINER_NAME or not BLOB_NAME_DATASET or not BLOB_NAME_MODELS:
    raise ValueError("❌ Missing required Azure secrets! Check .env file.")

# ✅ Helper function to download blob only if not already present
def download_blob(blob_name, destination_path):
    """Downloads a blob from Azure to a local file if it doesn't already exist."""
    if os.path.exists(destination_path):
        print(f"📦 {blob_name} already exists. Skipping download.")
        return
    
    try:
        print(f"⬇️ Downloading {blob_name} from Azure Blob Storage...")
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)

        with open(destination_path, "wb") as f:
            f.write(blob_client.download_blob().readall())
        
        print(f"✅ {blob_name} downloaded successfully.")
    except Exception as e:
        print(f"❌ Failed to download {blob_name}: {e}")

# ✅ Helper function to extract tar.gz files safely
def extract_tar_gz(tar_path, extract_to):
    """Extracts a tar.gz file to a given directory."""
    if os.path.exists(extract_to) and os.listdir(extract_to):
        print(f"📦 Models already extracted in {extract_to}. Skipping extraction.")
        return

    try:
        print(f"📂 Extracting {tar_path}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_to)
        print(f"✅ Extracted models to {extract_to}")
    except Exception as e:
        print(f"❌ Failed to extract {tar_path}: {e}")

# ✅ Download and Extract both models & dataset
MODEL_DIR = os.path.join(os.getcwd(), "models")
DATASET_PATH = os.path.join(os.getcwd(), "updated_cloud_cost_dataset.csv")

os.makedirs(MODEL_DIR, exist_ok=True)  # Ensure models directory exists

download_blob(BLOB_NAME_MODELS, "models.tar.gz")
extract_tar_gz("models.tar.gz", MODEL_DIR)

download_blob(BLOB_NAME_DATASET, DATASET_PATH)

# ✅ Load Models
arima_model_path = os.path.join(MODEL_DIR, "arima_model.pkl")
iso_forest_path = os.path.join(MODEL_DIR, "iso_forest.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")  # ✅ Add scaler

arima_fit, iso_forest, scaler = None, None, None
try:
    if os.path.exists(arima_model_path):
        arima_fit = joblib.load(arima_model_path)
    if os.path.exists(iso_forest_path):
        iso_forest = joblib.load(iso_forest_path)
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)

    print("✅ All models loaded successfully.")
except Exception as e:
    print(f"❌ Error loading models: {e}")

# ✅ Load Dataset
df = None
try:
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
        df.set_index('Date', inplace=True)
        print("✅ Dataset loaded successfully.")
    else:
        print("⚠️ Dataset file not found!")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")

# ✅ Render the HTML Dashboard
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    if df is None:
        return "❌ Dataset not loaded!", 500

    latest_cost = df["Total Monthly Cost"].iloc[-1]  
    avg_cost = df["Total Monthly Cost"].mean()  

    return render_template(
        "dashboard.html",
        latest_cost=latest_cost,
        avg_cost=avg_cost,
    )

# ✅ Predict Future Costs
@app.route("/predict_cost", methods=["POST"])
def predict_cost():
    if not arima_fit:
        return jsonify({"error": "❌ ARIMA model not loaded"}), 500

    data = request.get_json()
    future_steps = int(data.get("days", 30))

    if len(df) < future_steps:
        return jsonify({"error": "❌ Not enough historical data for prediction"}), 400

    arima_pred = arima_fit.forecast(steps=future_steps).tolist()
    return jsonify({"Future Costs": arima_pred})

# ✅ Detect Anomalies
@app.route("/detect_anomalies", methods=["POST"])
def detect_anomalies():
    if not iso_forest:
        return jsonify({"error": "Anomaly detection model not loaded"}), 500
    
    data = request.get_json()
    if not data or "Total Monthly Cost" not in data:
        return jsonify({"error": "Missing 'Total Monthly Cost' in request data"}), 400
    
    cost = np.array(data["Total Monthly Cost"]).reshape(1, -1)
    anomaly = iso_forest.predict(cost)
    return jsonify({"Anomaly": "Yes" if anomaly[0] == -1 else "No"})


# ✅ Run Flask on Azure-Compatible Port
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
