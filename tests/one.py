# ────────────────────────────────────────────────────────────────
#   Read Vertex AI Registered Model in Dataproc Jupyter Notebook
# ────────────────────────────────────────────────────────────────

import os
from google.cloud import aiplatform
import pickle           # or joblib, tensorflow, torch, etc.
from pathlib import Path

# ── 1. Configuration ─────────────────────────────────────────────
PROJECT_ID = "your-project-id"
LOCATION   = "europe-west4"          # or us-central1, etc.
MODEL_ID   = "orange-model"          # display name or numeric ID

# Choose how to reference the model (recommended order)
MODEL_REFERENCE = f"projects/{PROJECT_ID}/locations/{LOCATION}/models/{MODEL_ID}@champion"
# alternatives:
# MODEL_REFERENCE = f"projects/{PROJECT_ID}/locations/{LOCATION}/models/{MODEL_ID}@5"     # specific version
# MODEL_REFERENCE = f"projects/{PROJECT_ID}/locations/{LOCATION}/models/{MODEL_ID}"      # latest

# Where to download model artifacts locally in the notebook VM
LOCAL_MODEL_DIR = "/tmp/vertex_model"

# ── 2. Initialize & Get model metadata ───────────────────────────
aiplatform.init(project=PROJECT_ID, location=LOCATION)

print(f"Loading model from registry: {MODEL_REFERENCE}\n")

model = aiplatform.Model(model_name=MODEL_REFERENCE)

print("Model information:")
print(f"  Display name    : {model.display_name}")
print(f"  Version ID      : {model.version_id}")
print(f"  Version aliases : {model.version_aliases}")
print(f"  Artifact URI    : {model.gca_resource.artifact_uri}")
print(f"  Container       : {model.gca_resource.container_spec.image_uri or '—'}")

# Most important line:
GCS_ARTIFACT_URI = model.gca_resource.artifact_uri   # gs://bucket/path/to/model/

# ── 3. Download model artifacts to local disk ────────────────────
print(f"\nDownloading artifacts from:\n  {GCS_ARTIFACT_URI}\n  →  {LOCAL_MODEL_DIR}")

# Option A - Fastest & most reliable (recommended in Dataproc)
!gsutil -m cp -r {GCS_ARTIFACT_URI}/* {LOCAL_MODEL_DIR}/

# Option B - Using Python (gcsfs) - good when you want more control
# import gcsfs
# fs = gcsfs.GCSFileSystem()
# fs.get(GCS_ARTIFACT_URI.replace("gs://",""), LOCAL_MODEL_DIR, recursive=True)

print("Download finished!\n")

# ── 4. Load the model - choose your framework ────────────────────

LOCAL_MODEL_PATH = Path(LOCAL_MODEL_DIR)

#  ┌─────────────────────────────┐
#  │ Choose ONE of the following │
#  └─────────────────────────────┘

# A. Scikit-learn / joblib / pickle models (very common)
model_path = LOCAL_MODEL_PATH / "model.joblib"          # or model.pkl
import joblib
loaded_model = joblib.load(model_path)
# loaded_model = pickle.load(open(model_path, "rb"))    # alternative

# B. TensorFlow SavedModel
# import tensorflow as tf
# loaded_model = tf.saved_model.load(str(LOCAL_MODEL_PATH))

# C. PyTorch (torchscript or full model)
# import torch
# loaded_model = torch.jit.load(LOCAL_MODEL_PATH / "model.pt")
# # or: loaded_model = torch.load(LOCAL_MODEL_PATH / "model.pth")

# D. XGBoost (if saved as .json / .ubj / .model)
# import xgboost as xgb
# loaded_model = xgb.Booster()
# loaded_model.load_model(LOCAL_MODEL_PATH / "model.bst")

print("\nModel loaded successfully!")
print("Type:", type(loaded_model))

# ── 5. (Optional) Quick test / predict ───────────────────────────
# Example for scikit-learn style model
# sample = [[5.1, 3.5, 1.4, 0.2]]   # your feature vector(s)
# prediction = loaded_model.predict(sample)
# print("Sample prediction:", prediction)
