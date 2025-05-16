from flask import Flask, request, jsonify
from flask_cors import CORS # Make sure this is in your requirements.txt
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input
from sklearn.preprocessing import StandardScaler
import joblib
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": [
    "http://localhost:3000",
    "https://jd-frontend-s11e.onrender.com"
]}})

# FRONTEND_URL_FROM_ENV = os.environ.get("FRONTEND_APP_URL") 
# if FRONTEND_URL_FROM_ENV and FRONTEND_URL_FROM_ENV != "ALLOW_ALL_FOR_SETUP":
#     print(f"* Configuring CORS for specific origin: {FRONTEND_URL_FROM_ENV}")
#     CORS(app, resources={r"/*": {"origins": FRONTEND_URL_FROM_ENV}})
# else:
#     print("* FRONTEND_APP_URL not set. Configuring CORS to allow all origins (for initial setup/dev).")
#     CORS(app) 

CNN_MODEL_PATH = 'jackfruit_cnn_feature_extractor.h5'
SCALER_PATH = 'jackfruit_feature_scaler.pkl'
SVM_MODEL_PATH = 'jackfruit_svm_classifier.pkl'

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
CLASS_NAMES = ['Fruit_borer', 'Fruit_fly', 'Healthy', 'Rhizopus_rot']
MODEL_PREPROCESS_INPUT = mobilenet_v2_preprocess_input

cnn_feature_extractor = None
scaler = None
svm_classifier = None

try:
    if os.path.exists(CNN_MODEL_PATH):
        cnn_feature_extractor = tf.keras.models.load_model(CNN_MODEL_PATH)
        print(f"* CNN Feature Extractor model loaded successfully from {CNN_MODEL_PATH}")
    else:
        print(f"!!! CNN Model file not found at {CNN_MODEL_PATH}. Check if it's in the repository and the path is correct.")
except Exception as e:
    print(f"!!! Error loading CNN Feature Extractor model: {e}")

try:
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print(f"* Scaler loaded successfully from {SCALER_PATH}")
    else:
        print(f"!!! Scaler file not found at {SCALER_PATH}. Check if it's in the repository and the path is correct.")
except Exception as e:
    print(f"!!! Error loading scaler: {e}")

try:
    if os.path.exists(SVM_MODEL_PATH):
        svm_classifier = joblib.load(SVM_MODEL_PATH)
        print(f"* SVM Classifier model loaded successfully from {SVM_MODEL_PATH}")
    else:
        print(f"!!! SVM Classifier file not found at {SVM_MODEL_PATH}. Check if it's in the repository and the path is correct.")
except Exception as e:
    print(f"!!! Error loading SVM Classifier model: {e}")


# --- Preprocessing Function (Unchanged) ---
def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        preprocessed_img_array = MODEL_PREPROCESS_INPUT(img_array_expanded.copy())
        return preprocessed_img_array
    except Exception as e:
        app.logger.error(f"Image preprocessing error: {e}", exc_info=True)
        return None

# Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if cnn_feature_extractor is None:
        app.logger.error("CNN Feature Extractor model is not loaded.")
        return jsonify({"error": "Analysis service error: CNN model not available"}), 503
    if scaler is None:
        app.logger.error("Scaler is not loaded.")
        return jsonify({"error": "Analysis service error: Scaler not available"}), 503
    if svm_classifier is None:
        app.logger.error("SVM Classifier model is not loaded.")
        return jsonify({"error": "Analysis service error: Classifier not available"}), 503

    if 'file' not in request.files:
        return jsonify({"error": "No image file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for upload"}), 400

    if file:
        try:
            image_bytes = file.read()
            preprocessed_img = preprocess_image(image_bytes)
            if preprocessed_img is None:
                return jsonify({"error": "Image preprocessing failed. Check server logs."}), 400

            app.logger.debug("Extracting features...")
            features = cnn_feature_extractor.predict(preprocessed_img)
            app.logger.debug(f"Raw features shape: {features.shape}")

            features_reshaped = features.reshape(1, -1) if features.ndim == 1 else features
            scaled_features = scaler.transform(features_reshaped)
            app.logger.debug(f"Scaled features shape: {scaled_features.shape}")

            prediction_numeric_array = svm_classifier.predict(scaled_features)
            predicted_class_index = int(prediction_numeric_array[0])
            app.logger.debug(f"Predicted class index: {predicted_class_index}")

            if 0 <= predicted_class_index < len(CLASS_NAMES):
                label = CLASS_NAMES[predicted_class_index]
                app.logger.info(f"Prediction successful: Index={predicted_class_index}, Label='{label}'")
                return jsonify({"prediction": label, "class_index": predicted_class_index})
            else:
                app.logger.error(f"Predicted class index {predicted_class_index} is out of bounds.")
                return jsonify({"error": "Prediction resulted in an invalid class."}), 500
        except Exception as e:
            app.logger.error(f"An error occurred during prediction: {e}", exc_info=True)
            return jsonify({"error": "An unexpected error occurred on the server during prediction."}), 500
    else:
        return jsonify({"error": "Invalid file provided"}), 400

# Health Check Endpoint
@app.route('/health', methods=['GET'])
def health_check():
    models_loaded = cnn_feature_extractor is not None and scaler is not None and svm_classifier is not None
    if models_loaded:
        return jsonify({"status": "UP", "message": "Analysis service is running and models are loaded."}), 200
    else:

        missing_models = []
        if cnn_feature_extractor is None: missing_models.append("CNN")
        if scaler is None: missing_models.append("Scaler")
        if svm_classifier is None: missing_models.append("SVM")
        return jsonify({
            "status": "DEGRADED",
            "message": f"Analysis service is running but failed to load: {', '.join(missing_models)}."
        }), 503

# App run
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)