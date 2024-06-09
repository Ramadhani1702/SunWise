from flask import Flask, request, jsonify
from flask_cors import CORS
from fastai.vision.all import *
import os
import tempfile
import joblib
import pandas as pd

# Set up the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Path to the models
cnn_model_path = 'model_fastai.pkl'
decision_tree_model_path = 'decision_tree_model.pkl'

# Load the models
learn = load_learner(cnn_model_path)
model = joblib.load(decision_tree_model_path)
print("Models loaded successfully.")

# Define a function to predict estimation time
def predict_estimation_time(skin_type, uv_index):
    # Convert skin_type to numerical value
    skin_type_mapping = {
        'Type I & II': 0,
        'Type III': 1,
        'Type IV': 2,
        'Type V': 3,
        'Type VI': 4
    }
    
    if skin_type not in skin_type_mapping:
        return "Invalid skin type"
    
    skin_type_numeric = skin_type_mapping[skin_type]
    
    # Create a DataFrame with the input values
    input_data = {'skin_type': skin_type_numeric, 'uv_index': uv_index}
    input_df = pd.DataFrame([input_data])

    # Predict estimation_time
    estimation_time = model.predict(input_df)[0]
    return estimation_time

@app.route('/predict', methods=['GET', 'POST'])
def predictedtime():
    prediction = None
    estimation = None
    try:
        if request.method == 'POST':
            # Check if the POST request has the file part and UV index
            if 'imagefile' not in request.files or 'uv_index' not in request.form:
                return jsonify({
                    "data": None,
                    "status": {
                        "code": 400,
                        "message": "No file part or UV index provided"
                    }
                }), 400
            
            file = request.files['imagefile']
            uv_index = request.form['uv_index']
            
            # Validate UV index
            if not uv_index.isdigit():
                return jsonify({
                    "data": None,
                    "status": {
                        "code": 400,
                        "message": "Invalid UV index provided"
                    }
                }), 400

            uv_index = int(uv_index)

            # If the user does not select a file
            if file.filename == '':
                return jsonify({
                    "data": None,
                    "status": {
                        "code": 400,
                        "message": "No selected file"
                    }
                }), 400

            # Make prediction using CNN
            img = PILImage.create(file.stream)
            pred, pred_idx, probs = learn.predict(img)

            prediction = f'{pred} with probability {max(probs).item():.4f}'

            # Convert the predicted skin type to match the format expected by the decision tree model
            skin_type_mapping = {
                'I & II': 'Type I & II',
                'III': 'Type III',
                'IV': 'Type IV',
                'V': 'Type V',
                'VI': 'Type VI'
            }

            # Check if the prediction is valid and then map it
            pred_mapped = skin_type_mapping.get(pred)
            if (pred_mapped):
                # Use the mapped prediction as input to the decision tree model
                estimation = predict_estimation_time(pred_mapped, uv_index)
            else:
                estimation = "Invalid skin type"

        return jsonify({
            "data": {
                "Prediction": prediction,
                "Estimation": estimation
            },
            "status": {
                "code": 200,
                "message": "Successfully predicted"
            }
        }), 200

    except Exception as e:
        return jsonify({
            "data": None,
            "status": {
                "code": 500,
                "message": f"An error occurred: {str(e)}"
            }
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
