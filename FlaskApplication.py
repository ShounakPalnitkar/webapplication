from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load pre-trained models
with open('trad_model.pkl', 'rb') as f:
    trad_model = pickle.load(f)

with open('llm_model.pkl', 'rb') as f:
    llm_model = pickle.load(f)

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Expecting a JSON input

    # Convert input data to DataFrame for the model
    input_data = pd.DataFrame([data])

    # Encode categorical variables (if any)
    for col in input_data.select_dtypes(include=['object']).columns:
        input_data[col] = label_encoder.fit_transform(input_data[col])

    # Make predictions using both models
    trad_predictions = trad_model.predict(input_data)
    llm_predictions = llm_model.predict(input_data)

    # Combine predictions from both models (simple average of probabilities)
    hybrid_predictions_prob = (trad_model.predict_proba(input_data) + llm_model.predict_proba(input_data)) / 2
    hybrid_prediction = hybrid_predictions_prob.argmax(axis=1)

    # Return predictions as a response
    response = {
        'traditional_model': trad_predictions[0],
        'llm_model': llm_predictions[0],
        'hybrid_model': hybrid_prediction[0]
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)