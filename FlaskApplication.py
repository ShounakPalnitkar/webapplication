from flask import Flask, request, jsonify, render_template_string
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

# Load pre-trained LabelEncoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# HTML content inside the Python file
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Flask Prediction App</title>
</head>
<body>

<h1>Predict with Traditional and LLM Models</h1>

<form id="predictForm">
    <label for="feature1">Feature 1:</label>
    <input type="text" id="feature1" placeholder="Enter Feature 1" required><br><br>

    <label for="feature2">Feature 2:</label>
    <input type="text" id="feature2" placeholder="Enter Feature 2" required><br><br>

    <!-- Add more features here as needed -->
    
    <button type="submit">Predict</button>
</form>

<div id="response" style="margin-top: 20px;">
    <h3>Prediction Results:</h3>
    <pre id="result"></pre>
</div>

<script>
    document.getElementById('predictForm').onsubmit = async function(event) {
        event.preventDefault();

        // Collecting data from the form
        const data = {
            feature1: document.getElementById('feature1').value,
            feature2: document.getElementById('feature2').value,
            // Add more features here if needed
        };

        try {
            // Send POST request with input data
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            // Handle the response
            const result = await response.json();
            if (response.ok) {
                // Display the predictions in the UI
                document.getElementById('result').innerText = JSON.stringify(result, null, 2);
            } else {
                // Show error message if something goes wrong
                document.getElementById('result').innerText = 'Error: ' + result.error;
            }
        } catch (error) {
            // Handle any fetch errors
            document.getElementById('result').innerText = 'Error: ' + error.message;
        }
    };
</script>

</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(html_content)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Check if data is present
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Convert input data to DataFrame
        input_data = pd.DataFrame([data])

        # Encode categorical variables using the pre-trained label_encoder
        for col in input_data.select_dtypes(include=['object']).columns:
            input_data[col] = label_encoder.transform(input_data[col].astype(str))

        # Make predictions using both models
        trad_predictions = trad_model.predict(input_data)
        llm_predictions = llm_model.predict(input_data)

        # Average predicted probabilities for hybrid model
        hybrid_predictions_prob = (
            trad_model.predict_proba(input_data) + llm_model.predict_proba(input_data)
        ) / 2
        hybrid_prediction = hybrid_predictions_prob.argmax(axis=1)

        response = {
            'traditional_model': int(trad_predictions[0]),
            'llm_model': int(llm_predictions[0]),
            'hybrid_model': int(hybrid_prediction[0])
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
