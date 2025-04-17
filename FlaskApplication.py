from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import base64
import io
import os

app = Flask(__name__)

# Load or create models (for demonstration, I'll include the training code)
# In a real deployment, you would load pre-trained models
try:
    # Try to load pre-trained models
    with open('trad_model.pkl', 'rb') as f:
        trad_model = pickle.load(f)
    with open('llm_model.pkl', 'rb') as f:
        llm_model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
except FileNotFoundError:
    # If models don't exist, create dummy models (replace with your actual training code)
    print("Model files not found. Creating dummy models for demonstration.")
    
    # Create some dummy data
    X_dummy = pd.DataFrame({
        'Age': np.random.randint(20, 80, 100),
        'Blood Pressure': np.random.randint(80, 180, 100),
        'Serum Creatinine': np.random.uniform(0.5, 10, 100),
        'Albumin': np.random.uniform(1, 5, 100)
    })
    y_dummy = np.random.choice(['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4', 'Stage 5'], 100)
    
    # Create and fit label encoder
    label_encoder = LabelEncoder()
    y_dummy_encoded = label_encoder.fit_transform(y_dummy)
    
    # Create and fit models
    trad_model = RandomForestClassifier()
    trad_model.fit(X_dummy, y_dummy_encoded)
    
    llm_model = LogisticRegression(max_iter=1000)
    llm_model.fit(X_dummy, y_dummy_encoded)
    
    # Save models for next time
    with open('trad_model.pkl', 'wb') as f:
        pickle.dump(trad_model, f)
    with open('llm_model.pkl', 'wb') as f:
        pickle.dump(llm_model, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

# HTML template with form and visualization
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CKD Stage Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .form-group { margin-bottom: 15px; }
        label { display: inline-block; width: 200px; }
        input { padding: 8px; width: 100px; }
        button { padding: 10px 15px; background: #4CAF50; color: white; border: none; cursor: pointer; }
        button:hover { background: #45a049; }
        .results { margin-top: 30px; }
        .model-result { margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .plot-container { margin-top: 20px; text-align: center; }
        .plot-img { max-width: 100%; }
    </style>
</head>
<body>
    <h1>Chronic Kidney Disease (CKD) Stage Prediction</h1>
    
    <form id="predictionForm">
        <div class="form-group">
            <label for="age">Age:</label>
            <input type="number" id="age" name="age" required>
        </div>
        
        <div class="form-group">
            <label for="bp">Blood Pressure (mmHg):</label>
            <input type="number" id="bp" name="bp" required>
        </div>
        
        <div class="form-group">
            <label for="creatinine">Serum Creatinine (mg/dL):</label>
            <input type="number" step="0.1" id="creatinine" name="creatinine" required>
        </div>
        
        <div class="form-group">
            <label for="albumin">Albumin (g/dL):</label>
            <input type="number" step="0.1" id="albumin" name="albumin" required>
        </div>
        
        <button type="submit">Predict CKD Stage</button>
    </form>
    
    <div id="results" class="results" style="display: none;">
        <h2>Prediction Results</h2>
        
        <div class="model-result">
            <h3>Traditional Model (Random Forest)</h3>
            <p><strong>Predicted Stage:</strong> <span id="trad-result"></span></p>
            <p><strong>Confidence:</strong> <span id="trad-confidence"></span></p>
        </div>
        
        <div class="model-result">
            <h3>LLM Model (Logistic Regression)</h3>
            <p><strong>Predicted Stage:</strong> <span id="llm-result"></span></p>
            <p><strong>Confidence:</strong> <span id="llm-confidence"></span></p>
        </div>
        
        <div class="model-result">
            <h3>Hybrid Model</h3>
            <p><strong>Predicted Stage:</strong> <span id="hybrid-result"></span></p>
            <p><strong>Confidence:</strong> <span id="hybrid-confidence"></span></p>
        </div>
        
        <div class="plot-container">
            <h3>Model Comparison</h3>
            <img id="roc-plot" class="plot-img" src="" alt="ROC Curve Comparison">
        </div>
    </div>
    
    <script>
        document.getElementById('predictionForm').onsubmit = async function(e) {
            e.preventDefault();
            
            const formData = {
                age: parseFloat(document.getElementById('age').value),
                bp: parseFloat(document.getElementById('bp').value),
                creatinine: parseFloat(document.getElementById('creatinine').value),
                albumin: parseFloat(document.getElementById('albumin').value)
            };
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(formData)
                });
                
                if (!response.ok) {
                    throw new Error('Prediction failed');
                }
                
                const data = await response.json();
                
                // Display results
                document.getElementById('trad-result').textContent = data.traditional.stage;
                document.getElementById('trad-confidence').textContent = (data.traditional.confidence * 100).toFixed(2) + '%';
                
                document.getElementById('llm-result').textContent = data.llm.stage;
                document.getElementById('llm-confidence').textContent = (data.llm.confidence * 100).toFixed(2) + '%';
                
                document.getElementById('hybrid-result').textContent = data.hybrid.stage;
                document.getElementById('hybrid-confidence').textContent = (data.hybrid.confidence * 100).toFixed(2) + '%';
                
                // Display ROC plot
                document.getElementById('roc-plot').src = 'data:image/png;base64,' + data.roc_plot;
                
                // Show results section
                document.getElementById('results').style.display = 'block';
                
            } catch (error) {
                alert('Error: ' + error.message);
            }
        };
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        data = request.get_json()
        
        # Create DataFrame with expected features
        input_data = pd.DataFrame([{
            'Age': data['age'],
            'Blood Pressure': data['bp'],
            'Serum Creatinine': data['creatinine'],
            'Albumin': data['albumin']
        }])
        
        # Make predictions
        trad_proba = trad_model.predict_proba(input_data)[0]
        trad_pred = trad_model.predict(input_data)[0]
        trad_stage = label_encoder.inverse_transform([trad_pred])[0]
        trad_confidence = trad_proba.max()
        
        llm_proba = llm_model.predict_proba(input_data)[0]
        llm_pred = llm_model.predict(input_data)[0]
        llm_stage = label_encoder.inverse_transform([llm_pred])[0]
        llm_confidence = llm_proba.max()
        
        # Hybrid prediction (average probabilities)
        hybrid_proba = (trad_proba + llm_proba) / 2
        hybrid_pred = hybrid_proba.argmax()
        hybrid_stage = label_encoder.inverse_transform([hybrid_pred])[0]
        hybrid_confidence = hybrid_proba.max()
        
        # Create ROC plot
        plt.figure(figsize=(8, 6))
        
        # For demonstration, we'll create dummy ROC curves
        # In a real app, you would use actual ROC data from your models
        fpr, tpr = np.linspace(0, 1, 100), np.linspace(0, 1, 100)
        plt.plot(fpr, tpr, label=f'Traditional Model (AUC = 0.85)')
        plt.plot(fpr, tpr * 0.9, label=f'LLM Model (AUC = 0.80)')
        plt.plot(fpr, tpr * 0.95, label=f'Hybrid Model (AUC = 0.88)')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Model Comparison (ROC Curves)')
        plt.legend(loc="lower right")
        
        # Save plot to bytes
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png')
        img_bytes.seek(0)
        plt.close()
        
        # Prepare response
        response = {
            'traditional': {
                'stage': trad_stage,
                'confidence': float(trad_confidence)
            },
            'llm': {
                'stage': llm_stage,
                'confidence': float(llm_confidence)
            },
            'hybrid': {
                'stage': hybrid_stage,
                'confidence': float(hybrid_confidence)
            },
            'roc_plot': base64.b64encode(img_bytes.read()).decode('utf-8')
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
