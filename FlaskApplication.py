from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import base64
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__, static_url_path='/static')

# Initialize models and encoder
trad_model = RandomForestClassifier()
llm_model = LogisticRegression(max_iter=10000)
label_encoder = LabelEncoder()

# Sample training data structure (minimum required to initialize models)
def initialize_models():
    # Create minimal synthetic data just to initialize models
    X_demo = pd.DataFrame({
        'Age': [50, 60, 70],
        'Blood Pressure': [120, 140, 160],
        'Serum Creatinine': [1.2, 1.8, 3.5],
        'Albumin': [3.5, 3.0, 2.5]
    })
    y_demo = ['Stage 1', 'Stage 2', 'Stage 3']
    
    # Fit encoder and models
    y_encoded = label_encoder.fit_transform(y_demo)
    trad_model.fit(X_demo, y_encoded)
    llm_model.fit(X_demo, y_encoded)

# Initialize models at startup
initialize_models()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>CKD Health Advisor</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f9fc;
        }
        .header {
            background-color: #1a5276;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            text-align: center;
        }
        .container {
            display: flex;
            gap: 30px;
        }
        .prediction-card, .info-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            flex: 1;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #2c3e50;
        }
        input, select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        button {
            background-color: #2980b9;
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
            margin-top: 10px;
        }
        button:hover {
            background-color: #3498db;
        }
        .results {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #eee;
        }
        .model-result {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 15px;
        }
        .plot-container {
            margin-top: 20px;
        }
        .info-section {
            margin-bottom: 20px;
        }
        h3 {
            color: #1a5276;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }
        .hidden {
            display: none;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Chronic Kidney Disease Health Advisor</h1>
        <p>Get personalized predictions and health information</p>
    </div>
    
    <div class="container">
        <div class="prediction-card">
            <h2>CKD Stage Prediction</h2>
            <form id="predictionForm">
                <div class="form-group">
                    <label for="age">Age (years):</label>
                    <input type="number" id="age" name="Age" min="18" max="120" required>
                </div>
                
                <div class="form-group">
                    <label for="bp">Blood Pressure (mmHg):</label>
                    <input type="number" id="bp" name="Blood Pressure" min="50" max="250" required>
                </div>
                
                <div class="form-group">
                    <label for="creatinine">Serum Creatinine (mg/dL):</label>
                    <input type="number" step="0.1" id="creatinine" name="Serum Creatinine" min="0.1" max="15" required>
                </div>
                
                <div class="form-group">
                    <label for="albumin">Albumin (g/dL):</label>
                    <input type="number" step="0.1" id="albumin" name="Albumin" min="1" max="5" required>
                </div>
                
                <button type="submit">Predict CKD Stage</button>
            </form>
            
            <div id="results" class="results hidden">
                <h3>Prediction Results</h3>
                
                <div class="model-result">
                    <h4>Traditional Model (Random Forest)</h4>
                    <p><strong>Predicted Stage:</strong> <span id="trad-stage"></span></p>
                    <p><strong>Confidence:</strong> <span id="trad-conf"></span></p>
                </div>
                
                <div class="model-result">
                    <h4>LLM Model (Logistic Regression)</h4>
                    <p><strong>Predicted Stage:</strong> <span id="llm-stage"></span></p>
                    <p><strong>Confidence:</strong> <span id="llm-conf"></span></p>
                </div>
                
                <div class="model-result">
                    <h4>Hybrid Model</h4>
                    <p><strong>Predicted Stage:</strong> <span id="hybrid-stage"></span></p>
                    <p><strong>Confidence:</strong> <span id="hybrid-conf"></span></p>
                </div>
                
                <div class="plot-container">
                    <img id="roc-plot" src="" alt="Model Performance" style="max-width: 100%;">
                </div>
            </div>
        </div>
        
        <div class="info-card">
            <h2>CKD Health Information</h2>
            
            <div class="info-section">
                <h3>About Chronic Kidney Disease</h3>
                <p>Chronic Kidney Disease (CKD) is a progressive loss of kidney function over time. Early detection and proper management can slow its progression.</p>
            </div>
            
            <div class="info-section">
                <h3>Common Symptoms</h3>
                <ul>
                    <li>Fatigue and weakness</li>
                    <li>Swelling in legs, ankles or feet</li>
                    <li>Shortness of breath</li>
                    <li>Poor appetite</li>
                    <li>Changes in urination patterns</li>
                </ul>
            </div>
            
            <div class="info-section">
                <h3>Prevention Tips</h3>
                <ul>
                    <li>Control blood pressure</li>
                    <li>Manage blood sugar if diabetic</li>
                    <li>Maintain a healthy diet low in salt</li>
                    <li>Stay physically active</li>
                    <li>Avoid excessive use of NSAIDs</li>
                </ul>
            </div>
            
            <div class="info-section">
                <h3>When to See a Doctor</h3>
                <p>Consult a nephrologist if you experience persistent symptoms or have risk factors like diabetes, high blood pressure, or family history of kidney disease.</p>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = {
                Age: parseFloat(document.getElementById('age').value),
                'Blood Pressure': parseFloat(document.getElementById('bp').value),
                'Serum Creatinine': parseFloat(document.getElementById('creatinine').value),
                Albumin: parseFloat(document.getElementById('albumin').value)
            };
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(formData)
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    document.getElementById('trad-stage').textContent = data.traditional.stage;
                    document.getElementById('trad-conf').textContent = (data.traditional.confidence * 100).toFixed(2) + '%';
                    
                    document.getElementById('llm-stage').textContent = data.llm.stage;
                    document.getElementById('llm-conf').textContent = (data.llm.confidence * 100).toFixed(2) + '%';
                    
                    document.getElementById('hybrid-stage').textContent = data.hybrid.stage;
                    document.getElementById('hybrid-conf').textContent = (data.hybrid.confidence * 100).toFixed(2) + '%';
                    
                    document.getElementById('roc-plot').src = 'data:image/png;base64,' + data.roc_plot;
                    document.getElementById('results').classList.remove('hidden');
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                alert('Error submitting form: ' + error.message);
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        
        # Get predictions
        trad_proba = trad_model.predict_proba(input_df)[0]
        trad_pred = trad_model.predict(input_df)[0]
        trad_stage = label_encoder.inverse_transform([trad_pred])[0]
        
        llm_proba = llm_model.predict_proba(input_df)[0]
        llm_pred = llm_model.predict(input_df)[0]
        llm_stage = label_encoder.inverse_transform([llm_pred])[0]
        
        # Hybrid prediction
        hybrid_proba = (trad_proba + llm_proba) / 2
        hybrid_pred = hybrid_proba.argmax()
        hybrid_stage = label_encoder.inverse_transform([hybrid_pred])[0]
        
        # Generate ROC plot
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Model Performance Comparison')
        
        # Save plot to bytes
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png')
        img_bytes.seek(0)
        plt.close()
        
        return jsonify({
            'traditional': {
                'stage': trad_stage,
                'confidence': float(trad_proba.max())
            },
            'llm': {
                'stage': llm_stage,
                'confidence': float(llm_proba.max())
            },
            'hybrid': {
                'stage': hybrid_stage,
                'confidence': float(hybrid_proba.max())
            },
            'roc_plot': base64.b64encode(img_bytes.read()).decode('utf-8')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
