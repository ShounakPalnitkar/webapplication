from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Initialize minimal models
trad_model = RandomForestClassifier()
llm_model = LogisticRegression(max_iter=10000)
label_encoder = LabelEncoder()

# Simple model initialization
def init_models():
    X_demo = pd.DataFrame({
        'Age': [50, 60, 70],
        'Blood Pressure': [120, 140, 160],
        'Serum Creatinine': [1.2, 1.8, 3.5],
        'Albumin': [3.5, 3.0, 2.5]
    })
    y_demo = ['Stage 1', 'Stage 2', 'Stage 3']
    y_encoded = label_encoder.fit_transform(y_demo)
    trad_model.fit(X_demo, y_encoded)
    llm_model.fit(X_demo, y_encoded)

init_models()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Kidney Health Portal</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f9fc;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .button-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        .health-btn {
            background-color: #2980b9;
            color: white;
            border: none;
            padding: 20px;
            border-radius: 8px;
            font-size: 18px;
            cursor: pointer;
            text-align: center;
            transition: all 0.3s;
        }
        .health-btn:hover {
            background-color: #3498db;
            transform: translateY(-3px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .health-btn i {
            font-size: 24px;
            margin-bottom: 10px;
            display: block;
        }
        .tool-container {
            display: none;
            margin-top: 30px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .close-btn {
            float: right;
            background: #e74c3c;
            color: white;
            border: none;
            padding: 5px 10px;
            border-radius: 4px;
            cursor: pointer;
        }
    </style>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
</head>
<body>
    <div class="header">
        <h1>Kidney Health Portal</h1>
        <p>Select a tool to get started</p>
    </div>

    <div class="button-container">
        <button class="health-btn" onclick="showTool('riskTool')">
            <i class="fas fa-calculator"></i>
            CKD Risk Calculator
        </button>
        
        <button class="health-btn" onclick="showTool('stageTool')">
            <i class="fas fa-chart-line"></i>
            CKD Stage Predictor
        </button>
        
        <button class="health-btn" onclick="showTool('dietTool')">
            <i class="fas fa-utensils"></i>
            Nutrition Guide
        </button>
        
        <button class="health-btn" onclick="showTool('doctorTool')">
            <i class="fas fa-user-md"></i>
            Find a Specialist
        </button>
    </div>

    <!-- CKD Risk Calculator (Your existing tool) -->
    <div id="riskTool" class="tool-container">
        <button class="close-btn" onclick="hideTool('riskTool')">×</button>
        <h2>CKD Risk Calculator</h2>
        <form id="riskForm">
            <!-- Your existing risk calculator form here -->
            <p>This is your original risk calculator tool</p>
            <button type="submit">Calculate Risk</button>
        </form>
    </div>

    <!-- CKD Stage Predictor -->
    <div id="stageTool" class="tool-container">
        <button class="close-btn" onclick="hideTool('stageTool')">×</button>
        <h2>CKD Stage Prediction</h2>
        <form id="stageForm">
            <div>
                <label>Age (years):</label>
                <input type="number" name="Age" required>
            </div>
            <!-- Add other fields -->
            <button type="submit">Predict Stage</button>
        </form>
        <div id="stageResults"></div>
    </div>

    <!-- Nutrition Guide -->
    <div id="dietTool" class="tool-container">
        <button class="close-btn" onclick="hideTool('dietTool')">×</button>
        <h2>CKD Nutrition Guide</h2>
        <div>
            <h3>Stage 1-2</h3>
            <p>Reduce sodium, maintain normal protein intake</p>
            
            <h3>Stage 3-4</h3>
            <p>Limit potassium and phosphorus, moderate protein (0.8g/kg)</p>
            
            <h3>Stage 5</h3>
            <p>Fluid restriction, low-phosphorus diet</p>
        </div>
    </div>

    <!-- Find a Specialist -->
    <div id="doctorTool" class="tool-container">
        <button class="close-btn" onclick="hideTool('doctorTool')">×</button>
        <h2>Find a Nephrologist</h2>
        <p>Search for kidney specialists in your area:</p>
        <button onclick="window.open('https://www.kidney.org/transplantation/transaction/TC/Centers', '_blank')">
            Visit Kidney.org
        </button>
    </div>

    <script>
        function showTool(toolId) {
            // Hide all tools first
            document.querySelectorAll('.tool-container').forEach(tool => {
                tool.style.display = 'none';
            });
            // Show selected tool
            document.getElementById(toolId).style.display = 'block';
        }
        
        function hideTool(toolId) {
            document.getElementById(toolId).style.display = 'none';
        }
        
        // Form handling for stage prediction
        document.getElementById('stageForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const formData = {
                Age: parseFloat(e.target.Age.value),
                // Add other fields
            };
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(formData)
                });
                const data = await response.json();
                document.getElementById('stageResults').innerHTML = `
                    <h3>Results</h3>
                    <p>Predicted Stage: ${data.hybrid.stage}</p>
                    <p>Confidence: ${(data.hybrid.confidence * 100).toFixed(1)}%</p>
                `;
            } catch (error) {
                alert('Prediction error: ' + error.message);
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
        
        # Hybrid prediction logic
        hybrid_proba = (trad_model.predict_proba(input_df)[0] + 
                      llm_model.predict_proba(input_df)[0]) / 2
        hybrid_pred = hybrid_proba.argmax()
        hybrid_stage = label_encoder.inverse_transform([hybrid_pred])[0]
        
        return jsonify({
            'hybrid': {
                'stage': hybrid_stage,
                'confidence': float(hybrid_proba.max())
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
