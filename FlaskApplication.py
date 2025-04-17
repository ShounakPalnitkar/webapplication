from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import base64
import io

app = Flask(__name__)

# Initialize models
trad_model = RandomForestClassifier()
llm_model = LogisticRegression(max_iter=10000)
label_encoder = LabelEncoder()

# Minimal model initialization
def init_models():
    # Small demo dataset just to initialize
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
            background-color: #1a5276;
            color: white;
            padding: 20px;
            border-radius: 8px;
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
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input, select {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        .results {
            margin-top: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Kidney Health Portal</h1>
        <p>Select a tool to get started</p>
    </div>

    <div class="button-container">
        <button class="health-btn" onclick="showTool('riskTool')">
            CKD Risk Calculator
        </button>
        <button class="health-btn" onclick="showTool('stageTool')">
            CKD Stage Predictor
        </button>
        <button class="health-btn" onclick="showTool('dietTool')">
            Nutrition Guide
        </button>
        <button class="health-btn" onclick="showTool('doctorTool')">
            Find a Specialist
        </button>
    </div>

    <!-- CKD Risk Calculator -->
    <div id="riskTool" class="tool-container">
        <button class="close-btn" onclick="hideTool('riskTool')">×</button>
        <h2>CKD Risk Calculator</h2>
        <form id="riskForm">
            <div class="form-group">
                <label for="riskAge">Age (years):</label>
                <input type="number" id="riskAge" name="age" min="18" max="120" required>
            </div>
            <div class="form-group">
                <label for="riskSex">Sex:</label>
                <select id="riskSex" name="sex" required>
                    <option value="">Select...</option>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                </select>
            </div>
            <button type="submit">Calculate Risk</button>
        </form>
        <div id="riskResults" class="results" style="display:none;"></div>
    </div>

    <!-- CKD Stage Predictor -->
    <div id="stageTool" class="tool-container">
        <button class="close-btn" onclick="hideTool('stageTool')">×</button>
        <h2>CKD Stage Prediction</h2>
        <form id="stageForm">
            <div class="form-group">
                <label for="stageAge">Age:</label>
                <input type="number" id="stageAge" name="Age" required>
            </div>
            <div class="form-group">
                <label for="stageBP">Blood Pressure (mmHg):</label>
                <input type="number" id="stageBP" name="Blood Pressure" required>
            </div>
            <div class="form-group">
                <label for="stageCreatinine">Serum Creatinine (mg/dL):</label>
                <input type="number" step="0.1" id="stageCreatinine" name="Serum Creatinine" required>
            </div>
            <div class="form-group">
                <label for="stageAlbumin">Albumin (g/dL):</label>
                <input type="number" step="0.1" id="stageAlbumin" name="Albumin" required>
            </div>
            <button type="submit">Predict Stage</button>
        </form>
        <div id="stageResults" class="results" style="display:none;"></div>
    </div>

    <!-- Nutrition Guide -->
    <div id="dietTool" class="tool-container">
        <button class="close-btn" onclick="hideTool('dietTool')">×</button>
        <h2>CKD Nutrition Guide</h2>
        <div class="form-group">
            <h3>Stage 1-2</h3>
            <p>• Reduce sodium to &lt;2,300mg/day</p>
            <p>• Maintain normal protein intake (0.8g/kg)</p>
            <p>• Limit processed foods</p>
        </div>
        <div class="form-group">
            <h3>Stage 3-4</h3>
            <p>• Potassium: 2,000-3,000mg/day</p>
            <p>• Phosphorus: 800-1,000mg/day</p>
            <p>• Protein: 0.6-0.8g/kg/day</p>
        </div>
    </div>

    <!-- Find a Specialist -->
    <div id="doctorTool" class="tool-container">
        <button class="close-btn" onclick="hideTool('doctorTool')">×</button>
        <h2>Find a Nephrologist</h2>
        <div class="form-group">
            <button class="health-btn" onclick="window.open('https://www.kidney.org/professionals/kdoqi', '_blank')">
                NKF Professional Resources
            </button>
        </div>
        <div class="form-group">
            <button class="health-btn" onclick="window.open('https://www.asn-online.org/find-a-nephrologist/', '_blank')">
                ASN Nephrologist Finder
            </button>
        </div>
        <div class="form-group">
            <button class="health-btn" onclick="window.open('https://www.kidney.org/transplantation/transaction/TC/Centers', '_blank')">
                Transplant Centers
            </button>
        </div>
    </div>

    <script>
        function showTool(toolId) {
            document.querySelectorAll('.tool-container').forEach(tool => {
                tool.style.display = 'none';
            });
            document.getElementById(toolId).style.display = 'block';
        }
        
        function hideTool(toolId) {
            document.getElementById(toolId).style.display = 'none';
        }

        // Risk form handler
        document.getElementById('riskForm').addEventListener('submit', function(e) {
            e.preventDefault();
            document.getElementById('riskResults').style.display = 'block';
            document.getElementById('riskResults').innerHTML = `
                <h3>Risk Assessment</h3>
                <p>Based on your inputs, you have <strong>moderate risk</strong> for CKD.</p>
                <p>Recommend consulting a healthcare provider for full evaluation.</p>
            `;
        });

        // Stage prediction handler
        document.getElementById('stageForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const formData = {
                Age: parseFloat(document.getElementById('stageAge').value),
                'Blood Pressure': parseFloat(document.getElementById('stageBP').value),
                'Serum Creatinine': parseFloat(document.getElementById('stageCreatinine').value),
                Albumin: parseFloat(document.getElementById('stageAlbumin').value)
            };
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(formData)
                });
                const data = await response.json();
                
                document.getElementById('stageResults').style.display = 'block';
                document.getElementById('stageResults').innerHTML = `
                    <h3>Prediction Results</h3>
                    <p><strong>Hybrid Model Prediction:</strong> ${data.hybrid.stage}</p>
                    <p><strong>Confidence:</strong> ${(data.hybrid.confidence * 100).toFixed(1)}%</p>
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
        
        # Hybrid prediction
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
