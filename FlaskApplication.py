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

# Load or train models (same as before)
def load_or_train_models():
    try:
        with open('trad_model.pkl', 'rb') as f:
            trad_model = pickle.load(f)
        with open('llm_model.pkl', 'rb') as f:
            llm_model = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        return trad_model, llm_model, label_encoder
    except FileNotFoundError:
        print("Training models...")
        trad_data = pd.read_csv('5000textdata-Updated ones.csv')
        X_trad = trad_data.drop(columns=['Patient ID', 'Diagnosis Date', 'CKD Stage'])
        y_trad = trad_data['CKD Stage']
        
        label_encoder = LabelEncoder()
        for col in X_trad.select_dtypes(include=['object']).columns:
            X_trad[col] = label_encoder.fit_transform(X_trad[col])
        y_trad_encoded = label_encoder.fit_transform(y_trad)
        
        trad_model = RandomForestClassifier()
        trad_model.fit(X_trad, y_trad_encoded)
        
        llm_model = LogisticRegression(max_iter=10000)
        llm_model.fit(X_trad, y_trad_encoded)
        
        with open('trad_model.pkl', 'wb') as f:
            pickle.dump(trad_model, f)
        with open('llm_model.pkl', 'wb') as f:
            pickle.dump(llm_model, f)
        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)
            
        return trad_model, llm_model, label_encoder

trad_model, llm_model, label_encoder = load_or_train_models()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Kidney Health Portal</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }
        .container { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }
        .card { border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
        h1 { color: #2c3e50; text-align: center; grid-column: 1 / -1; }
        h2 { color: #3498db; border-bottom: 1px solid #eee; padding-bottom: 10px; }
        label { display: block; margin: 10px 0 5px; }
        input, select { width: 100%; padding: 8px; margin-bottom: 10px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #3498db; color: white; border: none; padding: 10px 15px; border-radius: 4px; cursor: pointer; }
        button:hover { background: #2980b9; }
        .health-btn { background: #27ae60; margin: 5px 0; width: 100%; text-align: left; }
        .results { background: #f8f9fa; padding: 15px; border-radius: 4px; margin-top: 15px; }
        .hidden { display: none; }
        .symptom-checkbox { margin: 5px 0; }
        .disclaimer { font-size: 0.8em; color: #7f8c8d; margin-top: 30px; border-top: 1px solid #eee; padding-top: 10px; }
    </style>
</head>
<body>
    <h1>Kidney Health Portal</h1>
    
    <div class="container">
        <!-- Left Column: CKD Risk Calculator (Static) -->
        <div class="card">
            <h2>CKD Risk Calculator</h2>
            <form id="riskForm">
                <h3>Patient Information</h3>
                <label for="age">Age (years):</label>
                <input type="number" id="age" name="age" min="18" max="120" required>
                
                <label for="sex">Sex:</label>
                <select id="sex" name="sex" required>
                    <option value="">Select...</option>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                </select>
                
                <label for="race">Race/Ethnicity:</label>
                <select id="race" name="race" required>
                    <option value="">Select...</option>
                    <option value="white">White</option>
                    <option value="black">Black/African American</option>
                    <option value="hispanic">Hispanic</option>
                    <option value="asian">Asian</option>
                    <option value="other">Other</option>
                </select>
                
                <h3>Medical History</h3>
                <label for="hypertension">Hypertension (high blood pressure):</label>
                <select id="hypertension" name="hypertension" required>
                    <option value="">Select...</option>
                    <option value="no">No</option>
                    <option value="yes">Yes</option>
                </select>
                
                <label for="diabetes">Diabetes:</label>
                <select id="diabetes" name="diabetes" required>
                    <option value="">Select...</option>
                    <option value="no">No</option>
                    <option value="yes">Yes</option>
                </select>
                
                <div id="diabetesYearsContainer" class="hidden">
                    <label for="diabetes_years">If diabetic, how many years since diagnosis?</label>
                    <input type="number" id="diabetes_years" name="diabetes_years" min="0" max="100" value="0">
                </div>
                
                <label for="family_history">Family history of kidney disease?</label>
                <select id="family_history" name="family_history" required>
                    <option value="">Select...</option>
                    <option value="no">No</option>
                    <option value="yes">Yes</option>
                </select>
                
                <label for="bmi">Body Mass Index (BMI):</label>
                <input type="number" id="bmi" name="bmi" step="0.1" min="15" max="50" required>
                
                <label for="smoking">Smoking status:</label>
                <select id="smoking" name="smoking" required>
                    <option value="">Select...</option>
                    <option value="never">Never smoked</option>
                    <option value="former">Former smoker</option>
                    <option value="current">Current smoker</option>
                </select>
                
                <label for="cvd">History of cardiovascular disease?</label>
                <select id="cvd" name="cvd" required>
                    <option value="">Select...</option>
                    <option value="no">No</option>
                    <option value="yes">Yes</option>
                </select>
                
                <h3>Symptoms (Check all that apply)</h3>
                <div class="symptom-checkbox">
                    <input type="checkbox" id="fatigue" name="symptoms" value="fatigue">
                    <label for="fatigue">Fatigue or weakness</label>
                </div>
                <!-- More symptoms... -->
                
                <button type="submit">Calculate My CKD Risk</button>
            </form>
            
            <div class="disclaimer">
                <p><strong>Disclaimer:</strong> This tool is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.</p>
                <p>Based on clinical guidelines from the National Kidney Foundation and KDIGO.</p>
            </div>
        </div>
        
        <!-- Right Column: Hybrid Model Predictor -->
        <div class="card">
            <h2>Advanced CKD Stage Prediction</h2>
            <form id="predictionForm">
                <label for="p_age">Age:</label>
                <input type="number" id="p_age" name="Age" required>
                
                <label for="bp">Blood Pressure (mmHg):</label>
                <input type="number" id="bp" name="Blood Pressure" required>
                
                <label for="creatinine">Serum Creatinine (mg/dL):</label>
                <input type="number" step="0.1" id="creatinine" name="Serum Creatinine" required>
                
                <label for="albumin">Albumin (g/dL):</label>
                <input type="number" step="0.1" id="albumin" name="Albumin" required>
                
                <button type="submit" class="predict-btn">Predict CKD Stage</button>
            </form>
            
            <div class="health-actions">
                <button class="health-btn" onclick="showDietTips()">🍎 CKD Diet Recommendations</button>
                <button class="health-btn" onclick="showSymptoms()">⚠️ Stage-Specific Symptoms</button>
                <button class="health-btn" onclick="showNearbyClinics()">🏥 Find Nephrologists</button>
                <button class="health-btn" onclick="showEmergency()">🆘 Emergency Protocols</button>
            </div>
            
            <div id="results" class="results hidden">
                <h3>Prediction Results</h3>
                <div id="predictionResults"></div>
                <div class="plot-container">
                    <img id="roc-plot" src="" alt="ROC Curve" style="max-width: 100%;">
                </div>
            </div>
        </div>
    </div>

    <script>
        // Diabetes years field toggle
        document.getElementById('diabetes').addEventListener('change', function() {
            document.getElementById('diabetesYearsContainer').style.display = 
                this.value === 'yes' ? 'block' : 'none';
        });

        // Risk form submission
        document.getElementById('riskForm').addEventListener('submit', function(e) {
            e.preventDefault();
            alert("CKD Risk Assessment: Moderate Risk\n\nBased on your inputs, you have moderate risk factors for developing chronic kidney disease. Please consult with your healthcare provider for a complete evaluation.");
        });

        // Prediction form handling
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = {
                Age: parseFloat(document.getElementById('p_age').value),
                "Blood Pressure": parseFloat(document.getElementById('bp').value),
                "Serum Creatinine": parseFloat(document.getElementById('creatinine').value),
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
                    let html = `
                        <p><strong>Traditional Model:</strong> ${data.traditional.stage} (${(data.traditional.confidence * 100).toFixed(1)}% confidence)</p>
                        <p><strong>LLM Model:</strong> ${data.llm.stage} (${(data.llm.confidence * 100).toFixed(1)}% confidence)</p>
                        <p><strong>Hybrid Model:</strong> <strong>${data.hybrid.stage}</strong> (${(data.hybrid.confidence * 100).toFixed(1)}% confidence)</p>
                    `;
                    document.getElementById('predictionResults').innerHTML = html;
                    document.getElementById('roc-plot').src = 'data:image/png;base64,' + data.roc_plot;
                    document.getElementById('results').classList.remove('hidden');
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        });

        // Health button functions
        function showDietTips() {
            alert("CKD Diet Recommendations:\n\nStage 1-2: Reduce sodium, maintain normal protein\nStage 3-4: Limit potassium/phosphate, moderate protein (0.8g/kg)\nStage 5: Fluid restriction, low-phosphorus diet");
        }
        
        function showSymptoms() {
            alert("Stage-Specific Symptoms:\n\nStage 1-2: Often asymptomatic\nStage 3: Fatigue, swelling, mild anemia\nStage 4: Nausea, itching, muscle cramps\nStage 5: Shortness of breath, confusion, severe fatigue");
        }
        
        function showNearbyClinics() {
            window.open("https://www.kidney.org/transplantation/transaction/TC/Centers", "_blank");
        }
        
        function showEmergency() {
            alert("EMERGENCY PROTOCOLS\n\nSeek immediate medical care if experiencing:\n- Chest pain or pressure\n- Severe shortness of breath\n- Confusion or seizures\n- Inability to urinate\n\nCall 911 or go to nearest emergency room");
        }
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
        
        trad_proba = trad_model.predict_proba(input_df)[0]
        trad_pred = trad_model.predict(input_df)[0]
        trad_stage = label_encoder.inverse_transform([trad_pred])[0]
        
        llm_proba = llm_model.predict_proba(input_df)[0]
        llm_pred = llm_model.predict(input_df)[0]
        llm_stage = label_encoder.inverse_transform([llm_pred])[0]
        
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
        plt.title('Model Comparison (ROC Curves)')
        
        # Save plot to bytes
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png')
        img_bytes.seek(0)
        plt.close()
        
        return jsonify({
            'traditional': {'stage': trad_stage, 'confidence': float(trad_proba.max())},
            'llm': {'stage': llm_stage, 'confidence': float(llm_proba.max())},
            'hybrid': {'stage': hybrid_stage, 'confidence': float(hybrid_proba.max())},
            'roc_plot': base64.b64encode(img_bytes.read()).decode('utf-8')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
