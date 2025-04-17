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
from datetime import datetime
import random

app = Flask(__name__)

# Initialize models and databases
trad_model = RandomForestClassifier()
llm_model = LogisticRegression(max_iter=10000)
label_encoder = LabelEncoder()

# Medication database
MED_DB = {
    "ibuprofen": {"ckd_risk": "High", "alternatives": ["acetaminophen"]},
    "metformin": {"ckd_risk": "Moderate", "warning": "Adjust dose if eGFR <45"},
    "lisinopril": {"ckd_risk": "Low", "benefit": "Kidney protective"}
}

# Initialize models
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
    <title>Advanced Kidney Health Portal</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            max-width: 1200px;
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
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .health-btn {
            background-color: #2980b9;
            color: white;
            border: none;
            padding: 15px;
            border-radius: 8px;
            cursor: pointer;
            margin: 5px;
            width: 100%;
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
        #radarChart, #progressionChart {
            max-width: 100%;
            margin: 0 auto;
        }
        .badge {
            display: inline-block;
            background: #27ae60;
            color: white;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 12px;
            margin-right: 5px;
        }
    </style>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="header">
        <h1>Advanced Kidney Health Portal</h1>
        <p>Comprehensive CKD risk assessment and management</p>
    </div>

    <div class="dashboard">
        <!-- Left Column -->
        <div>
            <!-- 1. Interactive Risk Visualization -->
            <div class="card">
                <h2>Your Kidney Health Radar</h2>
                <img id="radarChart" src="" alt="Risk Factors Radar Chart">
            </div>

            <!-- 2. Personalized Action Plan -->
            <div class="card">
                <h2>Your Action Plan</h2>
                <ul id="actionPlan"></ul>
            </div>

            <!-- 3. AR Kidney Scan (Placeholder) -->
            <div class="card">
                <h2>Lab Report Scanner <span class="badge">NEW</span></h2>
                <button class="health-btn" onclick="startARScan()">
                    <i class="fas fa-camera"></i> Scan Lab Report
                </button>
                <div id="arResults"></div>
            </div>
        </div>

        <!-- Right Column -->
        <div>
            <!-- 4. Voice Symptom Checker -->
            <div class="card">
                <h2>Voice Symptom Checker</h2>
                <button class="health-btn" id="voiceBtn" onclick="startVoiceRecording()">
                    <i class="fas fa-microphone"></i> Describe Your Symptoms
                </button>
                <div id="symptomResults"></div>
            </div>

            <!-- 5. Progression Timeline -->
            <div class="card">
                <h2>Projected Kidney Function</h2>
                <canvas id="progressionChart"></canvas>
            </div>

            <!-- 6. Medication Checker -->
            <div class="card">
                <h2>Medication Safety Check</h2>
                <input type="text" id="medInput" placeholder="Enter medication name">
                <button class="health-btn" onclick="checkMedication()">
                    Check Safety
                </button>
                <div id="medResults"></div>
            </div>
        </div>
    </div>

    <!-- 7. 3D Kidney Explorer Modal -->
    <div id="kidney3dModal" style="display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.8); z-index:1000; padding:50px;">
        <button class="close-btn" onclick="hideModal('kidney3dModal')">×</button>
        <iframe src="https://www.visiblebody.com/learn/urinary/kidneys" style="width:100%; height:90%; border:none;"></iframe>
    </div>

    <!-- 8. Nutrition AI Modal -->
    <div id="nutritionModal" style="display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.8); z-index:1000; padding:50px; color:white;">
        <button class="close-btn" onclick="hideModal('nutritionModal')">×</button>
        <h2>Personalized Nutrition Plan</h2>
        <div id="nutritionPlan"></div>
    </div>

    <!-- 9. Telehealth Modal -->
    <div id="telehealthModal" style="display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.8); z-index:1000; padding:50px; color:white;">
        <button class="close-btn" onclick="hideModal('telehealthModal')">×</button>
        <h2>Book Nephrologist Consultation</h2>
        <div id="doctorSlots"></div>
    </div>

    <!-- 10. Gamification Dashboard -->
    <div style="position:fixed; bottom:20px; right:20px; background:white; padding:15px; border-radius:8px; box-shadow:0 2px 10px rgba(0,0,0,0.2);">
        <h3>Your Kidney Health Score: <span id="healthScore">0</span></h3>
        <div id="badges"></div>
    </div>

    <script>
        // Global health tracker
        const healthTracker = {
            score: 0,
            badges: [],
            addPoints: function(points, area) {
                this.score += points;
                document.getElementById('healthScore').textContent = this.score;
                
                if (area === 'diet' && !this.badges.includes('Nutrition Pro') && this.score > 20) {
                    this.badges.push('Nutrition Pro');
                    updateBadges();
                }
            },
            updateBadges: function() {
                const badgeContainer = document.getElementById('badges');
                badgeContainer.innerHTML = this.badges.map(b => 
                    `<span class="badge">${b}</span>`).join('');
            }
        };

        // 1. Risk Visualization
        function updateRadarChart() {
            fetch('/get_risk_chart')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('radarChart').src = 
                        'data:image/png;base64,' + data.chart;
                });
        }

        // 2. Action Plan Generator
        function generateActionPlan(riskLevel) {
            const plans = {
                low: ["Annual checkup", "Maintain healthy diet"],
                moderate: ["3-month BP checks", "Reduce sodium intake"],
                high: ["Nephrologist consult", "Monthly lab tests"]
            };
            document.getElementById('actionPlan').innerHTML = 
                plans[riskLevel].map(item => `<li>${item}</li>`).join('');
            healthTracker.addPoints(10, 'assessment');
        }

        // 3. AR Scan (simulated)
        function startARScan() {
            document.getElementById('arResults').innerHTML = `
                <p><i class="fas fa-mobile-alt"></i> Point your camera at lab results</p>
                <p>Simulated result: Creatinine 1.4 mg/dL (mild elevation)</p>
            `;
            healthTracker.addPoints(5, 'engagement');
        }

        // 4. Voice Symptom Checker
        function startVoiceRecording() {
            document.getElementById('voiceBtn').innerHTML = `
                <i class="fas fa-stop"></i> Stop Recording`;
            document.getElementById('symptomResults').innerHTML = `
                <p>Listening... Say your symptoms</p>`;
            
            // Simulate analysis after 3 seconds
            setTimeout(() => {
                document.getElementById('voiceBtn').innerHTML = `
                    <i class="fas fa-microphone"></i> Describe Your Symptoms`;
                document.getElementById('symptomResults').innerHTML = `
                    <p>Detected symptoms: fatigue, swelling</p>
                    <p>Possible condition: Stage 2 CKD</p>`;
                healthTracker.addPoints(8, 'symptoms');
            }, 3000);
        }

        // 5. Progression Timeline
        function renderProgressionChart(stage) {
            const ctx = document.getElementById('progressionChart');
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: ['Now', '1 Year', '5 Years', '10 Years'],
                    datasets: [{
                        label: 'Kidney Function',
                        data: stage === 1 ? [90, 85, 70, 60] : 
                              stage === 2 ? [70, 65, 50, 40] : [40, 35, 25, 15],
                        borderColor: '#e74c3c',
                        tension: 0.1
                    }]
                }
            });
        }

        // 6. Medication Checker
        function checkMedication() {
            const med = document.getElementById('medInput').value.toLowerCase();
            fetch('/check_meds', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({medications: [med]})
            })
            .then(response => response.json())
            .then(data => {
                const result = data[med];
                let html = `<p>No data found for ${med}</p>`;
                if (result && typeof result === 'object') {
                    html = `<p><strong>${med}</strong>: ${result.ckd_risk} risk</p>`;
                    if (result.warning) html += `<p>Warning: ${result.warning}</p>`;
                    if (result.alternatives) html += `<p>Alternatives: ${result.alternatives.join(', ')}</p>`;
                }
                document.getElementById('medResults').innerHTML = html;
                healthTracker.addPoints(3, 'medication');
            });
        }

        // 7. 3D Kidney Explorer
        function showKidney3D() {
            document.getElementById('kidney3dModal').style.display = 'block';
            healthTracker.addPoints(5, 'education');
        }

        // 8. Nutrition AI
        function getNutritionPlan() {
            fetch('/get_diet_plan', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({weight: 70, phosphorus: 4.2})
            })
            .then(response => response.json())
            .then(data => {
                let html = '<h3>Today\'s Recommendations</h3>';
                for (const [meal, suggestion] of Object.entries(data)) {
                    html += `<p><strong>${meal}:</strong> ${suggestion}</p>`;
                }
                document.getElementById('nutritionPlan').innerHTML = html;
                document.getElementById('nutritionModal').style.display = 'block';
                healthTracker.addPoints(7, 'diet');
            });
        }

        // 9. Telehealth Integration
        function bookTelehealth() {
            const availableSlots = {
                "Dr. Smith": ["Mon 10AM", "Wed 2PM"],
                "Dr. Lee": ["Tue 3PM", "Fri 11AM"]
            };
            let options = '';
            for (const [doctor, slots] of Object.entries(availableSlots)) {
                options += `<optgroup label="${doctor}">` +
                           slots.map(s => `<option>${s}</option>`).join('') +
                           '</optgroup>';
            }
            document.getElementById('doctorSlots').innerHTML = `
                <select>${options}</select>
                <button class="health-btn" onclick="confirmBooking()">Confirm Booking</button>`;
            document.getElementById('telehealthModal').style.display = 'block';
        }

        // Helper functions
        function hideModal(id) {
            document.getElementById(id).style.display = 'none';
        }

        function confirmBooking() {
            alert('Appointment booked successfully!');
            hideModal('telehealthModal');
            healthTracker.addPoints(15, 'consultation');
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            updateRadarChart();
            generateActionPlan('moderate');
            renderProgressionChart(2);
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

@app.route('/get_risk_chart', methods=['POST'])
def get_risk_chart():
    # Generate radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    categories = ['Age', 'BP', 'Diabetes', 'Kidney\nFunction', 'Family\nHistory']
    values = [7, 5, 3, 6, 2]
    
    ax.plot(categories, values, color='#2980b9', linewidth=2)
    ax.fill(categories, values, color='#2980b9', alpha=0.25)
    
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    plt.close()
    return jsonify({'chart': base64.b64encode(img_bytes.getvalue()).decode()})

@app.route('/check_meds', methods=['POST'])
def check_meds():
    user_meds = request.json.get('medications', [])
    results = {med: MED_DB.get(med.lower(), "No data") for med in user_meds}
    return jsonify(results)

@app.route('/get_diet_plan', methods=['POST'])
def get_diet_plan():
    data = request.json
    plan = {
        "breakfast": "Egg whites with spinach" if data.get('phosphorus', 0) > 4.5 else "Oatmeal",
        "lunch": "Grilled chicken with steamed vegetables",
        "dinner": "Salmon with quinoa",
        "fluids": f"{data.get('weight', 70)*30} ml/day" 
    }
    return jsonify(plan)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
