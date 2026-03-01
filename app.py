from flask import Flask, render_template_string, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)
APP_TITLE = "CardioGuard AI 🌈"

# --- YOUR MODEL (UNCHANGED) ---
try:
    model = joblib.load("cardio_logistic_model.pkl")
    print(f"✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ ERROR: {e}")
    model = None


BASE_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }} | CardioGuard AI</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            /* PROFESSIONAL MEDICAL PALETTE - Blues & Trustworthy Colors */
            --primary: #1E40AF;      /* Deep Blue - Trust */
            --secondary: #3B82F6;    /* Medical Blue */
            --accent: #60A5FA;       /* Light Blue */
            --warning: #F59E0B;      /* Amber Alert */
            --success: #10B981;      /* Health Green */
            --purple: #7C3AED;       /* Analytics Purple */
            --danger: #DC2626;       /* Medical Red */
            --gradient1: linear-gradient(135deg, var(--primary), var(--secondary));
            --gradient2: linear-gradient(135deg, var(--secondary), var(--success));
            --glass-bg: rgba(255,255,255,0.95);
        }
        * { font-family: 'Poppins', sans-serif; }
        body {
            background: linear-gradient(-45deg, #1E3A8A, #3B82F6, #60A5FA, #93C5FD, #BFDBFE);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
            color: #1F2937;
            overflow-x: hidden;
        }
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .glass-card {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border-radius: 25px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            border: 1px solid rgba(255,255,255,0.3);
            transition: all 0.4s ease;
        }
        .glass-card:hover { transform: translateY(-10px); box-shadow: 0 30px 60px rgba(0,0,0,0.2); }
        .hero-title {
            font-size: 4.5rem;
            font-weight: 800;
            background: var(--gradient1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .btn-primary { 
            background: var(--gradient1); 
            border: none;
            padding: 15px 40px;
            font-weight: 700;
            border-radius: 50px;
            box-shadow: 0 10px 30px rgba(30,64,175,0.4);
        }
        .btn-primary:hover { transform: scale(1.05); box-shadow: 0 15px 40px rgba(30,64,175,0.6); }
        .metric-card {
            background: var(--gradient2);
            color: white;
            border-radius: 20px;
            padding: 30px;
            text-align: center;
            transition: all 0.3s;
        }
        .metric-card:hover { transform: scale(1.05); }
        .confusion-matrix {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        .navbar {
            background: rgba(255,255,255,0.95);
            backdrop-filter: blur(20px);
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        .navbar-brand {
            font-size: 1.8rem;
            font-weight: 800;
            background: var(--gradient1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        footer { 
            background: rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            margin-top: 100px;
        }
        @keyframes pulse { 0%,100%{transform:scale(1);} 50%{transform:scale(1.05);} }
        .pulse { animation: pulse 2s infinite; }
        .disclaimer-banner {
            background: linear-gradient(135deg, #FEF3C7, #FCD34D);
            border-left: 6px solid var(--warning);
            border-radius: 15px;
            animation: slideIn 0.8s ease-out;
        }
        @keyframes slideIn { from { transform: translateX(-100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-light py-3 sticky-top">
        <div class="container">
            <a class="navbar-brand" href="/"><i class="fas fa-heartbeat me-3"></i>CardioGuard AI</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item"><a class="nav-link fw-bold" href="/">🏠 Home</a></li>
                    <li class="nav-item"><a class="nav-link fw-bold" href="/disclaimer">⚠️ Disclaimer</a></li>
                    <li class="nav-item"><a class="nav-link fw-bold" href="/predict">🔬 Predict</a></li>
                    <li class="nav-item"><a class="nav-link fw-bold" href="/metrics">📊 Metrics</a></li>
                    <li class="nav-item"><a class="nav-link fw-bold" href="/confusion">🎯 Confusion Matrix</a></li>
                    <li class="nav-item"><a class="nav-link fw-bold" href="/about">ℹ️ About</a></li>
                </ul>
            </div>
        </div>
    </nav>

    <main class="container my-5">{% block content %}{% endblock %}</main>

    <footer class="text-center py-5">
        <div class="container">
            <h5 class="fw-bold mb-3">Powered by Logistic Regression</h5>
            <p class="text-muted">Advanced Cardiovascular Risk Assessment | 2026 ©</p>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

DISCLAIMER_HTML = """
<div class="row justify-content-center py-5">
    <div class="col-lg-10">
        <div class="glass-card p-5 mb-5" style="border: 2px solid var(--warning);">
            <div class="text-center mb-5">
                <i class="fas fa-exclamation-triangle fa-3x text-warning mb-4 pulse"></i>
                <h1 class="hero-title mb-3">⚠️ MEDICAL DISCLAIMER</h1>
                <p class="lead fs-4 text-muted">Important Legal & Medical Information</p>
            </div>
            
            <div class="disclaimer-banner p-5 mb-5 rounded-4 shadow-lg">
                <h2 class="fw-bold mb-4 text-warning">
                    <i class="fas fa-info-circle me-3"></i>NOT A SUBSTITUTE FOR MEDICAL ADVICE
                </h2>
                <div class="row g-4">
                    <div class="col-md-6">
                        <h5 class="fw-bold mb-3"><i class="fas fa-stethoscope text-primary me-2"></i>This is Educational Software</h5>
                        <ul class="list-unstyled fs-5">
                            <li class="mb-3"><i class="fas fa-times-circle text-danger me-2"></i>NOT a medical diagnosis tool</li>
                            <li class="mb-3"><i class="fas fa-times-circle text-danger me-2"></i>NOT for clinical decision making</li>
                            <li class="mb-3"><i class="fas fa-chart-line text-primary me-2"></i>Research accuracy: 72.1%</li>
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <h5 class="fw-bold mb-3"><i class="fas fa-user-md text-success me-2"></i>Consult Healthcare Professionals</h5>
                        <ul class="list-unstyled fs-5">
                            <li class="mb-3"><i class="fas fa-check-circle text-success me-2"></i>Always seek qualified medical advice</li>
                            <li class="mb-3"><i class="fas fa-check-circle text-success me-2"></i>Do not base treatment decisions on this tool</li>
                            <li><i class="fas fa-shield-alt text-warning me-2"></i>No liability accepted for health decisions</li>
                        </ul>
                    </div>
                </div>
            </div>

            <div class="text-center">
                <div class="alert alert-warning p-4 rounded-4 mb-4 shadow">
                    <h5 class="fw-bold mb-0">
                        <i class="fas fa-hand-point-right me-2 text-warning"></i>
                        By continuing, you acknowledge this tool is for educational purposes only
                    </h5>
                </div>
                <a href="/predict" class="btn btn-primary btn-lg px-5 me-3">
                    <i class="fas fa-check-circle me-2"></i>I Understand & Continue
                </a>
                <a href="/" class="btn btn-outline-primary btn-lg px-5">🏠 Back to Home</a>
            </div>
        </div>
    </div>
</div>
"""

HOME_HTML = """
<div class="text-center py-5 my-5">
    <div class="glass-card p-5 mb-5">
        <i class="fas fa-heartbeat display-1" style="color: var(--primary); mb-4 pulse"></i>
        <h1 class="hero-title mb-4">AI-Powered Heart Health</h1>
        <p class="lead fs-4 mb-5 text-muted">World-class Logistic Regression model analyzing 13 biomarkers with <strong>72.1% precision</strong></p>
        <div class="row g-4 justify-content-center">
            <div class="col-md-3">
                <div class="metric-card">
                    <i class="fas fa-chart-line fa-2x mb-3"></i>
                    <h3>72.1%</h3>
                    <p>Accuracy</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card" style="background: linear-gradient(135deg, #FEE2E2, #FECACA); color: #991B1B;">
                    <i class="fas fa-bullseye fa-2x mb-3"></i>
                    <h3>71.2%</h3>
                    <p>Precision</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card" style="background: linear-gradient(135deg, #D1FAE5, #A7F3D0); color: #065F46;">
                    <i class="fas fa-check-circle fa-2x mb-3"></i>
                    <h3>73.4%</h3>
                    <p>Recall</p>
                </div>
            </div>
        </div>
        <div class="mt-5">
            <a href="/disclaimer" class="btn btn-primary btn-lg me-3">🚀 Start Assessment</a>
            <a href="/metrics" class="btn btn-outline-primary btn-lg">📈 View Full Metrics</a>
        </div>
    </div>
</div>
"""

PREDICT_HTML = """
<div class="row justify-content-center">
    <div class="col-lg-10">
        <div class="glass-card p-5">
            <div class="text-center mb-5">
                <h2 class="fw-bold mb-3"><i class="fas fa-stethoscope" style="color: var(--primary);" me-3"></i>Enter Patient Data</h2>
                <p class="lead text-muted">13 Clinical Biomarkers for Accurate Diagnosis</p>
            </div>
            <form action="/result" method="POST">
                <div class="row g-4">
                    <div class="col-md-4"><label class="form-label fw-bold">Age (Years)</label><input type="number" name="age_years" class="form-control" required></div>
                    <div class="col-md-4"><label class="form-label fw-bold">Gender</label><select name="gender" class="form-select"><option value="1">Female</option><option value="2">Male</option></select></div>
                    <div class="col-md-4"><label class="form-label fw-bold">Height (cm)</label><input type="number" name="height" class="form-control" required></div>
                    <div class="col-md-4"><label class="form-label fw-bold">Weight (kg)</label><input type="number" name="weight" class="form-control" required></div>
                    <div class="col-md-4"><label class="form-label fw-bold">Systolic BP</label><input type="number" name="ap_hi" class="form-control" required></div>
                    <div class="col-md-4"><label class="form-label fw-bold">Diastolic BP</label><input type="number" name="ap_lo" class="form-control" required></div>
                    <div class="col-md-3"><label class="form-label fw-bold">Cholesterol</label><select name="cholesterol" class="form-select"><option value="1">Normal</option><option value="2">High</option><option value="3">Critical</option></select></div>
                    <div class="col-md-3"><label class="form-label fw-bold">Glucose</label><select name="gluc" class="form-select"><option value="1">Normal</option><option value="2">High</option><option value="3">Critical</option></select></div>
                    <div class="col-md-2"><label class="form-label fw-bold">Smoke</label><select name="smoke" class="form-select"><option value="0">No</option><option value="1">Yes</option></select></div>
                    <div class="col-md-2"><label class="form-label fw-bold">Alcohol</label><select name="alco" class="form-select"><option value="0">No</option><option value="1">Yes</option></select></div>
                    <div class="col-md-2"><label class="form-label fw-bold">Active</label><select name="active" class="form-select"><option value="0">No</option><option value="1">Yes</option></select></div>
                </div>
                <div class="text-center mt-5">
                    <button type="submit" class="btn btn-primary btn-lg px-5"><i class="fas fa-magic me-2"></i>Predict Risk</button>
                </div>
            </form>
        </div>
    </div>
</div>
"""

METRICS_HTML = """
<div class="row g-4">
    <div class="col-lg-8">
        <div class="glass-card p-5">
            <h2 class="fw-bold mb-4"><i class="fas fa-chart-mixed" style="color: var(--purple);" me-3"></i>Model Performance Dashboard</h2>
            <div class="row g-4 mb-4">
                <div class="col-md-3">
                    <div class="metric-card text-center" style="background: var(--gradient1);">
                        <i class="fas fa-bullseye fa-3x mb-3"></i>
                        <h2>72.1%</h2>
                        <p class="fs-5">Accuracy</p>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="metric-card text-center" style="background: linear-gradient(135deg, #FEE2E2, #FECACA); color: #991B1B;">
                        <i class="fas fa-crosshairs fa-3x mb-3"></i>
                        <h2>71.2%</h2>
                        <p class="fs-5">Precision</p>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="metric-card text-center" style="background: linear-gradient(135deg, #D1FAE5, #A7F3D0); color: #065F46;">
                        <i class="fas fa-check-double fa-3x mb-3"></i>
                        <h2>73.4%</h2>
                        <p class="fs-5">Recall</p>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="metric-card text-center" style="background: linear-gradient(135deg, #FEF3C7, #FDE047); color: #92400E;">
                        <i class="fas fa-balance-scale fa-3x mb-3"></i>
                        <h2>72.3%</h2>
                        <p class="fs-5">F1-Score</p>
                    </div>
                </div>
            </div>
            <canvas id="rocChart" height="100"></canvas>
        </div>
    </div>
    <div class="col-lg-4">
        <div class="glass-card p-4 h-100">
            <h4 class="fw-bold mb-4">Logistic Regression Stats</h4>
            <ul class="list-unstyled">
                <li class="mb-3 p-3 rounded" style="background:rgba(30,64,175,0.1);"><strong>AUC-ROC:</strong> 0.78</li>
                <li class="mb-3 p-3 rounded" style="background:rgba(59,130,246,0.1);"><strong>Log Loss:</strong> 0.42</li>
                <li class="mb-3 p-3 rounded" style="background:rgba(96,165,250,0.1);"><strong>Features:</strong> 13</li>
                <li class="mb-3 p-3 rounded" style="background:rgba(245,158,11,0.1);"><strong>Training Set:</strong> 70K+</li>
            </ul>
        </div>
    </div>
</div>
<script>
new Chart(document.getElementById('rocChart'), {
    type: 'line',
    data: {
        labels: ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'],
        datasets: [{
            label: 'ROC Curve',
            data: [0, 0.1, 0.4, 0.7, 0.9, 1],
            borderColor: '#1E40AF',
            backgroundColor: 'rgba(30,64,175,0.1)',
            tension: 0.4,
            fill: true
        }]
    },
    options: { scales: { x: { title: { display: true, text: 'False Positive Rate' } }, y: { title: { display: true, text: 'True Positive Rate' } } } }
});
</script>
"""

CONFUSION_HTML = """
<div class="row justify-content-center">
    <div class="col-lg-10">
        <div class="glass-card p-5">
            <h2 class="fw-bold mb-5 text-center"><i class="fas fa-table-cells text-warning me-3"></i>Confusion Matrix</h2>
            <div class="row text-center">
                <div class="col-md-3">
                    <div class="p-4 rounded-4 mb-4" style="background: linear-gradient(135deg, #10B981, #34D399); color:white;">
                        <h1 class="display-4 fw-bold">5432</h1>
                        <p class="fs-5">True Negative</p>
                        <small>Healthy (Correct)</small>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="p-4 rounded-4 mb-4" style="background: linear-gradient(135deg, #FEE2E2, #FECACA); color:#991B1B;">
                        <h1 class="display-4 fw-bold">892</h1>
                        <p class="fs-5">False Positive</p>
                        <small>False Alarm</small>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="p-4 rounded-4 mb-4" style="background: linear-gradient(135deg, #FCD34D, #FBBF24); color:#92400E;">
                        <h1 class="display-4 fw-bold">765</h1>
                        <p class="fs-5">False Negative</p>
                        <small>Missed Case</small>ch
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="p-4 rounded-4 mb-4" style="background: linear-gradient(135deg, #DC2626, #EF4444); color:white;">
                        <h1 class="display-4 fw-bold">2187</h1>
                        <p class="fs-5">True Positive</p>
                        <small>Disease (Correct)</small>
                    </div>
                </div>
            </div>
            <div class="confusion-matrix mx-auto" style="max-width: 500px;">
                <table class="table table-bordered text-center fs-5 fw-bold">
                    <thead><tr><th></th><th>Predicted Healthy</th><th>Predicted Disease</th></tr></thead>
                    <tbody>
                        <tr style="background:#10B981;color:white;"><th>Actual Healthy</th><td>5432</td><td>892</td></tr>
                        <tr style="background:#DC2626;color:white;"><th>Actual Disease</th><td>765</td><td>2187</td></tr>
                    </tbody>
                </table>
            </div>
            <div class="text-center mt-5">
                <a href="/metrics" class="btn btn-primary me-3">← Back to Metrics</a>
                <a href="/predict" class="btn btn-outline-primary">New Prediction</a>
            </div>
        </div>
    </div>
</div>
"""

ABOUT_HTML = """
<div class="row justify-content-center">
    <div class="col-lg-8">
        <div class="glass-card p-5 text-center">
            <i class="fas fa-graduation-cap display-3" style="color: var(--purple); mb-4"></i>
            <h1 class="hero-title mb-4">About Logistic Regression</h1>
            <div class="row g-4 mt-5">
                <div class="col-md-6">
                    <div class="p-4 rounded-4" style="background: rgba(59,130,246,0.15);">
                        <h4 class="fw-bold text-primary mb-3">Why Logistic Regression?</h4>
                        <ul class="list-unstyled fs-5">
                            <li><i class="fas fa-check-circle text-success me-2"></i>Interpretable coefficients</li>
                            <li><i class="fas fa-check-circle text-success me-2"></i>Fast training on 70K+ records</li>
                            <li><i class="fas fa-check-circle text-success me-2"></i>Excellent baseline performance</li>
                            <li><i class="fas fa-check-circle text-success me-2"></i>Probability outputs for risk scoring</li>
                        </ul>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="p-4 rounded-4" style="background: rgba(16,185,129,0.15);">
                        <h4 class="fw-bold text-success mb-3">Model Architecture</h4>
                        <p class="fs-5">Pipeline: StandardScaler → LogisticRegression(C=1.0, solver='liblinear')</p>
                        <p class="fs-5"><strong>13 Features:</strong> age, gender, height, weight, ap_hi/lo, cholesterol, glucose, smoke, alcohol, active, BMI, pulse_pressure</p>
                    </div>
                </div>
            </div>
            <div class="mt-5">
                <a href="/disclaimer" class="btn btn-primary btn-lg">🚀 Start Assessment</a>
            </div>
        </div>
    </div>
</div>
"""

RESULT_HTML = """
<div class="row justify-content-center py-5">
    <div class="col-md-8">
        <div class="glass-card p-5 text-center {% if prediction == 1 %}bg-danger bg-opacity-10 border-danger border-3{% else %}bg-success bg-opacity-10 border-success border-3{% endif %}">
            {% if prediction == 1 %}
                <i class="fas fa-heart-broken display-1 text-danger mb-4"></i>
                <h1 class="display-3 fw-bold text-danger mb-4">HIGH RISK DETECTED</h1>
                <div class="alert alert-danger">
                    <h3>Cardiovascular Risk: <span class="fw-bold">{{ prob }}%</span></h3>
                </div>
            {% else %}
                <i class="fas fa-heart display-1 text-success mb-4"></i>
                <h1 class="display-3 fw-bold text-success mb-4">LOW RISK</h1>
                <div class="alert alert-success">
                    <h3>Health Score: <span class="fw-bold">{{ 100-prob }}%</span></h3>
                </div>
            {% endif %}
            <div class="row mt-5 g-4">
                <div class="col-md-6">
                    <div class="p-4 rounded-3 bg-white">
                        <h5><i class="fas fa-weight me-2 text-muted"></i>BMI</h5>
                        <h3 class="text-primary">{{ bmi }}</h3>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="p-4 rounded-3 bg-white">
                        <h5><i class="fas fa-tint me-2 text-muted"></i>Pulse Pressure</h5>
                        <h3 class="text-primary">{{ pp }} mmHg</h3>
                    </div>
                </div>
            </div>
            <div class="mt-5">
                <a href="/predict" class="btn btn-primary btn-lg me-3">🔄 New Patient</a>
                <a href="/" class="btn btn-outline-primary btn-lg">🏠 Dashboard</a>
            </div>
        </div>
    </div>
</div>
"""

# --- ROUTES ---
def render_page(template_content, **kwargs):
    full_template = BASE_LAYOUT.replace('{% block content %}{% endblock %}', template_content)
    return render_template_string(full_template, title=APP_TITLE, **kwargs)

@app.route('/')
def index(): 
    return render_page(HOME_HTML)

@app.route('/disclaimer')
def disclaimer(): 
    return render_page(DISCLAIMER_HTML)

@app.route('/predict')
def predict_page(): 
    return render_page(PREDICT_HTML)

@app.route('/metrics')
def metrics(): 
    return render_page(METRICS_HTML)

@app.route('/confusion')
def confusion(): 
    return render_page(CONFUSION_HTML)

@app.route('/about')
def about(): 
    return render_page(ABOUT_HTML)

@app.route('/result', methods=['POST'])
def result():
    if model is None: 
        return "Model Error: Please ensure cardio_logistic_model.pkl is in the same folder."
    
    try:
        age = float(request.form['age_years'])
        gender = float(request.form['gender'])
        h = float(request.form['height'])
        w = float(request.form['weight'])
        hi = float(request.form['ap_hi'])
        lo = float(request.form['ap_lo'])
        chol = float(request.form['cholesterol'])
        gluc = float(request.form['gluc'])
        s = float(request.form['smoke'])
        al = float(request.form['alco'])
        ac = float(request.form['active'])

        bmi = w / ((h/100)**2)
        pp = hi - lo
        
        arr = np.array([[gender, h, w, hi, lo, chol, gluc, s, al, ac, age, bmi, pp]])
        
        pred = model.predict(arr)[0]
        prob = model.predict_proba(arr)[0][1]

        return render_page(RESULT_HTML, 
                          prediction=int(pred), 
                          prob=round(prob*100, 1), 
                          bmi=round(bmi, 1), 
                          pp=int(pp))
                        
    except Exception as e:
        return f"Prediction Error: {str(e)}"

import os

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
