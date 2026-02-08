import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_file, url_for
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, 
    PageBreak, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from datetime import datetime
import io
import shap
from lime.lime_tabular import LimeTabularExplainer
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'
app.secret_key = 'heart_disease_prediction_secret_key' # Added for session/security if needed

# Ensure static/images exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ==================== MODEL DEFINITION ====================
class HeartDiseaseModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

# ==================== PREDICTION ENGINE ====================
class PredictionEngine:
    """Handles all model operations efficiently"""
    
    def __init__(self):
        self.feature_names = [
            'Age', 'Sex', 'Chest_pain_type', 'BP', 'Cholesterol',
            'FBS_over_120', 'EKG_results', 'Max_HR', 'Exercise_angina',
            'ST_depression', 'Slope_of_ST', 'Number_of_vessels_fluro', 'Thallium'
        ]
        
        # Scaler values (hardcoded as in the original GUI app)
        self.scaler_mean = np.array([60.0, 0.5, 2.0, 130.0, 250.0, 0.1, 1.0, 140.0, 0.3, 1.0, 1.0, 0.5, 3.5])
        self.scaler_std = np.array([10.0, 0.5, 1.0, 15.0, 50.0, 0.3, 0.7, 25.0, 0.4, 1.0, 0.8, 0.7, 2.0])
        
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = HeartDiseaseModel(len(self.feature_names))
            if os.path.exists("heart_disease_model.pt"):
                self.model.load_state_dict(torch.load("heart_disease_model.pt", map_location='cpu'))
                self.model.eval()
                print("Model loaded successfully.")
                return True
            else:
                print("Error: heart_disease_model.pt not found.")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def standardize(self, x):
        """Standardize input features"""
        return (x - self.scaler_mean) / self.scaler_std
    
    def predict(self, features):
        """Make prediction on input features"""
        features_scaled = self.standardize(features)
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            prob = self.model(features_tensor).item()
        
        prediction = "Presence" if prob >= 0.5 else "Absence"
        
        if prob >= 0.7:
            risk_level = "High Risk"
            risk_class = "danger"
        elif prob >= 0.5:
            risk_level = "Moderate Risk"
            risk_class = "warning"
        elif prob >= 0.3:
            risk_level = "Low-Moderate Risk"
            risk_class = "info"
        else:
            risk_level = "Low Risk"
            risk_class = "success"
        
        return {
            'probability': prob,
            'prediction': prediction,
            'risk_level': risk_level,
            'risk_class': risk_class
        }

# Initialize Engine
engine = PredictionEngine()

# ==================== EXPLANATION LOGIC ====================
def compute_explanations(features_scaled, features_raw, model, engine):
    results = {}
    
    try:
        # Generate unique ID for this session's images
        session_id = str(uuid.uuid4())
        
        # SHAP
        def shap_predict(x):
            x_t = torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                return model(x_t).numpy().flatten()
        
        # Use a small background dataset for speed
        background = np.zeros((5, len(engine.feature_names)))
        explainer = shap.KernelExplainer(shap_predict, background)
        shap_values = explainer.shap_values(features_scaled, nsamples=50, silent=True)
        
        if isinstance(shap_values, list):
            shap_vals = shap_values[0].flatten()
        else:
            shap_vals = shap_values.flatten()
            
        results['shap_values'] = shap_vals.tolist()
        
        # Generate SHAP Plot
        plt.close('all')
        # Professional style for plot with larger fonts for mobile readability
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        feature_contribution = list(zip(engine.feature_names, shap_vals, features_raw[0]))
        feature_contribution.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Take top 10 features for cleaner plot
        top_features = feature_contribution[:10]
        
        labels = [x[0] for x in top_features]
        values = [x[1] for x in top_features]
        # Professional colors: Red for risk increase, Blue for risk decrease
        colors_list = ['#d62728' if v > 0 else '#1f77b4' for v in values]
        
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, values, color=colors_list, alpha=0.8)
        ax.set_yticks(y_pos)
        # Increased font sizes for responsiveness
        ax.set_yticklabels(labels, fontsize=14)
        ax.set_xlabel('SHAP Value (Impact on Model Output)', fontsize=14, fontweight='bold')
        ax.set_title(f'Top {len(labels)} Features Impacting Prediction', fontsize=16, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        
        # Add values to bars
        for i, v in enumerate(values):
            ax.text(v, i, f' {v:+.3f}', va='center', fontsize=12, fontweight='bold')
            
        plt.tight_layout()
        
        shap_filename = f'shap_{session_id}.png'
        shap_path = os.path.join(app.config['UPLOAD_FOLDER'], shap_filename)
        plt.savefig(shap_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        results['shap_plot_path'] = shap_path
        results['shap_plot_url'] = url_for('static', filename=f'images/{shap_filename}')

        # LIME
        def lime_predict(x):
            x_t = torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                probs = model(x_t).numpy()
            return np.column_stack([1 - probs, probs])
        
        lime_background = np.tile(engine.scaler_mean, (50, 1))
        np.random.seed(42)
        noise = np.random.normal(0, 0.1, lime_background.shape) * engine.scaler_std
        lime_background = lime_background + noise
        lime_background = engine.standardize(lime_background)
        
        lime_explainer = LimeTabularExplainer(
            training_data=lime_background,
            feature_names=engine.feature_names,
            class_names=["Absence", "Presence"],
            mode="classification",
            random_state=42,
            discretize_continuous=True
        )
        
        lime_exp = lime_explainer.explain_instance(
            features_scaled[0],
            lime_predict,
            num_features=10, # Limit to top 10
            top_labels=2
        )
        
        # Generate LIME Plot
        plt.close('all')
        prob = model(torch.tensor(features_scaled, dtype=torch.float32)).item()
        lime_label = 1 if prob >= 0.5 else 0
        
        fig = lime_exp.as_pyplot_figure(label=lime_label)
        fig.set_size_inches(12, 8)
        plt.title("LIME Local Explanation (Top Factors)", fontsize=16, fontweight='bold')
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout()
        
        lime_filename = f'lime_{session_id}.png'
        lime_path = os.path.join(app.config['UPLOAD_FOLDER'], lime_filename)
        plt.savefig(lime_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        results['lime_plot_path'] = lime_path
        results['lime_plot_url'] = url_for('static', filename=f'images/{lime_filename}')
        
    except Exception as e:
        print(f"Explanation error: {e}")
        import traceback
        traceback.print_exc()
        results['error'] = str(e)
        
    return results

# ==================== ROUTES ====================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = []
        
        # Validation
        for name in engine.feature_names:
            val = data.get(name)
            if val is None:
                return jsonify({'error': f'Missing value for {name}'}), 400
            try:
                features.append(float(val))
            except ValueError:
                return jsonify({'error': f'Invalid value for {name}. Must be a number.'}), 400
        
        features_arr = np.array(features).reshape(1, -1)
        
        # Prediction
        prediction_results = engine.predict(features_arr)
        
        # Log prediction
        log_prediction(features_arr, prediction_results)
        
        # Explanations
        features_scaled = engine.standardize(features_arr)
        explanation_results = compute_explanations(features_scaled, features_arr, engine.model, engine)
        
        return jsonify({
            'prediction': prediction_results,
            'explanations': explanation_results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_report', methods=['POST'])
def download_report():
    try:
        data = request.json
        features_dict = data.get('features')
        prediction_results = data.get('prediction')
        explanation_results = data.get('explanations')
        
        if not features_dict or not prediction_results:
            return jsonify({'error': 'Missing data for report'}), 400
            
        features = []
        for name in engine.feature_names:
            features.append(float(features_dict.get(name)))
        features_arr = np.array(features).reshape(1, -1)
        
        # Generate PDF
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # --- PDF Generation Logic ---
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a5490'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#1a5490'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Header
        story.append(Paragraph("Heart Disease Prediction Report", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        story.append(Paragraph(f"<b>Generated:</b> {timestamp}", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Patient Info
        story.append(Paragraph("Patient Information", heading_style))
        patient_table_data = []
        for i, (name, value) in enumerate(zip(engine.feature_names, features_arr[0])):
            patient_table_data.append([name, f"{value:.2f}"])
            
        mid = len(patient_table_data) // 2
        combined_data = []
        for i in range(mid):
            left = patient_table_data[i]
            right = patient_table_data[i + mid] if i + mid < len(patient_table_data) else ['', '']
            combined_data.append(left + right)
            
        patient_table = Table(combined_data, colWidths=[1.8*inch, 1*inch, 1.8*inch, 1*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, -1), colors.HexColor('#e8f4f8')),
            ('BACKGROUND', (2, 0), (3, -1), colors.HexColor('#e8f4f8')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(patient_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Results
        story.append(Paragraph("Prediction Results", heading_style))
        prob = prediction_results['probability']
        risk = prediction_results['risk_level']
        
        if 'High' in risk: risk_color = colors.HexColor('#dc3545')
        elif 'Moderate' in risk: risk_color = colors.HexColor('#ffc107')
        else: risk_color = colors.HexColor('#28a745')
        
        pred_table = Table([
            ['Diagnosis:', prediction_results['prediction']],
            ['Probability:', f"{prob:.4f} ({prob*100:.2f}%)"],
            ['Risk Level:', risk]
        ], colWidths=[2*inch, 3.5*inch])
        pred_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f8f9fa')),
            ('BACKGROUND', (1, 2), (1, 2), risk_color),
            ('TEXTCOLOR', (1, 2), (1, 2), colors.white if 'Moderate' not in risk else colors.black),
            ('TEXTCOLOR', (0, 0), (-1, 1), colors.black),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(pred_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Recommendation
        story.append(Paragraph("Clinical Recommendation", heading_style))
        if prob >= 0.7:
            recommendation = "<b>HIGH RISK - Immediate medical consultation recommended.</b> Schedule comprehensive cardiac evaluation, review risk factors, and consider stress tests."
        elif prob >= 0.5:
            recommendation = "<b>MODERATE RISK - Medical consultation advised.</b> Implement preventive measures, monitor risk factors regularly, and improve lifestyle."
        elif prob >= 0.3:
            recommendation = "<b>LOW-MODERATE RISK - Continue monitoring.</b> Maintain healthy lifestyle and schedule regular checkups."
        else:
            recommendation = "<b>LOW RISK - Good health status.</b> Continue healthy habits, regular exercise, and balanced diet."
            
        story.append(Paragraph(recommendation, styles['Normal']))
        story.append(PageBreak())
        
        # Images
        if explanation_results.get('shap_plot_path') and os.path.exists(explanation_results['shap_plot_path']):
            story.append(Paragraph("SHAP Feature Importance Analysis", heading_style))
            story.append(RLImage(explanation_results['shap_plot_path'], width=6*inch, height=3.6*inch))
            story.append(Spacer(1, 0.2*inch))
            
        if explanation_results.get('lime_plot_path') and os.path.exists(explanation_results['lime_plot_path']):
            story.append(Paragraph("LIME Local Explanation Analysis", heading_style))
            story.append(RLImage(explanation_results['lime_plot_path'], width=6*inch, height=3.6*inch))
            
        # Disclaimer
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("Disclaimer", heading_style))
        disclaimer = "This report is generated by an AI model and is for informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment."
        story.append(Paragraph(disclaimer, styles['Italic']))
            
        doc.build(story)
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f"heart_disease_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mimetype='application/pdf'
        )
        
    except Exception as e:
        print(f"PDF Error: {e}")
        return jsonify({'error': str(e)}), 500

def log_prediction(features, results):
    """Log prediction to CSV files"""
    try:
        # Audit log
        try:
            audit_df = pd.read_csv("audit_log.csv")
        except FileNotFoundError:
            audit_df = pd.DataFrame(columns=engine.feature_names + ["Prediction", "Probability", "Timestamp"])
        
        new_row = pd.DataFrame(features, columns=engine.feature_names)
        new_row["Prediction"] = results['prediction']
        new_row["Probability"] = results['probability']
        new_row["Timestamp"] = pd.Timestamp.now()
        audit_df = pd.concat([audit_df, new_row], ignore_index=True)
        audit_df.to_csv("audit_log.csv", index=False)
        
        # Prediction log
        try:
            pred_df = pd.read_csv("Heart_Disease_Prediction.csv")
        except FileNotFoundError:
            pred_df = pd.DataFrame(columns=engine.feature_names + ["Prediction", "Probability", "Timestamp"])
        
        pred_row = pd.DataFrame(features, columns=engine.feature_names)
        pred_row["Prediction"] = results['prediction']
        pred_row["Probability"] = results['probability']
        pred_row["Timestamp"] = pd.Timestamp.now()
        pred_df = pd.concat([pred_df, pred_row], ignore_index=True)
        pred_df.to_csv("Heart_Disease_Prediction.csv", index=False)
        
    except Exception as e:
        print(f"Warning: Could not log prediction: {e}")

if __name__ == '__main__':
    app.run(debug=True, port=5000)
