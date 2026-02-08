"""
Heart Disease Prediction System - PyQt5 GUI (macOS Fixed)
Professional interface with PDF report generation
"""

import sys
import os
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# macOS-specific fixes - MUST BE BEFORE OTHER IMPORTS
if sys.platform == 'darwin':
    os.environ['QT_MAC_WANTS_LAYER'] = '1'
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    # Disable multiprocessing in SHAP
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

# Configure matplotlib BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QTabWidget,
    QGroupBox, QGridLayout, QScrollArea, QProgressBar, QMessageBox,
    QFileDialog, QSplitter, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor, QPixmap

# Import for PDF generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, 
    PageBreak, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

import warnings
warnings.filterwarnings('ignore')


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
        
        # Scaler values (replace with actual values from training)
        self.scaler_mean = np.array([60.0, 0.5, 2.0, 130.0, 250.0, 0.1, 1.0, 140.0, 0.3, 1.0, 1.0, 0.5, 3.5])
        self.scaler_std = np.array([10.0, 0.5, 1.0, 15.0, 50.0, 0.3, 0.7, 25.0, 0.4, 1.0, 0.8, 0.7, 2.0])
        
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = HeartDiseaseModel(len(self.feature_names))
            self.model.load_state_dict(torch.load("heart_disease_model.pt", map_location='cpu'))
            self.model.eval()
            return True
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
        
        prediction = "PRESENCE" if prob >= 0.5 else "ABSENCE"
        
        if prob >= 0.7:
            risk_level = "HIGH RISK"
        elif prob >= 0.5:
            risk_level = "MODERATE RISK"
        elif prob >= 0.3:
            risk_level = "LOW-MODERATE RISK"
        else:
            risk_level = "LOW RISK"
        
        return {
            'probability': prob,
            'prediction': prediction,
            'risk_level': risk_level
        }


# ==================== WORKER THREAD FOR EXPLANATIONS ====================
class ExplanationWorker(QThread):
    """Background thread for computing SHAP and LIME explanations"""
    
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, model, features_scaled, features_raw, engine):
        super().__init__()
        self.model = model
        self.features_scaled = features_scaled
        self.features_raw = features_raw
        self.engine = engine
        
    def run(self):
        """Compute explanations in background"""
        results = {}
        
        try:
            # Import here to avoid macOS fork issues
            import shap
            from lime.lime_tabular import LimeTabularExplainer
            
            # SHAP computation with error handling
            self.progress.emit("Computing SHAP values...")
            
            try:
                def shap_predict(x):
                    x_t = torch.tensor(x, dtype=torch.float32)
                    with torch.no_grad():
                        return self.model(x_t).numpy().flatten()
                
                # Use smaller background dataset for speed
                background = np.zeros((5, len(self.engine.feature_names)))
                
                # Use KernelExplainer with limited samples to avoid segfault
                explainer = shap.KernelExplainer(shap_predict, background)
                shap_values = explainer.shap_values(self.features_scaled, nsamples=50, silent=True)
                
                if isinstance(shap_values, list):
                    shap_vals = shap_values[0].flatten()
                else:
                    shap_vals = shap_values.flatten()
                
                expected_val = explainer.expected_value
                if isinstance(expected_val, (np.ndarray, list)):
                    expected_val = float(expected_val[0])
                else:
                    expected_val = float(expected_val)
                
                results['shap_values'] = shap_vals
                results['shap_expected'] = expected_val
                
            except Exception as e:
                print(f"SHAP computation failed: {e}")
                results['shap_error'] = str(e)
            
            # LIME computation
            self.progress.emit("Computing LIME values...")
            
            try:
                def lime_predict(x):
                    x_t = torch.tensor(x, dtype=torch.float32)
                    with torch.no_grad():
                        probs = self.model(x_t).numpy()
                    return np.column_stack([1 - probs, probs])
                
                # Create background with variance for LIME
                lime_background = np.tile(self.engine.scaler_mean, (50, 1))
                # Add small random noise to create variance
                np.random.seed(42)
                noise = np.random.normal(0, 0.1, lime_background.shape) * self.engine.scaler_std
                lime_background = lime_background + noise
                lime_background = self.engine.standardize(lime_background)
                
                lime_explainer = LimeTabularExplainer(
                    training_data=lime_background,
                    feature_names=self.engine.feature_names,
                    class_names=["Absence", "Presence"],
                    mode="classification",
                    random_state=42,
                    discretize_continuous=True
                )
                
                lime_exp = lime_explainer.explain_instance(
                    self.features_scaled[0],
                    lime_predict,
                    num_features=len(self.engine.feature_names),
                    top_labels=2
                )
                
                results['lime_exp'] = lime_exp
                
            except Exception as e:
                print(f"LIME computation failed: {e}")
                results['lime_error'] = str(e)
            
            self.progress.emit("Generating visualizations...")
            
            # Create SHAP plot if available
            if 'shap_values' in results:
                try:
                    # Close any existing figures
                    plt.close('all')
                    
                    fig = plt.figure(figsize=(10, 6))
                    
                    feature_contribution = list(zip(self.engine.feature_names, shap_vals, self.features_raw[0]))
                    feature_contribution.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    labels = [x[0] for x in feature_contribution]
                    values = [x[1] for x in feature_contribution]
                    colors_list = ['red' if v > 0 else 'blue' for v in values]
                    
                    y_pos = np.arange(len(labels))
                    plt.barh(y_pos, values, color=colors_list, alpha=0.6)
                    plt.yticks(y_pos, labels)
                    plt.xlabel('SHAP Value (Impact on Prediction)')
                    plt.title(f'Feature Importance Analysis (SHAP)')
                    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                    plt.tight_layout()
                    
                    shap_path = 'shap_explanation.png'
                    plt.savefig(shap_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    results['shap_plot_path'] = shap_path
                    
                except Exception as e:
                    print(f"SHAP plot generation failed: {e}")
                    results['shap_plot_error'] = str(e)
            
            # Create LIME plot if available
            if 'lime_exp' in results:
                try:
                    # Close any existing figures
                    plt.close('all')
                    
                    prob = self.model(torch.tensor(self.features_scaled, dtype=torch.float32)).item()
                    lime_label = 1 if prob >= 0.5 else 0
                    
                    fig = results['lime_exp'].as_pyplot_figure(label=lime_label)
                    plt.tight_layout()
                    
                    lime_path = 'lime_explanation.png'
                    plt.savefig(lime_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    results['lime_plot_path'] = lime_path
                    results['lime_label'] = lime_label
                    
                except Exception as e:
                    print(f"LIME plot generation failed: {e}")
                    results['lime_plot_error'] = str(e)
            
            # Always close all figures at the end
            plt.close('all')
            
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(str(e))
            plt.close('all')


# ==================== IMAGE DISPLAY WIDGET ====================
class ImageDisplay(QLabel):
    """Display matplotlib plots as images to avoid threading issues"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(False)  # Changed to False for better quality
        self.setStyleSheet("background-color: white; border: 1px solid #cccccc;")
        self.setMinimumSize(400, 300)
        
    def set_image(self, filepath):
        """Load and display image from file"""
        if os.path.exists(filepath):
            pixmap = QPixmap(filepath)
            scaled_pixmap = pixmap.scaled(
                self.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)
        else:
            self.setText("Image not available")
    
    def clear_image(self):
        """Clear the displayed image"""
        self.clear()
        self.setText("No analysis yet")
    
    def resizeEvent(self, event):
        """Handle resize events to rescale image"""
        super().resizeEvent(event)
        if self.pixmap() and not self.pixmap().isNull():
            # Re-scale the pixmap when widget is resized
            pass  # Qt handles this automatically with scaled contents


# ==================== PDF REPORT GENERATOR ====================
class PDFReportGenerator:
    """Generate professional PDF reports"""
    
    def __init__(self, patient_data, prediction_results, explanation_results, engine):
        self.patient_data = patient_data
        self.prediction = prediction_results
        self.explanations = explanation_results
        self.engine = engine
        
    def generate(self, filename):
        """Generate complete PDF report"""
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
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
        
        # Title
        story.append(Paragraph("Heart Disease Prediction Report", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Report info
        timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        story.append(Paragraph(f"<b>Generated:</b> {timestamp}", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Patient Information Section
        story.append(Paragraph("Patient Information", heading_style))
        
        patient_table_data = []
        for i, (name, value) in enumerate(zip(self.engine.feature_names, self.patient_data[0])):
            patient_table_data.append([name, f"{value:.2f}"])
        
        # Split into two columns
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
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(patient_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Prediction Results Section
        story.append(Paragraph("Prediction Results", heading_style))
        
        prob = self.prediction['probability']
        pred = self.prediction['prediction']
        risk = self.prediction['risk_level']
        
        # Color-coded risk box
        if 'HIGH' in risk:
            risk_color = colors.HexColor('#dc3545')
        elif 'MODERATE' in risk:
            risk_color = colors.HexColor('#ffc107')
        else:
            risk_color = colors.HexColor('#28a745')
        
        prediction_data = [
            ['Diagnosis:', pred],
            ['Probability:', f"{prob:.4f} ({prob*100:.2f}%)"],
            ['Risk Level:', risk]
        ]
        
        pred_table = Table(prediction_data, colWidths=[2*inch, 3.5*inch])
        pred_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f8f9fa')),
            ('BACKGROUND', (1, 2), (1, 2), risk_color),
            ('TEXTCOLOR', (1, 2), (1, 2), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, 1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 2), (1, 2), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(pred_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Clinical Recommendation
        story.append(Paragraph("Clinical Recommendation", heading_style))
        
        if prob >= 0.7:
            recommendation = """
            <b>⚠ HIGH RISK - Immediate medical consultation recommended</b><br/>
            • Schedule comprehensive cardiac evaluation<br/>
            • Review top risk factors with cardiologist<br/>
            • Consider stress test and imaging studies<br/>
            • Immediate lifestyle modifications required
            """
        elif prob >= 0.5:
            recommendation = """
            <b>⚠ MODERATE RISK - Medical consultation advised</b><br/>
            • Schedule appointment with primary care physician<br/>
            • Implement preventive measures<br/>
            • Regular monitoring of risk factors<br/>
            • Lifestyle modifications recommended
            """
        elif prob >= 0.3:
            recommendation = """
            <b>ℹ LOW-MODERATE RISK - Continue monitoring</b><br/>
            • Maintain healthy lifestyle<br/>
            • Regular health checkups<br/>
            • Monitor risk factors periodically<br/>
            • Consider preventive screening
            """
        else:
            recommendation = """
            <b>✓ LOW RISK - Good health status</b><br/>
            • Continue healthy habits<br/>
            • Annual health screening<br/>
            • Maintain current lifestyle<br/>
            • Regular exercise and balanced diet
            """
        
        story.append(Paragraph(recommendation, styles['Normal']))
        story.append(PageBreak())
        
        # Feature Importance Section
        story.append(Paragraph("Feature Importance Analysis", heading_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Top 5 SHAP features
        if 'shap_values' in self.explanations:
            shap_vals = self.explanations['shap_values']
            feature_contribution = list(zip(self.engine.feature_names, shap_vals, self.patient_data[0]))
            feature_contribution.sort(key=lambda x: abs(x[1]), reverse=True)
            
            story.append(Paragraph("Top Contributing Features (SHAP Analysis):", styles['Heading3']))
            
            shap_data = [['Rank', 'Feature', 'Value', 'Impact', 'Effect']]
            for i, (fname, sval, fval) in enumerate(feature_contribution[:5], 1):
                direction = "↑ Increases" if sval > 0 else "↓ Decreases"
                shap_data.append([
                    str(i),
                    fname,
                    f"{fval:.2f}",
                    f"{sval:+.4f}",
                    direction
                ])
            
            shap_table = Table(shap_data, colWidths=[0.5*inch, 2*inch, 1*inch, 1*inch, 1.5*inch])
            shap_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5490')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(shap_table)
            story.append(Spacer(1, 0.3*inch))
        
        # Add SHAP plot
        if 'shap_plot_path' in self.explanations and os.path.exists(self.explanations['shap_plot_path']):
            story.append(Paragraph("SHAP Feature Importance Visualization:", styles['Heading3']))
            img = RLImage(self.explanations['shap_plot_path'], width=6*inch, height=3.6*inch)
            story.append(img)
            story.append(PageBreak())
        
        # Add LIME plot
        if 'lime_plot_path' in self.explanations and os.path.exists(self.explanations['lime_plot_path']):
            story.append(Paragraph("LIME Explanation (Local Interpretability):", styles['Heading3']))
            story.append(Spacer(1, 0.1*inch))
            img = RLImage(self.explanations['lime_plot_path'], width=6*inch, height=3.6*inch)
            story.append(img)
            story.append(Spacer(1, 0.3*inch))
        
        # Disclaimer
        story.append(PageBreak())
        story.append(Paragraph("Important Disclaimer", heading_style))
        disclaimer_text = """
        This report is generated by an AI-powered prediction system and is intended for informational 
        purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, 
        or treatment. Always seek the advice of qualified health providers with any questions regarding 
        a medical condition. The predictions are based on statistical models and may not account for 
        all individual factors. Clinical judgment by healthcare professionals is essential for accurate 
        diagnosis and treatment planning.
        """
        story.append(Paragraph(disclaimer_text, styles['Normal']))
        
        # Build PDF
        doc.build(story)
        return filename


# ==================== MAIN GUI WINDOW ====================
class HeartDiseaseGUI(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize prediction engine
        self.engine = PredictionEngine()
        if not self.engine.model:
            QMessageBox.critical(self, "Error", "Failed to load model!")
            sys.exit(1)
        
        self.current_results = None
        self.explanation_results = None
        self.worker = None
        
        self.initUI()
        
    def initUI(self):
        """Initialize the user interface"""
        self.setWindowTitle("Heart Disease Prediction System - Professional Edition")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Input form
        left_panel = self.create_input_panel()
        
        # Right panel - Results with tabs
        right_panel = self.create_results_panel()
        
        # Splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        main_layout.addWidget(splitter)
        
        # Apply stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #1a5490;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2366a8;
            }
            QPushButton:pressed {
                background-color: #0d3d6b;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #cccccc;
                border-radius: 3px;
                background-color: white;
            }
            QLabel {
                color: #333333;
            }
            QTextEdit {
                font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
            }
        """)
        
    def create_input_panel(self):
        """Create the input form panel"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(400)
        
        container = QWidget()
        layout = QVBoxLayout(container)
        
        # Header
        header = QLabel("Patient Data Entry")
        header.setFont(QFont("Arial", 16, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # Input fields
        input_group = QGroupBox("Patient Information")
        input_layout = QGridLayout()
        
        self.input_fields = {}
        
        # Define fields with their properties
        fields = [
            ("Age", "years"),
            ("Sex", "1=Male, 0=Female"),
            ("Chest_pain_type", "0-4"),
            ("BP", "mm Hg"),
            ("Cholesterol", "mg/dl"),
            ("FBS_over_120", "1=Yes, 0=No"),
            ("EKG_results", "0-2"),
            ("Max_HR", "bpm"),
            ("Exercise_angina", "1=Yes, 0=No"),
            ("ST_depression", "units"),
            ("Slope_of_ST", "0-2"),
            ("Number_of_vessels_fluro", "0-3"),
            ("Thallium", "3, 6, or 7"),
        ]
        
        row = 0
        for field_name, hint in fields:
            label = QLabel(f"{field_name.replace('_', ' ')}:")
            label.setToolTip(hint)
            
            widget = QLineEdit()
            widget.setPlaceholderText(hint)
            
            self.input_fields[field_name] = widget
            input_layout.addWidget(label, row, 0)
            input_layout.addWidget(widget, row, 1)
            row += 1
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.predict_btn = QPushButton("🔍 Analyze")
        self.predict_btn.clicked.connect(self.run_prediction)
        
        self.clear_btn = QPushButton("🗑 Clear")
        self.clear_btn.clicked.connect(self.clear_inputs)
        
        button_layout.addWidget(self.predict_btn)
        button_layout.addWidget(self.clear_btn)
        layout.addLayout(button_layout)
        
        layout.addStretch()
        
        scroll.setWidget(container)
        return scroll
    
    def create_results_panel(self):
        """Create the results display panel"""
        container = QWidget()
        layout = QVBoxLayout(container)
        
        # Header
        header = QLabel("Analysis Results")
        header.setFont(QFont("Arial", 16, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # Tab widget
        self.tabs = QTabWidget()
        
        # Tab 1: Prediction Results
        self.results_tab = QWidget()
        results_layout = QVBoxLayout(self.results_tab)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        # Use system monospace font instead of Courier
        font = QFont()
        font.setFamily("Monospace")
        font.setStyleHint(QFont.TypeWriter)
        font.setPointSize(10)
        self.results_text.setFont(font)
        results_layout.addWidget(self.results_text)
        
        # Download button
        self.download_btn = QPushButton("📥 Download PDF Report")
        self.download_btn.clicked.connect(self.download_report)
        self.download_btn.setEnabled(False)
        results_layout.addWidget(self.download_btn)
        
        self.tabs.addTab(self.results_tab, "📊 Prediction")
        
        # Tab 2: SHAP Visualization
        self.shap_tab = QWidget()
        shap_layout = QVBoxLayout(self.shap_tab)
        
        self.shap_display = ImageDisplay(parent=self.shap_tab)
        self.shap_display.clear_image()
        shap_layout.addWidget(self.shap_display)
        
        self.tabs.addTab(self.shap_tab, "🔬 SHAP Analysis")
        
        # Tab 3: LIME Visualization
        self.lime_tab = QWidget()
        lime_layout = QVBoxLayout(self.lime_tab)
        
        self.lime_display = ImageDisplay(parent=self.lime_tab)
        self.lime_display.clear_image()
        lime_layout.addWidget(self.lime_display)
        
        self.tabs.addTab(self.lime_tab, "🔍 LIME Analysis")
        
        layout.addWidget(self.tabs)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready to analyze patient data")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        return container
    
    def get_input_values(self):
        """Extract values from input fields"""
        try:
            values = []
            for field_name in self.engine.feature_names:
                widget = self.input_fields[field_name]
                value = float(widget.text())
                values.append(value)
            return np.array(values).reshape(1, -1)
        except ValueError as e:
            raise ValueError(f"Invalid input: Please fill all fields with valid numbers")
    
    def run_prediction(self):
        """Run prediction and start explanation computation"""
        try:
            # Disable button during computation
            self.predict_btn.setEnabled(False)
            
            # Get input values
            features = self.get_input_values()
            
            # Make prediction
            self.status_label.setText("Making prediction...")
            QApplication.processEvents()  # Update UI
            
            results = self.engine.predict(features)
            self.current_results = results
            
            # Display results
            self.display_results(features, results)
            
            # Start explanation computation in background
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate
            self.status_label.setText("Computing explanations...")
            
            features_scaled = self.engine.standardize(features)
            
            # Clean up previous worker if exists
            if self.worker is not None:
                self.worker.quit()
                self.worker.wait()
            
            self.worker = ExplanationWorker(
                self.engine.model,
                features_scaled,
                features,
                self.engine
            )
            self.worker.progress.connect(self.update_status)
            self.worker.finished.connect(self.on_explanations_ready)
            self.worker.error.connect(self.on_explanation_error)
            self.worker.start()
            
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", str(e))
            self.predict_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed: {str(e)}")
            self.predict_btn.setEnabled(True)
    
    def display_results(self, features, results):
        """Display prediction results"""
        prob = results['probability']
        pred = results['prediction']
        risk = results['risk_level']
        
        # Format results text
        text = f"""
{'='*70}
HEART DISEASE PREDICTION RESULTS
{'='*70}

DIAGNOSIS:      {pred}
PROBABILITY:    {prob:.4f} ({prob*100:.2f}%)
RISK LEVEL:     {risk}

{'='*70}
PATIENT DATA
{'='*70}
"""
        
        for name, value in zip(self.engine.feature_names, features[0]):
            text += f"{name:30s}: {value:.2f}\n"
        
        text += f"\n{'='*70}\n"
        text += "CLINICAL RECOMMENDATION\n"
        text += f"{'='*70}\n\n"
        
        if prob >= 0.7:
            text += "⚠️  HIGH RISK - Immediate medical consultation recommended\n"
            text += "   → Schedule comprehensive cardiac evaluation\n"
            text += "   → Review top risk factors with cardiologist\n"
        elif prob >= 0.5:
            text += "⚠️  MODERATE RISK - Medical consultation advised\n"
            text += "   → Implement preventive measures\n"
            text += "   → Regular monitoring of risk factors\n"
        elif prob >= 0.3:
            text += "ℹ️  LOW-MODERATE RISK - Continue monitoring\n"
            text += "   → Maintain healthy lifestyle\n"
            text += "   → Regular health checkups\n"
        else:
            text += "✓ LOW RISK - Good health status\n"
            text += "   → Continue healthy habits\n"
            text += "   → Annual health screening\n"
        
        self.results_text.setText(text)
        self.download_btn.setEnabled(False)  # Enable after explanations
    
    def update_status(self, message):
        """Update status label"""
        self.status_label.setText(message)
    
    def on_explanations_ready(self, results):
        """Handle completed explanations"""
        self.explanation_results = results
        self.progress_bar.setVisible(False)
        self.status_label.setText("Analysis complete!")
        self.download_btn.setEnabled(True)
        self.predict_btn.setEnabled(True)
        
        # Update SHAP plot
        if 'shap_plot_path' in results:
            self.shap_display.set_image(results['shap_plot_path'])
        elif 'shap_error' in results:
            self.shap_display.setText(f"SHAP analysis failed:\n{results['shap_error']}")
        
        # Update LIME plot
        if 'lime_plot_path' in results:
            self.lime_display.set_image(results['lime_plot_path'])
        elif 'lime_error' in results:
            self.lime_display.setText(f"LIME analysis failed:\n{results['lime_error']}")
        
        # Add explanation text to results
        if 'shap_values' in results:
            self.add_explanation_text(results)
        
        QMessageBox.information(self, "Success", "Analysis complete! You can now download the full report.")
    
    def add_explanation_text(self, results):
        """Add explanation details to results text"""
        current_text = self.results_text.toPlainText()
        
        if 'shap_values' in results:
            shap_vals = results['shap_values']
            features = self.get_input_values()
            
            feature_contribution = list(zip(self.engine.feature_names, shap_vals, features[0]))
            feature_contribution.sort(key=lambda x: abs(x[1]), reverse=True)
            
            current_text += f"\n{'='*70}\n"
            current_text += "TOP CONTRIBUTING FEATURES (SHAP)\n"
            current_text += f"{'='*70}\n\n"
            
            for i, (fname, sval, fval) in enumerate(feature_contribution[:5], 1):
                direction = "↑ INCREASES" if sval > 0 else "↓ DECREASES"
                current_text += f"{i}. {fname:30s} = {fval:6.2f}\n"
                current_text += f"   Impact: {sval:+.4f} {direction} risk\n\n"
        
        self.results_text.setText(current_text)
    
    def on_explanation_error(self, error_msg):
        """Handle explanation computation error"""
        self.progress_bar.setVisible(False)
        self.status_label.setText("Explanation computation failed")
        self.predict_btn.setEnabled(True)
        QMessageBox.warning(self, "Warning", f"Could not compute explanations: {error_msg}\nPrediction is still available.")
        self.download_btn.setEnabled(True)  # Allow download without explanations
    
    def download_report(self):
        """Generate and download PDF report"""
        if not self.current_results:
            QMessageBox.warning(self, "No Results", "Please run a prediction first!")
            return
        
        try:
            # Get save location
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Report",
                f"heart_disease_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                "PDF Files (*.pdf)"
            )
            
            if filename:
                features = self.get_input_values()
                
                generator = PDFReportGenerator(
                    features,
                    self.current_results,
                    self.explanation_results or {},
                    self.engine
                )
                
                self.status_label.setText("Generating PDF report...")
                QApplication.processEvents()
                
                generator.generate(filename)
                self.status_label.setText("Report saved successfully!")
                
                QMessageBox.information(
                    self,
                    "Success",
                    f"Report saved to:\n{filename}"
                )
                
                # Log the prediction
                self.log_prediction(features)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate report: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def log_prediction(self, features):
        """Log prediction to CSV files"""
        try:
            # Audit log
            try:
                audit_df = pd.read_csv("audit_log.csv")
            except FileNotFoundError:
                audit_df = pd.DataFrame(columns=self.engine.feature_names + ["Prediction", "Probability", "Timestamp"])
            
            new_row = pd.DataFrame(features, columns=self.engine.feature_names)
            new_row["Prediction"] = self.current_results['prediction']
            new_row["Probability"] = self.current_results['probability']
            new_row["Timestamp"] = pd.Timestamp.now()
            audit_df = pd.concat([audit_df, new_row], ignore_index=True)
            audit_df.to_csv("audit_log.csv", index=False)
            
            # Prediction log
            try:
                pred_df = pd.read_csv("Heart_Disease_Prediction.csv")
            except FileNotFoundError:
                pred_df = pd.DataFrame(columns=self.engine.feature_names + ["Prediction", "Probability", "Timestamp"])
            
            pred_row = pd.DataFrame(features, columns=self.engine.feature_names)
            pred_row["Prediction"] = self.current_results['prediction']
            pred_row["Probability"] = self.current_results['probability']
            pred_row["Timestamp"] = pd.Timestamp.now()
            pred_df = pd.concat([pred_df, pred_row], ignore_index=True)
            pred_df.to_csv("Heart_Disease_Prediction.csv", index=False)
            
        except Exception as e:
            print(f"Warning: Could not log prediction: {e}")
    
    def clear_inputs(self):
        """Clear all input fields"""
        for widget in self.input_fields.values():
            widget.clear()
        
        self.results_text.clear()
        self.shap_display.clear_image()
        self.lime_display.clear_image()
        self.status_label.setText("Inputs cleared - Ready for new patient data")
        self.download_btn.setEnabled(False)
        self.current_results = None
        self.explanation_results = None
        self.predict_btn.setEnabled(True)
    
    def closeEvent(self, event):
        """Clean up when closing the application"""
        # Stop worker thread if running
        if self.worker is not None and self.worker.isRunning():
            self.worker.quit()
            self.worker.wait()
        
        # Close all matplotlib figures
        plt.close('all')
        
        event.accept()


# ==================== MAIN ENTRY POINT ====================
def main():
    # macOS-specific fixes
    if sys.platform == 'darwin':
        os.environ['QT_MAC_WANTS_LAYER'] = '1'
        os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
    
    app = QApplication(sys.argv)
    app.setApplicationName("Heart Disease Prediction System")
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    try:
        window = HeartDiseaseGUI()
        window.show()
        
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()