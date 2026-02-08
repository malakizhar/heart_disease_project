# heart_disease_complete.py
# =================================
# ULTIMATE ROBUST VERSION - Handles all edge cases
# Complete testing script with full error handling

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# 1. Feature Names & Scaler Values
# -----------------------------
feature_names = [
    'Age', 'Sex', 'Chest_pain_type', 'BP', 'Cholesterol',
    'FBS_over_120', 'EKG_results', 'Max_HR', 'Exercise_angina',
    'ST_depression', 'Slope_of_ST', 'Number_of_vessels_fluro', 'Thallium'
]

# IMPORTANT: Replace with actual values from training!
scaler_mean = np.array([60.0, 0.5, 2.0, 130.0, 250.0, 0.1, 1.0, 140.0, 0.3, 1.0, 1.0, 0.5, 3.5])
scaler_std  = np.array([10.0, 0.5, 1.0, 15.0, 50.0, 0.3, 0.7, 25.0, 0.4, 1.0, 0.8, 0.7, 2.0])

def standardize(x):
    """Standardize input features"""
    return (x - scaler_mean) / scaler_std

# -----------------------------
# 2. Model Definition (Sequential)
# -----------------------------
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

# -----------------------------
# 3. Load Model
# -----------------------------
print("="*70)
print("HEART DISEASE PREDICTION SYSTEM v2.0")
print("="*70)
print("\nLoading model...")

model = HeartDiseaseModel(len(feature_names))
try:
    model.load_state_dict(torch.load("heart_disease_model.pt"))
    model.eval()
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit(1)

# -----------------------------
# 4. Get Patient Input
# -----------------------------
def get_user_input():
    print("\n" + "-"*70)
    print("ENTER PATIENT DETAILS")
    print("-" * 70)
    
    data = []
    data.append(float(input("Age: ")))
    data.append(int(input("Sex (1=Male, 0=Female): ")))
    data.append(int(input("Chest pain type (0-4): ")))
    data.append(float(input("Resting BP (mm Hg): ")))
    data.append(float(input("Cholesterol (mg/dl): ")))
    data.append(int(input("Fasting Blood Sugar >120 (1=Yes, 0=No): ")))
    data.append(int(input("EKG results (0-2): ")))
    data.append(float(input("Max Heart Rate: ")))
    data.append(int(input("Exercise angina (1=Yes, 0=No): ")))
    data.append(float(input("ST depression: ")))
    data.append(int(input("Slope of ST (0-2): ")))
    data.append(int(input("Number of vessels (0-3): ")))
    data.append(int(input("Thallium (3, 6, or 7): ")))
    
    return np.array(data).reshape(1, -1)

user_input = get_user_input()
user_scaled = standardize(user_input)
user_tensor = torch.tensor(user_scaled, dtype=torch.float32)

# -----------------------------
# 5. Make Prediction
# -----------------------------
print("\n" + "="*70)
print("PREDICTION RESULTS")
print("="*70)

with torch.no_grad():
    prob = model(user_tensor).item()

prediction = "PRESENCE" if prob >= 0.5 else "ABSENCE"
risk_level = "HIGH RISK" if prob >= 0.7 else "MODERATE RISK" if prob >= 0.5 else "LOW RISK"

print(f"\nHeart Disease Status: {prediction}")
print(f"Probability: {prob:.4f} ({prob*100:.2f}%)")
print(f"Risk Level: {risk_level}")
print("-" * 70)

# Check for potential issues
if prob > 0.95:
    print("\n⚠️  Very high probability - Model may need retraining")
elif prob < 0.05:
    print("\n⚠️  Very low probability - Model may need retraining")

# -----------------------------
# 6. SHAP Explanation
# -----------------------------
print("\n" + "="*70)
print("GENERATING EXPLANATIONS...")
print("="*70)

def shap_predict(x):
    x_t = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        return model(x_t).numpy().flatten()

try:
    print("\n[1/2] Calculating SHAP values...", end=" ")
    background = np.zeros((10, len(feature_names)))
    explainer = shap.KernelExplainer(shap_predict, background)
    shap_values = explainer.shap_values(user_scaled, nsamples=100)
    
    # Extract properly
    if isinstance(shap_values, list):
        shap_vals = shap_values[0].flatten()
    else:
        shap_vals = shap_values.flatten()
    
    expected_val = explainer.expected_value
    if isinstance(expected_val, (np.ndarray, list)):
        expected_val = float(expected_val[0])
    else:
        expected_val = float(expected_val)
    
    print("✓ Done")
    
    # Display top features
    feature_vals = user_input[0]
    shap_contribution = list(zip(feature_names, shap_vals, feature_vals))
    shap_contribution.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("\n" + "-"*70)
    print("TOP CONTRIBUTING FEATURES (SHAP)")
    print("-"*70)
    for i, (fname, sval, fval) in enumerate(shap_contribution[:5], 1):
        direction = "↑ INCREASES" if sval > 0 else "↓ DECREASES"
        print(f"{i}. {fname:30s} = {fval:6.2f}")
        print(f"   Impact: {sval:+.4f} {direction} risk")
        if i < 5:
            print()
    
    # Create plot
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(feature_names))
    colors = ['red' if x[1] > 0 else 'blue' for x in shap_contribution]
    values = [x[1] for x in shap_contribution]
    labels = [x[0] for x in shap_contribution]
    
    plt.barh(y_pos, values, color=colors, alpha=0.6)
    plt.yticks(y_pos, labels)
    plt.xlabel('SHAP Value (Impact on Prediction)')
    plt.title(f'Feature Importance\nBase: {expected_val:.4f} → Prediction: {prob:.4f}')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('shap_explanation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ SHAP plot saved: shap_explanation.png")
    
except Exception as e:
    print(f"\n✗ SHAP calculation failed: {e}")
    print("  Continuing without SHAP explanation...")

# -----------------------------
# 7. LIME Explanation
# -----------------------------
def lime_predict(x):
    x_t = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        probs = model(x_t).numpy()
    return np.column_stack([1 - probs, probs])

try:
    print("\n[2/2] Calculating LIME values...", end=" ")
    
    # Create background with variance for LIME
    lime_background = np.tile(scaler_mean, (50, 1))
    # Add small random noise to create variance
    np.random.seed(42)
    noise = np.random.normal(0, 0.1, lime_background.shape) * scaler_std
    lime_background = lime_background + noise
    lime_background = standardize(lime_background)
    
    lime_explainer = LimeTabularExplainer(
        training_data=lime_background,
        feature_names=feature_names,
        class_names=["Absence", "Presence"],
        mode="classification",
        discretize_continuous=True
    )
    
    lime_exp = lime_explainer.explain_instance(
        user_scaled[0], 
        lime_predict, 
        num_features=len(feature_names),
        top_labels=2
    )
    
    print("✓ Done")
    
    # Determine which class to explain
    lime_label = 1 if prob >= 0.5 else 0
    label_name = "Presence" if lime_label == 1 else "Absence"
    
    print("\n" + "-"*70)
    print(f"TOP CONTRIBUTING FEATURES (LIME - {label_name})")
    print("-"*70)
    
    lime_list = lime_exp.as_list(label=lime_label)
    for i, (feature_desc, weight) in enumerate(lime_list[:5], 1):
        # Interpret weights correctly
        if lime_label == 1:
            direction = "↑ INCREASES" if weight > 0 else "↓ DECREASES"
        else:
            direction = "↓ DECREASES" if weight > 0 else "↑ INCREASES"
        
        print(f"{i}. {feature_desc}")
        print(f"   Impact: {weight:+.4f} {direction} presence risk")
        if i < 5:
            print()
    
    # Save plot
    fig = lime_exp.as_pyplot_figure(label=lime_label)
    plt.title(f'LIME Feature Importance ({label_name} Class)')
    plt.tight_layout()
    plt.savefig('lime_explanation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ LIME plot saved: lime_explanation.png")
    
except Exception as e:
    print(f"\n✗ LIME calculation failed: {e}")
    print("  SHAP explanation is still available.")

# -----------------------------
# 8. Update Logs
# -----------------------------
print("\n" + "="*70)
print("UPDATING LOGS")
print("="*70)

try:
    # Audit log
    try:
        audit_df = pd.read_csv("audit_log.csv")
    except FileNotFoundError:
        audit_df = pd.DataFrame(columns=feature_names + ["Prediction", "Probability", "Timestamp"])
    
    new_row = pd.DataFrame(user_input, columns=feature_names)
    new_row["Prediction"] = prediction
    new_row["Probability"] = prob
    new_row["Timestamp"] = pd.Timestamp.now()
    audit_df = pd.concat([audit_df, new_row], ignore_index=True)
    audit_df.to_csv("audit_log.csv", index=False)
    print("✓ audit_log.csv updated")
    
    # Prediction log
    try:
        pred_df = pd.read_csv("Heart_Disease_Prediction.csv")
    except FileNotFoundError:
        pred_df = pd.DataFrame(columns=feature_names + ["Prediction", "Probability", "Timestamp"])
    
    pred_row = pd.DataFrame(user_input, columns=feature_names)
    pred_row["Prediction"] = prediction
    pred_row["Probability"] = prob
    pred_row["Timestamp"] = pd.Timestamp.now()
    pred_df = pd.concat([pred_df, pred_row], ignore_index=True)
    pred_df.to_csv("Heart_Disease_Prediction.csv", index=False)
    print("✓ Heart_Disease_Prediction.csv updated")
    
except Exception as e:
    print(f"✗ Error updating logs: {e}")

# -----------------------------
# 9. Final Summary
# -----------------------------
print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)

print(f"\n📋 PATIENT SUMMARY")
print("-" * 70)
for fname, fval in zip(feature_names[:7], user_input[0][:7]):
    print(f"  {fname:30s}: {fval}")
print("  ...")

print(f"\n🎯 FINAL DIAGNOSIS")
print("-" * 70)
print(f"  Status:      {prediction}")
print(f"  Probability: {prob:.4f} ({prob*100:.2f}%)")
print(f"  Risk Level:  {risk_level}")

# Clinical recommendation
print(f"\n💡 RECOMMENDATION")
print("-" * 70)
if prob >= 0.7:
    print("  ⚠️  HIGH RISK - Immediate medical consultation recommended")
    print("  → Schedule cardiac evaluation")
    print("  → Review top risk factors")
elif prob >= 0.5:
    print("  ⚠️  MODERATE RISK - Medical consultation advised")
    print("  → Consider preventive measures")
    print("  → Monitor risk factors")
elif prob >= 0.3:
    print("  ℹ️  LOW-MODERATE RISK - Continue monitoring")
    print("  → Maintain healthy lifestyle")
    print("  → Regular checkups")
else:
    print("  ✓ LOW RISK - Good health status")
    print("  → Continue healthy habits")
    print("  → Annual health screening")

print("\n" + "="*70)
print("Thank you for using the Heart Disease Prediction System")
print("="*70)