"""
Complete ML Training with MLflow Tracking + Data Cleaning
Save as: 02_ML_Training_MLflow.py
Run: python 02_ML_Training_MLflow.py
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, classification_report,
                            mean_squared_error, mean_absolute_error, r2_score)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

print("=" * 80)
print("🚀 EMI PREDICTION - ML TRAINING WITH MLFLOW")
print("=" * 80)

# ============================================================================
# STEP 1: DATA LOADING & PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: DATA LOADING & PREPROCESSING")
print("=" * 80)

# Load data
print("\n📂 Loading dataset...")
df = pd.read_csv('data/EMI_dataset.csv')
print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ============================================================================
# DATA CLEANING - Convert to proper types
# ============================================================================
print("\n🧹 Cleaning data and converting types...")

# List of numeric columns
numeric_columns = [
    'age', 'monthly_salary', 'years_of_employment', 'monthly_rent',
    'family_size', 'dependents', 'school_fees', 'college_fees',
    'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
    'current_emi_amount', 'credit_score', 'bank_balance', 'emergency_fund',
    'requested_amount', 'requested_tenure', 'max_monthly_emi'
]

# Convert to numeric, forcing errors to NaN
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Fill NaN with median
        if df[col].isna().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

# Convert categorical columns to string
categorical_columns = [
    'gender', 'marital_status', 'education', 'employment_type',
    'company_type', 'house_type', 'existing_loans', 'emi_scenario',
    'emi_eligibility'
]

for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

print("   ✓ All data types corrected")

# Remove any remaining duplicates
initial_rows = len(df)
df = df.drop_duplicates()
if initial_rows > len(df):
    print(f"   ✓ Removed {initial_rows - len(df)} duplicate rows")

print(f"   ✓ Final dataset: {df.shape[0]:,} rows")

# ============================================================================
# Feature Engineering
# ============================================================================
print("\n🔧 Feature Engineering...")

# Create financial ratios with safe division
df['debt_to_income'] = df['current_emi_amount'] / (df['monthly_salary'] + 1)
df['expense_to_income'] = (df['monthly_rent'] + df['school_fees'] + 
                            df['college_fees'] + df['travel_expenses'] + 
                            df['groceries_utilities'] + df['other_monthly_expenses']) / (df['monthly_salary'] + 1)
df['savings_ratio'] = (df['bank_balance'] + df['emergency_fund']) / (df['monthly_salary'] + 1)
df['disposable_income'] = df['monthly_salary'] - (df['monthly_rent'] + df['current_emi_amount'] + 
                                                   df['groceries_utilities'] + df['other_monthly_expenses'])
df['affordability_ratio'] = df['disposable_income'] / (df['requested_amount'] + 1)

# Handle any infinite or extreme values
for col in ['debt_to_income', 'expense_to_income', 'savings_ratio', 'affordability_ratio']:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df[col].fillna(df[col].median(), inplace=True)
    # Cap extreme values at 99th percentile
    upper_limit = df[col].quantile(0.99)
    df[col] = df[col].clip(upper=upper_limit)

print("   ✓ Created 5 derived financial features")

# Separate features and targets
feature_cols = [col for col in df.columns if col not in ['emi_eligibility', 'max_monthly_emi']]
X = df[feature_cols].copy()
y_classification = df['emi_eligibility'].copy()
y_regression = df['max_monthly_emi'].copy()

print(f"\n📊 Feature Summary:")
print(f"   Total Features: {X.shape[1]}")
print(f"   Classification Target: {y_classification.name}")
print(f"   Regression Target: {y_regression.name}")

# ============================================================================
# Encode categorical variables
# ============================================================================
print("\n🔄 Encoding categorical variables...")
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"   Categorical columns: {len(categorical_cols)}")

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Encode target
le_target = LabelEncoder()
y_classification_encoded = le_target.fit_transform(y_classification)
class_names = le_target.classes_
print(f"   Target classes: {list(class_names)}")

# Verify no NaN values remain
if X.isna().sum().sum() > 0:
    print("\n   ⚠️  Warning: NaN values detected, filling with median...")
    X.fillna(X.median(), inplace=True)

if y_regression.isna().sum() > 0:
    print("   ⚠️  Warning: NaN in target, filling with median...")
    y_regression.fillna(y_regression.median(), inplace=True)

print("   ✓ All encoding complete, no NaN values")

# ============================================================================
# Split data
# ============================================================================
print("\n✂️ Splitting data (80% train, 20% test)...")
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_classification_encoded, y_regression, 
    test_size=0.2, random_state=42, stratify=y_classification_encoded
)
print(f"   Training: {X_train.shape[0]:,} samples")
print(f"   Testing: {X_test.shape[0]:,} samples")

# ============================================================================
# Scale features
# ============================================================================
print("\n📏 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save preprocessing objects
print("\n💾 Saving preprocessing objects...")
os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(label_encoders, 'models/label_encoders.pkl')
joblib.dump(le_target, 'models/target_encoder.pkl')
joblib.dump(feature_cols, 'models/feature_columns.pkl')
print("   ✓ Saved: scaler.pkl, label_encoders.pkl, target_encoder.pkl, feature_columns.pkl")

# ============================================================================
# STEP 2: MLFLOW SETUP
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: MLFLOW CONFIGURATION")
print("=" * 80)

mlflow.set_tracking_uri("file:./mlruns")
print("✅ MLflow tracking URI: ./mlruns")
print("💡 After training, run: mlflow ui")

# ============================================================================
# STEP 3: CLASSIFICATION MODELS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: TRAINING CLASSIFICATION MODELS (3 Models)")
print("=" * 80)

mlflow.set_experiment("EMI_Classification")
classification_results = {}

# MODEL 1: Logistic Regression
print("\n" + "-" * 80)
print("📊 [1/3] LOGISTIC REGRESSION")
print("-" * 80)

with mlflow.start_run(run_name="Logistic_Regression"):
    print("   Training model...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    lr_model.fit(X_train_scaled, y_class_train)
    
    # Predictions
    y_pred = lr_model.predict(X_test_scaled)
    y_pred_proba = lr_model.predict_proba(X_test_scaled)
    
    # Metrics
    accuracy = accuracy_score(y_class_test, y_pred)
    precision = precision_score(y_class_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_class_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_class_test, y_pred, average='weighted', zero_division=0)
    roc_auc = roc_auc_score(y_class_test, y_pred_proba, multi_class='ovr', average='weighted')
    
    # Log to MLflow
    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.sklearn.log_model(lr_model, "model")
    
    # Save locally
    joblib.dump(lr_model, 'models/logistic_regression_classifier.pkl')
    
    # Store results
    classification_results['Logistic Regression'] = {
        'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'f1_score': f1, 'roc_auc': roc_auc
    }
    
    print(f"   ✅ Accuracy:  {accuracy:.4f}")
    print(f"   ✅ Precision: {precision:.4f}")
    print(f"   ✅ Recall:    {recall:.4f}")
    print(f"   ✅ F1-Score:  {f1:.4f}")
    print(f"   ✅ ROC-AUC:   {roc_auc:.4f}")

# MODEL 2: Random Forest
print("\n" + "-" * 80)
print("📊 [2/3] RANDOM FOREST CLASSIFIER")
print("-" * 80)

with mlflow.start_run(run_name="Random_Forest_Classifier"):
    print("   Training model...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, 
                                     random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_class_train)
    
    # Predictions
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_class_test, y_pred)
    precision = precision_score(y_class_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_class_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_class_test, y_pred, average='weighted', zero_division=0)
    roc_auc = roc_auc_score(y_class_test, y_pred_proba, multi_class='ovr', average='weighted')
    
    # Log to MLflow
    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 20)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.sklearn.log_model(rf_model, "model")
    
    # Save locally
    joblib.dump(rf_model, 'models/random_forest_classifier.pkl')
    
    # Store results
    classification_results['Random Forest'] = {
        'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'f1_score': f1, 'roc_auc': roc_auc
    }
    
    print(f"   ✅ Accuracy:  {accuracy:.4f}")
    print(f"   ✅ Precision: {precision:.4f}")
    print(f"   ✅ Recall:    {recall:.4f}")
    print(f"   ✅ F1-Score:  {f1:.4f}")
    print(f"   ✅ ROC-AUC:   {roc_auc:.4f}")

# MODEL 3: XGBoost
print("\n" + "-" * 80)
print("📊 [3/3] XGBOOST CLASSIFIER")
print("-" * 80)

with mlflow.start_run(run_name="XGBoost_Classifier"):
    print("   Training model...")
    xgb_model = XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.1,
                              random_state=42, n_jobs=-1, eval_metric='mlogloss')
    xgb_model.fit(X_train, y_class_train)
    
    # Predictions
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_class_test, y_pred)
    precision = precision_score(y_class_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_class_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_class_test, y_pred, average='weighted', zero_division=0)
    roc_auc = roc_auc_score(y_class_test, y_pred_proba, multi_class='ovr', average='weighted')
    
    # Log to MLflow
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.sklearn.log_model(xgb_model, "model")
    
    # Save locally
    joblib.dump(xgb_model, 'models/xgboost_classifier.pkl')
    
    # Store results
    classification_results['XGBoost'] = {
        'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'f1_score': f1, 'roc_auc': roc_auc
    }
    
    print(f"   ✅ Accuracy:  {accuracy:.4f}")
    print(f"   ✅ Precision: {precision:.4f}")
    print(f"   ✅ Recall:    {recall:.4f}")
    print(f"   ✅ F1-Score:  {f1:.4f}")
    print(f"   ✅ ROC-AUC:   {roc_auc:.4f}")

# ============================================================================
# STEP 4: REGRESSION MODELS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: TRAINING REGRESSION MODELS (3 Models)")
print("=" * 80)

mlflow.set_experiment("EMI_Regression")
regression_results = {}

# MODEL 1: Linear Regression
print("\n" + "-" * 80)
print("📊 [1/3] LINEAR REGRESSION")
print("-" * 80)

with mlflow.start_run(run_name="Linear_Regression"):
    print("   Training model...")
    lin_model = LinearRegression(n_jobs=-1)
    lin_model.fit(X_train_scaled, y_reg_train)
    
    # Predictions
    y_pred = lin_model.predict(X_test_scaled)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred))
    mae = mean_absolute_error(y_reg_test, y_pred)
    r2 = r2_score(y_reg_test, y_pred)
    mape = np.mean(np.abs((y_reg_test - y_pred) / (y_reg_test + 1))) * 100
    
    # Log to MLflow
    mlflow.log_param("model_type", "Linear Regression")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mape", mape)
    mlflow.sklearn.log_model(lin_model, "model")
    
    # Save locally
    joblib.dump(lin_model, 'models/linear_regression.pkl')
    
    # Store results
    regression_results['Linear Regression'] = {
        'rmse': rmse, 'mae': mae, 'r2_score': r2, 'mape': mape
    }
    
    print(f"   ✅ RMSE:     ₹{rmse:.2f}")
    print(f"   ✅ MAE:      ₹{mae:.2f}")
    print(f"   ✅ R² Score: {r2:.4f}")
    print(f"   ✅ MAPE:     {mape:.2f}%")

# MODEL 2: Random Forest Regressor
print("\n" + "-" * 80)
print("📊 [2/3] RANDOM FOREST REGRESSOR")
print("-" * 80)

with mlflow.start_run(run_name="Random_Forest_Regressor"):
    print("   Training model...")
    rfr_model = RandomForestRegressor(n_estimators=100, max_depth=20,
                                     random_state=42, n_jobs=-1)
    rfr_model.fit(X_train, y_reg_train)
    
    # Predictions
    y_pred = rfr_model.predict(X_test)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred))
    mae = mean_absolute_error(y_reg_test, y_pred)
    r2 = r2_score(y_reg_test, y_pred)
    mape = np.mean(np.abs((y_reg_test - y_pred) / (y_reg_test + 1))) * 100
    
    # Log to MLflow
    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 20)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mape", mape)
    mlflow.sklearn.log_model(rfr_model, "model")
    
    # Save locally
    joblib.dump(rfr_model, 'models/random_forest_regressor.pkl')
    
    # Store results
    regression_results['Random Forest'] = {
        'rmse': rmse, 'mae': mae, 'r2_score': r2, 'mape': mape
    }
    
    print(f"   ✅ RMSE:     ₹{rmse:.2f}")
    print(f"   ✅ MAE:      ₹{mae:.2f}")
    print(f"   ✅ R² Score: {r2:.4f}")
    print(f"   ✅ MAPE:     {mape:.2f}%")

# MODEL 3: XGBoost Regressor
print("\n" + "-" * 80)
print("📊 [3/3] XGBOOST REGRESSOR")
print("-" * 80)

with mlflow.start_run(run_name="XGBoost_Regressor"):
    print("   Training model...")
    xgbr_model = XGBRegressor(n_estimators=100, max_depth=10, learning_rate=0.1,
                             random_state=42, n_jobs=-1)
    xgbr_model.fit(X_train, y_reg_train)
    
    # Predictions
    y_pred = xgbr_model.predict(X_test)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred))
    mae = mean_absolute_error(y_reg_test, y_pred)
    r2 = r2_score(y_reg_test, y_pred)
    mape = np.mean(np.abs((y_reg_test - y_pred) / (y_reg_test + 1))) * 100
    
    # Log to MLflow
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mape", mape)
    mlflow.sklearn.log_model(xgbr_model, "model")
    
    # Save locally
    joblib.dump(xgbr_model, 'models/xgboost_regressor.pkl')
    
    # Store results
    regression_results['XGBoost'] = {
        'rmse': rmse, 'mae': mae, 'r2_score': r2, 'mape': mape
    }
    
    print(f"   ✅ RMSE:     ₹{rmse:.2f}")
    print(f"   ✅ MAE:      ₹{mae:.2f}")
    print(f"   ✅ R² Score: {r2:.4f}")
    print(f"   ✅ MAPE:     {mape:.2f}%")

# ============================================================================
# STEP 5: MODEL COMPARISON & SELECTION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: MODEL COMPARISON & BEST MODEL SELECTION")
print("=" * 80)

# Classification Comparison
print("\n📊 CLASSIFICATION MODELS COMPARISON:")
print("-" * 80)
class_df = pd.DataFrame(classification_results).T
print(class_df.round(4))

best_classifier = class_df['f1_score'].idxmax()
print(f"\n🏆 BEST CLASSIFICATION MODEL: {best_classifier}")
print(f"   F1-Score: {class_df.loc[best_classifier, 'f1_score']:.4f}")
print(f"   Accuracy: {class_df.loc[best_classifier, 'accuracy']:.4f}")

# Regression Comparison
print("\n📊 REGRESSION MODELS COMPARISON:")
print("-" * 80)
reg_df = pd.DataFrame(regression_results).T
print(reg_df.round(2))

best_regressor = reg_df['r2_score'].idxmax()
print(f"\n🏆 BEST REGRESSION MODEL: {best_regressor}")
print(f"   R² Score: {reg_df.loc[best_regressor, 'r2_score']:.4f}")
print(f"   RMSE: ₹{reg_df.loc[best_regressor, 'rmse']:.2f}")

# Save comparison
comparison_results = {
    'classification': classification_results,
    'regression': regression_results,
    'best_classifier': best_classifier,
    'best_regressor': best_regressor
}
joblib.dump(comparison_results, 'models/model_comparison.pkl')

print("\n" + "=" * 80)
print("✅ TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 80)
print("\n📁 Models saved in: models/")
print("📊 MLflow experiments in: mlruns/")
print("\n💡 Next Steps:")
print("   1. View experiments: mlflow ui")
print("   2. Open browser: http://localhost:5000")
print("   3. Run Streamlit app: streamlit run app.py")
print("=" * 80)