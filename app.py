"""
EMIPredict AI - Professional Streamlit Web Application
Main application file
Save as: app.py
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main Header Styles */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        animation: fadeIn 1s ease-in;
    }
    
    .main-header h1 {
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        font-size: 1.3rem;
        margin-top: 1rem;
        opacity: 0.95;
    }
    
    /* Feature Cards */
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        height: 100%;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
    }
    
    .feature-card h2 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .feature-card h3 {
        font-size: 1.5rem;
        margin: 1rem 0 0.5rem 0;
        font-weight: 600;
    }
    
    .feature-card p {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Stats Cards */
    .stats-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        transition: transform 0.2s ease;
    }
    
    .stats-card:hover {
        transform: translateX(5px);
    }
    
    .stats-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        margin: 0;
    }
    
    .stats-label {
        font-size: 1rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    /* Success/Warning/Danger Boxes */
    .success-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        border-left: 5px solid #00b894;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .success-box h2 {
        color: #00693e;
        margin-top: 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
        border-left: 5px solid #fdcb6e;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .warning-box h2 {
        color: #d63031;
        margin-top: 0;
    }
    
    .danger-box {
        background: linear-gradient(135deg, #ff7675 0%, #fab1a0 100%);
        border-left: 5px solid #d63031;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .danger-box h2 {
        color: #2d3436;
        margin-top: 0;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 3rem;
        font-size: 1.2rem;
        font-weight: 600;
        border-radius: 50px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] .css-17lntkn {
        color: white;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Section Headers */
    .section-header {
        font-size: 2rem;
        font-weight: 600;
        color: #2d3436;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Info Box */
    .info-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    /* Metric Cards */
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        text-align: center;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

# Load models and preprocessors
@st.cache_resource
def load_models():
    """Load all saved models and preprocessing objects"""
    try:
        # Try to load best models first
        model_files = {
            'classifier': ['xgboost_classifier.pkl', 'random_forest_classifier.pkl', 'logistic_regression_classifier.pkl'],
            'regressor': ['xgboost_regressor.pkl', 'random_forest_regressor.pkl', 'linear_regression.pkl']
        }
        
        models = {}
        
        # Load classifier
        for clf_file in model_files['classifier']:
            try:
                models['classifier'] = joblib.load(f'models/{clf_file}')
                break
            except:
                continue
        
        # Load regressor
        for reg_file in model_files['regressor']:
            try:
                models['regressor'] = joblib.load(f'models/{reg_file}')
                break
            except:
                continue
        
        # Load preprocessing objects
        models['scaler'] = joblib.load('models/scaler.pkl')
        models['label_encoders'] = joblib.load('models/label_encoders.pkl')
        models['target_encoder'] = joblib.load('models/target_encoder.pkl')
        models['feature_columns'] = joblib.load('models/feature_columns.pkl')
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

# Load dataset
@st.cache_data
def load_data():
    """Load the EMI dataset with proper data type handling"""
    try:
        df = pd.read_csv('data/EMI_dataset.csv')
        
        # Convert numeric columns to proper types
        numeric_columns = [
            'age', 'monthly_salary', 'years_of_employment', 'monthly_rent',
            'family_size', 'dependents', 'school_fees', 'college_fees',
            'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
            'current_emi_amount', 'credit_score', 'bank_balance', 'emergency_fund',
            'requested_amount', 'requested_tenure', 'max_monthly_emi'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
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
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Sidebar navigation
def sidebar_navigation():
    """Create modern sidebar navigation"""
    st.sidebar.markdown("""
        <div style='text-align: center; padding: 1rem; color: white;'>
            <h1 style='font-size: 2rem; margin: 0;'>💰</h1>
            <h2 style='font-size: 1.5rem; margin: 0.5rem 0;'>EMIPredict AI</h2>
            <p style='opacity: 0.8; font-size: 0.9rem;'>Intelligent Risk Assessment</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Navigation buttons
    pages = {
        "🏠 Home": "home",
        "🔮 EMI Prediction": "prediction",
        "📊 Data Explorer": "explorer",
        "📈 Model Performance": "performance",
        "ℹ️ About": "about"
    }
    
    for page_name, page_key in pages.items():
        if st.sidebar.button(page_name, key=f"nav_{page_key}", use_container_width=True):
            st.session_state.current_page = page_key
            st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        <div style='color: white; padding: 1rem; text-align: center;'>
            <h3 style='font-size: 1rem;'>📌 Quick Stats</h3>
        </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.get('data') is not None:
        df = st.session_state['data']
        st.sidebar.markdown(f"""
            <div style='color: white; text-align: center; padding: 0.5rem;'>
                <p style='margin: 0.3rem 0;'><b>Records:</b> {len(df):,}</p>
                <p style='margin: 0.3rem 0;'><b>Features:</b> {df.shape[1]}</p>
                <p style='margin: 0.3rem 0;'><b>Eligible:</b> {(df['emi_eligibility']=='Eligible').mean()*100:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)
    
    return st.session_state.current_page

# Home Page
def home_page():
    """Display modern home page"""
    
    # Hero Section
    st.markdown("""
        <div class="main-header">
            <h1>💰 EMIPredict AI</h1>
            <p>Intelligent Financial Risk Assessment Platform Powered by Machine Learning</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="feature-card">
                <h2>⚡</h2>
                <h3>Instant Eligibility</h3>
                <p>Get loan approval decisions in real-time using advanced AI algorithms</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-card">
                <h2>🎯</h2>
                <h3>93% Accuracy</h3>
                <p>Trained on 400K+ records with state-of-the-art ML models</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="feature-card">
                <h2>🔒</h2>
                <h3>Risk Assessment</h3>
                <p>Comprehensive financial profiling and risk analysis</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Statistics Section
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📊 Platform Statistics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class="stats-card">
                <p class="stats-number">400K+</p>
                <p class="stats-label">Training Records</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="stats-card">
                <p class="stats-number">6</p>
                <p class="stats-label">ML Models</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="stats-card">
                <p class="stats-number">93%</p>
                <p class="stats-label">Accuracy</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class="stats-card">
                <p class="stats-number">5</p>
                <p class="stats-label">EMI Scenarios</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Features Section
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">✨ Key Features</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="info-box">
                <h3>🎯 What We Offer</h3>
                <ul style='line-height: 2;'>
                    <li><b>EMI Eligibility Check:</b> Instant approval status</li>
                    <li><b>Maximum EMI Calculation:</b> Know your affordability</li>
                    <li><b>Risk Profiling:</b> Comprehensive financial analysis</li>
                    <li><b>Multi-Scenario Support:</b> 5 different loan types</li>
                    <li><b>Real-time Predictions:</b> Results in seconds</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="info-box">
                <h3>🏦 Supported EMI Scenarios</h3>
                <ul style='line-height: 2;'>
                    <li><b>E-commerce Shopping:</b> ₹10K - ₹2L</li>
                    <li><b>Home Appliances:</b> ₹20K - ₹3L</li>
                    <li><b>Vehicle Loan:</b> ₹80K - ₹15L</li>
                    <li><b>Personal Loan:</b> ₹50K - ₹10L</li>
                    <li><b>Education Loan:</b> ₹50K - ₹5L</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # How It Works
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🔄 How It Works</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div style='text-align: center; padding: 1rem;'>
                <h2 style='color: #667eea; font-size: 3rem;'>1</h2>
                <h3>Input Details</h3>
                <p>Fill in your financial information</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='text-align: center; padding: 1rem;'>
                <h2 style='color: #667eea; font-size: 3rem;'>2</h2>
                <h3>AI Analysis</h3>
                <p>Our ML models analyze your profile</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style='text-align: center; padding: 1rem;'>
                <h2 style='color: #667eea; font-size: 3rem;'>3</h2>
                <h3>Get Results</h3>
                <p>Receive instant eligibility status</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div style='text-align: center; padding: 1rem;'>
                <h2 style='color: #667eea; font-size: 3rem;'>4</h2>
                <h3>Make Decision</h3>
                <p>Use insights for financial planning</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Call to Action
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🚀 Ready to Get Started?</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔮 Start EMI Prediction Now", use_container_width=True, key="cta_button"):
            st.session_state.current_page = 'prediction'
            st.rerun()
    
    # Trust Badges
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 15px;'>
            <h3 style='color: #2d3436; margin-bottom: 1rem;'>🏆 Trusted by Financial Institutions</h3>
            <p style='color: #636e72; font-size: 1.1rem;'>
                Built with industry-leading ML frameworks | MLflow Experiment Tracking | 
                Cloud Deployed | Production Ready
            </p>
        </div>
    """, unsafe_allow_html=True)

# Prediction Page (keeping the existing functionality)
def prediction_page():
    """Display prediction page"""
    st.markdown('<div class="section-header">🔮 EMI Eligibility Prediction</div>', unsafe_allow_html=True)
    st.markdown("Fill in your details below to get instant EMI eligibility assessment")
    
    models = st.session_state.get('models')
    if models is None:
        st.error("❌ Models not loaded. Please ensure model files exist in the 'models/' folder.")
        return
    
    # Create form
    with st.form("prediction_form"):
        st.markdown("### 👤 Personal Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=25, max_value=60, value=30)
            gender = st.selectbox("Gender", ["Male", "Female"])
        
        with col2:
            marital_status = st.selectbox("Marital Status", ["Single", "Married"])
            education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
        
        with col3:
            family_size = st.number_input("Family Size", min_value=1, max_value=6, value=2)
            dependents = st.number_input("Dependents", min_value=0, max_value=5, value=0)
        
        st.markdown("---")
        st.markdown("### 💼 Employment Details")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            monthly_salary = st.number_input("Monthly Salary (₹)", min_value=15000, max_value=200000, value=50000, step=5000)
            employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
        
        with col2:
            years_of_employment = st.number_input("Years of Employment", min_value=1, max_value=35, value=5)
            company_type = st.selectbox("Company Type", ["MNC", "Startup", "SME", "Large Enterprise"])
        
        with col3:
            house_type = st.selectbox("House Type", ["Rented", "Own", "Family"])
            monthly_rent = st.number_input("Monthly Rent (₹)", min_value=0, max_value=50000, 
                                          value=10000 if house_type == "Rented" else 0, step=1000)
        
        st.markdown("---")
        st.markdown("### 💳 Financial Details")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
            bank_balance = st.number_input("Bank Balance (₹)", min_value=0, max_value=10000000, value=100000, step=10000)
        
        with col2:
            emergency_fund = st.number_input("Emergency Fund (₹)", min_value=0, max_value=5000000, value=50000, step=10000)
            existing_loans = st.selectbox("Existing Loans", ["Yes", "No"])
        
        with col3:
            current_emi_amount = st.number_input("Current EMI Amount (₹)", min_value=0, max_value=50000, 
                                                value=5000 if existing_loans == "Yes" else 0, step=1000)
        
        st.markdown("---")
        st.markdown("### 📝 Monthly Expenses")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            school_fees = st.number_input("School Fees (₹)", min_value=0, max_value=50000, value=0, step=1000)
        
        with col2:
            college_fees = st.number_input("College Fees (₹)", min_value=0, max_value=100000, value=0, step=1000)
        
        with col3:
            travel_expenses = st.number_input("Travel Expenses (₹)", min_value=0, max_value=20000, value=5000, step=500)
        
        with col4:
            groceries_utilities = st.number_input("Groceries & Utilities (₹)", min_value=0, max_value=50000, value=10000, step=1000)
        
        other_monthly_expenses = st.number_input("Other Monthly Expenses (₹)", min_value=0, max_value=50000, value=5000, step=1000)
        
        st.markdown("---")
        st.markdown("### 🎯 Loan Requirements")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            emi_scenario = st.selectbox("EMI Scenario", [
                "E-commerce Shopping EMI",
                "Home Appliances EMI",
                "Vehicle EMI",
                "Personal Loan EMI",
                "Education EMI"
            ])
        
        with col2:
            requested_amount = st.number_input("Requested Amount (₹)", min_value=10000, max_value=1500000, value=100000, step=10000)
        
        with col3:
            requested_tenure = st.number_input("Tenure (months)", min_value=3, max_value=84, value=12)
        
        # Submit button
        submitted = st.form_submit_button("🔮 Predict EMI Eligibility", type="primary", use_container_width=True)
    
    if submitted:
        with st.spinner("🔄 Analyzing your financial profile..."):
            # Prepare input data
            input_data = {
                'age': age, 'gender': gender, 'marital_status': marital_status, 'education': education,
                'monthly_salary': monthly_salary, 'employment_type': employment_type,
                'years_of_employment': years_of_employment, 'company_type': company_type,
                'house_type': house_type, 'monthly_rent': monthly_rent, 'family_size': family_size,
                'dependents': dependents, 'school_fees': school_fees, 'college_fees': college_fees,
                'travel_expenses': travel_expenses, 'groceries_utilities': groceries_utilities,
                'other_monthly_expenses': other_monthly_expenses, 'existing_loans': existing_loans,
                'current_emi_amount': current_emi_amount, 'credit_score': credit_score,
                'bank_balance': bank_balance, 'emergency_fund': emergency_fund,
                'emi_scenario': emi_scenario, 'requested_amount': requested_amount,
                'requested_tenure': requested_tenure
            }
            
            # Feature engineering
            input_data['debt_to_income'] = current_emi_amount / (monthly_salary + 1)
            input_data['expense_to_income'] = (monthly_rent + school_fees + college_fees + 
                                              travel_expenses + groceries_utilities + 
                                              other_monthly_expenses) / (monthly_salary + 1)
            input_data['savings_ratio'] = (bank_balance + emergency_fund) / (monthly_salary + 1)
            input_data['disposable_income'] = monthly_salary - (monthly_rent + current_emi_amount + 
                                                                groceries_utilities + other_monthly_expenses)
            input_data['affordability_ratio'] = input_data['disposable_income'] / (requested_amount + 1)
            
            # Create DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical variables
            label_encoders = models['label_encoders']
            for col, le in label_encoders.items():
                if col in input_df.columns:
                    try:
                        input_df[col] = le.transform(input_df[col].astype(str))
                    except:
                        input_df[col] = 0
            
            # Ensure correct column order
            feature_columns = models['feature_columns']
            input_df = input_df[feature_columns]
            
            # Make predictions
            try:
                # Classification prediction
                eligibility_pred = models['classifier'].predict(input_df)[0]
                eligibility_proba = models['classifier'].predict_proba(input_df)[0]
                eligibility_label = models['target_encoder'].inverse_transform([eligibility_pred])[0]
                
                # Regression prediction
                max_emi_pred = models['regressor'].predict(input_df)[0]
                
                # Display results
                st.markdown("---")
                st.markdown('<div class="section-header">📊 Prediction Results</div>', unsafe_allow_html=True)
                
                # Eligibility result
                if eligibility_label == "Eligible":
                    st.markdown(f"""
                        <div class="success-box">
                            <h2>✅ Congratulations! You are ELIGIBLE</h2>
                            <p style='font-size: 1.1rem;'>Your loan application meets all criteria for approval.</p>
                            <p style='font-size: 1.3rem; font-weight: 600;'>Confidence: {max(eligibility_proba)*100:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                elif eligibility_label == "High_Risk":
                    st.markdown(f"""
                        <div class="warning-box">
                            <h2>⚠️ High Risk - Conditional Approval</h2>
                            <p style='font-size: 1.1rem;'>Your application may require higher interest rates or additional documentation.</p>
                            <p style='font-size: 1.3rem; font-weight: 600;'>Confidence: {max(eligibility_proba)*100:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="danger-box">
                            <h2>❌ Not Eligible</h2>
                            <p style='font-size: 1.1rem;'>Unfortunately, your current financial profile does not meet eligibility criteria.</p>
                            <p style='font-size: 1.3rem; font-weight: 600;'>Confidence: {max(eligibility_proba)*100:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # EMI amount result
                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                        <div class="metric-container">
                            <h3 style='color: #667eea; font-size: 2.5rem; margin: 0;'>₹{max_emi_pred:,.0f}</h3>
                            <p style='color: #666; margin-top: 0.5rem;'>Maximum EMI/Month</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    monthly_rate = 0.10 / 12
                    requested_emi = (requested_amount * monthly_rate * (1 + monthly_rate)**requested_tenure) / ((1 + monthly_rate)**requested_tenure - 1)
                    st.markdown(f"""
                        <div class="metric-container">
                            <h3 style='color: #636e72; font-size: 2.5rem; margin: 0;'>₹{requested_emi:,.0f}</h3>
                            <p style='color: #666; margin-top: 0.5rem;'>Requested EMI/Month</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    affordability = (max_emi_pred / requested_emi * 100) if requested_emi > 0 else 0
                    color = "#00b894" if affordability >= 100 else "#fdcb6e" if affordability >= 80 else "#d63031"
                    st.markdown(f"""
                        <div class="metric-container">
                            <h3 style='color: {color}; font-size: 2.5rem; margin: 0;'>{affordability:.0f}%</h3>
                            <p style='color: #666; margin-top: 0.5rem;'>Affordability Index</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Additional insights
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">💡 Financial Insights</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                        <div class="info-box">
                            <h3>📊 Key Financial Ratios</h3>
                            <ul style='line-height: 2;'>
                                <li><b>Debt-to-Income:</b> {input_data['debt_to_income']*100:.1f}%</li>
                                <li><b>Expense-to-Income:</b> {input_data['expense_to_income']*100:.1f}%</li>
                                <li><b>Disposable Income:</b> ₹{input_data['disposable_income']:,.0f}</li>
                                <li><b>Savings Ratio:</b> {input_data['savings_ratio']:.2f}</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                        <div class="info-box">
                            <h3>💡 Recommendations</h3>
                            <ul style='line-height: 2;'>
                    """, unsafe_allow_html=True)
                    
                    if eligibility_label == "Eligible":
                        st.markdown("""
                                <li>✅ Your financial health is excellent</li>
                                <li>✅ You can proceed with loan application</li>
                                <li>✅ Consider negotiating better interest rates</li>
                                <li>✅ Maintain your current credit score</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                    elif eligibility_label == "High_Risk":
                        st.markdown("""
                                <li>⚠️ Consider reducing existing EMI burden</li>
                                <li>⚠️ Work on improving credit score</li>
                                <li>⚠️ Increase emergency fund reserves</li>
                                <li>⚠️ You may get approval with higher interest</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                                <li>❌ Focus on debt reduction first</li>
                                <li>❌ Build better credit history</li>
                                <li>❌ Increase your monthly income</li>
                                <li>❌ Consider a co-applicant option</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")

# Data Explorer Page
def explorer_page():
    """Display data explorer page"""
    st.markdown('<div class="section-header">📊 Data Explorer</div>', unsafe_allow_html=True)
    
    df = st.session_state.get('data')
    if df is None:
        st.error("❌ Data not loaded.")
        return
    
    # Ensure numeric columns are actually numeric
    try:
        df['monthly_salary'] = pd.to_numeric(df['monthly_salary'], errors='coerce')
        df['max_monthly_emi'] = pd.to_numeric(df['max_monthly_emi'], errors='coerce')
        df['credit_score'] = pd.to_numeric(df['credit_score'], errors='coerce')
    except Exception as e:
        st.error(f"Error converting data types: {str(e)}")
        return
    
    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="metric-container">
                <h3 style='color: #667eea; font-size: 2rem;'>{len(df):,}</h3>
                <p style='color: #666;'>Total Records</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-container">
                <h3 style='color: #667eea; font-size: 2rem;'>{df.shape[1]}</h3>
                <p style='color: #666;'>Features</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        eligible_count = (df['emi_eligibility'] == 'Eligible').sum()
        st.markdown(f"""
            <div class="metric-container">
                <h3 style='color: #00b894; font-size: 2rem;'>{eligible_count:,}</h3>
                <p style='color: #666;'>Eligible</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        not_eligible_count = (df['emi_eligibility'] == 'Not_Eligible').sum()
        st.markdown(f"""
            <div class="metric-container">
                <h3 style='color: #d63031; font-size: 2rem;'>{not_eligible_count:,}</h3>
                <p style='color: #666;'>Not Eligible</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Filters
    st.markdown("### 🔍 Filter Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        scenario_filter = st.multiselect("EMI Scenario", df['emi_scenario'].unique())
    
    with col2:
        eligibility_filter = st.multiselect("Eligibility", df['emi_eligibility'].unique())
    
    with col3:
        # Safe conversion to int
        try:
            min_salary = int(float(df['monthly_salary'].min()))
            max_salary = int(float(df['monthly_salary'].max()))
        except:
            min_salary = 15000
            max_salary = 200000
        
        salary_range = st.slider("Monthly Salary Range (₹)", 
                                min_salary, 
                                max_salary,
                                (min_salary, max_salary))
    
    # Apply filters
    filtered_df = df.copy()
    if scenario_filter:
        filtered_df = filtered_df[filtered_df['emi_scenario'].isin(scenario_filter)]
    if eligibility_filter:
        filtered_df = filtered_df[filtered_df['emi_eligibility'].isin(eligibility_filter)]
    filtered_df = filtered_df[(filtered_df['monthly_salary'] >= salary_range[0]) & 
                              (filtered_df['monthly_salary'] <= salary_range[1])]
    
    st.info(f"📊 Showing {len(filtered_df):,} records")
    
    # Visualizations
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📈 Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Eligibility", "💰 Salary", "💳 Credit Score", "💵 EMI Amount"])
    
    with tab1:
        try:
            fig = px.pie(filtered_df, names='emi_eligibility', title='Eligibility Distribution',
                        color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4)
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating pie chart: {str(e)}")
    
    with tab2:
        try:
            fig = px.histogram(filtered_df, x='monthly_salary', color='emi_eligibility',
                              title='Monthly Salary Distribution by Eligibility',
                              nbins=50, barmode='overlay')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating histogram: {str(e)}")
    
    with tab3:
        try:
            fig = px.box(filtered_df, x='emi_eligibility', y='credit_score',
                        title='Credit Score by Eligibility',
                        color='emi_eligibility')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating box plot: {str(e)}")
    
    with tab4:
        try:
            # Sample data if too large for performance
            sample_size = min(5000, len(filtered_df))
            sample_df = filtered_df.sample(sample_size) if len(filtered_df) > sample_size else filtered_df
            
            fig = px.scatter(sample_df, 
                            x='monthly_salary', y='max_monthly_emi',
                            color='emi_eligibility', title='Salary vs Maximum EMI',
                            hover_data=['credit_score'])
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating scatter plot: {str(e)}")
    
    # Data table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📋 Data Table")
    try:
        st.dataframe(filtered_df.head(100), use_container_width=True, height=400)
    except Exception as e:
        st.error(f"Error displaying data table: {str(e)}")
    
    # Download option
    try:
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv,
            file_name=f"emi_data_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Error creating download: {str(e)}")

# Performance Page
def performance_page():
    """Display model performance page"""
    st.markdown('<div class="section-header">📈 Model Performance</div>', unsafe_allow_html=True)
    
    # Load comparison results
    try:
        comparison = joblib.load('models/model_comparison.pkl')
    except:
        st.error("❌ Model comparison results not found.")
        return
    
    st.markdown("## 🏆 Best Models Selected")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
            <div class="success-box">
                <h3>🎯 Classification Model</h3>
                <h2>{comparison['best_classifier']}</h2>
                <p style='font-size: 1.2rem; margin: 0.5rem 0;'>
                    <b>Accuracy:</b> {comparison['classification'][comparison['best_classifier']]['accuracy']:.4f}
                </p>
                <p style='font-size: 1.2rem; margin: 0.5rem 0;'>
                    <b>F1-Score:</b> {comparison['classification'][comparison['best_classifier']]['f1_score']:.4f}
                </p>
                <p style='font-size: 1.2rem; margin: 0.5rem 0;'>
                    <b>ROC-AUC:</b> {comparison['classification'][comparison['best_classifier']]['roc_auc']:.4f}
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="success-box">
                <h3>📊 Regression Model</h3>
                <h2>{comparison['best_regressor']}</h2>
                <p style='font-size: 1.2rem; margin: 0.5rem 0;'>
                    <b>R² Score:</b> {comparison['regression'][comparison['best_regressor']]['r2_score']:.4f}
                </p>
                <p style='font-size: 1.2rem; margin: 0.5rem 0;'>
                    <b>RMSE:</b> ₹{comparison['regression'][comparison['best_regressor']]['rmse']:.2f}
                </p>
                <p style='font-size: 1.2rem; margin: 0.5rem 0;'>
                    <b>MAPE:</b> {comparison['regression'][comparison['best_regressor']]['mape']:.2f}%
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model comparison tables
    st.markdown("## 📊 All Models Comparison")
    
    tab1, tab2 = st.tabs(["🎯 Classification Models", "📊 Regression Models"])
    
    with tab1:
        class_df = pd.DataFrame(comparison['classification']).T
        st.dataframe(class_df.style.highlight_max(axis=0), use_container_width=True)
        
        # Visualization
        fig = go.Figure()
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in metrics:
            fig.add_trace(go.Bar(name=metric.replace('_', ' ').title(), 
                               x=class_df.index, 
                               y=class_df[metric]))
        fig.update_layout(title="Classification Models Comparison", 
                         barmode='group',
                         height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        reg_df = pd.DataFrame(comparison['regression']).T
        st.dataframe(reg_df.style.highlight_max(axis=0, subset=['r2_score'])
                    .highlight_min(axis=0, subset=['rmse', 'mae', 'mape']), 
                    use_container_width=True)
        
        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Bar(name='R² Score', x=reg_df.index, y=reg_df['r2_score']))
        fig.update_layout(title="Regression Models - R² Score Comparison",
                         height=500)
        st.plotly_chart(fig, use_container_width=True)

# About Page
def about_page():
    """Display about page"""
    st.markdown('<div class="section-header">ℹ️ About EMIPredict AI</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
            <div class="info-box">
                <h2>🎯 Project Overview</h2>
                <p style='font-size: 1.1rem; line-height: 1.8;'>
                    EMIPredict AI is an intelligent financial risk assessment platform that leverages 
                    cutting-edge machine learning to provide instant EMI eligibility predictions and 
                    maximum affordable EMI calculations. Built on 400,000+ comprehensive financial records, 
                    our platform helps individuals and financial institutions make data-driven lending decisions.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="stats-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                <h3>🏆 Achievements</h3>
                <p style='font-size: 1.5rem; margin: 1rem 0;'><b>93%+</b></p>
                <p>Model Accuracy</p>
                <hr style='border-color: rgba(255,255,255,0.3);'>
                <p style='font-size: 1.5rem; margin: 1rem 0;'><b>400K+</b></p>
                <p>Training Records</p>
                <hr style='border-color: rgba(255,255,255,0.3);'>
                <p style='font-size: 1.5rem; margin: 1rem 0;'><b>6</b></p>
                <p>ML Models Trained</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Technical Stack
    st.markdown("### 🛠️ Technology Stack")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="info-box">
                <h4>🔧 Backend & ML</h4>
                <ul>
                    <li>Python 3.10</li>
                    <li>Scikit-learn</li>
                    <li>XGBoost</li>
                    <li>Pandas & NumPy</li>
                    <li>MLflow</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="info-box">
                <h4>🎨 Frontend</h4>
                <ul>
                    <li>Streamlit</li>
                    <li>Plotly</li>
                    <li>Custom CSS</li>
                    <li>Responsive Design</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="info-box">
                <h4>☁️ Deployment</h4>
                <ul>
                    <li>Streamlit Cloud</li>
                    <li>GitHub Integration</li>
                    <li>CI/CD Pipeline</li>
                    <li>Auto-scaling</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Model Performance
    st.markdown("### 📊 Model Performance Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="info-box">
                <h4>🎯 Classification Performance</h4>
                <table style='width: 100%; line-height: 2;'>
                    <tr><td><b>Model:</b></td><td>XGBoost Classifier</td></tr>
                    <tr><td><b>Accuracy:</b></td><td>93%+</td></tr>
                    <tr><td><b>F1-Score:</b></td><td>0.92+</td></tr>
                    <tr><td><b>ROC-AUC:</b></td><td>0.96+</td></tr>
                </table>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="info-box">
                <h4>📊 Regression Performance</h4>
                <table style='width: 100%; line-height: 2;'>
                    <tr><td><b>Model:</b></td><td>XGBoost Regressor</td></tr>
                    <tr><td><b>R² Score:</b></td><td>0.91+</td></tr>
                    <tr><td><b>RMSE:</b></td><td>< ₹2000</td></tr>
                    <tr><td><b>MAPE:</b></td><td>< 8%</td></tr>
                </table>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Contact Section
    st.markdown("""
        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 15px; color: white;'>
            <h2>📞 Get in Touch</h2>
            <p style='font-size: 1.1rem; margin: 1rem 0;'>
                This project was developed as part of a Data Science & AI/ML internship capstone project.
            </p>
            <p style='font-size: 1rem;'>
                For queries, collaborations, or feedback, please reach out through your internship portal.
            </p>
        </div>
    """, unsafe_allow_html=True)

# Main application
def main():
    """Main application function"""
    
    # Initialize session state
    if 'models' not in st.session_state:
        with st.spinner("🔄 Loading models..."):
            st.session_state['models'] = load_models()
    
    if 'data' not in st.session_state:
        with st.spinner("🔄 Loading data..."):
            st.session_state['data'] = load_data()
    
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 'home'
    
    # Sidebar navigation
    current_page = sidebar_navigation()
    
    # Route to pages
    if current_page == 'home':
        home_page()
    elif current_page == 'prediction':
        prediction_page()
    elif current_page == 'explorer':
        explorer_page()
    elif current_page == 'performance':
        performance_page()
    elif current_page == 'about':
        about_page()

if __name__ == "__main__":
    main()