"""
Comprehensive EDA for EMI Dataset with Data Cleaning
Save as: 01_EDA_Analysis.py
Run: python 01_EDA_Analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("🔍 EMI DATASET - EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n📂 Loading dataset...")
df = pd.read_csv('data/EMI_dataset.csv')
print(f"✅ Initial load: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ============================================================================
# 2. DATA CLEANING & TYPE CONVERSION
# ============================================================================
print("\n" + "=" * 80)
print("🧹 DATA CLEANING & TYPE CONVERSION")
print("=" * 80)

# Display initial data types
print("\n📋 Initial Column Data Types:")
print(df.dtypes)

# List of columns that should be numeric
numeric_columns = [
    'age', 'monthly_salary', 'years_of_employment', 'monthly_rent',
    'family_size', 'dependents', 'school_fees', 'college_fees',
    'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
    'current_emi_amount', 'credit_score', 'bank_balance', 'emergency_fund',
    'requested_amount', 'requested_tenure', 'max_monthly_emi'
]

# Convert to numeric, forcing errors to NaN
print("\n🔄 Converting columns to numeric types...")
for col in numeric_columns:
    if col in df.columns:
        # Convert to numeric, non-numeric values become NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Check for NaN values created
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            print(f"   ⚠️  {col}: {nan_count} non-numeric values found and converted to NaN")
            # Fill NaN with median for that column
            df[col].fillna(df[col].median(), inplace=True)
            print(f"      ✓ Filled with median: {df[col].median():.2f}")

# Convert categorical columns to string type
categorical_columns = [
    'gender', 'marital_status', 'education', 'employment_type',
    'company_type', 'house_type', 'existing_loans', 'emi_scenario',
    'emi_eligibility'
]

print("\n🔄 Ensuring categorical columns are strings...")
for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

print("\n✅ Data types converted successfully!")

# ============================================================================
# 3. DATA QUALITY CHECK
# ============================================================================
print("\n" + "=" * 80)
print("🔎 DATA QUALITY ASSESSMENT")
print("=" * 80)

# Missing values check
print("\n🔍 Missing Values Check:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("   ✅ No missing values found!")
else:
    print("   ⚠️  Found missing values:")
    print(missing[missing > 0])

# Duplicates check
duplicates = df.duplicated().sum()
print(f"\n🔁 Duplicate Records: {duplicates}")
if duplicates > 0:
    print(f"   ⚠️  Removing {duplicates} duplicates...")
    df = df.drop_duplicates()
    print(f"   ✅ Cleaned dataset: {df.shape[0]:,} rows")

# Final dataset info
print(f"\n📊 Final Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ============================================================================
# 4. BASIC INFORMATION
# ============================================================================
print("\n" + "=" * 80)
print("📊 DATASET OVERVIEW")
print("=" * 80)

print("\n📋 Column Names:")
print(df.columns.tolist())

print("\n📈 Statistical Summary (Numeric Columns):")
print(df.describe())

print("\n🔍 First 5 Rows:")
print(df.head())

# ============================================================================
# 5. TARGET VARIABLE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("🎯 TARGET VARIABLES ANALYSIS")
print("=" * 80)

# Classification Target
print("\n📊 Classification Target (emi_eligibility):")
eligibility_counts = df['emi_eligibility'].value_counts()
print(eligibility_counts)
print("\nPercentage Distribution:")
eligibility_pct = (df['emi_eligibility'].value_counts(normalize=True) * 100).round(2)
print(eligibility_pct)

# Regression Target
print("\n💰 Regression Target (max_monthly_emi):")
print(f"   Mean: ₹{df['max_monthly_emi'].mean():,.2f}")
print(f"   Median: ₹{df['max_monthly_emi'].median():,.2f}")
print(f"   Std Dev: ₹{df['max_monthly_emi'].std():,.2f}")
print(f"   Min: ₹{df['max_monthly_emi'].min():,.0f}")
print(f"   Max: ₹{df['max_monthly_emi'].max():,.0f}")

# ============================================================================
# 6. DEMOGRAPHIC ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("👥 DEMOGRAPHIC ANALYSIS")
print("=" * 80)

print("\n🎂 Age Distribution:")
print(f"   Mean Age: {df['age'].mean():.1f} years")
print(f"   Median Age: {df['age'].median():.1f} years")
print(f"   Age Range: {df['age'].min():.0f} - {df['age'].max():.0f} years")

print("\n🚻 Gender Distribution:")
print(df['gender'].value_counts())

print("\n💑 Marital Status:")
print(df['marital_status'].value_counts())

print("\n🎓 Education Levels:")
print(df['education'].value_counts())

# ============================================================================
# 7. FINANCIAL ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("💰 FINANCIAL METRICS ANALYSIS")
print("=" * 80)

print("\n💵 Monthly Salary:")
print(f"   Mean: ₹{df['monthly_salary'].mean():,.2f}")
print(f"   Median: ₹{df['monthly_salary'].median():,.2f}")
print(f"   Range: ₹{df['monthly_salary'].min():,.0f} - ₹{df['monthly_salary'].max():,.0f}")

print("\n💳 Credit Score:")
print(f"   Mean: {df['credit_score'].mean():.0f}")
print(f"   Median: {df['credit_score'].median():.0f}")
print(f"   Range: {df['credit_score'].min():.0f} - {df['credit_score'].max():.0f}")

# Credit score categories
try:
    credit_categories = pd.cut(df['credit_score'], 
                              bins=[300, 550, 650, 750, 850],
                              labels=['Poor (300-550)', 'Fair (550-650)', 
                                     'Good (650-750)', 'Excellent (750-850)'])
    print("\n📊 Credit Score Categories:")
    print(credit_categories.value_counts())
except Exception as e:
    print(f"\n⚠️  Could not categorize credit scores: {str(e)}")

print("\n🏦 Existing Loans:")
print(df['existing_loans'].value_counts())

# ============================================================================
# 8. EMI SCENARIO ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("📋 EMI SCENARIO ANALYSIS")
print("=" * 80)

print("\n🎯 EMI Scenarios Distribution:")
print(df['emi_scenario'].value_counts())

print("\n💵 Statistics by Scenario:")
scenario_stats = df.groupby('emi_scenario').agg({
    'requested_amount': ['mean', 'min', 'max'],
    'requested_tenure': ['mean', 'min', 'max'],
    'max_monthly_emi': 'mean'
}).round(0)
print(scenario_stats)

# ============================================================================
# 9. CORRELATION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("🔗 CORRELATION ANALYSIS")
print("=" * 80)

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print("\n🔝 Top 10 Features Correlated with max_monthly_emi:")
correlations = df[numerical_cols].corr()['max_monthly_emi'].sort_values(ascending=False)
print(correlations.head(10))

# ============================================================================
# 10. VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("📊 GENERATING VISUALIZATIONS")
print("=" * 80)

import os
os.makedirs('visualizations', exist_ok=True)

# 1. Target Variable Distribution
try:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Classification
    eligibility_counts = df['emi_eligibility'].value_counts()
    colors = {'Eligible': '#2ecc71', 'High_Risk': '#f39c12', 'Not_Eligible': '#e74c3c'}
    plot_colors = [colors.get(x, '#95a5a6') for x in eligibility_counts.index]
    
    eligibility_counts.plot(kind='bar', ax=axes[0], color=plot_colors)
    axes[0].set_title('EMI Eligibility Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Eligibility Status')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt='%d')
    
    # Regression
    axes[1].hist(df['max_monthly_emi'].dropna(), bins=50, color='#3498db', edgecolor='black')
    axes[1].set_title('Maximum Monthly EMI Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Max Monthly EMI (₹)')
    axes[1].set_ylabel('Frequency')
    mean_emi = df['max_monthly_emi'].mean()
    axes[1].axvline(mean_emi, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: ₹{mean_emi:,.0f}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('visualizations/01_target_distributions.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: 01_target_distributions.png")
    plt.close()
except Exception as e:
    print(f"   ⚠️  Error in visualization 1: {str(e)}")

# 2. Salary by Eligibility
try:
    plt.figure(figsize=(12, 6))
    df.boxplot(column='monthly_salary', by='emi_eligibility', ax=plt.gca())
    plt.title('Monthly Salary Distribution by Eligibility Status', fontsize=14, fontweight='bold')
    plt.suptitle('')
    plt.xlabel('EMI Eligibility')
    plt.ylabel('Monthly Salary (₹)')
    plt.savefig('visualizations/02_salary_by_eligibility.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: 02_salary_by_eligibility.png")
    plt.close()
except Exception as e:
    print(f"   ⚠️  Error in visualization 2: {str(e)}")

# 3. Credit Score Analysis
try:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Box plot
    df.boxplot(column='credit_score', by='emi_eligibility', ax=axes[0])
    axes[0].set_title('Credit Score by Eligibility')
    axes[0].set_xlabel('Eligibility Status')
    axes[0].set_ylabel('Credit Score')
    
    # Histogram
    for status in df['emi_eligibility'].unique():
        subset = df[df['emi_eligibility'] == status]['credit_score'].dropna()
        axes[1].hist(subset, bins=30, alpha=0.5, label=status)
    axes[1].set_xlabel('Credit Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Credit Score Distribution by Eligibility')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('visualizations/03_credit_score_analysis.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: 03_credit_score_analysis.png")
    plt.close()
except Exception as e:
    print(f"   ⚠️  Error in visualization 3: {str(e)}")

# 4. Correlation Heatmap
try:
    top_features = ['age', 'monthly_salary', 'credit_score', 'bank_balance', 
                    'emergency_fund', 'current_emi_amount', 'requested_amount', 
                    'requested_tenure', 'max_monthly_emi']
    
    # Filter only existing columns
    top_features = [col for col in top_features if col in df.columns]
    
    plt.figure(figsize=(12, 10))
    corr_matrix = df[top_features].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
    plt.title('Correlation Heatmap - Key Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualizations/04_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: 04_correlation_heatmap.png")
    plt.close()
except Exception as e:
    print(f"   ⚠️  Error in visualization 4: {str(e)}")

# 5. EMI Scenario Analysis
try:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Average requested amount
    scenario_amounts = df.groupby('emi_scenario')['requested_amount'].mean().sort_values()
    scenario_amounts.plot(kind='barh', ax=axes[0], color='#16a085')
    axes[0].set_title('Average Requested Amount by EMI Scenario', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Average Amount (₹)')
    axes[0].set_ylabel('EMI Scenario')
    
    # Eligibility distribution
    scenario_eligibility = pd.crosstab(df['emi_scenario'], df['emi_eligibility'], 
                                       normalize='index') * 100
    scenario_eligibility.plot(kind='barh', stacked=True, ax=axes[1], 
                             color=['#2ecc71', '#f39c12', '#e74c3c'])
    axes[1].set_title('Eligibility Distribution by EMI Scenario (%)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Percentage')
    axes[1].set_ylabel('EMI Scenario')
    axes[1].legend(title='Eligibility', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    plt.savefig('visualizations/05_scenario_analysis.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: 05_scenario_analysis.png")
    plt.close()
except Exception as e:
    print(f"   ⚠️  Error in visualization 5: {str(e)}")

# 6. Age vs Salary Scatter
try:
    plt.figure(figsize=(12, 6))
    
    # Create numeric mapping for colors
    eligibility_map = {'Eligible': 0, 'High_Risk': 1, 'Not_Eligible': 2}
    colors = df['emi_eligibility'].map(eligibility_map)
    
    scatter = plt.scatter(df['age'], df['monthly_salary'], 
                         c=colors, cmap='RdYlGn_r', alpha=0.5, s=20)
    cbar = plt.colorbar(scatter, ticks=[0, 1, 2])
    cbar.set_label('Eligibility')
    cbar.ax.set_yticklabels(['Eligible', 'High Risk', 'Not Eligible'])
    
    plt.xlabel('Age (years)')
    plt.ylabel('Monthly Salary (₹)')
    plt.title('Age vs Monthly Salary (Colored by Eligibility)', fontsize=14, fontweight='bold')
    plt.savefig('visualizations/06_age_salary_scatter.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: 06_age_salary_scatter.png")
    plt.close()
except Exception as e:
    print(f"   ⚠️  Error in visualization 6: {str(e)}")

# ============================================================================
# 11. KEY INSIGHTS
# ============================================================================
print("\n" + "=" * 80)
print("💡 KEY INSIGHTS SUMMARY")
print("=" * 80)

print("\n✨ Important Findings:")
print(f"1. Dataset: {len(df):,} records × {df.shape[1]} features")
print(f"2. Data Quality: {100 - (df.isnull().sum().sum() / df.size * 100):.2f}% complete")

eligible_pct = (df['emi_eligibility'] == 'Eligible').mean() * 100
high_risk_pct = (df['emi_eligibility'] == 'High_Risk').mean() * 100
not_eligible_pct = (df['emi_eligibility'] == 'Not_Eligible').mean() * 100

print(f"\n3. Eligibility Distribution:")
print(f"   • Eligible: {eligible_pct:.1f}%")
print(f"   • High Risk: {high_risk_pct:.1f}%")
print(f"   • Not Eligible: {not_eligible_pct:.1f}%")

print(f"\n4. Financial Highlights:")
print(f"   • Average Salary: ₹{df['monthly_salary'].mean():,.0f}")
print(f"   • Average Credit Score: {df['credit_score'].mean():.0f}")
print(f"   • People with Existing Loans: {(df['existing_loans']=='Yes').mean()*100:.1f}%")

print(f"\n5. EMI Scenario Distribution:")
for scenario in df['emi_scenario'].value_counts().head().index:
    count = (df['emi_scenario'] == scenario).sum()
    print(f"   • {scenario}: {count:,} records")

print("\n" + "=" * 80)
print("✅ EDA COMPLETED SUCCESSFULLY!")
print("=" * 80)
print("\n📁 All visualizations saved in 'visualizations/' folder")
print("📊 Review the charts before proceeding to model training!")
print("=" * 80)

# Save cleaned data
df.to_csv('data/EMI_dataset_cleaned.csv', index=False)
print("\n💾 Cleaned dataset saved as: data/EMI_dataset_cleaned.csv")
print("\n🎯 Next step: Run 02_ML_Training_MLflow.py")