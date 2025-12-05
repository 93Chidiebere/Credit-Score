"""
Credit Scoring Application with TCGM
Real-time Prediction Interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="Credit Scoring System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .approved {
        color: #28a745;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .rejected {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .warning {
        color: #ffc107;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model_artifacts():
    """Load trained model and preprocessing artifacts"""
    try:
        model = joblib.load('tcgm_credit_scoring_model.pkl')
        scaler = joblib.load('scaler.pkl')
        
        # # Load feature names
        # with open('feature_names.txt', 'r') as f:
        #     feature_names = f.read().splitlines()
        
        return model, scaler #feature_names
    except FileNotFoundError as e:
        st.error(f"⚠️ Model files not found: {e}")
        st.info("Please ensure model files are in the same directory:\n- tcgm_credit_scoring_model.pkl\n- scaler.pkl\n- feature_names.txt")
        return None, None, None

# Feature engineering function
def engineer_features(data):
    """Apply feature engineering to input data"""
    df = data.copy()
    
    # Income per dependent
    df['IncomePerDependent'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)
    
    # Total delinquencies
    df['TotalDelinquencies'] = (
        df['NumberOfTime30-59DaysPastDueNotWorse'] + 
        df['NumberOfTime60-89DaysPastDueNotWorse'] + 
        df['NumberOfTimes90DaysLate']
    )
    
    # Credit utilization squared
    df['RevolvingUtilization_Squared'] = df['RevolvingUtilizationOfUnsecuredLines'] ** 2
    
    # Debt to income ratio
    df['DebtToIncomeRatio'] = df['DebtRatio'] * df['MonthlyIncome']
    
    # Has delinquency flag
    df['HasDelinquency'] = (df['TotalDelinquencies'] > 0).astype(int)
    
    return df

# Calculate financial metrics
def calculate_financial_metrics(probability, monthly_income, optimal_threshold=0.35):
    """Calculate financial exposure and risk metrics"""
    exposure = monthly_income * 3  # 3 months of income
    lgd = 0.60  # Loss Given Default
    
    # Expected loss if approved
    expected_loss = probability * exposure * lgd
    
    # Decision based on threshold
    decision = "APPROVED" if probability < optimal_threshold else "REJECTED"
    
    # Risk category
    if probability < 0.15:
        risk_category = "Low Risk"
        risk_color = "green"
    elif probability < 0.35:
        risk_category = "Medium Risk"
        risk_color = "orange"
    else:
        risk_category = "High Risk"
        risk_color = "red"
    
    return {
        'decision': decision,
        'exposure': exposure,
        'expected_loss': expected_loss,
        'risk_category': risk_category,
        'risk_color': risk_color
    }

# Main app
def main():
    # Header
    st.markdown('<p class="main-header">🕒💰 Credit Scoring System</p>', unsafe_allow_html=True)
    st.markdown("### Powered by TimeCost Gradient Machine (TCGM)")
    st.markdown("---")
    
    # Load model
    model, scaler, feature_names = load_model_artifacts()
    
    if model is None:
        st.stop()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model parameters
        st.subheader("Model Parameters")
        optimal_threshold = st.slider(
            "Decision Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.01,
            help="Probability threshold for loan approval"
        )
        
        cost_fp = st.number_input(
            "Cost of False Positive ($)",
            min_value=0,
            value=500,
            step=50,
            help="Cost of approving a bad loan"
        )
        
        cost_fn = st.number_input(
            "Cost of False Negative ($)",
            min_value=0,
            value=50,
            step=10,
            help="Cost of rejecting a good customer"
        )
        
        st.markdown("---")
        
        # Input mode selection
        st.subheader("Input Mode")
        input_mode = st.radio(
            "Choose input method:",
            ["Manual Entry", "Batch Upload", "Quick Test"]
        )
        
        st.markdown("---")
        st.info(f"**Model Version:** TCGM v1.0\n\n**Last Updated:** {datetime.now().strftime('%Y-%m-%d')}")
    
    # Main content area
    if input_mode == "Manual Entry":
        manual_entry_mode(model, scaler, feature_names, optimal_threshold)
    
    elif input_mode == "Batch Upload":
        batch_upload_mode(model, scaler, feature_names, optimal_threshold)
    
    elif input_mode == "Quick Test":
        quick_test_mode(model, scaler, feature_names, optimal_threshold)

def manual_entry_mode(model, scaler, feature_names, optimal_threshold):
    """Manual entry interface for single predictions"""
    st.header("📝 Manual Credit Application Entry")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35,
            help="Applicant's age in years"
        )
        
        monthly_income = st.number_input(
            "Monthly Income ($)",
            min_value=0,
            value=5000,
            step=100,
            help="Gross monthly income"
        )
        
        num_dependents = st.number_input(
            "Number of Dependents",
            min_value=0,
            max_value=20,
            value=0,
            help="Number of financial dependents"
        )
        
        st.subheader("Credit Profile")
        
        revolving_util = st.slider(
            "Revolving Utilization Rate",
            min_value=0.0,
            max_value=2.0,
            value=0.3,
            step=0.01,
            help="Total balance on credit cards / credit limits"
        )
        
        debt_ratio = st.slider(
            "Debt Ratio",
            min_value=0.0,
            max_value=5.0,
            value=0.5,
            step=0.01,
            help="Monthly debt payments / gross monthly income"
        )
    
    with col2:
        st.subheader("Credit History")
        
        num_open_credit = st.number_input(
            "Open Credit Lines & Loans",
            min_value=0,
            max_value=50,
            value=5,
            help="Number of open loans and credit lines"
        )
        
        num_real_estate = st.number_input(
            "Real Estate Loans",
            min_value=0,
            max_value=20,
            value=1,
            help="Number of mortgage and real estate loans"
        )
        
        st.subheader("Delinquency History")
        
        times_30_59_late = st.number_input(
            "30-59 Days Past Due",
            min_value=0,
            max_value=20,
            value=0,
            help="Number of times 30-59 days late in last 2 years"
        )
        
        times_60_89_late = st.number_input(
            "60-89 Days Past Due",
            min_value=0,
            max_value=20,
            value=0,
            help="Number of times 60-89 days late in last 2 years"
        )
        
        times_90_late = st.number_input(
            "90+ Days Past Due",
            min_value=0,
            max_value=20,
            value=0,
            help="Number of times 90+ days late in last 2 years"
        )
    
    st.markdown("---")
    
    # Predict button
    if st.button("🔮 Predict Credit Risk", type="primary", use_container_width=True):
        # Create input dataframe
        input_data = pd.DataFrame({
            'RevolvingUtilizationOfUnsecuredLines': [revolving_util],
            'age': [age],
            'NumberOfTime30-59DaysPastDueNotWorse': [times_30_59_late],
            'DebtRatio': [debt_ratio],
            'MonthlyIncome': [monthly_income],
            'NumberOfOpenCreditLinesAndLoans': [num_open_credit],
            'NumberOfTimes90DaysLate': [times_90_late],
            'NumberRealEstateLoansOrLines': [num_real_estate],
            'NumberOfTime60-89DaysPastDueNotWorse': [times_60_89_late],
            'NumberOfDependents': [num_dependents]
        })
        
        # Engineer features
        input_engineered = engineer_features(input_data)
        
        # Ensure correct feature order
        input_engineered = input_engineered[feature_names]
        
        # Scale features
        input_scaled = scaler.transform(input_engineered)
        
        # Make prediction
        probability = model.predict_proba(input_scaled)[0, 1]
        
        # Calculate financial metrics
        metrics = calculate_financial_metrics(probability, monthly_income, optimal_threshold)
        
        # Display results
        display_prediction_results(probability, metrics, monthly_income)

def batch_upload_mode(model, scaler, feature_names, optimal_threshold):
    """Batch upload interface for multiple predictions"""
    st.header("📤 Batch Upload for Multiple Predictions")
    
    st.info("Upload a CSV file with applicant data. The file should contain the following columns:\n"
            "- RevolvingUtilizationOfUnsecuredLines\n"
            "- age\n"
            "- NumberOfTime30-59DaysPastDueNotWorse\n"
            "- DebtRatio\n"
            "- MonthlyIncome\n"
            "- NumberOfOpenCreditLinesAndLoans\n"
            "- NumberOfTimes90DaysLate\n"
            "- NumberRealEstateLoansOrLines\n"
            "- NumberOfTime60-89DaysPastDueNotWorse\n"
            "- NumberOfDependents")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Loaded {len(df)} applications")
            
            # Show preview
            with st.expander("📋 Data Preview"):
                st.dataframe(df.head(10))
            
            if st.button("🚀 Process All Applications", type="primary"):
                # Engineer features
                df_engineered = engineer_features(df)
                
                # Ensure correct feature order
                df_engineered = df_engineered[feature_names]
                
                # Scale features
                df_scaled = scaler.transform(df_engineered)
                
                # Make predictions
                probabilities = model.predict_proba(df_scaled)[:, 1]
                
                # Add predictions to dataframe
                df['DefaultProbability'] = probabilities
                df['Decision'] = ['APPROVED' if p < optimal_threshold else 'REJECTED' for p in probabilities]
                df['RiskCategory'] = pd.cut(probabilities, 
                                           bins=[0, 0.15, 0.35, 1.0],
                                           labels=['Low Risk', 'Medium Risk', 'High Risk'])
                df['ExpectedLoss'] = probabilities * df['MonthlyIncome'] * 3 * 0.60
                
                # Display results
                st.subheader("📊 Batch Results Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Applications", len(df))
                
                with col2:
                    approved = (df['Decision'] == 'APPROVED').sum()
                    st.metric("Approved", approved, delta=f"{approved/len(df)*100:.1f}%")
                
                with col3:
                    rejected = (df['Decision'] == 'REJECTED').sum()
                    st.metric("Rejected", rejected, delta=f"{rejected/len(df)*100:.1f}%")
                
                with col4:
                    avg_prob = df['DefaultProbability'].mean()
                    st.metric("Avg Default Prob", f"{avg_prob:.1%}")
                
                # Visualizations
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Decision distribution
                    fig_decision = px.pie(df, names='Decision', title='Approval Distribution',
                                         color='Decision',
                                         color_discrete_map={'APPROVED': 'green', 'REJECTED': 'red'})
                    st.plotly_chart(fig_decision, use_container_width=True)
                
                with col2:
                    # Risk category distribution
                    fig_risk = px.histogram(df, x='RiskCategory', title='Risk Category Distribution',
                                           color='RiskCategory',
                                           color_discrete_map={'Low Risk': 'green', 
                                                             'Medium Risk': 'orange', 
                                                             'High Risk': 'red'})
                    st.plotly_chart(fig_risk, use_container_width=True)
                
                # Probability distribution
                fig_prob = px.histogram(df, x='DefaultProbability', 
                                       title='Default Probability Distribution',
                                       nbins=50)
                fig_prob.add_vline(x=optimal_threshold, line_dash="dash", 
                                  line_color="red", annotation_text="Threshold")
                st.plotly_chart(fig_prob, use_container_width=True)
                
                # Show detailed results
                st.subheader("📋 Detailed Results")
                st.dataframe(df[['age', 'MonthlyIncome', 'DefaultProbability', 
                                'Decision', 'RiskCategory', 'ExpectedLoss']])
                
                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"credit_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

def quick_test_mode(model, scaler, feature_names, optimal_threshold):
    """Quick test with preset scenarios"""
    st.header("⚡ Quick Test with Preset Scenarios")
    
    scenarios = {
        "Excellent Credit - Low Risk": {
            'RevolvingUtilizationOfUnsecuredLines': 0.1,
            'age': 45,
            'NumberOfTime30-59DaysPastDueNotWorse': 0,
            'DebtRatio': 0.2,
            'MonthlyIncome': 8000,
            'NumberOfOpenCreditLinesAndLoans': 10,
            'NumberOfTimes90DaysLate': 0,
            'NumberRealEstateLoansOrLines': 2,
            'NumberOfTime60-89DaysPastDueNotWorse': 0,
            'NumberOfDependents': 1
        },
        "Average Credit - Medium Risk": {
            'RevolvingUtilizationOfUnsecuredLines': 0.5,
            'age': 35,
            'NumberOfTime30-59DaysPastDueNotWorse': 1,
            'DebtRatio': 0.4,
            'MonthlyIncome': 5000,
            'NumberOfOpenCreditLinesAndLoans': 7,
            'NumberOfTimes90DaysLate': 0,
            'NumberRealEstateLoansOrLines': 1,
            'NumberOfTime60-89DaysPastDueNotWorse': 0,
            'NumberOfDependents': 2
        },
        "Poor Credit - High Risk": {
            'RevolvingUtilizationOfUnsecuredLines': 1.2,
            'age': 28,
            'NumberOfTime30-59DaysPastDueNotWorse': 3,
            'DebtRatio': 1.5,
            'MonthlyIncome': 3000,
            'NumberOfOpenCreditLinesAndLoans': 15,
            'NumberOfTimes90DaysLate': 2,
            'NumberRealEstateLoansOrLines': 0,
            'NumberOfTime60-89DaysPastDueNotWorse': 1,
            'NumberOfDependents': 3
        }
    }
    
    selected_scenario = st.selectbox("Choose a test scenario:", list(scenarios.keys()))
    
    if st.button("🧪 Run Test Scenario", type="primary", use_container_width=True):
        # Get scenario data
        scenario_data = scenarios[selected_scenario]
        
        # Create dataframe
        input_data = pd.DataFrame([scenario_data])
        
        # Engineer features
        input_engineered = engineer_features(input_data)
        
        # Ensure correct feature order
        input_engineered = input_engineered[feature_names]
        
        # Scale features
        input_scaled = scaler.transform(input_engineered)
        
        # Make prediction
        probability = model.predict_proba(input_scaled)[0, 1]
        
        # Calculate financial metrics
        metrics = calculate_financial_metrics(probability, scenario_data['MonthlyIncome'], optimal_threshold)
        
        # Display results
        display_prediction_results(probability, metrics, scenario_data['MonthlyIncome'])
        
        # Show input features
        with st.expander("📊 View Scenario Details"):
            st.json(scenario_data)

def display_prediction_results(probability, metrics, monthly_income):
    """Display prediction results with visualizations"""
    st.markdown("---")
    st.header("🎯 Prediction Results")
    
    # Main decision display
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.markdown("### Decision")
        if metrics['decision'] == "APPROVED":
            st.markdown(f'<p class="approved">✅ {metrics["decision"]}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="rejected">❌ {metrics["decision"]}</p>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Risk Level")
        st.markdown(f'<p style="color:{metrics["risk_color"]};font-size:1.5rem;font-weight:bold">{metrics["risk_category"]}</p>', 
                   unsafe_allow_html=True)
    
    with col3:
        st.markdown("### Default Probability")
        st.markdown(f'<p style="font-size:2rem;font-weight:bold;color:#1f77b4">{probability:.1%}</p>', 
                   unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Financial metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Credit Exposure",
            f"${metrics['exposure']:,.2f}",
            help="3 months of monthly income"
        )
    
    with col2:
        st.metric(
            "Expected Loss",
            f"${metrics['expected_loss']:,.2f}",
            help="Probability × Exposure × LGD (60%)"
        )
    
    with col3:
        st.metric(
            "Monthly Income",
            f"${monthly_income:,.2f}"
        )
    
    # Probability gauge chart
    st.markdown("---")
    st.subheader("📈 Risk Visualization")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Default Probability (%)", 'font': {'size': 24}},
        delta={'reference': 35, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 15], 'color': 'lightgreen'},
                {'range': [15, 35], 'color': 'yellow'},
                {'range': [35, 100], 'color': 'lightcoral'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 35
            }
        }
    ))
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendation
    st.markdown("---")
    st.subheader("💡 Recommendation")
    
    if probability < 0.15:
        st.success("""
        **Low Risk Applicant**
        - Strong credit profile
        - Minimal delinquency history
        - Recommend approval with standard terms
        """)
    elif probability < 0.35:
        st.warning("""
        **Medium Risk Applicant**
        - Acceptable credit profile
        - Some concerns in credit history
        - Consider approval with adjusted terms or higher interest rate
        - May require additional documentation
        """)
    else:
        st.error("""
        **High Risk Applicant**
        - Weak credit profile
        - Significant delinquency concerns
        - Recommend rejection or require substantial collateral
        - Consider financial counseling referral
        """)

# Run the app
if __name__ == "__main__":
    main()