# 🕒💰 Credit Scoring System with TCGM

A real-time credit scoring application powered by **TimeCost Gradient Machine (TCGM)** for intelligent loan approval decisions.

---

## 🚀 Features

### **Real-Time Predictions**
- Manual entry for single applicants
- Batch processing for multiple applications
- Quick test scenarios for demonstration

### **Financial Intelligence**
- Cost-sensitive decision making
- Expected loss calculations
- Credit exposure analysis
- Risk categorization (Low/Medium/High)

### **Interactive Visualizations**
- Default probability gauge
- Risk distribution charts
- Approval/rejection analytics
- Batch processing dashboards

### **Production-Ready**
- TCGM algorithm optimized for financial predictions
- Configurable decision thresholds
- CSV batch upload/download
- Audit trail ready

## 🎯 Quick Start

### **Run the Application**
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

---

## 📖 User Guide

### **1. Manual Entry Mode**

**Use Case:** Score individual loan applications

**Steps:**
1. Select "Manual Entry" in sidebar
2. Fill in applicant information:
   - **Personal Info:** Age, Income, Dependents
   - **Credit Profile:** Utilization rate, Debt ratio
   - **Credit History:** Open accounts, Real estate loans
   - **Delinquency:** Past due records
3. Click "🔮 Predict Credit Risk"
4. Review results and recommendation

**Output:**
- Approval/Rejection decision
- Default probability
- Risk category
- Expected loss
- Detailed recommendation


**Batch Results Include:**
- Approval summary statistics
- Risk distribution charts
- Probability histograms
- Downloadable results CSV

---

---

## ⚙️ Configuration

### **Sidebar Settings**

#### **Decision Threshold** (Default: 0.35)
- Range: 0.0 to 1.0
- Lower = More approvals (higher risk)
- Higher = Fewer approvals (lower risk)
- **Recommendation:** 0.30 - 0.40 for balanced decisions

#### **Cost of False Positive** (Default: $500)
- Cost of approving a defaulter
- Reflects actual loan loss

#### **Cost of False Negative** (Default: $50)
- Cost of rejecting a good customer
- Opportunity cost

---

## 📊 Understanding the Results

### **Decision Categories**

| Default Probability | Risk Level | Recommendation |
|---------------------|-----------|----------------|
| < 15% | **Low Risk** 🟢 | Approve with standard terms |
| 15% - 35% | **Medium Risk** 🟡 | Approve with adjusted terms |
| > 35% | **High Risk** 🔴 | Reject or require collateral |

### **Key Metrics Explained**

**Default Probability**
- Likelihood of 90+ days delinquency in next 2 years
- Based on TCGM's cost-sensitive predictions

**Credit Exposure**
- Estimated loan amount (3 months of income)
- Used for financial loss calculation

**Expected Loss**
- Probability × Exposure × LGD (60%)
- Monetary risk if loan is approved

**Risk Category**
- Low/Medium/High based on probability thresholds
- Color-coded for quick identification

## 🛠️ Troubleshooting

### **Common Issues**

#### **1. Model Files Not Found**
```
Error: FileNotFoundError: tcgm_credit_scoring_model.pkl
```
**Solution:** Ensure model files are in the same directory as `app.py`

#### **2. Feature Mismatch**
```
Error: ValueError: feature mismatch
```
**Solution:** Verify CSV columns match required features exactly

#### **3. TCGM Import Error**
```
Error: ModuleNotFoundError: No module named 'tcgm'
```
**Solution:** Install TCGM
```bash
pip install tcgm==0.1.3
```

#### **4. Port Already in Use**
```
Error: Port 8501 is already in use
```
**Solution:** Use a different port
```bash
streamlit run app.py --server.port 8502
```

---

## 📦 Project Structure

```
credit-scoring-app/
│
├── app.py                              # Main Streamlit application
├── README.md                           # Documentation
├── requirements.txt                    # Python dependencies
│
├── models/                             # Model artifacts
│   ├── tcgm_credit_scoring_model.pkl
│   ├── scaler.pkl
│   └── feature_names.txt
│
├── data/                               # Sample data
│   ├── sample_batch.csv
│   └── test_scenarios.json
│
└── notebooks/                          # Development notebooks
    ├── 01_EDA.ipynb
    └── 02_Model_Training.ipynb
```

---

## 🔄 Model Updates

### **Retraining the Model**

When you retrain your TCGM model:

1. **Save new artifacts:**
```python
import joblib

# Save model
joblib.dump(tcgm_model, 'tcgm_credit_scoring_model.pkl')

# Save scaler
joblib.dump(scaler, 'scaler.pkl')

# Save feature names
with open('feature_names.txt', 'w') as f:
    f.write('\n'.join(feature_names))
```

2. **Replace old files** in app directory

3. **Restart Streamlit** (it will auto-reload)

4. **Verify** with quick test scenarios

---

## 📈 Performance Metrics

### **Model Performance**
- **AUC:** 0.854 (Excellent discrimination)
- **Brier Score:** 0.067 (Well-calibrated probabilities)
- **Expected Loss:** Optimized through TCGM

---

## 🎓 About TCGM

**TimeCost Gradient Machine** is a specialized ML algorithm for financial predictions that:

✅ **Optimizes for monetary loss**, not just accuracy
✅ **Handles asymmetric costs** (FP ≠ FN)
✅ **Time-aware** for financial drift
✅ **Built-in boosting** without external frameworks
✅ **Regulatory-compliant** with interpretability

**Developed by:** Chidiebere V. Christopher  
**Learn More:** [LinkedIn Profile](https://www.linkedin.com/in/chidiebere-christopher/)

---

---

## 🚀 Deployment Options

### **1. Local Deployment**
```bash
streamlit run app.py
```

### **2. Streamlit Cloud (Free)**
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Deploy!


### **Contributing**
Pull requests welcome! Please:
1. Fork the repository
2. Create feature branch
3. Submit PR with description

---

---

## 🙏 Acknowledgments

- **TCGM Algorithm:** Chidiebere V. Christopher
- **Dataset:** Kaggle "Give Me Some Credit" Competition
- **Framework:** Streamlit for rapid deployment
- **Visualization:** Plotly for interactive charts

---

## 📚 Additional Resources

- [TCGM Documentation](https://pypi.org/project/tcgm/)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Credit Scoring Best Practices](https://www.federalreserve.gov/creditscoring)
- [Fair Lending Compliance](https://www.consumerfinance.gov/fair-lending/)


**Built with ❤️ using TimeCost Gradient Machine**

*Last Updated: December 5, 2025*
