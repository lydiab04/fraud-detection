# Unified Fraud Detection System for E-Commerce & Banking

An end-to-end machine learning pipeline built for Adey Innovations Inc. to identify fraudulent transactions across two distinct streams: contextual e-commerce activities and anonymized bank credit card transactions. 

This project balances the financial costs of false negatives (missed fraud) against the customer experience costs of false positives (blocking legitimate users) using advanced feature engineering, resampling techniques, and SHAP explainability.

---

1. **Task 2: Modeling Layer (`notebooks/modeling.ipynb`)**
   * Implemented baseline **Logistic Regression** and **Random Forest Ensembles** using Stratified 5-Fold Cross-Validation.
   * Mitigated extreme target skewness with balanced training iterations utilizing **SMOTE**.

2. **Task 3: SHAP Explainability (`notebooks/shap-explainability.ipynb`)**
   * Integrated a Game-Theoretic interpretability layer utilizing **SHAP** to trace structural global drivers and extract localized diagnostic features.
   * Global feature impacts are rendered and archived inside `reports/figures/`.

3. **Software Engineering & Continuous Integration (`.github/workflows/unittests.yml`)**
   * Migrated functional transformations out of loose blocks into modular architecture packages inside `src/`.
   * Enforced systematic development tracking using an automated **GitHub Actions CI Test Suite Pipeline**.

---

## 📂 Repository Architecture Map
```text
fraud-detection/
├── .github/workflows/
│   └── unittests.yml             <-- FIXED: Automated CI Pipeline Configuration
├── data/
│   ├── raw/
│   └── processed/
├── src/                          <-- FIXED: Modular Production Code base
│   ├── __init__.py
│   ├── preprocessing.py
│   └── imbalance.py
├── tests/                        <-- FIXED: Automated Unit Testing Framework
│   ├── __init__.py
│   └── test_processing.py
├── models/
│   └── random_forest_fraud.pkl   <-- FIXED: Serialized Best ML Model Object
├── notebooks/
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb            <-- FIXED: Model Selection & Metrics Training
│   └── shap-explainability.ipynb <-- FIXED: SHAP XAI Visualizations & Audits
├── reports/
│   ├── figures/
│   │   └── shap_global_summary.png
│   └── Final_Fraud_Detection_Report.md <-- FIXED: Comprehensive Analysis Case-Study
└── requirements.txt
🚀 Getting Started & Environment Setup1. Clone the RepositoryBashgit clone [https://github.com/YOUR_GITHUB_USERNAME/fraud-detection.git](https://github.com/YOUR_GITHUB_USERNAME/fraud-detection.git)
cd fraud-detection
2. Set Up a Virtual EnvironmentBash# Create environment
python3 -m venv .venv

# Activate environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
3. Install DependenciesBashpip install --upgrade pip
pip install -r requirements.txt
📊 Summary of Core Insights (Task 1)Severe Class Imbalance: Verified that fraud accounts for only a minor fraction of overall data across both streams, requiring specialized evaluation metrics ($AUC\text{-}PR$ and $F_1\text{-score}$) instead of standard classification accuracy.The "Golden Hour" Vector: E-commerce fraud evaluation highlighted an intense spike in activity where time_since_signup is near zero—indicating programmatic bot creation and immediate checkouts.Spatial Risk Hubs: Geolocation profiling mapped higher statistical concentrations of transaction risk to specific international node ranges.🛠️ Tech Stack & Key LibrariesData Core: pandas, numpyPlotting & Analytics: matplotlib, seabornMachine Learning & Resampling: scikit-learn, imbalanced-learnExplainable AI (Planned): shap