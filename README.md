# Unified Fraud Detection System for E-Commerce & Banking

An end-to-end machine learning pipeline built for Adey Innovations Inc. to identify fraudulent transactions across two distinct streams: contextual e-commerce activities and anonymized bank credit card transactions. 

This project balances the financial costs of false negatives (missed fraud) against the customer experience costs of false positives (blocking legitimate users) using advanced feature engineering, resampling techniques, and SHAP explainability.

---

## 📌 Project Status: Interim-1 Milestone (Completed)
As of June 7, 2026, **Task 1: Data Analysis and Preprocessing** is fully complete on the `task-1` branch.
* **Data Cleaning & Pipeline Integration:** Handled missing/duplicate logs and aligned data types.
* **Geolocation Enrichment:** Developed an efficient $O(N \log N)$ `merge_asof` lookup matching e-commerce IP addresses to spatial country boundaries without data leakage.
* **Feature Engineering:** Extracted raw behavioral metrics including account age velocity (`time_since_signup`), structural temporal patterns (`hour_of_day`, `day_of_week`), and transaction frequency loops (`device_count`).
* **Imbalance Isolation Strategy:** Documented a strict train-test validation split strategy using isolated SMOTE to avoid data leakage in upcoming model checkpoints.

---

## 📂 Repository Architecture

```text
fraud-detection/
├── data/                       # Local only (Ignored by Git)
│   ├── raw/                    # Original raw CSV files
│   └── processed/              # Engineered, cleaned, and normalized data
├── notebooks/
│   ├── eda-fraud-data.ipynb    # E-commerce EDA & Geolocation mapping
│   ├── eda-creditcard.ipynb    # Credit card EDA & Imbalance review
│   └── feature-engineering.ipynb # Feature extraction, scaling & encoding
├── src/                        # Modular source code scripts
├── tests/                      # Unit testing suite
├── models/                     # Saved model binaries (serialized artifacts)
├── requirements.txt            # Project environment specifications
└── README.md                   # Project documentation
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