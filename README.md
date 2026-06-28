# 🏥 Health Insurance Charges Prediction ML and Pythin libararies
<img width="1408" height="768" alt="info graph insurance" src="https://github.com/user-attachments/assets/ba381930-8357-4a78-945f-cd766d1ffb6b" />

An end-to-end Machine Learning project to analyze demographic and health indicators, perform exploratory data analysis (EDA), and build predictive regression models to estimate annual medical insurance charges for individuals.

## 📊 Project Overview
Predicting medical costs is critical for both insurance providers (for accurate premium pricing) and consumers (for financial planning). This project leverages a tabular dataset containing key demographic attributes to uncover driving factors behind health expenditures and builds an optimized predictive engine.

### Core Architecture Pipeline
1. **Data Ingestion & Cleaning:** Checking for structural integrity, missing values, and anomalies.
2. **Exploratory Data Analysis (EDA):** Univariate distribution profiles, statistical distribution checks, and bivariate correlation mapping.
3. **Feature Engineering:** - Feature scaling via `StandardScaler` to bring numeric values onto a uniform scale.
   - Categorical encoding (One-Hot Encoding) for nominal variables.
   - Domain-specific binning (e.g., Obese classifications).
4. **Machine Learning Regression:** Implementing and comparing structural, ensemble, and boosted tree models to minimize prediction errors.

---

## 📂 Dataset Feature Profile
The model relies on the following structural characteristics from `insurance.csv`:

* `age`: Age of primary beneficiary (Numerical)
* `sex`: Insurance contractor gender (`female`, `male`)
* `bmi`: Body mass index, providing an objective index of body weight relative to height ($kg / m^2$) (Numerical)
* `children`: Number of dependents covered by health insurance (Numerical)
* `smoker`: Smoking status (`yes`, `no`)
* `region`: The beneficiary's residential area in the US (`northeast`, `northwest`, `southeast`, `southwest`)
* `charges`: Individual medical costs billed by health insurance (Continuous **Target Variable**)

---

## 🛠️ Tech Stack & Environment
* **Language:** Python 3.8+
* **Environment:** Jupyter Notebook
* **Core Libraries:**
  * **Data Manipulation:** `pandas`, `numpy`
  * **Visualization:** `matplotlib`, `seaborn`
  * **Machine Learning Framework:** `scikit-learn`
  * **Gradient Boosting:** `xgboost`

---
## 👤 Author

**Anwarul karim**  
 · [LinkedIn]((https://www.linkedin.com/in/md-anwarul-karim-377a0b194/)
 
## 🚀 Getting Started

### Prerequisites
Ensure you have Python installed, then set up the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
