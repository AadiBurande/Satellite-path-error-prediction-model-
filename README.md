# Satellite Path Error Prediction Model 🛰️

## 📝 Project Overview
This project was developed for the **Smart India Hackathon (SIH) 2025** to address the challenge provided by the **Indian Space Research Organisation (ISRO)**. 

The goal is to develop an AI/ML-based model to predict time-varying patterns of the error build-up between uploaded (broadcast) and modeled values of both **satellite clock biases** and **ephemeris parameters** for navigation satellites (GEO/GSO and MEO).

### Problem Statement ID: 25176
Accurate GNSS positioning depends on precise satellite timing and orbital data. This model predicts future error build-ups at 15-minute intervals to enhance the reliability of navigation systems.

---

## 🚀 Features
- **Data Integration:** Seamless loading of multi-satellite datasets (GEO/MEO).
- **Advanced Preprocessing:** Automated data cleaning and handling of missing temporal values.
- **Time-Series Engineering:** Creation of lag features and rolling statistics to capture temporal dependencies.
- **High-Performance Modeling:** Optimized **XGBoost Regressor** for multi-step time-series forecasting.
- **Statistical Validation:** Evaluation of error distributions to ensure they approach a normal distribution (as per ISRO requirements).

---

## 📂 Project Structure
```text
sih_satellite_project/
├── data/               # Raw and cleaned CSV datasets
├── models/             # Trained .joblib models and performance metrics (JSON)
├── notebooks/          # Jupyter notebooks for Exploratory Data Analysis (EDA)
├── plots/              # Visualizations (Error distribution, QQ-plots, Predictions)
├── src/                # Modular Python scripts for the pipeline
├── requirements.txt    # Project dependencies
└── app.py              # Main application/demo script

🛠️ Technical Stack
Language: Python 3.x

Libraries: Pandas, NumPy, Scikit-learn, XGBoost, Joblib

Visualization: Matplotlib, Seaborn

Version Control: Git & GitHub

📊 Results & Performance
Our model evaluates error distribution closeness to a normal distribution.

Metrics: RMSE, MAE, and R² scores are stored in models/model_metrics.json.

Visuals: Check the plots/ folder for the predicted vs. actual error trends.

⚙️ Installation & Usage
Clone the repository:

Bash
git clone [https://github.com/AadiBurande/Satellite-path-error-prediction-model.git](https://github.com/AadiBurande/Satellite-path-error-prediction-model.git)
cd Satellite-path-error-prediction-model
Set up the virtual environment:

Bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
Install dependencies:

Bash
pip install -r requirements.txt
Run the data pipeline:

Bash
python Load_data.py
python train_model.py

