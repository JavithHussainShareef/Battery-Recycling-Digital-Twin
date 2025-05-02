# Battery Recycling Digital Twin with ML Models

This project leverages machine learning and digital twin concepts to analyze and predict chemical compositions from recycled battery blackmass, forecast metal prices, and detect anomalies in material composition.

## 📊 Project Overview

The goal is to simulate and optimize the battery recycling process by:
- Merging operational and chemical data from recycled battery bags.
- Predicting chemical compositions using Random Forest and Decision Tree regressors.
- Forecasting market prices for critical metals like Lithium, Cobalt, and Nickel.
- Detecting contaminated or anomalous chemical samples using Isolation Forest.
- Integrating a Digital Twin simulation for hypothetical scenario analysis.

---

## 🛠️ Tech Stack

- Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- Machine Learning: Random Forest, Decision Tree, Isolation Forest
- Data Preprocessing: StandardScaler, Time Parsing
- Visualization: Matplotlib, Seaborn

---

## 📂 Dataset Structure

- `Bag_dataset.csv`: Contains source ID, processing date, and weight of battery recycling bags.
- `Blackmass.csv`: Composition analysis results of processed blackmass.
- `Metal_Prices_2021_2023.csv`: Historical price data of key metals from 2021 to 2023.

---

## 🔍 Features

- Data merging and preprocessing
- Normalization of chemical elements
- Multi-output regression for predicting compositions
- Time series-based metal price forecasting
- Contamination detection using anomaly detection (Isolation Forest)
- Modular structure with clear exception handling

---

## 🚀 Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure datasets are in the same directory or update file paths
python main.py
