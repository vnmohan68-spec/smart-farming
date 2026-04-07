# smart-farming
AgriML India is a Streamlit-based smart farming app for paddy yield prediction, variety recommendation, fertilizer planning, pre-harvest estimation, and farm risk analysis using machine learning.
# 🌾 AgriML India — Smart Farming Intelligence

AgriML India is a powerful **Streamlit + Machine Learning** web application built for **Tamil Nadu paddy farmers** to improve crop decisions using real farm data.

This app provides:
- ⚖️ **Paddy Yield Prediction**
- 🌿 **Best Variety Recommendation**
- 🧪 **Fertilizer Dose Planning**
- 🔮 **Pre-Harvest Yield Estimation**
- 🚨 **Farm Risk / Anomaly Detection**
- 💬 **AgriBot Assistant**
- 🌐 **Multi-language Support** (English, Tamil, Hindi, Telugu, Malayalam)

---

## 🚀 Features

### 1) ⚖️ Yield Predictor
Predict exact paddy yield (in Kg) using farm inputs such as:
- Agriblock
- Variety
- Soil type
- Nursery type
- Farm size (Hectares)
- Fertilizer schedule
- Trash bundles
- Seed rate

### 2) 🌿 Variety Advisor
Recommends the **best paddy variety** for a specific:
- Agriblock
- Soil type
- Farm size

### 3) 🧪 Fertilizer Planner
Suggests optimized fertilizer doses for:
- DAP (Day 20)
- Urea (Day 40)
- Potash (Day 50)
- Micronutrients (Day 70)
- Weed Control
- Pest Control

### 4) 🔮 Pre-Harvest Predictor
Estimate final yield before harvest using:
- Trash bundle count
- Farm size

### 5) 🚨 Risk Monitor
Detect underperforming farms using:
- Isolation Forest anomaly detection
- Historical yield comparison
- Risk gap vs expected yield

### 6) 💬 AgriBot
Simple intelligent farm assistant trained on paddy farming insights:
- Variety guidance
- Fertilizer timing
- Yield tips
- Irrigation advice
- Soil guidance
- Risk awareness

### 7) 📊 Data Explorer
Explore:
- Raw farm dataset
- Charts and visualizations
- Statistics
- CSV export

---

## 🧠 Machine Learning Models Used

- **RandomForestRegressor** → Yield Prediction
- **RandomForestClassifier** → Variety Recommendation
- **RandomForestClassifier** → Fertilizer Dose Classification
- **RandomForestRegressor** → Pre-Harvest Yield Estimation
- **IsolationForest** → Farm Risk / Anomaly Detection
- **PCA** → Risk Visualization
- **LabelEncoder** → Categorical Encoding
- **StandardScaler** → Feature Scaling
- **KFold Cross Validation** → Model Validation

---

## 📂 Project Structure

```bash
AgriML-India/
│── app.py
│── paddydataset.csv
│── requirements.txt
│── README.md
