# 🚗 Regression Task: Car Price Prediction

## 📋 Project Overview

**Business Objective:** Predict used car prices in the Egyptian market based on vehicle specifications and condition.

**Machine Learning Task:** Regression (predicting continuous numerical values)

**Dataset:** Hatla2ee scraped car listings data (~18,000+ cars)

**Best Model Performance:** Random Forest Regressor with **R² = 88%** ⭐

---

## 📂 Project Structure

```
Regression Task/
├── Data/
│   ├── hatla2ee_scraped_data.csv      # Raw scraped data
│   └── processed_data.csv              # Cleaned & transformed data
├── Preprocessing/
│   ├── Preprocessing.ipynb             # Data cleaning & feature engineering
│   ├── label_encoder_Make.pkl          # Saved encoder for car brands
│   └── label_encoder_Model.pkl         # Saved encoder for car models
└── Model_training/
    ├── RandomForest.ipynb              # Best model (R²=88%)
    └── LinearRegression.ipynb          # Baseline model (R²=62%)
```

---

## 🔄 Workflow Pipeline

```
1. Data Loading → 2. Preprocessing → 3. Model Training → 4. Evaluation → 5. Deployment
      ↓                    ↓                  ↓                ↓              ↓
  Raw CSV         Clean & Transform    Train Multiple    Compare Models   Make Predictions
                  (Handle Missing,      Models (RF,     (R², MAE, RMSE)  (New Cars)
                   Outliers, Encode)    Ridge, etc.)
```

---

## 📓 Notebooks Explained

### 1️⃣ **Preprocessing/Preprocessing.ipynb** 
*Data cleaning and feature engineering*

#### **What It Does:**
- ✅ Loads raw car listing data from Hatla2ee
- ✅ Handles missing values using KNN Imputer and median imputation
- ✅ Cleans numeric columns (extracts numbers from "100,000 EGP" format)
- ✅ Removes outliers using IQR method
- ✅ Creates new feature: `Years_of_usage` (derived from Year)
- ✅ Encodes categorical variables (Brand, Model) using Label Encoding
- ✅ Converts boolean features to numeric (0/1)
- ✅ Applies log transformation to `Price` and `Mileage` (for better model performance)
- ✅ Saves processed data and encoders for future use

#### **Key Features Created:**
- **Years_of_usage**: Current year - Car year (age of the car)
- **Log-transformed Price**: Handles price skewness
- **Log-transformed Mileage**: Normalizes mileage distribution

#### **Output Files:**
- `processed_data.csv` - Clean dataset ready for modeling
- `label_encoder_Make.pkl` - Encoder for car brands (for predictions on new data)
- `label_encoder_Model.pkl` - Encoder for car models (for predictions on new data)

#### **Why This Matters:**
Clean data = Better models! This notebook ensures data quality and creates features that help models learn price patterns effectively.

---

### 2️⃣ **Model_training/RandomForest.ipynb** 🏆
*Best performing model for car price prediction*

#### **What It Does:**
- ✅ Loads preprocessed data and splits into train/test sets (70/30)
- ✅ Trains Random Forest Regressor (100 decision trees)
- ✅ Evaluates model performance using R², MAE, RMSE
- ✅ Visualizes predictions vs actual prices
- ✅ Demonstrates how to predict prices for new cars

#### **Model Performance:**
```
✅ R² Score: 0.88 (88%)  - Explains 88% of price variance!
✅ MAE: 0.21            - Low average error
✅ RMSE: 0.30           - Consistent predictions
```

#### **Why Random Forest?**
- Captures **non-linear relationships** (price doesn't change linearly with age)
- Handles **feature interactions** (Brand + Model + Age affect price together)
- **Robust to outliers** (luxury cars don't skew predictions)
- **High accuracy** for complex pricing patterns

#### **Example Prediction Workflow:**
```python
# New car features
new_car = {
    'Mileage': 50000,
    'Brand': 'Toyota',
    'Model': 'Corolla',
    'Automatic Transmission': True,
    'Years_of_usage': 5
}

# Model predicts: 450,000 EGP
```

#### **Business Value:**
- **Car Dealers**: Price inventory competitively
- **Buyers**: Identify overpriced/underpriced cars
- **Online Marketplaces**: Auto-suggest fair prices

---

### 3️⃣ **Model_training/LinearRegression.ipynb**
*Baseline model comparison using Ridge Regression*

#### **What It Does:**
- ✅ Compares multiple linear models (Linear, Ridge, Lasso, ElasticNet)
- ✅ Performs hyperparameter tuning using GridSearchCV
- ✅ Trains best linear model (Ridge with α=10)
- ✅ Evaluates performance and compares with Random Forest

#### **Model Performance:**
```
⚠️ R² Score: 0.62 (62%)  - Moderate performance
⚠️ MAE: 40.45           - Higher error than Random Forest
⚠️ RMSE: 53.16          - Less consistent predictions
```

#### **Models Tested:**
1. **Linear Regression**: Basic OLS (Ordinary Least Squares)
2. **Ridge Regression**: L2 regularization (prevents overfitting) ✅ Best
3. **Lasso Regression**: L1 regularization (feature selection)
4. **ElasticNet**: Combination of L1 and L2

#### **Why Ridge Won:**
- **α=10** provides strong regularization
- Prevents overfitting on training data
- More stable than basic linear regression

#### **Comparison with Random Forest:**
| Metric | Ridge | Random Forest | Winner |
|--------|-------|---------------|--------|
| R² | 62% | **88%** | Random Forest |
| MAE | 40.45 | **0.21** | Random Forest |
| RMSE | 53.16 | **0.30** | Random Forest |

#### **Why Random Forest Outperforms:**
- Car prices have **non-linear patterns** (Ridge assumes linearity)
- **Feature interactions** matter (Brand × Model × Age)
- Random Forest captures complex relationships automatically

#### **When to Use Ridge:**
- ✅ Need **interpretability** (see feature coefficients)
- ✅ Fast training required
- ✅ Building a **baseline** model

---

## 🎯 Key Results Summary

### **Final Model Selection: Random Forest** 🏆

**Performance:**
- **88% R²** - Explains 88% of price variance
- **Average Error: ~21%** (in log scale)
- **Predictions within reasonable range** for most cars

### **Model Comparison:**

| Model | R² | MAE | RMSE | Training Time | Use Case |
|-------|-----|-----|------|---------------|----------|
| **Random Forest** | **88%** | **0.21** | **0.30** | ~2 min | Production (Best accuracy) |
| Ridge Regression | 62% | 40.45 | 53.16 | ~1 sec | Baseline/Interpretability |

### **Business Impact:**
- **26% improvement** in variance explained (Random Forest vs Ridge)
- More accurate pricing → Better business decisions
- Can handle new car data with preprocessing pipeline

---

## 🚀 How to Use This Project

### **Step 1: Run Preprocessing**
```bash
Open: Preprocessing/Preprocessing.ipynb
Execute all cells
Output: processed_data.csv + encoders
```

### **Step 2: Train Model (Random Forest)**
```bash
Open: Model_training/RandomForest.ipynb
Execute all cells
Output: Trained model + evaluation metrics
```

### **Step 3: Make Predictions**
```python
# Example: Predict price for a new car
new_car_data = {
    'Mileage': 80000,
    'Brand': 'BMW',
    'Model': '320',
    'Automatic Transmission': True,
    'Air Conditioner': True,
    'Power Steering': True,
    'Remote Control': True,
    'Years_of_usage': 7
}

# Follow the prediction pipeline in RandomForest.ipynb
# Model will output predicted price in EGP
```

---

## 📊 Dataset Information

**Source:** Hatla2ee (Egyptian car marketplace)

**Features:**
- **Price** (Target): Car price in EGP
- **Mileage**: Kilometers driven
- **Brand**: Car manufacturer (Toyota, BMW, Mercedes, etc.)
- **Model**: Specific car model
- **Automatic Transmission**: Boolean
- **Air Conditioner**: Boolean
- **Power Steering**: Boolean
- **Remote Control**: Boolean
- **Years_of_usage**: Age of the car (derived feature)

**Dataset Size:** ~18,000+ car listings

---

## 🔑 Key Techniques Used

### **Data Preprocessing:**
- ✅ KNN Imputation for missing values
- ✅ Log transformation for skewed distributions
- ✅ Label Encoding for categorical variables
- ✅ IQR method for outlier removal
- ✅ Feature engineering (Years_of_usage)

### **Modeling:**
- ✅ Random Forest Regressor (ensemble learning)
- ✅ Ridge Regression (regularized linear model)
- ✅ Hyperparameter tuning with GridSearchCV
- ✅ Cross-validation for model selection

### **Evaluation:**
- ✅ R² Score (variance explained)
- ✅ MAE (Mean Absolute Error)
- ✅ RMSE (Root Mean Squared Error)
- ✅ Actual vs Predicted visualizations

---

## 💡 Insights & Learnings

### **What Works:**
1. **Log transformation** significantly improves model performance for skewed price data
2. **Years_of_usage** is a strong predictor (newer cars = higher prices)
3. **Random Forest** handles non-linearity better than linear models
4. **Encoding strategy** matters - Label Encoding works well for Brand/Model

### **Model Selection Criteria:**
- **Accuracy**: Random Forest wins (88% vs 62%)
- **Interpretability**: Ridge regression easier to explain
- **Speed**: Ridge trains faster (1 sec vs 2 min)
- **Production**: Random Forest recommended for deployment

### **Business Recommendations:**
1. **Use Random Forest** for production pricing system
2. **Monitor predictions** on luxury cars (potential outliers)
3. **Update encoders** when new brands/models appear
4. **Retrain quarterly** to capture market trends

---

## 📈 Future Improvements

### **Potential Enhancements:**
- [ ] Add more features (color, fuel type, transmission type)
- [ ] Try XGBoost or LightGBM (gradient boosting models)
- [ ] Implement feature importance analysis
- [ ] Build API for real-time predictions
- [ ] Add confidence intervals to predictions
- [ ] Handle new brands/models not in training data

### **Advanced Techniques:**
- [ ] Neural networks for complex patterns
- [ ] Time series analysis for price trends
- [ ] Ensemble multiple models (stacking)
- [ ] Geospatial features (location affects price)

---

## 📝 Notes

- All prices are in **Egyptian Pounds (EGP)**
- Log transformation is used throughout - remember to reverse it for actual prices
- Encoders must be saved and used for new predictions
- Model assumes similar data distribution to training set

---

## 🏁 Conclusion

This regression project successfully predicts used car prices with **88% accuracy** using Random Forest. The comprehensive preprocessing pipeline and model comparison ensure robust, production-ready predictions for the Egyptian car market.

**Next Steps:**
1. ✅ Deploy model as web API
2. ✅ Integrate with car listing platforms
3. ✅ Monitor model performance over time
4. ✅ Collect user feedback and retrain

---

*For questions or issues, refer to the detailed comments within each notebook.*
