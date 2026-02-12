# Credit Card Fraud Detection using XGBoost with Imbalanced Data

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5%2B-orange)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-red)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Project Overview

This project implements a robust credit card fraud detection system using XGBoost classifier, specifically designed to handle **highly imbalanced datasets**. The model achieves **98.2% ROC-AUC** and **89.6% PR-AUC** with a recall of **83%** for fraud transactions.

**Key Challenge**: Credit card fraud datasets typically have <0.1% fraud transactions. Using accuracy as a metric would be misleading (a model predicting "no fraud" always would achieve 99.8% accuracy but be completely useless).

## 🎯 Key Features

- ✅ **Proper handling of imbalanced data** using `scale_pos_weight` parameter
- ✅ **Correct evaluation metrics**: Precision, Recall, F1-score, ROC-AUC, PR-AUC
- ✅ **Stratified K-Fold Cross-Validation** for reliable performance estimation
- ✅ **Feature standardization** using StandardScaler
- ✅ **Comprehensive performance analysis** with confusion matrix

## 📊 Dataset

**Credit Card Fraud Detection Dataset**
- **Source**: [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Transactions**: 284,807
- **Features**: 30 (V1-V28 PCA transformed features, Time, Amount)
- **Target**: Class (0 = legitimate, 1 = fraud)
- **Fraud Ratio**: 0.17% (492 fraud transactions)

## 🛠️ Technical Stack

- **Python 3.8+**
- **XGBoost**: Gradient boosting framework
- **scikit-learn**: Data preprocessing, model evaluation, cross-validation
- **pandas/numpy**: Data manipulation
- **matplotlib/seaborn**: Visualization
- **imbalanced-learn** (optional): For additional resampling techniques

## 🚀 Model Architecture

```python
scale_pos_weight = 284315 / 492  # ≈ 577 (non-fraud / fraud ratio)

model = XGBClassifier(
    learning_rate=0.05,
    max_depth=6,
    n_estimators=800,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)
```

## 📈 Performance Metrics

### Cross-Validation Results (5-Fold Stratified)

| Metric | Score |
|--------|-------|
| **Mean ROC-AUC** | **0.9822** |
| Fold 1 | 0.9841 |
| Fold 2 | 0.9786 |
| Fold 3 | 0.9947 |
| Fold 4 | 0.9693 |
| Fold 5 | 0.9845 |

### Test Set Performance

#### Classification Report
```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.95      0.83      0.89        98

    accuracy                           1.00     56962
   macro avg       0.98      0.91      0.94     56962
weighted avg       1.00      1.00      1.00     56962
```

#### Confusion Matrix
```
[[56860     4]   # True Negatives, False Positives
 [   17    81]]   # False Negatives, True Positives
```

#### Key Metrics
- **PR-AUC (Precision-Recall AUC)**: **0.8962**
- **Recall (Fraud Detection Rate)**: **83%**
- **Precision**: **95%**
- **F1-Score**: **0.89**

## 💡 Why These Metrics Matter

| Metric | Why It's Important for Fraud Detection |
|--------|----------------------------------------|
| **Recall** | How many frauds we actually catch (minimize false negatives) |
| **Precision** | How many of our fraud alerts are correct (minimize false positives) |
| **PR-AUC** | Best metric for imbalanced data - focuses on positive class |
| **ROC-AUC** | Overall model discrimination ability |

## 🎯 Impact of Class Weighting

**Before `scale_pos_weight` (default):**
- Precision: 0.99
- Recall: 0.80
- F1: 0.88

**After `scale_pos_weight = 577`:**
- Precision: 0.95
- Recall: 0.83 ✓ (+3% improvement)
- F1: 0.89 ✓

## 🚦 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/ZaryabAhmad01/Credit-Card-Fraud-Detection-using-XGBoost-with-Imbalanced-Data.gt
cd Credit-Card-Fraud-Detection-using-XGBoost-with-Imbalanced-Data
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
- Download from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Place `creditcard.csv` in project root directory

### 4. Run the Model
```bash
python fraud_detection.py
```


## 📦 requirements.txt

```
pandas==1.3.5
numpy==1.21.6
matplotlib==3.5.1
seaborn==0.11.2
scikit-learn==1.0.2
xgboost==1.5.2
imbalanced-learn==0.9.1  # Optional for SMOTE
```

## 🔮 Future Improvements

1. **Hyperparameter Tuning**: GridSearchCV or RandomizedSearchCV for optimal parameters
2. **SMOTE/ADASYN**: Synthetic oversampling techniques
3. **Threshold Optimization**: Adjust decision threshold based on business costs
4. **Feature Engineering**: Additional derived features from Time and Amount
5. **Model Ensembling**: Combine XGBoost with LightGBM, Random Forest
6. **Deep Learning**: LSTM/CNN for sequence-based patterns
7. **Model Deployment**: Flask API + Docker containerization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📧 Contact



Project Link: [https://github.com/ZaryabAhmad01/Credit-Card-Fraud-Detection-using-XGBoost-with-Imbalanced-Data.)

## 🙏 Acknowledgments

- [Kaggle](https://www.kaggle.com/) for hosting the dataset
- [Machine Learning Group - ULB](http://mlg.ulb.ac.be) for providing the dataset
- XGBoost developers for the excellent library

---

**⭐ Star this repository if you find it helpful!** 

**Key Takeaway**: In imbalanced classification problems, **always prioritize PR-AUC and Recall over Accuracy**. This project demonstrates how proper handling of class imbalance can significantly improve fraud detection rates.
