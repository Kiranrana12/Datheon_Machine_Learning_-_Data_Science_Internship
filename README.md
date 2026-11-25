# Datheon_Machine_Learning_-_Data_Science_Internship
# README: Machine Learning Portfolio (Tasks 1–4)

This repository contains four end-to-end Machine Learning pipelines covering classification, regression, ensemble learning, and Natural Language Processing (NLP). Each task is implemented in Python using Scikit-Learn, Pandas, and visualisation libraries, and is designed to run in **Google Colab**.

---

## 📂 Repository Structure

```
├── task1_iris_classification.py       # Multi-class classification (Iris)
├── task2_california_housing.py        # Regression (California Housing)
├── task3_telco_churn.py               # Ensemble classification (Telco Churn)
├── task4_sentiment_analysis.py        # NLP text classification (Sentiment)
└── README.md
```

---

## 📊 Task 1 — Multi-Class Benchmark Classification (Iris)

**Goal:** Build end-to-end classification pipelines with normalisation, train/test splitting, and confusion matrices.

**Dataset:** `iris.csv` (150 samples, 4 features, 3 classes: setosa, versicolor, virginica)

**Workflow:**
1. Load dataset with Pandas and perform exploratory **pairplot** analysis.
2. Split data **80/20** with stratification to preserve class balance.
3. Normalise features using `StandardScaler` (fit on train, transform on test).
4. Train and benchmark two models:
   - **Logistic Regression**
   - **Decision Tree**
5. Evaluate using **Accuracy, Precision, Recall, F1-Score** (macro-averaged for multi-class).
6. Plot **confusion matrix heatmaps** for both models side-by-side.
7. Visualise **2D decision boundaries** (using scaled `petal_length` and `petal_width`).

**Key Concepts Demonstrated:**
- Data leakage prevention (scaling after split)
- Macro-averaging for multi-class metrics
- Dimensionality reduction for visualisation (2D projection of 4D data)

**Expected Results:**
| Model | Accuracy | F1-Score |
|---|---|---|
| Logistic Regression | ~0.97 | ~0.97 |
| Decision Tree | ~0.97 | ~0.97 |

---

## 🏠 Task 2 — California Housing Regression

**Goal:** Predict median house values using linear models with regularisation and preprocessing pipelines.

**Dataset:** `task2.csv` (20,640 samples, 10 features including categorical `ocean_proximity`)

**Workflow:**
1. Load the dataset and detect missing values (207 missing in `total_bedrooms`).
2. Generate a **correlation heatmap** (numeric columns only) to assess multicollinearity.
3. Engineer an interaction feature: `RoomsPerPerson = total_rooms / population`.
4. Split data **80/20**.
5. Build a `ColumnTransformer` pipeline:
   - **Numeric:** Median imputation → Standard scaling
   - **Categorical:** Mode imputation → One-Hot Encoding
6. Train and compare three models:
   - **OLS (Linear Regression)**
   - **Ridge (L2)** — tuned via `GridSearchCV`
   - **Lasso (L1)** — tuned via `GridSearchCV`
7. Evaluate using **RMSE, MAE, R²**.
8. Plot **Predicted vs Actual** scatter plots and **Residual Distributions**.

**Key Concepts Demonstrated:**
- Handling real missing data (`SimpleImputer`)
- Multicollinearity management via L1/L2 regularisation
- ColumnTransformer for mixed data types
- Data leakage prevention via Pipeline + GridSearchCV

---

## 📞 Task 3 — Telco Customer Churn Prediction

**Goal:** Predict customer churn using ensemble machine learning models with class imbalance handling.

**Dataset:** `task3.csv.txt` (7,043 samples, 20 features including categorical and numeric types)

**Workflow:**
1. Load data and clean `TotalCharges` (blank strings → `NaN` → median imputation).
2. Encode target `Churn` (Yes = 1, No = 0) and drop the `customerID` column.
3. Split data **80/20** with stratification.
4. Build a preprocessing pipeline:
   - **Numeric:** Standard scaling
   - **Categorical:** One-Hot Encoding
5. Apply **SMOTE** (Synthetic Minority Over-sampling Technique) **inside** the pipeline to prevent data leakage.
6. Train and cross-validate (5-Fold Stratified) two models:
   - **Random Forest Classifier**
   - **XGBoost Classifier**
7. Evaluate with **ROC-AUC** scores.
8. Plot **ROC curves** for both models.
9. Extract and visualise the **Top 10 feature importances** from XGBoost.

**Key Concepts Demonstrated:**
- Class imbalance handling via SMOTE
- ImbPipeline (`imblearn.pipeline`) for leakage-safe oversampling
- Cross-validation with `StratifiedKFold`
- Ensemble learning (bagging vs. boosting)
- Feature importance interpretation

**Expected Results:** Both models achieve **ROC-AUC > 0.80** (typically 0.84–0.86).

---

## 💬 Task 4 — Sentiment Analysis with NLP

**Goal:** Build a text classifier to identify sentiment in customer reviews using TF-IDF and Naive Bayes.

**Dataset:** `sentiment_reviews.csv` (10 reviews: 5 positive, 4 negative, 1 neutral)

**Workflow:**
1. Load dataset with Pandas.
2. Preprocess text with **NLTK**:
   - Lowercasing
   - Removing HTML tags and URLs (regex)
   - Removing punctuation and numbers
   - Tokenisation
   - Stopword removal
   - Lemmatisation
3. Split data **70/30** (stratification disabled due to small class sizes).
4. Build Pipelines using `TfidfVectorizer(max_features=5000)`:
   - **Multinomial Naive Bayes**
   - **Logistic Regression**
5. Evaluate with **Accuracy** and **Classification Report**.
6. Implement `predict_sentiment()` for **live custom user input**.

**Key Concepts Demonstrated:**
- Text preprocessing pipeline (clean → tokenise → lemmatise)
- TF-IDF vectorisation
- Pipeline-based data leakage prevention
- Handling edge cases (single-sample classes)
- Real-world deployment function for live inference

**⚠️ Note on Dataset Size:** With only 10 samples, evaluation metrics have high variance. The primary value here is in demonstrating the NLP pipeline and the live prediction function.

---

 🛠️ Technologies Used

| Category | Tools |
|---|---|
| **Data Manipulation** | Pandas, NumPy |
| **Visualisation** | Matplotlib, Seaborn |
| **Classical ML** | Scikit-Learn |
| **Ensemble ML** | XGBoost |
| **Imbalanced Data** | Imbalanced-Learn (SMOTE) |
| **NLP** | NLTK (tokenisation, lemmatisation, stopwords) |
| **Environment** | Google Colab |

---

🚀 How to Run

Each task is designed to run in **Google Colab**. General steps:

1. Open a new Colab notebook.
2. Install dependencies (where required):
   ```bash
   !pip install imbalanced-learn xgboost nltk scikit-learn pandas matplotlib seaborn
   ```
3. Upload the required dataset:
   ```python
   from google.colab import files
   files.upload()
   ```
4. Copy the corresponding task code into a cell.
5. Run all cells.

---

## 📈 Key Learning Outcomes

| Skill | Task(s) |
|---|---|
| Data cleaning & imputation | 1, 2, 3 |
| Feature engineering | 2, 3 |
| Data leakage prevention | 1, 2, 3, 4 |
| Multi-class evaluation | 1 |
| Regularisation (L1/L2) | 2 |
| Class imbalance handling | 3 |
| Ensemble learning | 3 |
| NLP text processing | 4 |
| Model interpretation | 1 (confusion matrix), 3 (feature importance) |
| Visualisation & reporting | All |

---

 👤 Author
Kiran Rana, Data science

---

 📄 License

This project is for educational purposes only.
