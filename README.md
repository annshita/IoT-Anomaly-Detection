# IoT Anomaly Detection
IoT Anomaly detection on 4 datasets using 4 basic ML algorithms.
2 datasets - IoT-23 (cleaned_data.csv) and NSL-KDD are provided. Edge-IIoT was too big. All datasets are taken from Kaggle.

## 📊 Datasets

### 1. **NSL-KDD**

* Standard dataset for network intrusion detection.
* Contains both normal and attack records.
* Binary classification

### 2. **IoT-23**

* Real-world IoT malware traffic traces.
* Multiclass classification.

### 3. **Edge-IIoT**

* Dataset for Industrial IoT systems.
* Used for binary classification

### 4. **Edge-IIoT (Multiclass)**

* Edge IIoT dataset used for multiclass classification with multiple attack categories.

---

## 🔧 Preprocessing

* Cleaned missing values.
* Encoded categorical features using Label Encoding or One-Hot Encoding.
* Standardized numerical features using Min-Max scaling.
* Split datasets into training and test sets (typically 80/20 split).

---

## 🧪 Implemented Models

Four classical machine learning algorithms were applied on all datasets:

1. **Decision Tree Classifier** (Scikit-learn `DecisionTreeClassifier`)
2. **Naive Bayes Classifier** (Scikit-learn `GaussianNB` or `MultinomialNB` depending on dataset)
3. **Support Vector Machine (SVM)** (Scikit-learn `SVC` with kernel tuning)
4. **Logistic Regression** (Scikit-learn `LogisticRegression`)

---

## 📈 Results

### Accuracy Comparison Table

| Dataset                | Model               | Accuracy |
| ---------------------- | ------------------- | -------- |
| NSL-KDD (Binary)       | Decision Tree       | XX.X%    |
| NSL-KDD (Binary)       | Naive Bayes         | XX.X%    |
| NSL-KDD (Binary)       | SVM                 | XX.X%    |
| NSL-KDD (Binary)       | Logistic Regression | XX.X%    |
| NSL-KDD (Multiclass)   | Decision Tree       | XX.X%    |
| NSL-KDD (Multiclass)   | Naive Bayes         | XX.X%    |
| NSL-KDD (Multiclass)   | SVM                 | XX.X%    |
| NSL-KDD (Multiclass)   | Logistic Regression | XX.X%    |
| IoT-23 (Binary)        | Decision Tree       | XX.X%    |
| IoT-23 (Binary)        | Naive Bayes         | XX.X%    |
| IoT-23 (Binary)        | SVM                 | XX.X%    |
| IoT-23 (Binary)        | Logistic Regression | XX.X%    |
| Edge-IIoT (Binary)     | Decision Tree       | XX.X%    |
| Edge-IIoT (Binary)     | Naive Bayes         | XX.X%    |
| Edge-IIoT (Binary)     | SVM                 | XX.X%    |
| Edge-IIoT (Binary)     | Logistic Regression | XX.X%    |
| Edge-IIoT (Multiclass) | Decision Tree       | XX.X%    |
| Edge-IIoT (Multiclass) | Naive Bayes         | XX.X%    |
| Edge-IIoT (Multiclass) | SVM                 | XX.X%    |
| Edge-IIoT (Multiclass) | Logistic Regression | XX.X%    |

(*Note: Replace XX.X% with actual accuracy scores after evaluation*)

---

## 📆 Key Observations

* The performance of models varied across datasets.
* Naive Bayes generally performed well for datasets with discrete categorical features.
* Decision Tree provided interpretable results but prone to overfitting.
* SVM often yielded good accuracy but was computationally expensive for large datasets.
* Logistic Regression performed consistently for well-separated feature spaces.

---

## 💾 Tools & Technologies

* Python 3.x
* Scikit-learn
* NumPy
* Pandas
* Matplotlib / Seaborn
* Jupyter Notebook

---

## 🔧 How to Run

1. Clone this repository:

```bash
git clone <your-repo-url>
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Prepare datasets in the `data/` folder.

4. Run training and evaluation scripts for each dataset:

```bash
python train_nslkdd_binary.py
python train_nslkdd_multiclass.py
python train_iot23.py
python train_edge_iiot.py
```

---

## 💡 Future Improvements

* Explore ensemble methods (e.g. Random Forest, XGBoost).
* Apply Deep Learning architectures for feature-rich datasets.
* Incorporate feature selection techniques to reduce dimensionality.
* Perform hyperparameter tuning for improved model performance.

