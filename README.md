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
| NSL-KDD (Binary)       | Decision Tree       | 98%    |
| NSL-KDD (Binary)       | Naive Bayes         | 90%    |
| NSL-KDD (Binary)       | SVM                 | 53%    |
| NSL-KDD (Binary)       | Logistic Regression | 95%    |
| IoT-23 (Multiclass)    | Decision Tree       | 81%    |
| IoT-23 (Multiclass)    | Naive Bayes         | 23%    |
| IoT-23 (Multiclass)    | SVM                 | 54%    |
| IoT-23 (Multiclass)    | Logistic Regression | 63%    |
| Edge-IIoT (Binary)     | Decision Tree       | 99%    |
| Edge-IIoT (Binary)     | Naive Bayes         | 89%    |
| Edge-IIoT (Binary)     | SVM                 | 91%    |
| Edge-IIoT (Binary)     | Logistic Regression | 89%    |
| Edge-IIoT (Multiclass) | Decision Tree       | 92%    |
| Edge-IIoT (Multiclass) | Naive Bayes         | 50%    |
| Edge-IIoT (Multiclass) | SVM                 | 29%    |
| Edge-IIoT (Multiclass) | Logistic Regression | 67%    |
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
