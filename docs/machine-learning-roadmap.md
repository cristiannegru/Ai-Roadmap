# 🤖 Machine Learning & Mathematics Roadmap

Welcome to the **Machine Learning & Mathematics Roadmap**. This guide covers Phase 1, Phase 2, and Phase 3 of the AI learning path. Follow this systematic guide to build a rock-solid foundation in Python, mathematics, data wrangling, and classical machine learning.

---

## 🗺️ Learning Path Overview
```
[Phase 1: Foundations] ──> [Phase 2: Data Essentials] ──> [Phase 3: Machine Learning]
       │                                │                                │
       ├─ Python Basics                 ├─ NumPy & Pandas                ├─ Feature Engineering
       └─ Linear Algebra & Stats        └─ Matplotlib & Seaborn          └─ Supervised & Unsupervised ML
```

---

## 📌 Phase 1: AI Foundations & Math

Before touching ML models, you must understand the language of data: linear algebra, calculus, probability, statistics, and Python.

### 1. Python Programming
Learn syntax, data structures, OOP, file handling, and virtual environments.
- 📺 **Python Full Course by CodeWithHarry**: [Watch Playlist](https://youtube.com/playlist?list=PLu0W_9lII9agICnT8t4iYVSZ3eykIAOME)
- 📺 **Jupyter Notebook & Google Colab Tutorial**: [Watch Video](https://youtu.be/5pf0_bpNbkw)

### 2. Linear Algebra
Focus on vectors, matrices, matrix multiplication, determinants, eigenvalues, and eigenvectors.
- 📺 **Linear Algebra Course by CampusX**: [Watch Playlist](https://youtube.com/playlist?list=PLKnIA16_RmvYu0fS_RuIB2eTbJcTFdrAA)

### 3. Statistics & Probability
Focus on descriptive statistics, distributions, hypothesis testing, Bayes' theorem, and conditional probability.
- 📺 **Statistics for Data Science by CampusX**: [Watch Video](https://youtube.com/watch?v=tPhzDKjQBpo)
- 📺 **Probability Course by CampusX**: [Watch Video](https://youtube.com/watch?v=Ty7knppVo9E)

---

## 📌 Phase 2: Data Manipulation & Visualization

Data scientists spend 80% of their time wrangling and visualizing data. Master these core libraries.

```
┌────────────────────────────────────────────────────────┐
│                     DATA TOOLKIT                       │
├───────────────┬────────────────────────────────────────┤
│ NumPy         │ Multi-dimensional array operations      │
├───────────────┼────────────────────────────────────────┤
│ Pandas        │ Structured data handling (DataFrames)  │
├───────────────┼────────────────────────────────────────┤
│ Matplotlib    │ Static charts and base plotting        │
├───────────────┼────────────────────────────────────────┤
│ Seaborn       │ Beautiful statistical visualizations   │
└───────────────┴────────────────────────────────────────┘
```

- 📺 **NumPy Complete Tutorial (freeCodeCamp)**: [Watch Video](https://www.youtube.com/watch?v=GB9ByFAIAH4)
- 📺 **Pandas Full Course (CampusX)**: [Watch Playlist](https://www.youtube.com/playlist?list=PLKnIA16_RmvbAlyx4_rdtR66B7EHX5k3z)
- 📺 **Data Visualization (Matplotlib & Seaborn) by CampusX**: [Watch Playlist](https://www.youtube.com/playlist?list=PLKnIA16_RmvbR85fgbfVRKOiMokUKVupy)

### 💻 Quick Code Sample: Data Wrangling
```python
import pandas as pd
import numpy as np

# Create a sample DataFrame
data = {'Feature_A': [1.0, np.nan, 3.5, 4.0], 'Feature_B': [10, 20, 15, 30]}
df = pd.DataFrame(data)

# Impute missing values with column mean
df['Feature_A'] = df['Feature_A'].fillna(df['Feature_A'].mean())
print(df)
```

---

## 📌 Phase 3: Machine Learning (100 Days of ML)

Master classical machine learning algorithms, how they work mathematically, and how to implement them from scratch and with Scikit-Learn.

- 📺 **100 Days of Machine Learning Playlist by CampusX**: [Watch Playlist](https://www.youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH)
- 📺 **StatQuest Machine Learning Videos**: [Watch Playlist](https://youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF)

---

### 🛡️ Core ML Modules & Videos

#### 1. Feature Engineering
Preparing raw data for ML models.
- **Topics**: Imputation, One-Hot Encoding, Ordinal Encoding, Scaling (Standardization, MinMax), Outlier Detection.
- 📺 **Feature Engineering Playlist (CampusX)**: [Watch Playlist](https://www.youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH)

#### 2. Regression Algorithms
Predicting continuous numerical values.
- **Topics**: Simple Linear Regression, Multiple Linear Regression, Ridge & Lasso (L1 & L2 Regularization).
- 📺 **Linear Regression from Scratch (CampusX)**: [Watch Video](https://www.youtube.com/watch?v=UZPfbG0jNec)
- 📺 **Ridge & Lasso Regression (CampusX)**: [Watch Video](https://www.youtube.com/watch?v=aEow1QoTLo0)

#### 3. Classification Algorithms
Predicting categorical labels.
- **Topics**: Logistic Regression, Decision Trees, Random Forests, Support Vector Machines (SVM), Naive Bayes, K-Nearest Neighbors (KNN).
- 📺 **Logistic Regression Tutorial (CampusX)**: [Watch Video](https://youtu.be/yIYKR4sgzI8)
- 📺 **Decision Trees Explained (StatQuest)**: [Watch Video](https://youtu.be/7VeUPuFGJHk)
- 📺 **Random Forests (StatQuest)**: [Watch Video](https://youtu.be/J4Wdy0Wc_xQ)
- 📺 **Support Vector Machines (SVM)**: [Watch Video](https://www.youtube.com/watch?v=HfrKYccB2DQ)

#### 4. Clustering & Unsupervised Learning
Finding hidden structures in unlabeled data.
- **Topics**: K-Means Clustering, Hierarchical Clustering, DBSCAN.
- 📺 **K-Means Clustering Explained**: [Watch Video](https://youtu.be/4b5d3muPQmA)
- 📺 **DBSCAN Clustering**: [Watch Video](https://youtu.be/C3r7tGRe2eI)

#### 5. Dimensionality Reduction
Reducing features while maintaining variance.
- **Topics**: Principal Component Analysis (PCA), t-SNE.
- 📺 **PCA Step-by-Step (StatQuest)**: [Watch Video](https://youtu.be/FgakZw6K1QQ)

---

### 📈 Evaluation Metrics
How to evaluate regression and classification models.
- **Regression**: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R-Squared ($R^2$).
- **Classification**: Confusion Matrix, Accuracy, Precision, Recall, F1-Score, ROC Curve, AUC.
- 📺 **Model Evaluation Metrics Video**: [Watch Video](https://www.youtube.com/watch?v=4jRBRDbJemM)

---

## 💻 Runnable Code in This Repo (Chunk 2)

Don't just watch — run. All Phase 1–2 concepts have tested implementations:

| Topic | Code | Try it |
|-------|------|--------|
| Python basics (chunking, counting, files, OOP) | `src/ai_roadmap/python_basics.py` | `notebooks/01_python_foundations.ipynb` |
| Math (cosine, Bayes, softmax, regression) | `src/ai_roadmap/math_utils.py` | `notebooks/02_numpy_pandas.ipynb` §§1–2 |
| Pandas cleaning pipeline (Milestone 2) | `src/ai_roadmap/data_wrangling.py` → `clean_dataframe()` | `notebooks/02_numpy_pandas.ipynb` §§3–6 |
| Datasets (offline, synthetic) | `data/samples/titanic_sample.csv`, `housing_sample.csv` | `pytest tests/test_data_wrangling.py -q` |

```bash
pip install -r requirements.txt
pytest -q                                   # 36 tests, all offline
python -m doctest src/ai_roadmap/python_basics.py -v
```

---

## 🛠️ Step-by-Step Practical Projects
Practice what you learn by building real projects.
1. **Housing Price Predictor (Regression)**
   - 📺 **Project Tutorial**: [Watch Video](https://www.youtube.com/watch?v=UZPfbG0jNec)
2. **Spam Email Classifier (Classification)**
   - 📺 **Project Tutorial**: [Watch Video](https://www.youtube.com/watch?v=YncZ0WwxyzU)
3. **Customer Segmentation (Clustering)**
   - 📺 **Project Tutorial**: [Watch Video](https://www.youtube.com/watch?v=UZPfbG0jNec)
