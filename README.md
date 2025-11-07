# 🧬 Cancer Detection from Gene Expression Data using PCA and Random Forest Algorithm

## 📘 Project Overview
This project aims to detect **cancer** using **gene expression data** by applying **Principal Component Analysis (PCA)** for dimensionality reduction and **Random Forest Classifier** for classification.

Gene expression datasets contain thousands of genetic features. PCA helps reduce the dimensionality while retaining the most important variance, and the Random Forest algorithm efficiently classifies samples as **Normal** or **Tumor** based on these principal components.

---

## 🎯 Objectives
- Load and preprocess gene expression data.  
- Normalize data using `StandardScaler`.  
- Apply **PCA** to reduce data to two components for visualization.  
- Train a **Random Forest Classifier** to detect cancer.  
- Evaluate the model using accuracy and classification metrics.  
- Visualize results using scatter and variance plots.

---

## 🧩 Technologies Used
| Category | Tools |
|-----------|-------|
| Programming Language | Python |
| Data Handling | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Algorithms | PCA, Random Forest Classifier |

---

## 📂 Dataset
You can use one of the following datasets:
1. **UCI Repository:** [Gene Expression Cancer RNA-Seq Dataset (ID: 401)](https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq)
2. **Kaggle Dataset:** Download manually as `data.csv`

The dataset should contain:
- Gene expression features (columns with numeric values)
- A target column named **`Class`**, representing cancer type or normal sample.

---

## ⚙️ Methodology

### 1️⃣ Data Preprocessing
- Load the dataset (`data.csv`) using Pandas.
- Separate the features (X) and target labels (y).
- Standardize the features using `StandardScaler`.

### 2️⃣ Dimensionality Reduction (PCA)
- Apply **PCA** to reduce thousands of gene features into 2 principal components.
- Visualize the reduced data using a scatter plot.

### 3️⃣ Model Training
- Split the data into training and testing sets (70%-30%).
- Train a **Random Forest Classifier** on the PCA-transformed data.

### 4️⃣ Model Evaluation
- Predict outcomes on test data.
- Measure **accuracy** and print the **classification report**.

### 5️⃣ Visualization
- **PCA Scatter Plot:** Shows separation between cancerous and non-cancerous samples.
- **Scree Plot:** Displays the cumulative explained variance by principal components.

---

## 🧠 Code Structure
```bash
📁 Cancer-Detection-PCA-RandomForest
│
├── data.csv                     # Dataset file
├── cancer_detection_pca_rf.py   # Main code
├── README.md                    # Project documentation
└── plots/                       # (Optional) Visual outputs

## 💻 How to Run the Project

1. Clone the repository:
   - git clone https://github.com/<your-username>/Cancer-Detection-PCA-RandomForest.git


2. Navigate to the project folder:
   - cd Cancer-Detection-PCA-RandomForest


3. Install dependencies:
   - pip install pandas numpy matplotlib seaborn scikit-learn


4. Run the Python file:
   - python cancer_detection_pca_rf.py

## 📊 Results
Metric	Result
Accuracy	~90–98%
Model	Random Forest Classifier
Features Used	2 (via PCA)

## Outputs:

-PCA scatter plot showing clusters of cancer vs normal samples.
-Scree plot showing explained variance.
-Classification report printed in the console.

## 🧩 Key Learnings

-Handling and analyzing high-dimensional biological data.
-Understanding PCA for feature reduction and visualization.
-Building and evaluating ensemble models like Random Forest.

## Importance of normalization and dimensionality reduction in ML pipelines.

🚀 Future Enhancements

-Use t-SNE or LDA for advanced feature reduction.
-Apply deep learning models for better accuracy.
-Create a web interface (Streamlit/Flask) for real-time predictions.
-Perform hyperparameter tuning for improved performance.

👩‍💻 Author
Nilam Sudarshi
