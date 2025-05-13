# CODESOFT

Projects done as part of CodeSoft Internship
# TASK 1

# movie-genre-prediction
A machine learning project that predicts the genre of a movie based on its plot summary using text vectorization (TF-IDF) and Logistic Regression.

# Dataset
Files Used: train_data.txt, test_data.txt, test_data_solution.txt
The dataset contains movie plot summaries and their corresponding genres.
Tech Stack / Libraries
# Python
Scikit-learn (for TF-IDF, Logistic Regression, Model Evaluation)
Pandas (for data handling)
# Google Colab (for training the model)
How to Run the Project
Upload the dataset files (train_data.txt, test_data.txt, test_data_solution.txt) to your Colab environment.
Install necessary Python packages if not already installed:
run it on the google colab

# TASK 2
# Credit Card Fraud Detection 

This project aims to detect fraudulent credit card transactions using machine learning models like Logistic Regression, Decision Tree, and Random Forest. The dataset used is highly imbalanced, and we use SMOTE to balance it before training.


##  Machine Learning Models Used

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier



##  Technologies and Tools

- Python
- Scikit-learn
- Pandas, NumPy
- Matplotlib & Seaborn
- SMOTE (from imbalanced-learn)

## Evaluation Metrics

- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)
- ROC-AUC Score
- ROC Curve Plot

---

## Key Steps

1. Load and explore the dataset
2. Preprocess and scale features
3. Handle class imbalance using SMOTE
4. Train multiple ML models
5. Evaluate performance and visualize results

## Files

- `Task_2(CREDIT_CARD_FRAUD_DETECTION).ipynb` → Main Jupyter Notebook
- `README.md` → Project overview

---

## Output Sample

<img src="https://i.imgur.com/2bGgOVZ.png" width="500" alt="ROC Curve Sample" />

## Conclusion

Random Forest performed the best among the three models, with a higher accuracy and ROC-AUC score. This notebook can be extended further by applying deep learning models or anomaly detection techniques.
