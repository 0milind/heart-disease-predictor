
 ❤️ Heart Disease Prediction Using Machine Learning

This project is a machine learning-based predictive system designed to identify the presence of heart disease in patients using clinical health parameters. It includes a complete pipeline from data preprocessing and model training to evaluation and deployment using a simple Streamlit frontend.

## 📊 Project Overview

- **Goal**: Predict the presence of heart disease based on input features like age, cholesterol, blood pressure, etc.
- **Tech Stack**: Python, Scikit-learn, XGBoost, Keras, Pandas, Matplotlib, Streamlit
- **Models Used**:
  - Logistic Regression
  - Decision Tree
  - Random Forest 🌲
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
  - XGBoost ⚡
  - Neural Networks 🧠

## 🗂️ Project Structure

```
├── data/
│   └── heart.csv                  # UCI heart disease dataset
├── notebooks/
│   ├── Heart_disease_prediction.ipynb
│   └── ml_final_complete.ipynb
├── model/
│   └── best_model.pkl             # Serialized trained model
├── app/
│   └── streamlit_app.py           # Interactive web frontend
├── UML/
│   └── heart_disease_uml.png      # UML diagram
├── results/
│   └── model_accuracy_chart.png   # Accuracy comparison graph
├── README.md
└── requirements.txt
```

 ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit App**
   ```bash
   streamlit run app/streamlit_app.py
   ```

## 🧪 Model Evaluation

| Algorithm            | Accuracy (%) | Precision | Recall | F1 Score |
|----------------------|--------------|-----------|--------|----------|
| Logistic Regression  | 85.25        | 0.85      | 0.85   | 0.85     |
| Naive Bayes          | 85.25        | 0.86      | 0.85   | 0.85     |
| SVM (Linear)         | 81.97        | 0.82      | 0.82   | 0.82     |
| KNN                  | 67.21        | 0.68      | 0.67   | 0.67     |
| Decision Tree        | 81.97        | 0.82      | 0.82   | 0.82     |
| Random Forest        | 95.08        | 0.95      | 0.95   | 0.95     |
| XGBoost              | 85.25        | 0.85      | 0.85   | 0.85     |
| Neural Network       | 80.33        | 0.81      | 0.80   | 0.80     |

## 🧠 Features Used

- Age
- Sex
- Chest pain type (cp)
- Resting blood pressure (trestbps)
- Serum cholesterol (chol)
- Fasting blood sugar (fbs)
- Resting ECG results (restecg)
- Maximum heart rate (thalach)
- Exercise-induced angina (exang)
- Oldpeak
- Slope of peak exercise ST segment
- Number of major vessels (ca)
- Thalassemia (thal)

## 📈 Performance Insights

- **Random Forest** had the highest overall accuracy.
- **Neural Networks** performed well but needed careful tuning.
- **PCA** was used in one notebook to reduce dimensionality without sacrificing performance.

## 🖼 UML Diagram

![UML Diagram](UML/heart_disease_uml.png)

## 👨‍💻 Contributors

- Milind Mohapatra – Developer & Analyst
- Based on the project implementation by [Shreekant Gosavi](https://github.com/gosavi-2001)
- Data Source: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


