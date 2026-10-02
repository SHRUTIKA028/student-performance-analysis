# 🎓 Student Performance Analysis

An interactive **Streamlit** web application that explores, visualises, and predicts student exam scores using machine-learning models trained on the **Student Performance Factors** dataset.

---

## 📸 Features

| Page | Description |
|------|-------------|
| **📊 Overview** | Dataset snapshot — shape, missing values, descriptive statistics |
| **📈 EDA** | Score distribution, correlation heatmap, category breakdowns, scatter plots with trendlines |
| **🤖 Prediction** | Train Random Forest, Gradient Boosting, or Linear Regression; view MAE / RMSE / R², actual-vs-predicted plot, residuals, and feature importance |
| **🔍 Predict My Score** | Fill in student details and get an instant predicted exam score with grade badge |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/Student_Performance_Analysis.git
cd Student_Performance_Analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📂 Project Structure

```
Student_Performance_Analysis/
├── app.py                        # Main Streamlit application
├── StudentPerformanceFactors.csv # Dataset
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## 🧰 Tech Stack

- **[Streamlit](https://streamlit.io/)** — UI framework
- **[Pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/)** — Data manipulation
- **[Plotly](https://plotly.com/python/)** — Interactive charts
- **[Matplotlib](https://matplotlib.org/) / [Seaborn](https://seaborn.pydata.org/)** — Static plots
- **[scikit-learn](https://scikit-learn.org/)** — ML models (Random Forest, Gradient Boosting, Linear Regression)

---

## 📊 Dataset

The dataset (`StudentPerformanceFactors.csv`) contains student-level features such as hours studied, attendance, parental involvement, access to resources, and more, with `Exam_Score` as the target variable.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
