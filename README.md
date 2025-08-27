# 🎓 Student Grade Predictor

A Machine Learning web application that predicts a student's **final grade** based on academic performance and personal study habits.  
Built with **Python, Pandas, Scikit-learn, and Streamlit**, this project demonstrates **data preprocessing, feature engineering, model building, and deployment**.

---

## 🚀 Live Demo
👉 [Try the App](https://ahmadhussain99-student-grade--student-grade-predictorapp-ajqxdf.streamlit.app/)

---

## 📊 Dataset
- **Source:** UCI Machine Learning Repository – Student Performance Dataset  
- **Size:** 395 student records  
- **Features:** 33 attributes (academic + social factors)  
- **Target Variable:** Final Grade (0–20 scale)  

---

## 🛠️ Tech Stack
- **Python** – Core programming language  
- **Pandas** – Data analysis and preprocessing  
- **Scikit-learn** – Machine Learning (Linear Regression)  
- **Streamlit** – Interactive web app deployment  

---

## ⚙️ Features
✅ Load and explore dataset with Pandas  
✅ Preprocess and engineer features (handling object → numeric conversion)  
✅ Train and evaluate a **Linear Regression model**  
✅ Metrics used:
- R² Score: **0.78**
- MSE: **4.46**  
✅ User-friendly **Streamlit web app** to input values and get predictions  
✅ Deployed on **Streamlit Cloud**  

---

## 📸 Screenshots

### 🖥️ Web App
![App Screenshot](https://i.ibb.co/b6csJ2z/student-grade-app.png)

### 📈 Predicted vs Actual Grades
![Prediction Graph](https://i.ibb.co/8zmr2VY/predicted-vs-actual.png)

---

## 📂 Project Structure
Student-Grade-Predictor/
│
├── app.py # Flask/Streamlit app file
├── student_grade_predictor.pkl # Saved ML model
├── requirements.txt # Dependencies
├── templates/ # HTML templates (if using Flask)
│ └── index.html
├── static/ # CSS styles
│ └── style.css
└── dataset/
└── student-mat.csv # Dataset

---

## ⚡ Quick Start

Clone this repository:
```bash
git clone https://github.com/ahmadhussain99/student-grade-predictor.git
cd student-grade-predictor
Create and activate virtual environment:
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
streamlit run app.py
📌 Future Improvements

Add more advanced ML models (Random Forest, Gradient Boosting)

Feature selection & hyperparameter tuning

Batch predictions via CSV upload

Improved UI/UX for better visualization

👨‍💻 Author

Ahmad Hussain
📍 Fresher | Aspiring Software Developer | Data Science Enthusiast
🔗 LinkedIn
 | GitHub

⭐ If you found this project useful, don't forget to star the repo!
