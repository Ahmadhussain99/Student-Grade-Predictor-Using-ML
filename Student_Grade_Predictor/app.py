# app.py - Streamlit Student Grade Predictor with Batch CSV Support

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
MODEL_PATH = "student_grade_predictor.pkl"
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Student Grade Predictor", page_icon="🎓", layout="centered")

st.title("🎓 Student Grade Predictor")
st.markdown("Predict a student's **final grade** based on academic and behavioral features.")

# Tabs for Single & Batch prediction
tab1, tab2 = st.tabs(["📄 Single Prediction", "📂 Batch CSV Prediction"])

# =============================
# Single Prediction Tab
# =============================
with tab1:
    st.subheader("Single Student Prediction")

    g1 = st.number_input("First Period Grade (0-20)", min_value=0, max_value=20, step=1)
    g2 = st.number_input("Second Period Grade (0-20)", min_value=0, max_value=20, step=1)
    studytime = st.selectbox("Weekly Study Hours (coded 1-4)", [1, 2, 3, 4])
    failures = st.number_input("Number of Past Failures (0-4)", min_value=0, max_value=4, step=1)
    absences = st.number_input("Days Absent (0+)", min_value=0, step=1)

    if st.button("Predict Final Grade", key="single"):
        features = np.array([[g1, g2, studytime, failures, absences]])
        prediction = model.predict(features)[0]
        st.success(f"Predicted Final Grade: **{prediction:.2f} / 20**")

        # Performance message
        if prediction >= 15:
            st.markdown("✅ Excellent performance! Keep it up!")
        elif prediction >= 10:
            st.markdown("⚠️ Average performance, can be improved.")
        else:
            st.markdown("❌ Needs serious improvement.")

# =============================
# Batch CSV Prediction Tab
# =============================
with tab2:
    st.subheader("Batch Prediction for Multiple Students")
    st.markdown("Upload a **CSV file** with columns: `first_period_grade`, `second_period_grade`, `weekly_study_hours`, `past_failures`, `days_absent`.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)

            required_columns = [
                "first_period_grade",
                "second_period_grade",
                "weekly_study_hours",
                "past_failures",
                "days_absent",
            ]

            # Validate columns
            if not all(col in df.columns for col in required_columns):
                st.error(f"CSV must contain these columns: {required_columns}")
            else:
                # Predict for each row
                predictions = model.predict(df[required_columns])
                df["Predicted_Final_Grade"] = predictions.round(2)

                st.success("Batch prediction completed!")
                st.dataframe(df)

                # Download link
                csv_download = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv_download,
                    file_name="predicted_grades.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")

# Footer
st.markdown("---")
st.caption("Developed with ❤️ by Ahmad")
