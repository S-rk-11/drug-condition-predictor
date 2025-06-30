# app.py
import streamlit as st
import joblib

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
le = joblib.load('label_encoder.pkl')

st.title("Patient Condition Prediction from Drug Review")
user_input = st.text_area("Enter the patient's review:")

if st.button("Predict Condition"):
    if user_input.strip() == "":
        st.warning("Please enter a valid review.")
    else:
        X_new = vectorizer.transform([user_input])
        if X_new.nnz == 0:
            st.error("No known words in input. Try using more descriptive text.")
        else:
            pred_class = model.predict(X_new)[0]
            predicted_condition = le.inverse_transform([pred_class])_]()
