import streamlit as st
import joblib

# Load saved components
model = joblib.load('model (4).pkl')
vectorizer = joblib.load('vectorizer (4).pkl')
label_encoder = joblib.load('label_encoder.pkl')

# App title
st.title("Drug Review Condition Predictor")

# User input
user_input = st.text_area("Enter a patient's review below:")

# Predict button
if st.button("Predict Condition"):
    if not user_input.strip():
        st.warning("Please enter a review to predict the condition.")
    else:
        # Vectorize the input
        X_new = vectorizer.transform([user_input])

        # Check if the input contains known vocabulary
        if X_new.nnz == 0:  # nnz = non-zero features
            st.error("Review contains no recognizable words. Try rephrasing.")
        else:
            # Predict
            pred_class = model.predict(X_new)[0]
            predicted_condition = label_encoder.inverse_transform([pred_class])[0]

            # Display result
            st.success(f"Predicted Condition: **{predicted_condition}**")
