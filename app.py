import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

st.title("Heart Disease Prediction System")
st.write("Academic machine learning project using four selected UCI Heart Disease attributes.")

sex_label_encoder = LabelEncoder()
cp_label_encoder = LabelEncoder()

sex_label_encoder.fit(["Female", "Male"])
cp_label_encoder.fit([
    "Typical Angina",
    "Atypical Angina",
    "Non-anginal Pain",
    "Asymptomatic"
])

model = pickle.load(open("heart_disease_svc_model.pkl", "rb"))

age_input = st.number_input("Age", min_value=1, max_value=120, value=55)
sex_input = st.selectbox("Sex", ["Female", "Male"])
cp_input = st.selectbox(
    "Chest Pain Type",
    [
        "Typical Angina",
        "Atypical Angina",
        "Non-anginal Pain",
        "Asymptomatic"
    ]
)
thalach_input = st.number_input(
    "Maximum Heart Rate Achieved",
    min_value=50,
    max_value=250,
    value=150
)

if st.button("Predict"):
    user_input = pd.DataFrame({
        "age": [age_input],
        "sex": [sex_input],
        "cp": [cp_input],
        "thalach": [thalach_input]
    })

    user_input["sex"] = sex_label_encoder.transform(user_input["sex"])
    user_input["cp"] = cp_label_encoder.transform(user_input["cp"])

    prediction = model.predict(user_input)

    if prediction[0] == 1:
        st.error("Prediction: HEART DISEASE PRESENT")
    else:
        st.success("Prediction: NO HEART DISEASE")

st.caption("For academic demonstration only. This model is not a medical diagnostic tool.")
