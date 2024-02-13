import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

model = pickle.load(open('model.pkl', 'rb'))

train_data = pd.read_csv('train_data.csv')  
feature_names = train_data.columns.drop('Is high risk')

user_input = {}

encoder = LabelEncoder()

for feature in feature_names:
    input_value = st.text_input(f"Enter {feature}")
    if isinstance(input_value, str):
        input_value = encoder.fit_transform([input_value])[0]
    user_input[feature] = input_value

if st.button('Predict'):
    user_input_df = pd.DataFrame([user_input])

    prediction = model.predict(user_input_df)

    # Display the prediction
    if prediction[0] == 1:
        st.write('Credit Card Not Approved')
    else:
        st.write('Credit Card Approved')