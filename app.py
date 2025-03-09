import streamlit as st
import pandas as pd
import pickle

# Load the model
with open('titanic_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit app
st.title('Titanic Traveller Survival Prediction')

# Input fields
pclass = st.selectbox('Passenger Class', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.number_input('Age', min_value=0, max_value=100)
fare = st.number_input('Fare', min_value=0.0)
embarked = st.selectbox('Embarked', ['C', 'Q', 'S'])

# Convert input to DataFrame
sex = 0 if sex == 'male' else 1
embarked_q = 1 if embarked == 'Q' else 0
embarked_s = 1 if embarked == 'S' else 0

input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex],
    'Age': [age],
    'Fare': [fare],
    'Embarked_Q': [embarked_q],
    'Embarked_S': [embarked_s]
})

# Standardize numerical features
input_data[['Age', 'Fare']] = scaler.transform(input_data[['Age', 'Fare']])

# Make prediction
if st.button('Predict'):
    prediction = model.predict(input_data)
    survival = 'The Traveller Survived' if prediction[0] == 1 else 'The Traveller Did Not Survive'
    st.write(f'Prediction: {survival}')