import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import pickle
import numpy as np
import streamlit as st

#load the model
model= tf.keras.models.load_model('model.h5')

#load the pickle files
model= load_model('model.h5')


with open('one_hot_encoder_geo.pkl', 'rb') as file:
    one_hot_encoder_geo= pickle.load(file)

with open('label_encoder.pkl', 'rb') as file:
    label_encoder= pickle.load(file)

with open('Scaler.pkl', 'rb') as file:
    scaler= pickle.load(file)

# streamlit app
st.title('Customer Churn Prediction')

#Input Features

Geography= st.selectbox ('Geography', one_hot_encoder_geo.categories_[0])
Gender= st.selectbox ('Gender', ['Male', 'Female'])
Age= st.slider ('Age', 18, 100)
Tenure= st.slider( 'Tenure', 0, 10)
CreditScore=  st.number_input('CreditScore')
Balance= st.number_input('Balance')
NumOfProducts= st.slider('NumOfProducts', 1, 4)
HasCrCard= st.selectbox('HasCrCard', [0, 1])
IsActiveMember= st.selectbox('IsActiveMember', [0, 1])
EstimatedSalary = st.number_input('EstimatedSalary')

# Preprocess Input Features
input_data= pd.DataFrame({
'CreditScore': [CreditScore],
'Gender': [1 if Gender=='Male' else 0],	
'Age': [Age],	
'Tenure': [Tenure],
'Balance': [Balance],
'NumOfProducts': [NumOfProducts],
'HasCrCard': [HasCrCard],
'IsActiveMember': [IsActiveMember],
'EstimatedSalary': [EstimatedSalary]})

encoded_geo = one_hot_encoder_geo.transform([[Geography]]).toarray()
geo_encoded_df = pd.DataFrame(encoded_geo,columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))


input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df], axis=1)

input_data = input_data[scaler.feature_names_in_]

scaled_input= scaler.transform(input_data)

prediction= model.predict(scaled_input)
churn_proba= prediction[0][0]
st.write(f'Churn Probability: {churn_proba:.4f}')

if churn_proba>0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is unlikely to churn.")