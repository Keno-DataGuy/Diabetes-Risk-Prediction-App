# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:55:59 2023

@author: KENO
"""

# importing the necessary library
import pandas as pd
import streamlit as st
import pickle

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the one-hot encoded mapping
with open('pre-process.pkl', 'rb') as file:
    ohe_process = pickle.load(file)
    ohe_process = ohe_process.drop('class_Negative', axis='columns')
    ohe_process = ohe_process.drop('class_Positive', axis='columns')
    
def preprocess_input(input_data):
    # Convert the input data into a DataFrame
    df = pd.DataFrame([input_data])
    
    # Get the column names with object data type
    object_columns = df.select_dtypes(include='object').columns

    # Convert the columns to categorical
    df[object_columns] = df[object_columns].astype('category')

    # Apply one-hot encoding using the loaded mapping
    df_encoded = pd.get_dummies(df).reindex(columns=ohe_process.columns, fill_value=0)

    return df_encoded

def main():
    st.title("Diabetes Risk Prediction App :hospital: :male-doctor:")

    # Collect user inputs
    input_data = {}
   
    with st.container():
        st.write("Patient Bio data")
        # input fields
        input_data['Age'] = st.number_input("Enter your age:", min_value=18, max_value=100)
        input_data['Gender'] = st.selectbox("Select your gender:", ['Male', 'Female'])
    
    
    
    input_data['Polyuria'] = st.selectbox("Frequent Urination with passage of large volume (polyuria):", ['No', 'Yes'])
    input_data['Polydipsia'] = st.selectbox("Excessive thirst (Polydipsia):", ['No', 'Yes'])
    input_data['sudden weight loss'] = st.selectbox("Sudden weight loss:", ['No', 'Yes'])
    input_data['weakness'] = st.selectbox(" Signs of Weakness:", ['No', 'Yes'])
    input_data['Polyphagia'] = st.selectbox(" Consumption of excessive amounts of food (Polyphagia):", ['No', 'Yes'])
    input_data['Genital thrush'] = st.selectbox(" Signs of yeast infection (Genital thrush):", ['No', 'Yes'])
    input_data['visual blurring'] = st.selectbox(" Visual blurring:", ['No', 'Yes'])
    input_data['Itching'] = st.selectbox(" Experiences itching:", ['No', 'Yes'])
    input_data['Irritability'] = st.selectbox(" Experiences irritability:", ['No', 'Yes'])
    input_data['delayed healing'] = st.selectbox(" Symptoms of delayed healing:", ['No', 'Yes'])
    input_data['partial paresis'] = st.selectbox(" partial weaking of muscles (partial paresis):", ['No', 'Yes'])
    input_data['muscle stiffness'] = st.selectbox(" Symptoms of muscle stiffness:", ['No', 'Yes'])
    input_data['Alopecia'] = st.selectbox(" Hair loss (alopecia):", ['No', 'Yes'])
    input_data['Obesity'] = st.selectbox("Obesity:", ['No', 'Yes'])


    # Preprocess the user inputs
    processed_input = preprocess_input(input_data)
    

    # Make predictions using the loaded model
    if st.button('Diagnose'):
        prediction = model.predict(processed_input)
        
        # Display the prediction to the user
        if prediction == 0:
            st.success("The Patient is not at risk for Diabetes")
        else:
            st.success("The Patient is at risk for Diabetes")
        
if __name__ == "__main__":
    main()
