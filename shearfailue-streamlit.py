import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import streamlit as st
from PIL import Image

st.write("""
# Failure mode and effects analysis of RC Circular members
This app predicts the **Failure mode and effects analysis of RC Circular members based on machinelearning**!
Data obtained from the (Mangalathu, Sujith, Seong-Hoon Hwang, and Jong-Su Jeon. "Failure mode and effects analysis of RC members based 
on machine-learning-based SHapley Additive exPlanations (SHAP) approach." Engineering Structures 219 (2020): 110927.)
***
""")
st.write('---')
image=Image.open(r'Shear-failure-circular.png')
st.image(image, use_column_width=True)
req_col_names = ["a_over_D", "axial_load_ratio", " longitudinal_reinforcement_index", " transverse_reinforcement_index ", "Failure_mode)" ]
def get_input_features():
    a_over_D = st.sidebar.slider('a_over_D', 1.0,10.0,1.5)
    axial_load_ratio = st.sidebar.slider('axial_load_ratio', 0.0,0.81,0.5)
    longitudinal_reinforcement_index = st.sidebar.slider('longitudinal_reinforcement_index',0.07,0.75,0.5)
    transverse_reinforcement_index = st.sidebar.slider('(transverse_reinforcement_index', 0.034,3.454,0.050)


    data_user = {'(a/D)': a_over_D,
            'axial_load_ratio)': axial_load_ratio,
            'longitudinal_reinforcement_index': longitudinal_reinforcement_index,
            'transverse_reinforcement_index': transverse_reinforcement_index,}
    features = pd.DataFrame(data_user, index=[0])
    return features

df = get_input_features()
# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')
import pickle
# Reads in saved classification model
load_clf = pickle.load(open('rfrshearfailure.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)



st.subheader('Prediction Probability')
st.write(prediction_proba)
st.subheader('Prediction')
penguins_species = np.array(['Flexure','Flexure-shear','Shear'])
st.write(penguins_species[prediction])
st.write(prediction)
st.write('---')
st.header('Confusion matrix with accuracy=0.91')
image3=Image.open(r'Figure_3.png')
st.image(image3, use_column_width=True)