import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn import tree
from sklearn.tree import plot_tree
import pickle

st.write("# Simple Iris Flower Prediction App")
st.write("This app predicts the **Iris flower** type!")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

modeldt = pickle.load(open("irisdt6.h5", "rb")) #rb: read binary

prediction = modeldt.predict(df)
prediction_proba = modeldt.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(y.unique())

st.subheader('Prediction')
st.write(prediction)
