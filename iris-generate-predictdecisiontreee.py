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
    sepal_length = st.sidebar.slider('Sepal length', 0.0,1.0,0.0)
    sepal_width = st.sidebar.slider('Sepal width', 0.0,1.0,0.0)
    petal_length = st.sidebar.slider('Petal length', 0.0,1.0,0.0)
    petal_width = st.sidebar.slider('Petal width', 0.0, 1.0,0.0)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

data = sns.load_dataset('iris')
Y = data.species.copy()

st.subheader('User Input parameters')
st.write(df)

modeldt = pickle.load(open("irisdt7.h5", "rb")) #rb: read binary

prediction = modeldt.predict(df)
prediction_proba = modeldt.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(Y.unique())

st.subheader('Prediction')
st.write(prediction)
if prediction == 0 :
 st.write('setosa')
elif prediction == 1 :
 st.write('versicolor')
elif prediction == 2 :
 st.write('virginica')
else :
 st.write('enter the value correctly')
        
st.subheader('Prediction Probability')
st.write(prediction_proba)
