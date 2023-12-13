import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

try:
    import sklearn
except ModuleNotFoundError:
    print("Please install the 'scikit-learn' library before running this script. You can do this by running the following command:")
    print("\n pip install -U scikit-learn \n")
    exit()

# The rest of your code goes here

df = pd.read_csv('dataset.csv')

# Check the features in the dataframe
print("Features: ", df.columns)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Splitting the data into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Define the model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Create an interface for the user to input their details
with st.form(key='user_details'):
    age = st.slider('Age', 1, 100, 30)
    sex = st.selectbox('Sex', [0, 1])
    cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3])
    trestbps = st.slider('Resting Blood Pressure', 50, 250, 100)
    chol = st.slider('Serum Cholestoral', 50, 300, 150)
    fbs = st.selectbox('Fasting Blood Sugar', [0, 1])
    restecg = st.selectbox('Resting Electrocardiographic Results', [0, 1, 2])
    thalach = st.slider('Maximum Heart Rate Achieved', 50, 200, 120)
    exang = st.selectbox('Exercise-Induced Angina', [0, 1])
    oldpeak = st.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 4.0, 0.5)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', [0, 1, 2])
    ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', [0, 1, 2, 3])
    thal = st.selectbox('Thalassemia', [0, 1, 2, 3])
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    input_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
    input_data = sc.transform(input_data)
    prediction = knn.predict(input_data)
    if prediction == 1:
        st.write('Heart rate is above 100 bpm.')
    else:
        st.write('Heart rate is not above 100 bpm.')