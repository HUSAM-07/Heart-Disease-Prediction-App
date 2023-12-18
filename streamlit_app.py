import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

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

# App Settings and Title
st.title('Heart Rate Prediction using Machine Learning')

# Input explanations
explanations = {
    'age': 'The person’s age in years',
    'sex': 'The person’s sex (1 = male, 0 = female)',
    'cp': 'Chest pain type:\n0: asymptomatic\n1: atypical angina\n2: non-anginal pain\n3: typical angina',
    'trestbps': 'The person’s resting blood pressure (mm Hg on admission to the hospital)',
    'chol': 'The person’s cholesterol measurement in mg/dl',
    'fbs': 'The person’s fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)',
    'restecg': 'Resting electrocardiographic results:\n0: showing probable or definite left ventricular hypertrophy by Estes’ criteria\n1: normal\n2: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)',
    'thalach': 'The person’s maximum heart rate achieved',
    'exang': 'Exercise-induced angina (1 = yes; 0 = no)',
    'oldpeak': 'ST depression induced by exercise relative to rest',
    'slope': 'The slope of the peak exercise ST segment:\n0: downsloping\n1: flat\n2: upsloping',
    'ca': 'The number of major vessels (0–3)',
    'thal': 'A blood disorder called thalassemia\n0: NULL (dropped from the dataset previously)\n1: fixed defect (no blood flow in some part of the heart)\n2: normal blood flow\n3: reversible defect (a blood flow is observed but it is not normal)'
}

# Create an interface for the user to input their details
with st.form(key='user_details'):
    for feature in explanations:
        value = st.slider(explanations[feature], min_value=min(df[feature]), max_value=max(df[feature]), value=df[feature].median()) if feature != 'sex' and feature != 'fbs' and feature != 'exang' else st.selectbox(explanations[feature], options=[0, 1], index=df[feature].mode()[0])
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    input_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
    input_data = sc.transform(input_data)
    prediction = knn.predict(input_data)
    if prediction == 1:
        st.write('High Probability of presence of Heart Disease.')
    else:
        st.write('Low Probability of presence of Heart Disease.')
st.caption('This Application is Developed by Mohammed Zubair Ahmed & Mohammed Husamuddin')
