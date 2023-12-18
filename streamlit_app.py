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

# Create an interface for the user to input their details
with st.form(key='user_details'):
    age = st.slider('Age', 1, 100, 30)
    
    # Sex: 0 = female, 1 = male
    sex = st.selectbox('Sex (0 = female, 1 = male)', [0, 1])
    
    # Chest Pain Type
    cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3], format_func=lambda x: f'{x}: {"Asymptomatic" if x == 0 else "Atypical Angina" if x == 1 else "Non-Anginal Pain" if x == 2 else "Typical Angina"}')
    
    trestbps = st.slider('Resting Blood Pressure (mm Hg)', 50, 250, 100)
    chol = st.slider('Serum Cholestoral (mg/dl)', 50, 300, 150)
    
    # Fasting Blood Sugar: 0 = false, 1 = true
    fbs = st.selectbox('Fasting Blood Sugar (> 120 mg/dl, 1 = true; 0 = false)', [0, 1])
    
    # Resting Electrocardiographic Results
    restecg = st.selectbox('Resting Electrocardiographic Results', [0, 1, 2], format_func=lambda x: f'{x}: {"Probable/Definite LVH" if x == 0 else "Normal" if x == 1 else "ST-T Wave Abnormality"}')
    
    thalach = st.slider('Maximum Heart Rate Achieved', 50, 200, 120)
    
    # Exercise-Induced Angina: 0 = no, 1 = yes
    exang = st.selectbox('Exercise-Induced Angina (0 = no, 1 = yes)', [0, 1])
    
    oldpeak = st.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 4.0, 0.5)
    
    # Slope of the Peak Exercise ST Segment: 0 = downsloping, 1 = flat, 2 = upsloping
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', [0, 1, 2], format_func=lambda x: f'{x}: {"Downsloping" if x == 0 else "Flat" if x == 1 else "Upsloping"}')
    
    # Number of Major Vessels Colored by Fluoroscopy
    ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', [0, 1, 2, 3])
    
    # Thalassemia
    thal = st.selectbox('Thalassemia', [0, 1, 2, 3], format_func=lambda x: f'{x}: {"NULL" if x == 0 else "Fixed Defect" if x == 1 else "Normal Blood Flow" if x == 2 else "Reversible Defect"}')
    
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
