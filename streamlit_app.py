import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st

st.title('Heart Disease Prediction App')

sex = st.selectbox('Sex', ('Male', 'Female'))
cp = st.selectbox('Chest Pain Type', ('Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'))
fbs = st.selectbox('Fasting Blood Sugar', ('> 120 mg/dl', '< 120 mg/dl'))
restecg = st.selectbox('Resting ECG', ('Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'))
exang = st.selectbox('Exercise-induced Angina', ('Yes', 'No'))
slope = st.selectbox('Slope of the Peak Exercise ST Segment', ('Upsloping', 'Flat', 'Downsloping'))
ca = st.selectbox('ST Segment Number Abnormal', ('0', '1', '2', '3', '4 or more'))
thal = st.selectbox('Thalassemia', ('fixed defect', 'reversable defect', 'normal'))
age = st.slider('Age', 20, 100, 30)
trestbps = st.slider('Resting Blood Pressure', 60, 250, 90)
chol = st.slider('Serum Cholestoral in mg/dl', 100, 600, 200)
thalach = st.slider('Maximum Heart Rate Achieved', 50, 200, 120)
oldpeak = st.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 8.0, 1.0)


def predict_heart_rate(sex, cp, fbs, restecg, exang, slope, ca, thal, age, trestbps, chol, thalach, oldpeak):
    feature_vector = np.array([sex, cp, fbs, restecg, exang, slope, ca, thal, age, trestbps, chol, thalach, oldpeak])
    feature_vector = feature_vector.reshape(1, -1)
    prediction = knn_classifier.predict(feature_vector)
    return prediction[0]

# Load your data into a DataFrame
df = pd.read_csv('dataset.csv')

# Remove rows with non-numeric values in 'target' column
df = df[df['target'].apply(lambda x: str(x).isnumeric())]

# Convert the 'target' column to numeric type
df['target'] = pd.to_numeric(df['target'])


# Load the dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data', header=None)

# Assign the features and the target to the corresponding variables
X = df.iloc[:, :-1] # Features
y = df.iloc[:, -1] # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the kNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Train the kNN classifier
knn_classifier.fit(X_train, y_train)

# Now, the 'X' variable is defined and can be used to fit the kNN classifier

# Load the trained KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X, y)

# Get user input
user_input = {'sex': [sex],
              'cp': [cp],
              'fbs': [fbs],
              'restecg': [restecg],
              'exang': [exang],
              'slope': [slope],
              'ca': [ca],
              'thal': [thal],
              'age': [age],
              'trestbps': [trestbps],
              'chol': [chol],
              'thalach': [thalach],
              'oldpeak': [oldpeak]}

# Create a dataframe
user_df = pd.DataFrame(user_input)

# Convert categorical values to numerical values
user_df = user_df.replace({'sex': {'Male': 1, 'Female': 0},
                           'cp': {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3},
                           'fbs': {'> 120 mg/dl': 1, '< 120 mg/dl': 0},
                           'restecg': {'Normal': 0, 'ST-T wave abnormality': 1, 'Left ventricular hypertrophy': 2},
                           'exang': {'Yes': 1, 'No': 0},
                           'slope': {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2},
                           'ca': {'0': 0, '1': 1, '2': 2, '3': 3, '4 or more': 4},
                           'thal': {'fixed defect': 0, 'reversable defect': 1, 'normal': 2}}
                          )

# Print the prediction
prediction = predict_heart_rate(sex, cp, fbs, restecg, exang, slope, ca, thal, age, trestbps, chol, thalach, oldpeak)
st.write('The predicted probability of heart disease is:', prediction)

