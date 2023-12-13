import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
heart_data = pd.read_csv('dataset.csv')

# Page layout
st.set_page_config(
    page_title="Heart Disease Prediction App",
    page_icon="❤️",
    layout="wide",
)

# Title
st.title("Heart Disease Prediction App")
# Sidebar
st.sidebar.subheader("Dataset Exploration")

# Show dataset
if st.sidebar.checkbox("Show Dataset"):
    st.dataframe(heart_data)

# Explore correlation between features
st.sidebar.subheader("Correlation Matrix Heatmap")

# Correlation matrix
correlation_matrix = heart_data.corr()
top_corr_features = correlation_matrix.index
plt.figure(figsize=(20, 20))
sns.heatmap(heart_data[top_corr_features].corr(), annot=True, cmap="RdYlGn")
st.pyplot()

# Data preprocessing
st.sidebar.subheader("Data Preprocessing")

# Dummies
heart_data = pd.get_dummies(heart_data, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

# Feature scaling
scaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
heart_data[columns_to_scale] = scaler.fit_transform(heart_data[columns_to_scale])
# Model building and evaluation
st.sidebar.subheader("Model Building and Evaluation")

# KNN model
knn_scores = []
X = heart_data.drop('target', axis=1)
y = heart_data['target']
for k in range(1, 21):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn_classifier, X, y, cv=10)
    knn_scores.append(score.mean())

# Plot the scores for different K values
st.subheader("K Neighbors Classifier scores for different K values")
fig, ax = plt.subplots()
ax.plot([k for k in range(1, 21)], knn_scores, color='red')
for i in range(1, 21):
    ax.text(i, knn_scores[i - 1], (i, knn_scores[i - 1]))
ax.set_xticks([k for k in range(1, 21)])
ax.set_xlabel('Number of Neighbors (K)')
ax.set_ylabel('Scores')
ax.set_title('K Neighbors Classifier scores for different K values')
st.pyplot(fig)
# User input and prediction
def predict_heart_rate():
    user_input = {
        'age': st.number_input('Age', min_value=20, max_value=80),
        'sex': st.selectbox('Sex', ['Male', 'Female']),
        # ... add input for other features ...
    }

    # Preprocess user input
    user_input_df = pd.DataFrame(user_input, index=[0])
    user_input_df = pd.get_dummies(user_input_df, columns=['sex'])

    # Apply scaling to numerical features
    user_input_df[columns_to_scale] = scaler.transform(user_input_df[columns_to_scale])

    # Predict heart rate
    prediction = knn_classifier.predict(user_input_df)[0]

    # Display prediction
    st.write(f"Predicted Heart Rate: {prediction:.2f} bpm")
    st.write(f"Note: This prediction is for informational purposes only and should not be used for medical diagnosis.")

# Button to trigger prediction
if st.button("Predict Heart Rate"):
    predict_heart_rate()
# Output and Visualization

def visualize_results(user_input_df, prediction):
    # Plot heart rate distribution
    heart_rates = heart_data['target']
    fig, ax = plt.subplots()
    ax.hist(heart_rates, bins=20, edgecolor='black')
    ax.axvline(prediction, color='red', linestyle='--', label=f"Predicted: {prediction:.2f} bpm")
    ax.set_xlabel('Heart Rate (bpm)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Heart Rate in Dataset')
    ax.legend()
    st.pyplot(fig)

    # Show nearest neighbors
    distances, indices = knn_classifier.kneighbors(user_input_df)
    nearest_neighbors = heart_data.iloc[indices[0]]
    st.subheader("Nearest Neighbors:")
    st.dataframe(nearest_neighbors)

# User Input and Prediction

def predict_heart_rate():
    global user_input_df
    user_input = {
        'age': st.number_input('Age', min_value=20, max_value=80),
        'sex': st.selectbox('Sex', ['Male', 'Female']),
        # ... add input for other features ...
    }

    # Preprocess user input
    user_input_df = pd.DataFrame(user_input, index=[0])
    user_input_df = pd.get_dummies(user_input_df, columns=['sex'])

    # Apply scaling to numerical features
    user_input_df[columns_to_scale] = scaler.transform(user_input_df[columns_to_scale])

    # Predict heart rate
    global prediction
    prediction = knn_classifier.predict(user_input_df)[0]

    # Display prediction
    st.write(f"Predicted Heart Rate: {prediction:.2f} bpm")
    st.write(f"Note: This prediction is for informational purposes only and should not be used for medical diagnosis.")

    # Run visualization function
    visualize_results(user_input_df, prediction)

# Button to trigger prediction
if st.button("Predict Heart Rate"):
    predict_heart_rate()

