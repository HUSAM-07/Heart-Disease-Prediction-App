import streamlit as st
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from matplotlib.cm import rainbow
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Suppressing warnings
warnings.filterwarnings('ignore')

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
ax.set_xticks([i for i in range(1, 21)])
ax.set_xlabel('Number of Neighbors (K)')
ax.set_ylabel('Scores')
ax.set_title('K Neighbors Classifier scores for different K values')
st.pyplot(fig)
st.set_option('deprecation.showPyplotGlobalUse', False)