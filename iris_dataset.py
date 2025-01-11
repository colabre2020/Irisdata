
# Author: SP

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Loading the Iris dataset

def load_iris_data():
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target_names[iris.target]
    return df, iris

# creating the function for the Streamlit UI layout

def app():
    # Title of the app
    
    st.title("Interactive ML with Streamlit & Plotly")

    # Left Pane for Dataset and Task Selection
    
    st.sidebar.header("Select Dataset and Task")

    # Select Dataset (currently only Iris dataset)
    
    dataset = st.sidebar.selectbox("Select Dataset", ["Iris"])

    # Select Task (EDA, Classification, Clustering, Dimensionality Reduction)
    
    task = st.sidebar.selectbox("Select Task", ["EDA", "Classification", "Clustering", "Dimensionality Reduction", "Model Evaluation"])

    # Load Dataset (for now, only Iris)
    
    if dataset == "Iris":
        df, iris = load_iris_data()

    # Display dataset overview in main area
    
    st.write(f"### {dataset} Dataset")
    st.write("Here is a preview of the dataset:")
    st.write(df.head())

    # Task-based functionalities
    
    if task == "EDA":
        st.header("Exploratory Data Analysis (EDA)")
        # Pairplot for visualization (using Plotly)
        st.subheader("Pairplot of features")
        fig = px.scatter_matrix(df, dimensions=iris.feature_names, color="species", title="Iris Dataset Pairplot")
        st.plotly_chart(fig)

        # Correlation Heatmap using Plotly
        
        st.subheader("Correlation Heatmap")
        corr_matrix = df.drop('species', axis=1).corr()
        fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Heatmap")
        st.plotly_chart(fig)

    elif task == "Classification":
        st.header("Classification (K-Nearest Neighbors)")
        # Prepare data
        X = iris.data
        y = iris.target

        # Split data into train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a KNN classifier
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)

        # Make predictions
        y_pred = knn.predict(X_test)

        # Display results
        st.subheader("Model Evaluation")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

    elif task == "Clustering":
        st.header("Clustering (K-Means)")
        # Prepare data
        X = iris.data

        # Apply K-Means clustering
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X)

        # Display results using Plotly scatter plot
        
        st.subheader("K-Means Clustering Result")
        fig = px.scatter(x=X[:, 0], y=X[:, 1], color=clusters, labels={'x': 'Sepal Length', 'y': 'Sepal Width'}, title="K-Means Clustering of Iris Dataset")
        st.plotly_chart(fig)

    elif task == "Dimensionality Reduction":
        st.header("Dimensionality Reduction (PCA)")
        # Apply PCA to reduce to 2D
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(iris.data)

        # Display results using Plotly scatter plot
        st.subheader("PCA Result (2D)")
        fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=iris.target, labels={'x': 'Principal Component 1', 'y': 'Principal Component 2'}, title="PCA of Iris Dataset")
        st.plotly_chart(fig)

    elif task == "Model Evaluation":
        st.header("Model Evaluation")
        # Prepare data
        X = iris.data
        y = iris.target

        # Split data into train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a KNN classifier
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)

        # Make predictions
        y_pred = knn.predict(X_test)

        # Display metrics
        st.subheader("Accuracy")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

# Run the app
if __name__ == "__main__":
    app()
